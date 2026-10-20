"""
Baseline knowledge retrieval: similarity search of the question over chunks.

StandardRAG, the reference point both prior works report. No graph is touched
at all: the corpus passages are embedded, the question is embedded, the top-k
nearest passages are returned. It is what retrieval looks like when the
hypergraph contributes nothing, so the gap between it and any graph method is
what the graph is worth.

Reads the chunk file produced by graph_construction, writes a knowledge file
that text_generation consumes.

    python -m knowledge_retrieval.baselines.chunk_simsearch \
        --chunks ../output/Mix/chunks.json \
        --questions ../datasets/Mix/questions.json \
        --out ../output/Mix/retrieval/standardrag.json

OUTPUT CONTRACT

Same shape as every other retriever, with one section filled:

    "knowledge": {"hyperedges": [],       # empty, by construction
                  "chunks":     [{"text", "score", "rank",
                                  "chunk_id", "chunk_index"}, ...]}

The `hyperedges` key stays present and empty rather than absent, so a consumer
that reads it finds the same contract here as everywhere else. Running
generation with `--knowledge all` therefore shows the model only the chunks,
which is exactly what StandardRAG is.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 10

Q_FIELDS = ("question", "query", "text", "input")
CHUNK_FIELDS = ("content", "text", "chunk", "passage", "body")


# ------------------------------------------------------------------ utils
def first_field(obj, fields, default=""):
    for f in fields:
        if isinstance(obj, dict) and obj.get(f):
            v = obj[f]
            return " ".join(map(str, v)) if isinstance(v, list) else str(v)
    return default


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def load_chunks(path):
    """[{chunk_id, chunk_index, text}] in corpus order. Both shapes are read."""
    data = json.loads(Path(path).read_text())
    items = (list(data.items()) if isinstance(data, dict)
             else [((c.get("chunk_id", str(i)) if isinstance(c, dict) else str(i)), c)
                   for i, c in enumerate(data)])
    out = []
    for i, (cid, c) in enumerate(items):
        text = c if isinstance(c, str) else first_field(c, CHUNK_FIELDS, "")
        text = text.strip()
        if text:
            out.append({"chunk_id": str(cid), "chunk_index": i, "text": text})
    if not out:
        raise SystemExit(f"no chunk texts in {path}")
    return out


# --------------------------------------------------------------- the index
class ChunkIndex:
    """Embeddings over chunk texts, with the cache keyed by content.

    Identical passages are embedded once and merged, so a corpus with repeated
    boilerplate does not spend the budget on the same vector many times over
    and does not fill a neighbour list with copies of one passage.
    """

    def __init__(self, records, model_name=DEFAULT_MODEL, backend="exact"):
        merged = {}
        for r in records:
            slot = merged.setdefault(norm(r["text"]), {"text": r["text"], "records": []})
            slot["records"].append(r)
        self.units = list(merged.values())
        self.model_name, self.backend = model_name, backend
        self.emb = self._encode([u["text"] for u in self.units])
        if backend == "hnsw":
            self._build_hnsw()

    def _encode(self, texts):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.model_name)
        return np.asarray(model.encode(texts, normalize_embeddings=True,
                                       show_progress_bar=len(texts) > 200),
                          dtype=np.float32)

    def _build_hnsw(self):
        from hnsw import HNSW
        self.hnsw = HNSW(space="cosine", M=16, ef_construction=200)
        for uid, vec in enumerate(self.emb):
            self.hnsw.insert(vec, {"uid": uid})

    def encode_queries(self, texts):
        return self._encode(list(texts))

    def search(self, vec, k):
        """[(unit index, cosine similarity)], best first."""
        if self.backend == "hnsw":
            from hnsw import Node
            hits = self.hnsw.knn_search(Node(vec, 0), k)
            uids = [n.metadata["uid"] for _, n in hits]
            scored = sorted(((u, float(self.emb[u] @ vec)) for u in uids),
                            key=lambda x: -x[1])
            return scored[:k]
        sims = self.emb @ vec
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        return [(int(i), float(sims[i])) for i in top[np.argsort(-sims[top])]]

    # ---- cache, so re-running a retriever costs no embedding time
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"units": self.units, "emb": self.emb,
                         "model": self.model_name, "version": 1}, f)

    @classmethod
    def load(cls, path, model_name, backend, fingerprint):
        if not Path(path).exists():
            return None
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
        except Exception:
            return None          # unreadable cache: rebuild rather than die
        if state.get("model") != model_name or state.get("version") != 1:
            return None
        obj = cls.__new__(cls)
        obj.units, obj.emb = state["units"], state["emb"]
        obj.model_name, obj.backend = model_name, backend
        if len(obj.units) != fingerprint:
            return None                      # corpus changed since the cache
        if backend == "hnsw":
            obj._build_hnsw()
        return obj


# ------------------------------------------------------------------- main
def retrieve(index, questions, top_k, min_score):
    vecs = index.encode_queries(first_field(q, Q_FIELDS, f"(question {i})")
                                for i, q in enumerate(questions))
    results = []
    for qi, (q, vec) in enumerate(zip(questions, vecs)):
        hits = []
        for rank, (uid, score) in enumerate(index.search(vec, top_k), 1):
            if score < min_score:
                continue
            unit = index.units[uid]
            rec = unit["records"][0]
            hits.append({"text": unit["text"], "score": round(score, 4),
                         "rank": rank, "chunk_id": rec["chunk_id"],
                         "chunk_index": rec["chunk_index"],
                         "duplicate_of": [r["chunk_id"] for r in unit["records"][1:]]})
        results.append({
            "qi": qi,
            "question": first_field(q, Q_FIELDS, f"(question {qi})"),
            "knowledge": {"hyperedges": [], "chunks": hits},
            "trace": {"n_hyperedges": 0, "n_chunks": len(hits)},
        })
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--chunks", required=True, help="chunks.json of the corpus")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="drop chunks below this cosine similarity")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--index", choices=["exact", "hnsw"], default="exact",
                    help="exact is a single matrix product and has no approximation "
                         "error; hnsw matches the production retriever")
    ap.add_argument("--cache", default=None,
                    help="embedding cache (default: alongside --out)")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--run-name", default="standardrag")
    args = ap.parse_args()

    t0 = time.time()
    records = load_chunks(args.chunks)
    questions = json.loads(Path(args.questions).read_text())
    print(f"corpus     : {len(records)} chunks")
    print(f"questions  : {len(questions)}")

    cache = Path(args.cache or (Path(args.out).parent /
                                f".{args.run_name}_index.pkl"))
    index = (None if args.rebuild else
             ChunkIndex.load(cache, args.model, args.index,
                             len({norm(r["text"]) for r in records})))
    if index is None:
        index = ChunkIndex(records, args.model, args.index)
        index.save(cache)
        print(f"index      : built, {len(index.units)} distinct passages")
    else:
        print(f"index      : cached, {len(index.units)} distinct passages")

    results = retrieve(index, questions, args.top_k, args.min_score)
    nc = [len(r["knowledge"]["chunks"]) for r in results]
    tops = [r["knowledge"]["chunks"][0]["score"] for r in results
            if r["knowledge"]["chunks"]]
    summary = {"questions": len(results),
               "mean_knowledge": 0,
               "mean_chunks": round(float(np.mean(nc)), 2) if nc else 0,
               "empty": sum(1 for x in nc if x == 0),
               "mean_top_score": round(float(np.mean(tops)), 4) if tops else None,
               "seconds": round(time.time() - t0, 1)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": {"run": args.run_name, "method": "chunk-simsearch",
                    "graph": None, "questions": args.questions,
                    "chunks": args.chunks, "knowledge": "chunks",
                    "model": args.model, "index": args.index,
                    "top_k": args.top_k, "chunk_top_k": args.top_k,
                    "min_score": args.min_score,
                    "knowledge_sections": ["chunks"],
                    "llm_calls_per_question": 0},
         "summary": summary, "results": results}, indent=2, ensure_ascii=False))

    print(f"chunks     : {summary['mean_chunks']} per question, "
          f"{summary['empty']} empty, top score {summary['mean_top_score']}")
    print(f"hyperedges : none, the graph is not used")
    print(f"wrote {out}  ({summary['seconds']}s)")


if __name__ == "__main__":
    main()