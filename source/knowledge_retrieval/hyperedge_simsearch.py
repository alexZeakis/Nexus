"""
Baseline knowledge retrieval: similarity search of the question over hyperedges.

The simplest thing a hypergraph can do: embed every hyperedge description,
embed the question, return the top-k nearest. No planning, no traversal, no
joins, no LLM call. It is the floor that Nexus retrieval has to beat, and the
condition every other retriever is compared against.

Reads the graph produced by graph_construction, writes a knowledge file that
text_generation consumes.

    python -m knowledge_retrieval.baselines.hyperedge_simsearch \
        --graph ../output/Mix/graph/graph.graphml \
        --questions ../datasets/Mix/questions.json \
        --chunks ../output/Mix/chunks.json \
        --out ../output/Mix/retrieval/simsearch.json

OUTPUT CONTRACT  (every retriever, baseline or ours, writes this shape)

    {
      "config": {"run", "method", "graph", "model", "top_k", ...},
      "summary": {"questions", "mean_knowledge", "mean_chunks", ...},
      "results": [
        {"qi": 0,
         "question": "...",
         "knowledge": {
            "hyperedges": [                  # ranked, best first
               {"text": "...",               # what generation reads
                "score": 0.71,               # method's own confidence, [0,1]
                "rank": 1,
                "node_id": "hyperedge-12",   # provenance in the graph
                "participants": ["Thompson", "Cape Fear"],
                "chunk_ids": ["chunk-ab12"],
                "chunk_indices": [27]}       # only when --chunks is given
            ],
            "chunks": [                      # only when --chunks is given
               {"text": "...",               # the passage, verbatim
                "score": 0.71,               # its best hyperedge's score
                "rank": 1,
                "chunk_id": "chunk-ab12",
                "chunk_index": 27,
                "from_hyperedges": 2,        # how many hits came out of it
                "node_ids": ["hyperedge-12", "hyperedge-55"]}
            ]},
         "trace": {}}                        # method-specific, free-form
      ]
    }

`text` and `score` are the only fields text_generation requires; the rest is
provenance, so a later method can add its own `trace` (a join tree, a reasoning
path) without breaking the consumer.

TWO SECTIONS OF KNOWLEDGE

The chunks are not retrieved separately and cost no extra search. Each one is
a passage a returned hyperedge was extracted from, inheriting that hyperedge's
similarity, so the section only widens each hit back to the text it came from.
That is the kind of context HyperGraphRAG hands its reader, so writing both
sections is what makes the comparison a comparison of retrieval rather than of
how much text each method happens to supply.

Both sections are written whenever --chunks is given. `generate_answers
--knowledge {hyperedges,chunks,all}` decides which the model actually sees, so
one retrieval file serves every variant and narrowing it here is only useful
to keep the file small.
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

CHUNK_SEPS = re.compile(r"<SEP>|\s\|\s|\|")
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
    """chunk_id -> (position in the corpus, text).

    Both chunk file shapes are read: a dict keyed by chunk id, and a list of
    records with a `chunk_id`. The position makes results comparable across
    graphs; the text is what goes in the `chunks` section of the contract.
    """
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text())
    items = (list(data.items()) if isinstance(data, dict)
             else [((c.get("chunk_id", str(i)) if isinstance(c, dict) else str(i)), c)
                   for i, c in enumerate(data)])
    out = {}
    for i, (cid, c) in enumerate(items):
        text = c if isinstance(c, str) else first_field(c, CHUNK_FIELDS, "")
        out[str(cid)] = (i, text.strip())
    return out


def chunks_of(hyperedges, chunks, top_k=None):
    """The source chunks of the retrieved hyperedges, ranked.

    A chunk is as relevant as the best hyperedge that came out of it, so it
    inherits that score; ties go to the chunk more hyperedges point at.
    """
    best = {}
    for item in hyperedges:
        for cid in item.get("chunk_ids", []):
            position, text = chunks.get(str(cid), (None, ""))
            if not text:
                continue
            entry = best.get(cid)
            if entry is None:
                entry = best[cid] = {"text": text, "chunk_id": str(cid),
                                     "score": 0.0, "from_hyperedges": 0,
                                     "node_ids": []}
                if position is not None:
                    entry["chunk_index"] = position
            entry["score"] = max(entry["score"], float(item["score"]))
            entry["from_hyperedges"] += 1
            if item.get("node_id"):
                entry["node_ids"].append(item["node_id"])

    ordered = sorted(best.values(),
                     key=lambda c: (-c["score"], -c["from_hyperedges"], c["chunk_id"]))
    ordered = ordered[:top_k] if top_k else ordered
    for rank, c in enumerate(ordered, 1):
        c["rank"] = rank
        c["score"] = round(c["score"], 4)
    return ordered


# ------------------------------------------------------------------ graph
def load_hyperedges(graph_path, kind_attr="kind", kind_value="hyperedge",
                    text_attr="name", chunk_attr="chunks"):
    """[{node_id, text, participants, chunk_ids}] for every hyperedge in the graph."""
    import networkx as nx
    G = nx.read_graphml(graph_path)
    out = []
    for node, data in G.nodes(data=True):
        if data.get(kind_attr) != kind_value:
            continue
        text = str(data.get(text_attr) or node).strip()
        if not text:
            continue
        participants = [G.nodes[n].get("name", n) for n in G.neighbors(node)
                        if G.nodes[n].get(kind_attr) != kind_value]
        chunk_ids = [c.strip() for c in CHUNK_SEPS.split(str(data.get(chunk_attr, "")))
                     if c.strip()]
        out.append({"node_id": node, "text": text,
                    "participants": participants, "chunk_ids": chunk_ids})
    if not out:
        kinds = sorted({str(d.get(kind_attr)) for _, d in G.nodes(data=True)})[:8]
        raise SystemExit(f"no nodes with {kind_attr}={kind_value!r}; values present: {kinds}")
    print(f"graph      : {G.number_of_nodes()} nodes, {len(out)} hyperedges")
    return out


# --------------------------------------------------------------- the index
class HyperedgeIndex:
    """Embeddings over hyperedge descriptions, with the cache keyed by content.

    Identical descriptions are embedded once and merged, which keeps the
    provenance of every copy and, with --index hnsw, avoids the failure mode
    where many identical vectors fill each other's neighbour lists.
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
            return None                      # graph changed since the cache was written
        if backend == "hnsw":
            obj._build_hnsw()
        return obj


# ------------------------------------------------------------------- main
def retrieve(index, questions, top_k, min_score, chunks,
             want_edges=True, want_chunks=True, chunk_top_k=None):
    vecs = index.encode_queries(first_field(q, Q_FIELDS, f"(question {i})")
                                for i, q in enumerate(questions))
    results = []
    for qi, (q, vec) in enumerate(zip(questions, vecs)):
        hyperedges = []
        for rank, (uid, score) in enumerate(index.search(vec, top_k), 1):
            if score < min_score:
                continue
            unit = index.units[uid]
            rec = unit["records"][0]
            chunk_ids = sorted({c for r in unit["records"] for c in r["chunk_ids"]})
            item = {"text": unit["text"], "score": round(score, 4), "rank": rank,
                    "node_id": rec["node_id"], "participants": rec["participants"],
                    "chunk_ids": chunk_ids}
            if chunks:
                item["chunk_indices"] = sorted({chunks[c][0] for c in chunk_ids
                                                if c in chunks})
            hyperedges.append(item)

        # rolled up from the hyperedges above, so a chunk only appears because
        # one of the returned hits came out of it
        chunk_hits = (chunks_of(hyperedges, chunks, chunk_top_k)
                      if want_chunks and chunks else [])

        results.append({
            "qi": qi,
            "question": first_field(q, Q_FIELDS, f"(question {qi})"),
            "knowledge": {"hyperedges": hyperedges if want_edges else [],
                          "chunks": chunk_hits},
            "trace": {"n_hyperedges": len(hyperedges) if want_edges else 0,
                      "n_chunks": len(chunk_hits)},
        })
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--graph", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", default=None,
                    help="chunks.json, for the chunk section and the corpus indices")
    ap.add_argument("--knowledge", choices=["all", "hyperedges", "chunks"], default="all",
                    help="which sections to write (default: all, both of them). "
                         "generate_answers picks which to SHOW the model, so there is "
                         "no reason to narrow it here.")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--chunk-top-k", type=int, default=None,
                    help="cap the chunk list (default: every chunk the returned "
                         "hyperedges came from)")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="drop hyperedges below this cosine similarity")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--index", choices=["exact", "hnsw"], default="exact",
                    help="exact is a single matrix product and has no approximation "
                         "error; hnsw matches the production retriever")
    ap.add_argument("--cache", default=None,
                    help="embedding cache (default: alongside --out)")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--run-name", default="simsearch")
    args = ap.parse_args()

    t0 = time.time()
    records = load_hyperedges(args.graph)
    questions = json.loads(Path(args.questions).read_text())
    chunks = load_chunks(args.chunks)
    print(f"questions  : {len(questions)}")

    want_edges = args.knowledge in ("all", "hyperedges")
    want_chunks = args.knowledge in ("all", "chunks")
    if chunks:
        print(f"corpus     : {len(chunks)} chunks")
    elif want_chunks:
        print("!! no chunk texts loaded"
              + (f" from {args.chunks}" if args.chunks else " (--chunks not given)")
              + ": the chunks section will be empty")

    cache = Path(args.cache or (Path(args.out).parent /
                                f".{args.run_name}_index.pkl"))
    index = (None if args.rebuild else
             HyperedgeIndex.load(cache, args.model, args.index,
                                 len({norm(r["text"]) for r in records})))
    if index is None:
        index = HyperedgeIndex(records, args.model, args.index)
        index.save(cache)
        print(f"index      : built, {len(index.units)} distinct descriptions")
    else:
        print(f"index      : cached, {len(index.units)} distinct descriptions")

    results = retrieve(index, questions, args.top_k, args.min_score, chunks,
                       want_edges, want_chunks, args.chunk_top_k)
    n = [len(r["knowledge"]["hyperedges"]) for r in results]
    nc = [len(r["knowledge"]["chunks"]) for r in results]
    tops = [r["knowledge"]["hyperedges"][0]["score"] for r in results
            if r["knowledge"]["hyperedges"]]
    summary = {"questions": len(results),
               "mean_knowledge": round(float(np.mean(n)), 2) if n else 0,
               "mean_chunks": round(float(np.mean(nc)), 2) if nc else 0,
               "empty": sum(1 for x in n if x == 0),
               "mean_top_score": round(float(np.mean(tops)), 4) if tops else None,
               "seconds": round(time.time() - t0, 1)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": {"run": args.run_name, "method": "hyperedge-simsearch",
                    "graph": args.graph, "questions": args.questions,
                    "chunks": args.chunks, "knowledge": args.knowledge,
                    "model": args.model, "index": args.index,
                    "top_k": args.top_k, "chunk_top_k": args.chunk_top_k,
                    "min_score": args.min_score,
                    "knowledge_sections": ["hyperedges", "chunks"],
                    "llm_calls_per_question": 0},
         "summary": summary, "results": results}, indent=2, ensure_ascii=False))

    print(f"hyperedges : {summary['mean_knowledge']} per question, "
          f"{summary['empty']} empty, top score {summary['mean_top_score']}")
    print(f"chunks     : {summary['mean_chunks']} per question "
          f"(the passages those hyperedges were extracted from)")
    print(f"wrote {out}  ({summary['seconds']}s)")


if __name__ == "__main__":
    main()