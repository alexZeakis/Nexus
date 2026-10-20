"""
Oracle baseline: the question's own gold context is the knowledge.

No retrieval happens. Each question's required sentences are handed straight
through as knowledge, in the retrieval contract, so text_generation consumes
them exactly like any real method's output. That makes this the CEILING for
the whole pipeline: whatever a retriever finds, it cannot beat knowing the
answer sentences in advance.

Its failures are the useful part. Where this baseline answers wrong, the gold
context itself does not contain the answer ("Partial filmography.", "Economic
statistics"), so the question is unanswerable from context and no retriever
can be blamed for missing it. Evaluation can then report accuracy over all
questions and over the solvable ones separately.

Two sources of "golden sentence":

    --source context   (default)  the `context` strings from questions.json,
                                  verbatim. The purest ceiling: nothing but
                                  the benchmark's own annotation.
    --source matched              the sentences those contexts were matched to
                                  in the corpus, read from a golden file
                                  (eval's golden_chunks.json). Closer to what
                                  a perfect retriever could actually return,
                                  since it is real corpus text, and it carries
                                  chunk provenance.

TWO SECTIONS OF KNOWLEDGE

`knowledge` is a dict, matching every other retriever:

    "knowledge": {"hyperedges": [...],    # the gold statements
                  "chunks":     [...]}    # the passages they live in

The `hyperedges` key holds the gold statements even though nothing here is a
hyperedge: the key is the contract, and `generate_answers --knowledge
hyperedges` has to find this baseline's knowledge where it finds every other
method's. A ceiling measured on a different field is not a ceiling.

The chunks are located in the corpus, not retrieved: each gold statement is
matched back to the passage containing it, so the oracle supplies the same
KIND of context as a chunk retriever and `--knowledge all` compares like with
like. --chunks is what makes that section possible.

    python -m knowledge_retrieval.baselines.golden_context \
        --questions ../datasets/Mix/questions.json \
        --chunks ../output/Mix/chunks.json \
        --out ../output/Mix/retrieval/golden.json

    python -m knowledge_retrieval.baselines.golden_context \
        --questions ../datasets/Mix/questions.json \
        --source matched --golden-file ../eval/Mix/golden_chunks.json \
        --chunks ../output/Mix/chunks.json \
        --out ../output/Mix/retrieval/golden-matched.json

Then, unchanged except for which section the model is shown:

    python -m text_generation.generate_answers \
        --retrieval ../output/Mix/retrieval/golden.json \
        --out ../output/Mix/answers/golden.json \
        --questions ../datasets/Mix/questions.json

    python -m text_generation.generate_answers \
        --retrieval ../output/Mix/retrieval/golden.json \
        --out ../output/Mix/answers/golden_all.json \
        --questions ../datasets/Mix/questions.json --knowledge all
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import OrderedDict
from pathlib import Path

Q_FIELDS = ("question", "query", "text", "input")
C_FIELDS = ("context", "contexts", "supporting_facts", "evidence")
CHUNK_FIELDS = ("content", "text", "chunk", "passage", "body")

#: A context can hold several statements joined by ';' (and the benchmark
#: sometimes repeats the same one several times over).
SUB_SEP = ";"
#: Shorter than this is a heading, not a statement: "Partial filmography."
MIN_STATEMENT_CHARS = 20
MIN_STATEMENT_WORDS = 4

#: Locating a gold statement in the corpus. A verbatim substring is the
#: answer when it exists; otherwise the best token overlap above this is
#: accepted, which covers punctuation and whitespace drift between the
#: benchmark's annotation and the corpus text.
OVERLAP_T = 0.6
STOP = frozenset(
    "the a an of and or to in on at for with by is was were be been as from "
    "that this these those it its he she they his her their which who whom "
    "but not had has have do did does are am will would can could".split())


def first_field(obj, fields, default=""):
    for f in fields:
        if isinstance(obj, dict) and obj.get(f):
            v = obj[f]
            return " ".join(map(str, v)) if isinstance(v, list) else str(v)
    return default


def contexts_of(q):
    """The question's gold context strings, whatever shape the field takes."""
    for f in C_FIELDS:
        v = q.get(f) if isinstance(q, dict) else None
        if not v:
            continue
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            out = []
            for item in v:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):           # {"sentence": ...} etc.
                    out.append(first_field(item, ("sentence", "text", "context")))
                elif isinstance(item, (list, tuple)):  # ["title", "sentence"]
                    out.append(" ".join(map(str, item)))
            return [s for s in out if s]
    return []


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def statements(context):
    """Split a context into distinct statements, deduplicated, in order."""
    seen = OrderedDict()
    for part in str(context).split(SUB_SEP):
        part = part.strip()
        if part:
            seen.setdefault(norm(part), part)
    return list(seen.values()) or [str(context).strip()]


def is_heading(text):
    t = norm(text)
    return len(t) < MIN_STATEMENT_CHARS or len(t.split()) < MIN_STATEMENT_WORDS


# ------------------------------------------------------- corpus provenance
def load_chunks(path):
    """chunk_id -> (position in the corpus, text). Both chunk shapes are read."""
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


def content_tokens(text):
    return [t for t in dict.fromkeys(norm(text).split())
            if len(t) >= 3 and t not in STOP]


def build_inverted(chunks):
    """token -> {chunk_id}, so locating a statement scans a handful of chunks."""
    inv = {}
    for cid, (_, text) in chunks.items():
        for tok in content_tokens(text):
            inv.setdefault(tok, set()).add(cid)
    return inv


def locate(statement, chunks, inv):
    """(chunk_id, how) for the corpus passage this gold statement came from.

    The rarest few tokens narrow the candidates, then a verbatim substring
    decides. Where the annotation differs from the corpus in punctuation or
    whitespace, the best token overlap stands in and is reported separately,
    so the two kinds of match never get confused in the counts.
    """
    s = norm(statement)
    toks = content_tokens(statement)
    if not s or not toks:
        return None, None

    rare = sorted(toks, key=lambda t: len(inv.get(t, ())))[:3]
    cands = None
    for t in rare:
        ids = inv.get(t, set())
        cands = set(ids) if cands is None else (cands & ids)
        if not cands:
            break
    if not cands:                       # no chunk has all three: widen
        cands = set(inv.get(rare[0], set()))
    if not cands:
        return None, None

    for cid in sorted(cands):
        if s in norm(chunks[cid][1]):
            return cid, "exact"

    want = set(toks)
    best, best_score = None, 0.0
    for cid in sorted(cands):
        have = set(norm(chunks[cid][1]).split())
        score = len(want & have) / len(want)
        if score > best_score:
            best, best_score = cid, score
    return (best, "overlap") if best_score >= OVERLAP_T else (None, None)


def chunks_of(items, chunks, top_k=None):
    """The passages the kept statements came from, deduplicated and ranked.

    Every gold statement scores 1.0, so chunks are ordered by how many of them
    they contain: a passage holding two required sentences is the more useful
    one to read.
    """
    best = {}
    for item in items:
        for cid in item.get("chunk_ids", []):
            position, text = chunks.get(str(cid), (None, ""))
            if not text:
                continue
            entry = best.get(cid)
            if entry is None:
                entry = best[cid] = {"text": text, "chunk_id": str(cid),
                                     "score": 1.0, "from_statements": 0,
                                     "node_ids": []}
                if position is not None:
                    entry["chunk_index"] = position
            entry["from_statements"] += 1
            entry["node_ids"].append(item["node_id"])

    ordered = sorted(best.values(),
                     key=lambda c: (-c["from_statements"], c.get("chunk_index", 0)))
    ordered = ordered[:top_k] if top_k else ordered
    for rank, c in enumerate(ordered, 1):
        c["rank"] = rank
    return ordered


# ------------------------------------------------------------ the sources
def from_questions(questions):
    """Gold contexts, verbatim from questions.json."""
    for qi, q in enumerate(questions):
        items = []
        for ci, ctx in enumerate(contexts_of(q)):
            for si, text in enumerate(statements(ctx)):
                items.append({"text": text, "context": ctx,
                              "node_id": f"gold-{qi}-{ci}-{si}",
                              "chunk_ids": [], "chunk_indices": []})
        yield qi, first_field(q, Q_FIELDS, f"(question {qi})"), items


def from_golden_file(path, questions):
    """The corpus sentences those contexts were matched to, with provenance."""
    blob = json.loads(Path(path).read_text())
    by_qi = {q["qi"]: q for q in blob["questions"]}
    n = max(len(questions or []), max(by_qi, default=-1) + 1)
    for qi in range(n):
        q = by_qi.get(qi)
        if q is None:
            yield qi, (first_field(questions[qi], Q_FIELDS, f"(question {qi})")
                       if questions and qi < len(questions) else f"(question {qi})"), []
            continue
        items = []
        for ci, c in enumerate(q.get("contexts", [])):
            text = (c.get("matched") or "").strip()
            if not text:
                continue
            items.append({"text": text, "context": c.get("context", ""),
                          "node_id": f"golden-{qi}-{ci}",
                          "chunk_ids": c.get("golden_ids", []),
                          "chunk_indices": c.get("golden_indices", []),
                          "match_distance": c.get("distance")})
        question = q.get("question") or (
            first_field(questions[qi], Q_FIELDS, f"(question {qi})")
            if questions and qi < len(questions) else f"(question {qi})")
        yield qi, question, items


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Oracle baseline: gold context as knowledge")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", choices=["context", "matched"], default="context",
                    help="context: the gold strings themselves; "
                         "matched: the corpus sentences they were matched to")
    ap.add_argument("--golden-file", default=None,
                    help="golden_chunks.json, required for --source matched")
    ap.add_argument("--chunks", default=None,
                    help="chunks.json. Needed for the chunks section: each gold "
                         "statement is located in the corpus and the passage "
                         "holding it is attached.")
    ap.add_argument("--knowledge", choices=["all", "hyperedges", "chunks"], default="all",
                    help="which sections to write (default: all, both of them). "
                         "generate_answers picks which to SHOW the model, so there "
                         "is no reason to narrow it here.")
    ap.add_argument("--chunk-top-k", type=int, default=None,
                    help="cap the chunk list (default: every passage the gold "
                         "statements came from)")
    ap.add_argument("--drop-headings", action="store_true",
                    help="omit statements too short to be one ('Partial filmography.'); "
                         "off by default, since those failures are the point")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    if args.source == "matched" and not args.golden_file:
        ap.error("--source matched needs --golden-file")

    t0 = time.time()
    questions = json.loads(Path(args.questions).read_text())
    run = args.run_name or ("golden" if args.source == "context" else "golden-matched")
    print(f"questions : {len(questions)}")
    print(f"source    : {args.source}"
          + (f"  ({args.golden_file})" if args.source == "matched" else ""))

    want_edges = args.knowledge in ("all", "hyperedges")
    want_chunks = args.knowledge in ("all", "chunks")
    chunks = load_chunks(args.chunks)
    inverted = build_inverted(chunks) if chunks else {}
    if chunks:
        print(f"corpus    : {len(chunks)} chunks, {len(inverted)} index terms")
    elif want_chunks:
        print("!! no chunk texts loaded"
              + (f" from {args.chunks}" if args.chunks else " (--chunks not given)")
              + ": the chunks section will be empty")

    stream = (from_questions(questions) if args.source == "context"
              else from_golden_file(args.golden_file, questions))

    results, n_dropped = [], 0
    located = {"exact": 0, "overlap": 0, "carried": 0, "missing": 0}
    n_chunks = 0
    for qi, question, items in stream:
        kept = []
        for item in items:
            if args.drop_headings and is_heading(item["text"]):
                n_dropped += 1
                continue
            kept.append(item)

        # --source matched already carries chunk ids; --source context does not,
        # so the statement is matched back to the passage that holds it
        if chunks and want_chunks:
            for item in kept:
                if item.get("chunk_ids"):
                    located["carried"] += 1
                    continue
                cid, how = locate(item["text"], chunks, inverted)
                if cid:
                    item["chunk_ids"] = [cid]
                    item["chunk_indices"] = [chunks[cid][0]]
                    item["located"] = how
                    located[how] += 1
                else:
                    located["missing"] += 1

        hyperedges = []
        for item in kept:
            entry = {"text": item["text"], "score": 1.0,
                     "rank": len(hyperedges) + 1, "node_id": item["node_id"],
                     "participants": [], "chunk_ids": item.get("chunk_ids", [])}
            if item.get("chunk_indices"):
                entry["chunk_indices"] = item["chunk_indices"]
            hyperedges.append(entry)

        chunk_hits = (chunks_of(hyperedges, chunks, args.chunk_top_k)
                      if want_chunks and chunks else [])
        n_chunks += len(chunk_hits)

        results.append({
            "qi": qi, "question": question,
            "knowledge": {"hyperedges": hyperedges if want_edges else [],
                          "chunks": chunk_hits},
            "trace": {"source": args.source,
                      "contexts": [i.get("context", "") for i in items],
                      "n_statements": len(hyperedges),
                      "n_chunks": len(chunk_hits),
                      **({"match_distances": [i["match_distance"] for i in items
                                              if i.get("match_distance") is not None]}
                         if args.source == "matched" else {})}})

    n = [len(r["knowledge"]["hyperedges"]) for r in results]
    nc = [len(r["knowledge"]["chunks"]) for r in results]
    headings = sum(1 for r in results for k in r["knowledge"]["hyperedges"]
                   if is_heading(k["text"]))
    summary = {"questions": len(results),
               "mean_knowledge": round(sum(n) / max(len(n), 1), 2),
               "mean_chunks": round(sum(nc) / max(len(nc), 1), 2),
               "empty": sum(1 for x in n if x == 0),
               "headings_kept": headings,
               "headings_dropped": n_dropped,
               "statements_located": located,
               "seconds": round(time.time() - t0, 1)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": {"run": run, "method": f"golden-{args.source}",
                    "questions": args.questions, "golden_file": args.golden_file,
                    "chunks": args.chunks, "knowledge": args.knowledge,
                    "chunk_top_k": args.chunk_top_k,
                    "drop_headings": args.drop_headings, "oracle": True,
                    "knowledge_sections": ["hyperedges", "chunks"],
                    "llm_calls_per_question": 0},
         "summary": summary, "results": results}, indent=2, ensure_ascii=False))

    print(f"statements: {summary['mean_knowledge']} per question, "
          f"{summary['empty']} empty")
    print(f"chunks    : {summary['mean_chunks']} per question "
          f"(the passages those statements live in)")
    if want_chunks and chunks:
        total = sum(located.values())
        print(f"located   : {located['exact']} verbatim, {located['overlap']} by overlap, "
              f"{located['carried']} already had ids, {located['missing']} not found "
              f"of {total}")
        if located["missing"]:
            print("            a statement that is not in the corpus contributes no "
                  "chunk, so the oracle is thinner there than a chunk retriever")
    print(f"headings  : {headings} kept"
          + (f", {n_dropped} dropped" if n_dropped else "")
          + "  (a question whose only context is a heading cannot be answered)")
    print(f"wrote {out}  ({summary['seconds']}s)")


if __name__ == "__main__":
    main()