"""
Step 1 of 3 - Query Entity Extraction, over the questions.

    <data_dir>/questions.json      input: the questions (see load_questions for shapes)
    --schema PATH                  input: the frozen schema, the same label set the
                                   graph was built with
    <out_dir>/query_entities.json  output: typed mentions per question

Each question becomes a set of tokens: the entities it mentions. That set is
what the token join queries against the hyperedge sets. The label set must be
the one the graph was built with, otherwise the query tokens and the graph
tokens do not come from the same vocabulary.

This is the first step of retrieval. A question is used as it is, with no
decomposition and no rewriting, so retrieval costs no LLM call at all.

Usage:
    python run_query_entity_extraction.py \
        --data-dir ../../datasets/Mix \
        --out-dir ../../output/Mix/knowledge \
        --schema ../../output/Mix/graph/schema.json --device cuda:0

Resumable: questions already in query_entities.json are skipped on re-run.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# The retrieval steps reuse the graph construction modules rather than keeping a
# second copy of each. One definition of the schema loader and the IO.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "graph_construction"))

from entity_extraction import DEFAULT_MODEL, GLiNERExtractor, load_schema
from utils import read_json, write_json


def load_questions(path: Path) -> list[dict]:
    """Accept a list of strings, a list of objects, or {"questions": [...]}.

    question_id is the id the dataset gives, falling back to the position. The
    position in this file is what generate_answers indexes by, so the two have
    to be read the same way everywhere.
    """
    data = read_json(path)
    if isinstance(data, dict):
        data = data.get("questions", data.get("data", []))
    out = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            out.append({"question_id": str(i), "qi": i, "question": item})
        else:
            text = item.get("question") or item.get("query") or item.get("text")
            if text:
                out.append({
                    "question_id": str(item.get("id", item.get("question_id", i))),
                    "qi": i,
                    "question": text,
                })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Nexus Query Entity Extraction")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--data-dir", type=Path, default=None,
                   help="dataset directory holding questions.json")
    p.add_argument("--questions", type=Path, default=None,
                   help="defaults to <data-dir>/questions.json")
    p.add_argument("--schema", type=Path, default=None,
                   help="path to schema.json; defaults to <out-dir>/schema.json")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default=None, help="e.g. cuda:0")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="a question is short, so it gives the model less evidence than a fact")
    p.add_argument("--label-batch-size", type=int, default=15)
    p.add_argument("--no-descriptions", action="store_true")
    p.add_argument("--report-every", type=int, default=200)
    p.add_argument("--restart", action="store_true", help="ignore existing query_entities.json")
    args = p.parse_args()

    if not args.questions and not args.data_dir:
        raise SystemExit("pass --questions or --data-dir")

    out_dir: Path = args.out_dir
    questions_path = args.questions or args.data_dir / "questions.json"
    questions = load_questions(questions_path)
    schema_path = args.schema or out_dir / "schema.json"
    schema = load_schema(read_json(schema_path))
    out_path = out_dir / "query_entities.json"

    done: dict[str, dict] = {}
    if out_path.exists() and not args.restart:
        done = read_json(out_path).get("queries", {})
        print(f"resuming    : {len(done)} questions already done")
    print(f"questions   : {len(questions)} from {questions_path}")
    print(f"schema      : {len(schema)} labels from {schema_path}")

    from gliner2 import GLiNER2
    model = GLiNER2.from_pretrained(args.model)
    if args.device and hasattr(model, "to"):
        model = model.to(args.device)

    extractor = GLiNERExtractor(
        schema,
        threshold=args.threshold,
        label_batch_size=args.label_batch_size,
        use_descriptions=not args.no_descriptions,
        extractor=model,
    )
    print(f"extractor   : {args.model} on {args.device or 'default device'}, "
          f"threshold {args.threshold}")

    queries = dict(done)
    for index, record in enumerate(questions, 1):
        qid = record["question_id"]
        if qid in queries:
            continue
        # The extractor names its unit a chunk; here the unit is a question.
        mentions = extractor.extract_chunk(
            {"chunk_id": qid, "content": record["question"]}, index)
        queries[qid] = {
            "question_id": qid,
            "qi": record["qi"],
            "question": record["question"],
            "tokens": [m.text for m in mentions],
            "mentions": [
                {"label": m.label, "text": m.text, "start": m.start, "end": m.end,
                 "confidence": round(float(m.confidence), 4)}
                for m in mentions
            ],
        }
        if index % args.report_every == 0:
            write_json(out_path, {"queries": queries})
            print(f"  {index:>6}/{len(questions)} questions", flush=True)

    label_counts = Counter(m["label"] for q in queries.values() for m in q["mentions"])
    total = sum(label_counts.values())
    empty = [qid for qid, q in queries.items() if not q["tokens"]]
    single = [qid for qid, q in queries.items() if len(q["tokens"]) == 1]

    print("\n" + "=" * 66)
    print(f"tokens            : {total} ({total / max(1, len(queries)):.2f} per question)")
    print(f"empty             : {len(empty)} ({len(empty) / max(1, len(queries)):.1%}) "
          "- these cannot be answered by the set join")
    print(f"single-token      : {len(single)} ({len(single) / max(1, len(queries)):.1%}) "
          "- set similarity is weak evidence at size 1")
    print(f"descriptions      : {'sent' if extractor.descriptions_ok else 'not supported'}")
    print(f"llm calls         : 0")
    print("=" * 66)

    print("\ntokens per label")
    for label in schema:
        count = label_counts.get(label, 0)
        print(f"  {label:<24} {count:>6}{'   <- never fires' if count == 0 else ''}")

    if empty:
        print("\nquestions with no entity (first 5)")
        for qid in empty[:5]:
            print(f"  {queries[qid]['question'][:70]!r}")

    write_json(out_path, {
        "config": {
            "unit": "question", "schema": str(schema_path),
            "questions": str(questions_path),
            "model": args.model, "threshold": args.threshold,
            "label_batch_size": args.label_batch_size,
            "descriptions": bool(extractor.descriptions_ok),
        },
        "stats": {
            "n_questions": len(queries), "n_tokens": total,
            "empty": empty, "single_token": len(single),
            "per_label": dict(label_counts),
        },
        "queries": queries,
    })
    print(f"\nwrote {out_path}")

    if len(empty) / max(1, len(queries)) > 0.05:
        print("\n[!] more than a twentieth of the questions produced no entity. "
              "Lower --threshold, or fall back to the question text for those.")


if __name__ == "__main__":
    main()
