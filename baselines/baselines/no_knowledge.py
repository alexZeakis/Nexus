"""
Baseline knowledge retrieval: nothing is retrieved.

PlainLLM, the floor of the whole benchmark. The model is asked the question
with no knowledge attached, so whatever it answers correctly is what it already
knew. Every retriever's score has to be read against this one: a method that
scores 60 on a benchmark the bare model scores 55 on has retrieved almost
nothing of value.

No corpus is read and no graph is touched. The only input is the question list,
and the only work is writing the contract with both sections empty.

    python -m knowledge_retrieval.baselines.no_knowledge \
        --questions ../datasets/Mix/questions.json \
        --out ../output/Mix/retrieval/plainllm.json

OUTPUT CONTRACT

    "knowledge": {"hyperedges": [], "chunks": []}

Both keys are present and empty rather than absent, so the same consumer reads
this file as reads every other. Generation must be run in a mode that answers
from the question alone: a prompt that instructs the model to answer only from
the given knowledge will refuse every question here, and the result would
measure the prompt rather than the model. Check the `insufficient` and
`no_knowledge` counters of the answers file before trusting this baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

Q_FIELDS = ("question", "query", "text", "input")


def first_field(obj, fields, default=""):
    for f in fields:
        if isinstance(obj, dict) and obj.get(f):
            v = obj[f]
            return " ".join(map(str, v)) if isinstance(v, list) else str(v)
    return default


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-name", default="plainllm")
    args = ap.parse_args()

    t0 = time.time()
    questions = json.loads(Path(args.questions).read_text())
    print(f"questions  : {len(questions)}")

    results = [{"qi": qi,
                "question": first_field(q, Q_FIELDS, f"(question {qi})"),
                "knowledge": {"hyperedges": [], "chunks": []},
                "trace": {"n_hyperedges": 0, "n_chunks": 0}}
               for qi, q in enumerate(questions)]

    summary = {"questions": len(results), "mean_knowledge": 0, "mean_chunks": 0,
               "empty": len(results), "mean_top_score": None,
               "seconds": round(time.time() - t0, 1)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": {"run": args.run_name, "method": "no-knowledge",
                    "graph": None, "questions": args.questions,
                    "chunks": None, "knowledge": "none",
                    "model": None, "index": None,
                    "top_k": 0, "chunk_top_k": 0, "min_score": None,
                    "knowledge_sections": [],
                    "llm_calls_per_question": 0},
         "summary": summary, "results": results}, indent=2, ensure_ascii=False))

    print("knowledge  : none, by construction")
    print("note       : generation must be allowed to answer without knowledge, "
          "or this measures the prompt")
    print(f"wrote {out}  ({summary['seconds']}s)")


if __name__ == "__main__":
    main()