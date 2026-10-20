"""
Step 3 of 7 - Label Extraction (Algorithm 1), over the extracted facts.

    <out_dir>/facts.json            input: [{"chunk_id", "facts"}] (run_fact_extraction.py)
    <out_dir>/fact_units.json       output: deduplicated facts with their chunk ids
    <out_dir>/fact_embeddings.npy   cached fact embeddings
    <out_dir>/schema.json           output: the frozen schema
    <out_dir>/trace.json            output: the traversal trace
    <out_dir>/raw_labels.json       output: the labels before consolidation

raw_labels.json is written just before the single consolidation call, which is
the one point where a crash would otherwise lose every extraction call made so
far. schema.json is only written once consolidation has returned.

Two parameters control cost and they are not the same thing. --n-medoids sets
how finely the corpus is sampled, so it should be chosen for coverage.
--max-calls is the budget. Lowering --n-medoids to save calls makes the k-means
coarse and risks missing a label permanently, since the schema is frozen.

Embeddings are cached, so re-running to retune tau or the prompts only repeats
the LLM calls.

Usage:
    export NEXUS_LLM_BASE_URL=https://api.openai.com/v1
    export NEXUS_LLM_API_KEY=sk-...
    python run_label_extraction.py --out-dir ./output/mydataset --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from embeddings import DEFAULT_MODEL as EMBED_MODEL, load_embedder
from label_extraction import label_extraction, normalize_label
from llm import LLMClient, make_consolidate_fn, make_extract_fn
from utils import read_json, write_json


def prepare_facts(out_dir: Path) -> list[dict]:
    """Flatten facts.json, dropping duplicates from chunk overlap and empty headings."""
    units: dict[str, dict] = {}
    for record in read_json(out_dir / "facts.json"):
        for fact in record["facts"]:
            if fact.endswith(":"):
                continue                          # "X's filmography includes the following films:"
            key = " ".join(fact.lower().split())
            if key not in units:
                units[key] = {"fact_id": len(units), "content": fact, "chunk_ids": []}
            if record["chunk_id"] not in units[key]["chunk_ids"]:
                units[key]["chunk_ids"].append(record["chunk_id"])
    facts = list(units.values())
    print(f"facts       : {len(facts)} unique")
    return facts


def prepare_embeddings(out_dir: Path, facts: list[dict], args) -> np.ndarray:
    cache, units_path = out_dir / "fact_embeddings.npy", out_dir / "fact_units.json"
    if cache.exists() and units_path.exists() and not args.reembed:
        if read_json(units_path) == facts:
            emb = np.load(cache)
            print(f"embeddings  : {emb.shape} (cached)")
            return emb
        print("embeddings  : facts changed, recomputing")

    embedder = load_embedder(args.embed_model)
    start = time.time()
    emb = embedder([f["content"] for f in facts])
    np.save(cache, emb)
    write_json(units_path, facts)
    print(f"embeddings  : {emb.shape} in {time.time() - start:.1f}s")
    return emb


class Progress:
    """Wraps extract_fn to report progress every `every` calls, without changing the algorithm."""

    def __init__(self, extract_fn, total: int, every: int, window: int, tau: float):
        self.extract_fn, self.total, self.every = extract_fn, total, every
        self.window, self.tau = window, tau
        self.calls, self.failed, self.labels, self.yields = 0, 0, set(), []
        self.start = time.time()

    def __call__(self, text: str):
        raw = self.extract_fn(text)
        self.calls += 1
        if raw is None:
            self.failed += 1
        else:
            found = {normalize_label(k) for k in raw} - {""}
            self.yields.append(len(found - self.labels))
            self.labels |= found
        if self.calls % self.every == 0 or self.calls == self.total:
            recent = self.yields[-self.window:]
            mean = sum(recent) / max(1, len(recent))
            print(f"  {self.calls:>4}/{self.total} facts | {len(self.labels):>4} labels "
                  f"| window mean {mean:.2f} (stop < {self.tau}) "
                  f"| {self.failed} failed | {time.time() - self.start:.0f}s", flush=True)
        return raw


def main() -> None:
    p = argparse.ArgumentParser(description="Nexus Label Extraction over facts")
    p.add_argument("--out-dir", required=True, type=Path)
    # embeddings
    p.add_argument("--embed-model", default=EMBED_MODEL)
    p.add_argument("--reembed", action="store_true")
    # traversal: a fact yields fewer labels than a chunk, so a longer window and lower tau
    p.add_argument("--n-medoids", type=int, default=300,
                   help="size of the medoid pool, i.e. how finely the corpus is sampled")
    p.add_argument("--window", type=int, default=10,
                   help="steps in the stopping window")
    p.add_argument("--tau", type=float, default=0.3,
                   help="stop when mean new labels over the window falls below this")
    p.add_argument("--max-calls", type=int, default=100,
                   help="hard ceiling on LLM calls, independent of the pool size")
    p.add_argument("--seed", type=int, default=0,
                   help="random_state of the k-means that builds the pool")
    # llm
    p.add_argument("--model", default=None, help="overrides NEXUS_LLM_MODEL")
    p.add_argument("--max-labels-per-fact", type=int, default=5)
    p.add_argument("--timeout", type=float, default=60, help="seconds per LLM call")
    p.add_argument("--report-every", type=int, default=20)
    args = p.parse_args()

    out_dir: Path = args.out_dir
    facts = prepare_facts(out_dir)
    embeddings = prepare_embeddings(out_dir, facts, args)

    client = LLMClient(**({"model": args.model} if args.model else {}))
    client.client = client.client.with_options(timeout=args.timeout, max_retries=2)
    print(f"llm         : {client.model} at {client.client.base_url}")

    budget = min(args.max_calls or args.n_medoids, args.n_medoids, len(facts))
    print(f"budget      : {budget} calls (pool {args.n_medoids}, max-calls {args.max_calls})")
    extract_fn = Progress(
        make_extract_fn(client, max_labels=args.max_labels_per_fact, verbose=False),
        total=budget, every=args.report_every, window=args.window, tau=args.tau,
    )
    base_consolidate = make_consolidate_fn(client, verbose=True)

    def consolidate_fn(labels, support, examples):
        write_json(out_dir / "raw_labels.json",
                   {"raw_labels": sorted(labels), "support": dict(support)})
        print(f"\nconsolidating {len(labels)} labels "
              f"(raw labels saved to raw_labels.json)...", flush=True)
        t = time.time()
        schema = base_consolidate(labels, support, examples)
        print(f"consolidation done in {time.time() - t:.0f}s", flush=True)
        return schema

    for stale in ("schema.json", "trace.json", "raw_labels.json"):
        (out_dir / stale).unlink(missing_ok=True)   # a leftover file must not look like a finished run
    print(f"\ntraversal (report every {args.report_every} calls)")
    start = time.time()
    result = label_extraction(
        [f["content"] for f in facts],
        extract_fn=extract_fn,
        consolidate_fn=consolidate_fn,
        embeddings=embeddings,
        n_medoids=args.n_medoids,
        window=args.window,
        tau=args.tau,
        max_calls=args.max_calls,
        cluster_seed=args.seed,
    )
    elapsed = time.time() - start
    trace = result.trace

    print("\n" + "=" * 66)
    print(f"extraction calls : {trace.n_calls} of {len(facts)} facts "
          f"({trace.n_calls / len(facts):.1%}) in {elapsed:.1f}s")
    print(f"stopped early    : {trace.stopped_early}")
    print(f"failed calls     : {len(trace.failed)} ({trace.failure_rate:.0%})")
    print(f"raw labels       : {len(result.raw_labels)}")
    print(f"frozen schema    : {len(result.schema)} labels")
    print("=" * 66)

    print("\nsaturation curve")
    for step, (new, total) in enumerate(zip(trace.new_labels, trace.label_count), 1):
        print(f"  {step:>3}  +{new:<3} -> {total:<4} {'#' * new}")

    if not trace.stopped_early:
        print("\n[!] the budget stopped the traversal, not the saturation rule. "
              "Raise --max-calls to find where the curve actually flattens.")
    if trace.failure_rate > 0.2:
        print("\n[!] more than a fifth of the extraction calls failed. The schema "
              "below is not trustworthy.")
        if getattr(client, "last_bad_response", None):
            print(f"    last bad response: {client.last_bad_response[:300]!r}")

    print("\nfrozen schema")
    for label, desc in result.schema.items():
        print(f"  {label:<22} {desc}")

    write_json(out_dir / "schema.json", {
        "schema": result.schema,
        "aliases": getattr(base_consolidate, "aliases", {}),
        "raw_labels": sorted(result.raw_labels),
        "support": result.label_support,
        "examples": result.label_examples,
    })
    write_json(out_dir / "trace.json", {
        "config": {
            "unit": "fact", "n_medoids": args.n_medoids, "max_calls": args.max_calls,
            "window": args.window, "tau": args.tau, "seed": args.seed,
            "embed_model": args.embed_model, "llm_model": client.model,
        },
        "n_facts": len(facts),
        "visited_fact_ids": [facts[i]["fact_id"] for i in trace.visited],
        "visited_facts": [facts[i]["content"] for i in trace.visited],
        "new_labels": trace.new_labels,
        "failed_fact_ids": [facts[i]["fact_id"] for i in trace.failed],
        "label_count": trace.label_count,
        "stopped_early": trace.stopped_early,
        "elapsed_seconds": round(elapsed, 2),
    })
    print(f"\nwrote {out_dir / 'schema.json'} and {out_dir / 'trace.json'}")


if __name__ == "__main__":
    main()
