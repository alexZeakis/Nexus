"""
Step 2 of 3 - Hyperedge Retrieval: top-k fuzzy set similarity join of the
questions against the hyperedge collection.

The hypergraph is read as a collection of sets: one set per hyperedge, whose
elements are its participants. A question is the same kind of set, its elements
being the entities extracted from it. Retrieval is then a kNN fuzzy set
similarity join, k hyperedges per question, ranked by set similarity.

Matching is fuzzy at the element level, so a question naming "Weisblum" still
reaches a hyperedge holding "Andy Weisblum". The collection is indexed once with
tokenjoin_prepare and every question is answered against that index.

    --hyperedges PATH                 input: canonical hyperedges
                                      (defaults to <out_dir>/hyperedge_linking.json)
    --postings PATH                   input: which of them the graph kept
                                      (defaults to <out_dir>/postings.json)
    --query-entities PATH             input: the question token sets
                                      (defaults to <out_dir>/query_entities.json)
    <out_dir>/retrieved.json          output: ranked hyperedges per question

Nothing is re-ranked here. Binding reads this file, expands each hit into its
own query and produces the final per-question result.

Usage:
    python run_hyperedge_retrieval.py --out-dir ./output/mydataset/knowledge --k 10 \
        --hyperedges ./output/mydataset/graph/hyperedge_linking.json \
        --postings ./output/mydataset/graph/postings.json
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "graph_construction"))

from entity_extraction import TRIM_CHARS
from utils import read_json, write_json

_MODULES = {"jaccard": "pytokenjoin.jaccard.join_knn",
            "edit": "pytokenjoin.edit.join_knn"}


def clean_set(elements) -> list[str]:
    """Trim boundary punctuation and drop repeats, on both sides of the join.

    The join is fuzzy at the element level, so "Weak and the Wicked." and
    "Weak and the Wicked" are not the same element and the pair scores 0.6
    instead of 1.0. Entity extraction trims at the source; this repeats it here
    so a graph built before that change still joins correctly.
    """
    out = []
    for element in elements:
        element = str(element).strip(TRIM_CHARS).strip()
        if element:
            out.append(element)
    return list(dict.fromkeys(out))


def load_hyperedges(hyperedges_path: Path, postings_path: Path | None, tokens: str) -> list[dict]:
    """The hyperedge collection, as sets of participants."""
    canonical = read_json(hyperedges_path)["canonical"]

    keep = None
    if postings_path is not None and postings_path.exists():
        # postings.json lists exactly the hyperedges graph assembly wired in,
        # so anything missing from it is unreachable by retrieval anyway.
        keep = {hid for hids in read_json(postings_path).values() for hid in hids}

    field = "participant_names" if tokens == "names" else "participants"
    out = []
    for edge in canonical:
        if keep is not None and edge["canonical_id"] not in keep:
            continue
        elements = clean_set(edge.get(field, []))
        if not elements:
            continue
        out.append({
            "canonical_id": edge["canonical_id"],
            "elements": elements,
            "description": edge.get("official_description", ""),
            "participants": edge.get("participants", []),
            "src": edge.get("src", []),
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Nexus Hyperedge Retrieval (top-k token join)")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--hyperedges", type=Path, default=None,
                   help="canonical hyperedges; defaults to <out-dir>/hyperedge_linking.json")
    p.add_argument("--postings", type=Path, default=None,
                   help="graph posting lists; defaults to <out-dir>/postings.json")
    p.add_argument("--query-entities", type=Path, default=None,
                   help="question tokens; defaults to <out-dir>/query_entities.json")
    p.add_argument("--k", type=int, default=10, help="hyperedges kept per question")
    p.add_argument("--similarity", default="jaccard", choices=["jaccard", "edit"])
    p.add_argument("--tokens", default="names", choices=["names", "ids"],
                   help="'names' matches fuzzily; 'ids' needs the query entities linked first")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="drop results below this; k forces a full list even when nothing fits")
    p.add_argument("--batch-size", type=int, default=2000, help="questions per query call")
    p.add_argument("--no-graph-filter", action="store_true",
                   help="keep hyperedges that graph assembly dropped")
    args = p.parse_args()

    import importlib
    import pandas as pd

    out_dir: Path = args.out_dir
    hyperedges_path = args.hyperedges or out_dir / "hyperedge_linking.json"
    postings_path = args.postings or out_dir / "postings.json"
    query_path = args.query_entities or out_dir / "query_entities.json"

    edges = load_hyperedges(
        hyperedges_path, None if args.no_graph_filter else postings_path, args.tokens)
    queries = read_json(query_path)["queries"]

    print(f"hyperedges from: {hyperedges_path}")
    if not args.no_graph_filter:
        print(f"graph filter   : {postings_path}"
              f"{'' if postings_path.exists() else '  <- missing, no filter applied'}")
    print(f"queries from   : {query_path}")

    for q in queries.values():
        q["tokens"] = clean_set(q["tokens"])
    live = [(sid, q) for sid, q in queries.items() if q["tokens"]]
    empty = [sid for sid, q in queries.items() if not q["tokens"]]
    print(f"hyperedges   : {len(edges)} sets "
          f"({sum(len(e['elements']) for e in edges) / max(1, len(edges)):.2f} participants each)")
    print(f"questions    : {len(live)} with tokens, {len(empty)} empty (skipped)")
    if not edges or not live:
        print("nothing to join")
        return

    # ids are row positions: the library returns positions and then overwrites
    # the id columns in place, so positions are the one reading that holds.
    right = pd.DataFrame({"id": list(range(len(edges))),
                          "elements": [e["elements"] for e in edges]})

    module = importlib.import_module(_MODULES[args.similarity])
    tj = module.TokenJoin()
    start = time.time()
    tj.tokenjoin_prepare(right, "id", "elements")
    print(f"index        : {args.similarity} kNN over {len(edges)} sets "
          f"in {time.time() - start:.1f}s")

    results: dict[str, list[dict]] = {}
    start = time.time()
    for offset in range(0, len(live), args.batch_size):
        batch = live[offset: offset + args.batch_size]
        left = pd.DataFrame({
            "id": list(range(len(batch))),
            "elements": [clean_set(q["tokens"]) for _, q in batch],
        })
        frame = tj.tokenjoin_query(left, "id", "elements", k=args.k)
        for _, row in frame.iterrows():
            sid = batch[int(row["l_id"])][0]
            score = float(row["score"])
            if score < args.min_score:
                continue
            edge = edges[int(row["r_id"])]
            results.setdefault(sid, []).append({
                "hyperedge_id": edge["canonical_id"],
                "score": round(score, 6),
                "description": edge["description"],
                "participants": edge["participants"],
                "src": edge["src"],
            })
        print(f"  {min(offset + args.batch_size, len(live)):>6}/{len(live)} questions",
              flush=True)
    elapsed = time.time() - start

    # tokenjoin_query does not order within a left record.
    for sid, hits in results.items():
        hits.sort(key=lambda h: -h["score"])
        for rank, hit in enumerate(hits, 1):
            hit["rank"] = rank

    for sid in queries:
        results.setdefault(sid, [])

    sizes = [len(results[sid]) for sid, _ in live]
    tops = [results[sid][0]["score"] for sid, _ in live if results[sid]]
    no_hit = [sid for sid, _ in live if not results[sid]]

    print("\n" + "=" * 66)
    print(f"retrieval    : {sum(sizes)} results for {len(live)} questions "
          f"in {elapsed:.1f}s ({elapsed / max(1, len(live)) * 1000:.1f}ms each)")
    print(f"full lists   : {sum(1 for s in sizes if s == args.k)} of {len(live)} reached k={args.k}")
    print(f"no result    : {len(no_hit)} with tokens but nothing above --min-score")
    if tops:
        tops_sorted = sorted(tops)
        print(f"top-1 score  : median {tops_sorted[len(tops_sorted) // 2]:.3f}, "
              f"min {tops_sorted[0]:.3f}, max {tops_sorted[-1]:.3f}")
    print("=" * 66)

    print("\nexamples")
    for sid, q in live[:3]:
        print(f"  {q['question'][:66]!r}")
        print(f"    tokens: {q['tokens']}")
        for hit in results[sid][:3]:
            print(f"    {hit['score']:.3f}  {hit['description'][:60]!r}")

    write_json(out_dir / "retrieved.json", {
        "config": {
            "k": args.k, "similarity": args.similarity, "tokens": args.tokens,
            "min_score": args.min_score,
            "hyperedges": str(hyperedges_path),
            "postings": None if args.no_graph_filter else str(postings_path),
            "query_entities": str(query_path),
        },
        "stats": {
            "n_hyperedges": len(edges), "n_questions": len(queries),
            "n_with_tokens": len(live), "n_empty": len(empty),
            "n_results": sum(sizes), "seconds": round(elapsed, 2),
        },
        "results": results,
    })
    print(f"\nwrote {out_dir / 'retrieved.json'}")

    counts = Counter(len(q["tokens"]) for _, q in live)
    if counts.get(1, 0) / max(1, len(live)) > 0.3:
        print("\n[!] many questions carry a single token, where set similarity "
              "cannot separate the hyperedges that contain it.")


if __name__ == "__main__":
    main()
