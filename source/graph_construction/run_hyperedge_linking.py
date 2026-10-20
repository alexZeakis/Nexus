"""
Step 6 of 7 - Hyperedge Linking, where each fact is a hyperedge.

    <out_dir>/fact_units.json          input: fact texts and their source chunks
    <out_dir>/entities_linked.json     input: canonical participants per fact
    <out_dir>/chunks.json              input: corpus order for canonical choice
    <out_dir>/hyperedges.json          output: facts as hyperedges, before linking
    <out_dir>/hyperedge_linking.json   output: canonical hyperedges, statistics

Candidates come from a fuzzy set similarity join over the participant sets.
Adjudication is bidirectional entailment with an NLI cross-encoder over the
fact sentences, not the MATCH prompt used for entities: two facts about the
same people refer to the same things while stating different things, so the
entity question answers yes when it should answer no.

A fact with fewer than two distinct linked entities is not a hyperedge and is
dropped by load_records.

Usage:
    python run_hyperedge_linking.py --out-dir ./output/mydataset --device cuda:0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from hyperedge_linking import (
    DEFAULT_NLI_MODEL,
    NLIMatcher,
    build_components,
    build_hyperedges,
    candidate_pairs,
    canonicalize,
    load_records,
    order_pairs,
    summarize,
)
from utils import read_json, write_json


def main() -> None:
    p = argparse.ArgumentParser(description="Nexus Hyperedge Linking over facts")
    p.add_argument("--out-dir", required=True, type=Path)
    # candidate generation
    p.add_argument("--similarity", default="jaccard", choices=["jaccard", "edit"])
    p.add_argument("--join-threshold", type=float, default=0.85)
    p.add_argument("--max-pairs", type=int, default=None)
    # adjudication
    p.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    p.add_argument("--device", default=None, help="e.g. cuda:0")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--decision-threshold", type=float, default=0.7,
                   help="both entailment directions must reach this")
    # canonicalization
    p.add_argument("--canonical", default="first", choices=["first", "confidence"])
    p.add_argument("--clustering", default="star", choices=["star", "transitive"])
    p.add_argument("--max-component", type=int, default=15)
    p.add_argument("--cache", type=Path, default=None, help="optional JSON file of pair decisions")
    args = p.parse_args()

    out_dir: Path = args.out_dir
    facts = read_json(out_dir / "fact_units.json")
    linked_mentions = read_json(out_dir / "entities_linked.json")["mentions"]
    chunk_pos = {c["chunk_id"]: i for i, c in enumerate(read_json(out_dir / "chunks.json"))}

    payload = build_hyperedges(facts, linked_mentions)
    write_json(out_dir / "hyperedges.json", payload)

    # Corpus order of a fact: its first source chunk, then fact id.
    ranked = sorted(facts, key=lambda f: (min(chunk_pos.get(c, 10**9) for c in f["chunk_ids"]), f["fact_id"]))
    fact_order = {str(f["fact_id"]): r for r, f in enumerate(ranked)}

    records = load_records(payload, fact_order)
    print(f"facts             : {len(facts)}")
    print(f"hyperedges        : {len(records)} facts with 2+ distinct entities "
          f"({1 - len(records) / max(1, len(facts)):.1%} dropped)")
    if not records:
        print("nothing to link")
        return

    # ------------------------------------------------- candidate generation
    start = time.time()
    pairs, join_stats = candidate_pairs(records, threshold=args.join_threshold,
                                        similarity=args.similarity)
    print(f"token join        : {len(pairs)} candidate pairs at "
          f"{args.similarity} >= {args.join_threshold} over "
          f"{join_stats['sets']} sets in {time.time() - start:.1f}s")

    scored = order_pairs(pairs, join_stats.get("scores", {}))
    if args.max_pairs:
        scored = scored[: args.max_pairs]
        print(f"                    truncated to the top {len(scored)} by set similarity")

    # ------------------------------------------------------- adjudication
    decisions: dict[tuple[int, int], tuple[float, float, float]] = {}
    if scored:
        matcher = NLIMatcher(model_name=args.nli_model, batch_size=args.batch_size,
                             device=args.device, dtype=args.dtype)
        print(f"nli               : {args.nli_model} on {matcher.device} "
              f"({matcher.dtype}), {len(scored)} pairs, both directions")
        start = time.time()
        results = matcher.match([(records[a].description, records[b].description)
                                 for a, b, _ in scored])
        decisions = {(a, b): r for (a, b, _), r in zip(scored, results)}
        print(f"                    {len(scored)} pairs in {time.time() - start:.1f}s "
              f"({matcher.calls} forward passes)")

    confirmed = [(a, b) for (a, b), (score, _, _) in decisions.items()
                 if score >= args.decision_threshold]
    match_scores = {pair: d[0] for pair, d in decisions.items() if d[0] >= args.decision_threshold}
    one_way = sum(1 for score, f, b in decisions.values()
                  if score < args.decision_threshold and max(f, b) >= args.decision_threshold)
    print(f"confirmed         : {len(confirmed)} of {len(decisions)} "
          f"({len(confirmed) / max(1, len(decisions)):.0%})")
    print(f"one-way entailment: {one_way} pairs where one fact is more specific "
          f"than the other (not duplicates)")

    if args.cache is not None:
        write_json(args.cache, {f"{records[a].hyperedge_id}||{records[b].hyperedge_id}":
                                {"score": round(s, 4), "forward": round(f, 4), "backward": round(b_, 4)}
                                for (a, b), (s, f, b_) in decisions.items()})

    # ------------------------------------------------------ canonicalization
    components = build_components(
        records, confirmed, mode=args.clustering, scores=match_scores,
        max_size=args.max_component if args.clustering == "star" else None,
    )
    _, canonical = canonicalize(records, components, rule=args.canonical)
    stats = summarize(records, canonical, pairs, confirmed)
    stats["n_facts"] = len(facts)
    stats["one_way_pairs"] = one_way

    config = {
        "unit": "fact", "similarity": args.similarity,
        "join_threshold": args.join_threshold, "nli_model": args.nli_model,
        "decision_threshold": args.decision_threshold, "serialize": "fact sentence only",
        "canonical": args.canonical, "clustering": args.clustering,
    }
    write_json(out_dir / "hyperedge_linking.json", {
        "config": config, "stats": stats, "canonical": canonical,
    })

    print("\n" + "=" * 66)
    print(f"canonical         : {stats['n_canonical']} from {stats['n_hyperedges']} "
          f"({stats['reduction']:.1%} reduction)")
    print(f"merged groups     : {stats['n_merged_groups']} "
          f"({stats['cross_chunk_merges']} span more than one chunk)")
    print(f"largest component : {stats['largest_component']} ({args.clustering} clustering)")
    print("=" * 66)

    print("\nlargest merges (check these by hand)")
    for entry in sorted(canonical, key=lambda c: -c["n_merged"])[:8]:
        if entry["n_merged"] < 2:
            continue
        print(f"  x{entry['n_merged']} {entry['official_description'][:70]!r}")
        for variant in entry["variants"][:2]:
            print(f"        <- {variant['description'][:66]!r}")

    print(f"\nwrote {out_dir / 'hyperedges.json'} and {out_dir / 'hyperedge_linking.json'}")

    if args.clustering == "transitive":
        oversized = [c for c in canonical if c["n_merged"] > args.max_component]
        if oversized:
            print(f"\n[!] {len(oversized)} components exceed {args.max_component} members. "
                  "Use --clustering star.")
    if stats["n_candidate_pairs"] and stats["confirm_rate"] > 0.9:
        print("\n[!] the NLI model confirms nearly everything the join proposes, so "
              "it is not discriminating.")


if __name__ == "__main__":
    main()
