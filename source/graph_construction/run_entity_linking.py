"""
Step 5 of 7 - Entity Linking, over the fact-level mentions.

    <out_dir>/entities.json         input, never modified (run_entity_extraction.py)
    <out_dir>/fact_units.json       input: fact texts and their chunk ids
    <out_dir>/chunks.json           input: chunk order, for choosing official names
    <out_dir>/entities_linked.json  output: the mentions, plus official_name
    <out_dir>/linking.json          output: canonical entities and statistics

Blocking and matching serialize a mention differently on purpose. Blocking
embeds the name and its type, since that is what should decide which pairs are
worth comparing at all. Matching reads --serialize, context by default, since
the surrounding snippet is the evidence the matcher adjudicates on.

Usage:
    python run_entity_linking.py --out-dir ./output/mydataset --device cuda:0
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from entity_linking import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_MATCH_MODEL,
    FlanT5Matcher,
    build_clusters,
    build_mentions,
    candidate_pairs,
    canonicalize,
    group_by_exact,
    serialize,
)
from utils import read_json, write_json

#: How a mention is rendered for the blocking encoder. Name and type only.
BLOCK_SERIALIZE = "label"


def main() -> None:
    p = argparse.ArgumentParser(description="Nexus Entity Linking over facts")
    p.add_argument("--out-dir", required=True, type=Path)
    # blocking
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--k", type=int, default=5, help="top-k for the ANN self-join")
    p.add_argument("--threshold", type=float, default=0.85, help="ANN similarity cutoff")
    p.add_argument("--lexical-threshold", type=float, default=0.85)
    p.add_argument("--serialize", default="context", choices=["name", "label", "context"],
                   help="how a mention is rendered for the MATCHER")
    # matching
    p.add_argument("--match-model", default=DEFAULT_MATCH_MODEL)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--decision-threshold", type=float, default=0.5)
    p.add_argument("--max-component", type=int, default=25)
    p.add_argument("--device", default=None, help="e.g. cuda:0")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--cache", type=Path, default=None, help="optional JSON file of pair decisions")
    args = p.parse_args()

    out_dir: Path = args.out_dir
    entities_path = out_dir / "entities.json"
    linked_path = out_dir / "entities_linked.json"

    payload = read_json(entities_path)
    facts = read_json(out_dir / "fact_units.json")
    chunk_pos = {c["chunk_id"]: i for i, c in enumerate(read_json(out_dir / "chunks.json"))}

    # Corpus order of facts: position of their first source chunk, then fact id.
    # canonicalize() names each entity after its first mention in this order.
    ranked = sorted(facts, key=lambda f: (min(chunk_pos.get(c, 10**9) for c in f["chunk_ids"]), f["fact_id"]))
    rank = {str(f["fact_id"]): r for r, f in enumerate(ranked)}
    chunks_of = {str(f["fact_id"]): f["chunk_ids"] for f in facts}

    # The linking module calls its unit a chunk; here each fact is one.
    units = [{"chunk_id": str(f["fact_id"]), "content": f["content"]} for f in facts]
    ordered_payload = {"mentions": {
        fid: [{**m, "chunk_index": rank.get(fid, 0)} for m in ms]
        for fid, ms in payload["mentions"].items()
    }}

    mentions = build_mentions(ordered_payload, units)
    groups = group_by_exact(mentions)
    print(f"mentions          : {len(mentions)}")
    print(f"after exact merge : {len(groups)} groups "
          f"({1 - len(groups) / max(1, len(mentions)):.1%} collapsed lexically)")

    # ------------------------------------------------------------- blocking
    from embeddings import load_embedder

    embedder = load_embedder(args.embed_model, device=args.device)
    start = time.time()
    pairs, block_stats = candidate_pairs(
        groups, embed_fn=embedder, k=args.k, threshold=args.threshold,
        lexical_threshold=args.lexical_threshold, serialize_mode=BLOCK_SERIALIZE,
    )
    print(f"candidate pairs   : {len(pairs)} "
          f"({block_stats['ann']} from ANN, {block_stats['lexical']} lexical) "
          f"across {block_stats['labels']} labels in {time.time() - start:.1f}s")

    # ------------------------------------------------------------- matching
    cache: dict[str, float] = {}
    if args.cache is not None and args.cache.exists():
        cache = read_json(args.cache)
        print(f"cache             : {len(cache)} decisions reused")

    def cache_key(i: int, j: int) -> str:
        return f"{groups[i].label}||{groups[i].normalized}||{groups[j].normalized}"

    todo = [(i, j) for i, j in pairs if cache_key(i, j) not in cache]
    if todo:
        matcher = FlanT5Matcher(
            model_name=args.match_model, batch_size=args.batch_size,
            decision_threshold=args.decision_threshold, device=args.device, dtype=args.dtype,
        )
        print(f"matcher           : {args.match_model} on {matcher.device} "
              f"({matcher.dtype}), {len(todo)} pairs to adjudicate")
        start = time.time()
        records = [(serialize(groups[i], args.serialize), serialize(groups[j], args.serialize))
                   for i, j in todo]
        for (i, j), (_, prob) in zip(todo, matcher.match(records)):
            cache[cache_key(i, j)] = round(float(prob), 4)
        print(f"                    {len(todo)} pairs in {time.time() - start:.1f}s "
              f"({matcher.calls} forward passes)")

    confirmed = [(i, j) for i, j in pairs if cache.get(cache_key(i, j), 0.0) >= args.decision_threshold]
    print(f"confirmed matches : {len(confirmed)} of {len(pairs)} "
          f"({len(confirmed) / max(1, len(pairs)):.0%})")
    if args.cache is not None:
        write_json(args.cache, cache)

    # ------------------------------------------------- clusters and output
    clusters = build_clusters(groups, confirmed)
    assignment, entities = canonicalize(groups, clusters)
    for e in entities:                              # "context" holds fact ids here
        e["fact_ids"] = e.pop("context")
        e["chunk_ids"] = sorted({c for fid in e["fact_ids"] for c in chunks_of.get(fid, [])},
                                key=lambda c: chunk_pos.get(c, 10**9))

    linked_mentions = {
        fid: [{**m, **assignment.get((fid, pos), {})} for pos, m in enumerate(ms)]
        for fid, ms in payload["mentions"].items()
    }
    config = {
        "unit": "fact", "embed_model": args.embed_model, "match_model": args.match_model,
        "k": args.k, "threshold": args.threshold, "lexical_threshold": args.lexical_threshold,
        "serialize_blocking": BLOCK_SERIALIZE, "serialize_matching": args.serialize,
        "decision_threshold": args.decision_threshold,
    }
    write_json(linked_path, {
        "source": "entities.json",
        "config": config,
        "stats": {
            "n_mentions": len(mentions), "n_groups": len(groups), "n_entities": len(entities),
            "n_candidate_pairs": len(pairs), "n_confirmed": len(confirmed),
        },
        "mentions": linked_mentions,
    })
    write_json(out_dir / "linking.json", {"config": config, "entities": entities})

    sizes = Counter(len(v) for v in clusters.values())
    merged = [e for e in entities if e["n_groups"] > 1]
    print("\n" + "=" * 66)
    print(f"canonical entities: {len(entities)} from {len(mentions)} mentions "
          f"({1 - len(entities) / max(1, len(mentions)):.1%} collapsed overall)")
    print(f"semantic merges   : {len(merged)} entities span more than one group")
    print("component sizes   : " +
          ", ".join(f"{size}x{count}" for size, count in sorted(sizes.items())[:8]))
    print("=" * 66)

    print("\nlargest merges (check these by hand)")
    for e in sorted(entities, key=lambda e: -e["n_groups"])[:10]:
        if e["n_groups"] >= 2:
            print(f"  {e['official_name']!r} ({e['label']}) <- {e['aliases'][:6]}")

    oversized = [e for e in entities if e["n_groups"] > args.max_component]
    if oversized:
        print(f"\n[!] {len(oversized)} components exceed {args.max_component} groups. "
              "Merging is transitive, so inspect these before trusting the graph:")
        for e in oversized[:5]:
            print(f"    {e['official_name']!r}: {e['n_groups']} groups, {e['n_mentions']} mentions")

    print(f"\nwrote {linked_path} and {out_dir / 'linking.json'}")
    print(f"      {entities_path} was not modified")


if __name__ == "__main__":
    main()
