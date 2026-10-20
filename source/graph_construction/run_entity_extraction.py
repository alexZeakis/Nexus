"""
Step 4 of 7 - Entity Extraction, over the extracted facts.

    <out_dir>/schema.json         input: the frozen schema (run_label_extraction.py)
    <out_dir>/fact_units.json     input: deduplicated facts with their chunk ids
    <out_dir>/entities.json       output: typed mentions per fact, plus stats

Resumable: facts already in entities.json are skipped on re-run.

Usage:
    python run_entity_extraction.py --out-dir ./output/mydataset --device cuda:0
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from entity_extraction import DEFAULT_MODEL, GLiNERExtractor, extract_corpus, load_schema
from utils import read_json, write_json


def as_fact_mentions(per_fact: dict) -> dict:
    """The extractor names its unit 'chunk'; here the unit is a fact."""
    out = {}
    for fact_id, mentions in per_fact.items():
        out[fact_id] = [
            {"fact_id": fact_id,
             **{k: v for k, v in m.items() if k not in ("chunk_id", "chunk_index", "fact_id")}}
            for m in mentions
        ]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Nexus Entity Extraction over facts")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default=None, help="e.g. cuda:0")
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--label-batch-size", type=int, default=15)
    p.add_argument("--no-descriptions", action="store_true")
    p.add_argument("--report-every", type=int, default=500)
    p.add_argument("--restart", action="store_true", help="ignore existing entities.json")
    args = p.parse_args()

    out_dir: Path = args.out_dir
    facts = [{"chunk_id": str(f["fact_id"]), "content": f["content"]}
             for f in read_json(out_dir / "fact_units.json")]
    schema = load_schema(read_json(out_dir / "schema.json"))
    out_path = out_dir / "entities.json"

    done = {}
    if out_path.exists() and not args.restart:
        done = read_json(out_path).get("mentions", {})
        print(f"resuming    : {len(done)} facts already done")
    print(f"facts       : {len(facts)}")
    print(f"schema      : {len(schema)} labels")

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
          f"threshold {args.threshold}, {args.label_batch_size} labels per pass")

    def on_progress(index: int, total: int) -> None:
        if index % args.report_every == 0:
            print(f"  {index:>6}/{total} facts", flush=True)

    def on_flush(per_fact: dict) -> None:
        write_json(out_path, {"mentions": as_fact_mentions(per_fact)})

    per_fact, stats = extract_corpus(
        facts, extractor, done=done, on_progress=on_progress,
        flush_every=args.report_every, on_flush=on_flush,
    )
    per_fact = as_fact_mentions(per_fact)

    label_counts = Counter(m["label"] for ms in per_fact.values() for m in ms)
    total = sum(label_counts.values())
    empty = [fid for fid, ms in per_fact.items() if not ms]

    print("\n" + "=" * 66)
    print(f"facts processed  : {stats.n_chunks} in {stats.seconds:.1f}s "
          f"({stats.seconds / max(1, stats.n_chunks) * 1000:.0f}ms per fact)")
    print(f"mentions         : {total} ({total / max(1, len(per_fact)):.1f} per fact)")
    print(f"empty facts      : {len(empty)} ({len(empty) / max(1, len(per_fact)):.1%})")
    print(f"descriptions     : {'sent' if extractor.descriptions_ok else 'not supported'}")
    print("=" * 66)

    print("\nmentions per label")
    for label in schema:
        count = label_counts.get(label, 0)
        print(f"  {label:<24} {count:>6}{'   <- never fires' if count == 0 else ''}")

    write_json(out_path, {
        "config": {
            "unit": "fact",
            "model": args.model,
            "threshold": args.threshold,
            "label_batch_size": args.label_batch_size,
            "descriptions": bool(extractor.descriptions_ok),
        },
        "stats": {
            "n_facts": len(per_fact),
            "n_mentions": total,
            "empty_facts": empty,
            "per_label": dict(label_counts),
            "seconds": round(stats.seconds, 2),
        },
        "mentions": per_fact,
    })
    print(f"\nwrote {out_path}")

    if len(empty) / max(1, len(per_fact)) > 0.1:
        print("\n[!] more than a tenth of the facts produced no entity. "
              "Lower --threshold or revisit the schema.")


if __name__ == "__main__":
    main()
