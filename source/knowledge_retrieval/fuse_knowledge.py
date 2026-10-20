"""
Fuse the semantic and structural retrieval files into one.

Both retrievers run over the same graph and return the same kind of object.
They differ only in how they select it: the semantic one matches the question
embedding against hyperedge embeddings in one step, the structural one links
query entities, joins them against hyperedge participants and expands along
shared participants. This writes the union, so a question is answered from
both views at once.

The output is a retrieval file like any other, so generate_answers.py and
score_knowledge.py read it with no changes and --top-k / --chunk-top-k still
apply downstream.

HYPEREDGES  (--fuse)

    rrf   (default)  reciprocal rank fusion: each list votes for an item with
                     weight 1/(k + its rank), k = 60, and the votes add up
    max              the item keeps the higher of its two scores
    sum              the two scores are added

max is the obvious reading of "keep the best score" and it is the right one
only if the two scores are on the same scale. They are not. The embedding
cosine sits around a median of 0.58 and the join score, a jaccard carried
through a decayed product during expansion, sits around 0.06. Under max almost
every semantic item outranks almost every structural one, so the merged list
is the semantic ranking with the structural leftovers underneath and the
fusion does nothing.

rrf never compares the two numbers, only the positions, so the scales never
have to meet. Its useful property is that an item both retrievers returned
beats an item only one of them found, which is the entire reason to fuse. With
k = 60 the gap between first and tenth place is small, so agreement between
the lists counts for more than being first in either one.

CHUNKS  (--chunks)

    derive  (default)  rebuild the chunk list from the fused hyperedges
    fuse               fuse the two chunk lists directly, like the hyperedges

A chunk is not a retrieval result in either file. It is a shadow of the
hyperedge list: the retrievers give each chunk the best score among the
retrieved hyperedges extracted from it, so a chunk is never ranked against the
question itself. Fusing the two chunk lists separately therefore produces a
chunk ranking that no longer matches the hyperedges the model is shown, and a
chunk can survive the cap while every hyperedge pointing at it was dropped.

`derive` applies the retrievers' own rule to the fused list instead:

    score = the best fused score among the hyperedges pointing at the chunk
    tie   = how many of them point at it

which keeps the two sections consistent and puts the chunk score on the rrf
scale, where a chunk backed by a hyperedge both retrievers agreed on outranks
a chunk backed by one retriever's top hit.

IDENTITY

Hyperedges are matched on `node_id` and chunks on `chunk_id`, falling back to
the normalized text when a file carries neither. The run reports how many
merges matched on an id and how many on text alone, so a schema without ids
does not quietly stop deduplicating.

WHAT THE SUMMARY IS FOR

`jaccard` and `both` are results, not diagnostics. They say how much of what
one retriever returns the other already had. High overlap explains a variant
that changes nothing. Low overlap with no gain in accuracy is a different and
stronger claim: the two views disagree about which facts matter and it makes
no difference to the answer.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import OrderedDict
from pathlib import Path

LABELS = ("semantic", "structural")
DEFAULT_FUSE = "rrf"
DEFAULT_CHUNKS = "derive"
DEFAULT_RRF_K = 60

HYPEREDGES, CHUNKS = "hyperedges", "chunks"

#: tried in order, the first with a value wins. node_id keys a hyperedge,
#: chunk_id keys a chunk. `node_ids` (plural, on chunks) is not an id.
ID_FIELDS = ("node_id", "chunk_id", "hyperedge_id", "fact_id", "edge_id", "id")


# ------------------------------------------------------------------ helpers
def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def has_text(item):
    return bool((item.get("text") or "").strip())


def sections_of(result):
    """A result's knowledge as {section: [item, ...]}, flat list included."""
    know = result.get("knowledge", [])
    if isinstance(know, dict):
        return OrderedDict((k, list(v or [])) for k, v in know.items())
    return OrderedDict([(HYPEREDGES, list(know or []))])


def item_key(item):
    """(key, matched_on_an_id). Falls back to the normalized text."""
    for f in ID_FIELDS:
        v = item.get(f)
        if v not in (None, "", []):
            return f"{f}={v}", True
    return f"text={norm(item.get('text'))}", False


def score_of(item):
    try:
        return float(item.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def rank_of(item, position):
    """The rank the retriever wrote, or the item's position in its list."""
    try:
        r = int(item.get("rank") or position)
        return r if r > 0 else position
    except (TypeError, ValueError):
        return position


def results_by_qi(blob, path):
    """{qi: result}. Accepts a list or a dict of results, as the files vary."""
    res = blob.get("results", blob)
    if isinstance(res, dict):
        res = [res[k] for k in sorted(res, key=lambda x: int(x))]
    if not isinstance(res, list):
        raise SystemExit(f"{path}: no 'results' list")
    out = OrderedDict()
    for i, r in enumerate(res):
        out[int(r.get("qi", i))] = r
    return out


# -------------------------------------------------------------------- fuse
def fuse_lists(per_source, mode, rrf_k):
    """Several ranked lists of the same kind of item, merged into one.

    `per_source` is [(label, [item, ...]), ...]. Each item keeps the fields it
    arrived with, `score` becomes the fused score, and the per-source scores
    and ranks are kept beside it so the merge can be read back without
    re-running anything.
    """
    merged, on_id_merges, on_text_merges = OrderedDict(), 0, 0

    for label, items in per_source:
        for position, item in enumerate(items, start=1):
            if not has_text(item):
                continue
            key, on_id = item_key(item)
            rank = rank_of(item, position)
            entry = merged.get(key)
            if entry is None:
                entry = merged[key] = {"item": dict(item), "scores": {}, "ranks": {}}
            else:
                on_id_merges += on_id
                on_text_merges += not on_id
                # the two retrievers may truncate text differently, and one
                # may carry a field the other left out
                if len(item.get("text") or "") > len(entry["item"].get("text") or ""):
                    entry["item"]["text"] = item["text"]
                for k, v in item.items():
                    if (k not in ("score", "text", "rank")
                            and entry["item"].get(k) in (None, "", [])):
                        entry["item"][k] = v
            # a list naming the same item twice keeps its better position
            entry["scores"][label] = max(entry["scores"].get(label, float("-inf")),
                                         score_of(item))
            entry["ranks"][label] = min(entry["ranks"].get(label, rank), rank)

    out = []
    for entry in merged.values():
        scores, ranks = entry["scores"], entry["ranks"]
        if mode == "rrf":
            fused = sum(1.0 / (rrf_k + r) for r in ranks.values())
        elif mode == "sum":
            fused = sum(scores.values())
        else:
            fused = max(scores.values())
        item = entry["item"]
        item.pop("rank", None)
        item["score"] = round(fused, 6)
        item["fused_from"] = sorted(ranks)
        item["source_scores"] = {k: round(v, 6) for k, v in scores.items()}
        item["source_ranks"] = dict(ranks)
        out.append(item)

    out.sort(key=lambda it: (-it["score"], -len(it["fused_from"]), norm(it.get("text"))))
    for i, item in enumerate(out, start=1):
        item["rank"] = i
    return out, on_id_merges, on_text_merges


def derive_chunks(fused_hyperedges, known):
    """The chunk list implied by the fused hyperedges.

    This is the retrievers' own rule, fed the fused ranking: a chunk takes the
    best score among the hyperedges extracted from it, and ties go to the
    chunk more of them point at. `known` maps chunk_id to the chunk entry
    either file wrote, which is where the text comes from.

    Returns (chunks, missing), where `missing` counts chunk ids a hyperedge
    named but no file carried text for. Those are dropped, since a chunk with
    no text is nothing to show the model.
    """
    best, missing = OrderedDict(), 0
    for he in fused_hyperedges:
        for cid in he.get("chunk_ids") or []:
            source = known.get(cid)
            if source is None or not has_text(source):
                missing += 1
                continue
            entry = best.get(cid)
            if entry is None:
                entry = best[cid] = {
                    "text": source["text"],
                    "chunk_id": cid,
                    "score": float("-inf"),
                    "from_hyperedges": 0,
                    "node_ids": [],
                    "fused_from": set(),
                    **({"chunk_index": source["chunk_index"]}
                       if source.get("chunk_index") is not None else {}),
                }
            entry["score"] = max(entry["score"], he["score"])
            entry["from_hyperedges"] += 1
            if he.get("node_id"):
                entry["node_ids"].append(he["node_id"])
            entry["fused_from"].update(he.get("fused_from") or [])

    chunks = sorted(best.values(),
                    key=lambda c: (-c["score"], -c["from_hyperedges"], c["chunk_id"]))
    for i, c in enumerate(chunks, start=1):
        c["score"] = round(c["score"], 6)
        c["fused_from"] = sorted(c["fused_from"])
        c["rank"] = i
    return chunks, missing


# ----------------------------------------------------------------- reports
def keys_of(items):
    return {item_key(i)[0] for i in items if has_text(i)}


def mean(xs, nd=2):
    return round(statistics.fmean(xs), nd) if xs else 0.0


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="Fuse the semantic and structural retrieval files into one")
    ap.add_argument("--semantic", required=True,
                    help="retrieval file from the embedding retriever")
    ap.add_argument("--structural", required=True,
                    help="retrieval file from the join and expansion retriever")
    ap.add_argument("--out", default=None, help="output file")
    ap.add_argument("--out-dir", default=None,
                    help="output directory; the file is named by --name")
    ap.add_argument("--name", default=None,
                    help="run name, and the file name under --out-dir "
                         "(default: the structural file's name plus -fused)")
    ap.add_argument("--fuse", choices=["rrf", "max", "sum"], default=DEFAULT_FUSE,
                    help="how two hyperedge lists become one (default: rrf). rrf "
                         "uses only the ranks, so the two score scales do not have "
                         "to agree. max and sum are meaningful only if they do.")
    ap.add_argument("--chunks", choices=["derive", "fuse"], default=DEFAULT_CHUNKS,
                    help="derive (default) rebuilds the chunk list from the fused "
                         "hyperedges, keeping the two sections consistent. fuse "
                         "merges the two chunk lists directly, for comparison.")
    ap.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K,
                    help=f"rrf constant (default {DEFAULT_RRF_K}). Larger means "
                         f"agreement between the lists counts for more than rank.")
    args = ap.parse_args()

    if not args.out and not args.out_dir:
        ap.error("pass --out or --out-dir")

    paths = {"semantic": Path(args.semantic), "structural": Path(args.structural)}
    for label, p in paths.items():
        if not p.is_file():
            raise SystemExit(f"missing {label} file: {p}")

    name = args.name or f"{paths['structural'].stem}-fused"
    out_path = Path(args.out) if args.out else Path(args.out_dir) / f"{name}.json"

    blobs = {k: json.loads(p.read_text()) for k, p in paths.items()}
    byqi = {k: results_by_qi(b, paths[k]) for k, b in blobs.items()}

    qis = sorted(set(byqi["semantic"]) | set(byqi["structural"]))
    only = {k: sorted(set(byqi[k]) - set(byqi[other]))
            for k, other in (("semantic", "structural"), ("structural", "semantic"))}

    for label in LABELS:
        cfg = blobs[label].get("config", {})
        print(f"{label:<11}: {paths[label]}  "
              f"[{cfg.get('run', '?')}, {len(byqi[label])} questions]")
    print(f"fuse       : {args.fuse}"
          + (f" (k={args.rrf_k})" if args.fuse == "rrf" else "")
          + f", chunks={args.chunks}")
    print(f"questions  : {len(qis)} in the union")
    if only["semantic"] or only["structural"]:
        print(f"  !! {len(only['semantic'])} only in the semantic file, "
              f"{len(only['structural'])} only in the structural file. Those are "
              f"fused from one side, so they are not a comparison. Check the two "
              f"runs covered the same questions before putting this in a table.")

    results = []
    id_merges = text_merges = missing_text = 0
    stat = {HYPEREDGES: {"a": [], "b": [], "m": [], "o": [], "j": [], "both": []},
            CHUNKS: {"a": [], "b": [], "m": [], "o": [], "j": [], "both": []}}
    raw = {label: {HYPEREDGES: [], CHUNKS: []} for label in LABELS}
    empty = {label: 0 for label in LABELS}

    for qi in qis:
        res = {label: byqi[label].get(qi, {}) for label in LABELS}
        secs = {label: sections_of(r) for label, r in res.items()}
        got = {label: {sec: secs[label].get(sec, []) for sec in (HYPEREDGES, CHUNKS)}
               for label in LABELS}

        for label in LABELS:
            if qi in byqi[label] and not got[label][HYPEREDGES]:
                empty[label] += 1
            for sec in (HYPEREDGES, CHUNKS):
                raw[label][sec].extend(score_of(i) for i in got[label][sec]
                                       if has_text(i))

        hyper, bid, btext = fuse_lists(
            [(label, got[label][HYPEREDGES]) for label in LABELS],
            args.fuse, args.rrf_k)
        id_merges += bid
        text_merges += btext

        if args.chunks == "derive":
            known = {}
            for label in LABELS:
                for c in got[label][CHUNKS]:
                    cid = c.get("chunk_id")
                    if cid and has_text(c) and cid not in known:
                        known[cid] = c
            chunks, miss = derive_chunks(hyper, known)
            missing_text += miss
        else:
            chunks, bid, btext = fuse_lists(
                [(label, got[label][CHUNKS]) for label in LABELS],
                args.fuse, args.rrf_k)
            id_merges += bid
            text_merges += btext

        knowledge = OrderedDict([(HYPEREDGES, hyper), (CHUNKS, chunks)])

        for sec, fused in knowledge.items():
            ka, kb = keys_of(got["semantic"][sec]), keys_of(got["structural"][sec])
            d = stat[sec]
            d["a"].append(len(ka))
            d["b"].append(len(kb))
            d["m"].append(len(fused))
            d["o"].append(len(ka & kb))
            if ka or kb:
                d["j"].append(len(ka & kb) / len(ka | kb))
            d["both"].append(sum(1 for i in fused if len(i.get("fused_from", [])) > 1))

        base = res["structural"] or res["semantic"]
        results.append({
            "qi": qi,
            "question": base.get("question") or res["semantic"].get("question", ""),
            "knowledge": knowledge,
            "n_knowledge": {k: len(v) for k, v in knowledge.items()},
            "sources": [label for label in LABELS if qi in byqi[label]],
        })

    summary = {
        "questions": len(qis),
        "only_semantic": len(only["semantic"]),
        "only_structural": len(only["structural"]),
        "empty_semantic": empty["semantic"],
        "empty_structural": empty["structural"],
        "merged_on_id": id_merges,
        "merged_on_text": text_merges,
        "chunk_ids_without_text": missing_text,
        "mean_knowledge": mean([sum(r["n_knowledge"].values()) for r in results]),
        "sections": {
            sec: {"mean_semantic": mean(d["a"]),
                  "mean_structural": mean(d["b"]),
                  "mean_fused": mean(d["m"]),
                  "mean_overlap": mean(d["o"]),
                  "mean_jaccard": mean(d["j"], 3),
                  "mean_from_both": mean(d["both"])}
            for sec, d in stat.items()},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"config": {"run": name, "method": "fused",
                    "fuse": args.fuse,
                    "rrf_k": args.rrf_k if args.fuse == "rrf" else None,
                    "chunks": args.chunks,
                    "semantic": str(paths["semantic"]),
                    "structural": str(paths["structural"]),
                    "knowledge_sections": [HYPEREDGES, CHUNKS],
                    "llm_calls_per_question": 0},
         "summary": summary,
         "results": results}, indent=2, ensure_ascii=False))

    print()
    print(f"  {'':<12}{'sem':>7}{'struct':>8}{'fused':>7}"
          f"{'shared':>8}{'jaccard':>9}{'both':>7}")
    for sec, d in summary["sections"].items():
        print(f"  {sec:<12}{d['mean_semantic']:>7}{d['mean_structural']:>8}"
              f"{d['mean_fused']:>7}{d['mean_overlap']:>8}"
              f"{d['mean_jaccard']:>9}{d['mean_from_both']:>7}")
    print("  (per question. shared = in both input lists, both = fused items "
          "backed by both)")

    if empty["semantic"] or empty["structural"]:
        print(f"\n  no hyperedges at all: {empty['semantic']} questions in the "
              f"semantic file, {empty['structural']} in the structural one")
    print(f"  merges: {id_merges} on an id, {text_merges} on text alone")
    if text_merges and not id_merges:
        print("  !! nothing merged on an id, so deduplication is text only. "
              "Check ID_FIELDS against the ids your files carry.")
    if missing_text:
        print(f"  !! {missing_text} chunk ids were named by a hyperedge but no "
              f"file carried their text, so they were dropped")

    print("\n  score distributions (min / median / max):")
    for label in LABELS:
        for sec in (HYPEREDGES, CHUNKS):
            xs = raw[label][sec]
            if xs:
                print(f"    {label:<11}{sec:<12}{min(xs):.3f} / "
                      f"{statistics.median(xs):.3f} / {max(xs):.3f}   n={len(xs)}")
    ma, mb = (raw["semantic"][HYPEREDGES], raw["structural"][HYPEREDGES])
    if ma and mb:
        a, b = statistics.median(ma), statistics.median(mb)
        ratio = max(a, b) / max(min(a, b), 1e-9)
        if ratio > 1.25:
            higher = "structural" if b > a else "semantic"
            tail = ("" if args.fuse == "rrf" else "  Use --fuse rrf.")
            print(f"  {'  ' if args.fuse == 'rrf' else '!!'} hyperedge medians "
                  f"differ by {ratio:.1f}x ({a:.3f} against {b:.3f}), so the "
                  f"{higher} retriever would take most of the ranking under max "
                  f"or sum.{tail}")

    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()