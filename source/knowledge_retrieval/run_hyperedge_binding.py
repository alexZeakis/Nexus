"""
Step 3 of 3 - Hyperedge Binding, and the per-question result.

BINDING

The token join can only retrieve what a question names. "What is the title of
one of the works that they collaborated on, alongside Weak and the Wicked?"
names no person, so the fact holding the answer shares nothing with it.

Binding closes that structurally. Every retrieved hyperedge names its
participants, so the entities of a hit are added to the query and the join runs
again, with no model and no rewriting: it is a second pass over the same
prepared index.

One query is built PER RESULT, not one per question. With Q = [A, B] and two
hits [C, A] and [B, D], union binding issues two queries, [A, B, C] and
[A, B, D]. It never issues the aggregate [A, B, C, D], which would be a larger
and far less selective set than either branch.

    --expand-tokens union         Q + S, everything
                    drop-matched  (Q \\ M) + (S \\ M), dropping what the seed
                                  matched on. "What else, besides this one."
                    entities-only S \\ M, the seed's new entities alone. Highest
                                  recall on the bridge, no question-side filter.

    --max-element-df N            drop query elements whose rarest token appears
                                  in more than N hyperedges. "title" and "works"
                                  are element-level stopwords and drag in every
                                  fact that happens to use the word.

ROUNDS

--rounds R repeats the operation. Each result of round r seeds round r+1 and
carries its own branch query set forward, so a branch that reached a fact
through [A, B, C] expands from [A, B, C] and not from the original question.
Branching is bounded by --seeds, which caps how many results of a round go on
to seed the next one, and by --max-added, which caps what a question gains per
round. A bound hit scores its seed's score times its own set similarity, so a
fact found three rounds out ranks below the fact that led to it.

R = 1 reaches any fact two hops from the question, which covers what the
benchmarks ask.

KNOWLEDGE, TWO SECTIONS

`knowledge` is a dict, not a list:

    "knowledge": {"hyperedges": [{"text", "score", "rank", "node_id",
                                  "participants", "chunk_ids"}, ...],
                  "chunks":     [{"text", "score", "rank", "chunk_id",
                                  "from_hyperedges", "node_ids"}, ...]}

The chunks are not retrieved separately. Each one is a passage a kept hyperedge
was extracted from, and it inherits that hyperedge's score, so the section costs
no extra retrieval: it only widens each hit back to the text it came from. That
is the context HyperGraphRAG hands its reader, so writing both sections is what
makes the comparison a comparison of retrieval rather than of how much text each
method happens to supply. Both are written by default and
`generate_answers --knowledge` decides which the model actually sees.

    <knowledge>/retrieved.json         input: ranked hyperedges per question
    <knowledge>/query_entities.json    input: each question's own tokens and qi
    <graph>/hyperedge_linking.json     input: the collection
    <graph>/postings.json              input: which hyperedges the graph kept
    <graph>/chunks.json                input: the chunk section
    --out                              output: one ranked list per question

Usage:
    python run_hyperedge_binding.py \\
        --retrieved      ../../output/Mix/knowledge/retrieved.json \\
        --query-entities ../../output/Mix/knowledge/query_entities.json \\
        --hyperedges     ../../output/Mix/graph/hyperedge_linking.json \\
        --postings       ../../output/Mix/graph/postings.json \\
        --chunks         ../../output/Mix/graph/chunks.json \\
        --out            ../../output/Mix/retrieval/nexus_union.json \\
        --expand-tokens union --seeds 10 --k 10 --rounds 1
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "graph_construction"))

from utils import read_json, write_json

from run_hyperedge_retrieval import clean_set, load_hyperedges

_MODULES = {"jaccard": "pytokenjoin.jaccard.join_knn",
            "edit": "pytokenjoin.edit.join_knn"}

_WORD = re.compile(r"[a-z0-9']+")

CHUNK_FIELDS = ("content", "text", "chunk", "passage", "body")


# --------------------------------------------------------------------------- #
# Query expansion
# --------------------------------------------------------------------------- #


def tokens_of(element: str) -> list[str]:
    return _WORD.findall(element.lower())


def build_token_df(edges: list[dict]) -> Counter:
    """How many hyperedges each token appears in, over the whole collection."""
    df = Counter()
    for edge in edges:
        seen = set()
        for element in edge["elements"]:
            seen.update(tokens_of(element))
        df.update(seen)
    return df


def drop_generic(elements: list[str], df: Counter, limit: int | None) -> list[str]:
    """Remove elements whose rarest token is common across the collection.

    An element is only as specific as its most distinctive token: "works" has
    one token and it is everywhere, "Weak and the Wicked" has "wicked", which
    is not. Never returns empty, since a query with no elements retrieves
    nothing at all.
    """
    if not limit:
        return elements
    kept = [e for e in elements
            if (toks := tokens_of(e)) and min(df.get(t, 0) for t in toks) <= limit]
    return kept or elements


def matched_elements(query: list[str], seed_elements: list[str]) -> set[str]:
    """Query elements the seed shares, by token overlap rather than identity.

    The join is fuzzy, so "Weak and the Wicked" in the question matches
    "Weak and the Wicked." in the graph and both have to count as matched.
    """
    seed_tokens = [set(tokens_of(e)) for e in seed_elements]
    out = set()
    for element in query:
        qt = set(tokens_of(element))
        if not qt:
            continue
        if any(len(qt & st) / len(qt | st) >= 0.5 for st in seed_tokens if st):
            out.add(element)
    return out


def expand_query(query: list[str], seed_elements: list[str], mode: str) -> list[str]:
    """The next query set of ONE branch, from its own query and one hit."""
    if mode == "union":
        return clean_set(list(query) + list(seed_elements))
    matched = matched_elements(query, seed_elements)
    new = [e for e in seed_elements if e not in matched]
    if mode == "entities-only":
        return clean_set(new)
    return clean_set([e for e in query if e not in matched] + new)   # drop-matched


# --------------------------------------------------------------------------- #
# Chunks
# --------------------------------------------------------------------------- #


def first_field(obj, fields, default=""):
    for f in fields:
        if isinstance(obj, dict) and obj.get(f):
            v = obj[f]
            return " ".join(map(str, v)) if isinstance(v, list) else str(v)
    return default


def load_chunks(path):
    """chunk_id -> (position in the corpus, text).

    Both chunk file shapes are read: a dict keyed by chunk id, and a list of
    records with a `chunk_id`. The position makes results comparable across
    graphs; the text is what goes in the `chunks` section of the contract.
    """
    if not path or not Path(path).exists():
        return {}
    data = read_json(path)
    items = (list(data.items()) if isinstance(data, dict)
             else [((c.get("chunk_id", str(i)) if isinstance(c, dict) else str(i)), c)
                   for i, c in enumerate(data)])
    out = {}
    for i, (cid, c) in enumerate(items):
        text = c if isinstance(c, str) else first_field(c, CHUNK_FIELDS, "")
        out[str(cid)] = (i, text.strip())
    return out


def load_hyperedge_text(path):
    """canonical_id -> (description, participant_names), to fill what retrieval omitted."""
    if not path or not Path(path).exists():
        return {}
    blob = read_json(path)
    return {e["canonical_id"]: (e.get("official_description", ""),
                                e.get("participant_names", []))
            for e in blob.get("canonical", [])}


def chunks_of(hyperedges, chunks, top_k=None):
    """The source chunks of the retrieved hyperedges, ranked.

    A chunk is as relevant as the best hyperedge that came out of it, so it
    inherits that score; ties go to the chunk more hyperedges point at. This
    adds no retrieval of its own, it only widens each hit back to the passage
    it was extracted from, which is the context the baselines hand the reader.
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


# --------------------------------------------------------------------------- #
# Binding
# --------------------------------------------------------------------------- #


def bind(results, queries, edges, by_id, df, args) -> dict:
    """Expand every hit into its own query, round by round.

    A frontier entry is (question_id, hit, the query set of the branch that
    found it). Round r turns each entry into one expanded query, joins them all
    in one pass, and the hits that come back become the frontier of round r+1.
    """
    import importlib
    import pandas as pd

    module = importlib.import_module(_MODULES[args.similarity])
    tj = module.TokenJoin()
    right = pd.DataFrame({"id": list(range(len(edges))),
                          "elements": [e["elements"] for e in edges]})
    tj.tokenjoin_prepare(right, "id", "elements")

    # Round 1 starts from the retrieved hits, each carrying the question's own set.
    frontier = []
    for qid, hits in results.items():
        q0 = clean_set(queries.get(qid, {}).get("tokens", []))
        for hit in hits[: args.seeds]:
            frontier.append((qid, hit, q0))

    added_per_round: list[int] = []
    n_merged = 0
    gained: set[str] = set()

    for round_no in range(1, args.rounds + 1):
        plan = []                       # (qid, seed, expanded query)
        for qid, seed, qset in frontier:
            edge = by_id.get(seed["hyperedge_id"])
            if edge is None:
                continue
            expanded = expand_query(qset, edge["elements"], args.expand_tokens)
            expanded = drop_generic(expanded, df, args.max_element_df)
            if expanded:
                plan.append((qid, seed, expanded))

        if not plan:
            print(f"round {round_no}       : nothing left to expand")
            break

        left = pd.DataFrame({"id": list(range(len(plan))),
                             "elements": [q for _, _, q in plan]})
        # k+1, because a seed always wins its own expanded query and is then skipped.
        start = time.time()
        frame = tj.tokenjoin_query(left, "id", "elements", k=args.k + 1)

        added: dict[str, dict[str, dict]] = {}
        for _, row in frame.iterrows():
            qid, seed, expanded = plan[int(row["l_id"])]
            edge = edges[int(row["r_id"])]
            hid = edge["canonical_id"]
            if hid == seed["hyperedge_id"]:
                continue                               # the seed wins its own query
            score = (float(seed.get("score", 0.0)) * float(row["score"])
                     if args.score_rule == "product" else float(row["score"]) * args.decay)
            if score < args.min_score:
                continue
            via = {"seed": seed["hyperedge_id"], "round": round_no,
                   "score": round(float(row["score"]), 4)}
            existing = {h["hyperedge_id"] for h in results[qid]}
            if hid in existing:
                for h in results[qid]:
                    if h["hyperedge_id"] == hid:
                        h.setdefault("via", []).append(via)
                n_merged += 1
                continue
            slot = added.setdefault(qid, {})
            if hid in slot:
                if round(score, 6) > slot[hid]["score"]:
                    slot[hid]["score"] = round(score, 6)
                    slot[hid]["_qset"] = expanded
                slot[hid]["via"].append(via)
            else:
                slot[hid] = {"hyperedge_id": hid, "score": round(score, 6),
                             "description": edge["description"],
                             "participants": edge["participants"], "src": edge["src"],
                             "expanded": True, "round": round_no, "via": [via],
                             "_qset": expanded}

        # Keep the best of each question's gains, and let them seed the next round.
        frontier = []
        n_added = 0
        for qid, slot in added.items():
            new = sorted(slot.values(), key=lambda h: -h["score"])[: args.max_added]
            for hit in new:
                hit["via"] = hit["via"][:3]          # the route is a diagnostic
            results[qid].extend(new)
            n_added += len(new)
            gained.add(qid)
            for hit in new[: args.seeds]:
                frontier.append((qid, hit, hit["_qset"]))

        added_per_round.append(n_added)
        print(f"round {round_no}       : {len(plan)} queries -> {n_added} added "
              f"in {time.time() - start:.1f}s")
        if not n_added:
            break

    for hits in results.values():
        hits.sort(key=lambda h: (-float(h.get("score", 0.0)), h.get("expanded", False)))
        for rank, hit in enumerate(hits, 1):
            hit["rank"] = rank
            hit.pop("_qset", None)                   # branch state, not output

    total_added = sum(added_per_round)
    print("\n" + "=" * 66)
    print(f"added         : {total_added} hyperedges "
          f"({total_added / max(1, len(results)):.2f} per question) "
          f"over {len(added_per_round)} round(s): {added_per_round}")
    print(f"already there : {n_merged} bindings reached a hyperedge the join had")
    print(f"questions     : {len(gained)} of {len(results)} gained something")
    print("=" * 66)

    example = next((q for q, h in results.items() if any(x.get("expanded") for x in h)), None)
    if example:
        print(f"\nexample: {example}")
        for hit in results[example][:8]:
            mark = f"  +{hit['round']}" if hit.get("expanded") else "   "
            print(f"{mark} {hit['score']:.4f}  {hit.get('description', '')[:60]!r}")

    return {"n_added": total_added, "added_per_round": added_per_round,
            "n_merged": n_merged, "questions_gained": len(gained)}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(
        description="Nexus Hyperedge Binding, one ranked list per question")
    # inputs
    p.add_argument("--retrieved", required=True, type=Path)
    p.add_argument("--query-entities", required=True, type=Path,
                   help="query_entities.json, for each question's own tokens and qi")
    p.add_argument("--hyperedges", required=True, type=Path)
    p.add_argument("--postings", type=Path, default=None,
                   help="keep the collection identical to retrieval; omit to use all hyperedges")
    p.add_argument("--chunks", type=Path, default=None,
                   help="chunks.json, for the chunk section and the corpus indices")
    p.add_argument("--out", required=True, type=Path)
    # what to bind
    p.add_argument("--expand-tokens", default="union",
                   choices=["union", "drop-matched", "entities-only"])
    p.add_argument("--rounds", type=int, default=1,
                   help="how many times to repeat the binding, each result seeding the next")
    p.add_argument("--max-element-df", type=int, default=None,
                   help="drop query elements whose rarest token appears in more than N hyperedges")
    p.add_argument("--seeds", type=int, default=5,
                   help="results of a round that go on to seed the next, per question")
    p.add_argument("--k", type=int, default=10, help="results per expanded query")
    p.add_argument("--max-added", type=int, default=20,
                   help="cap what one question gains per round")
    # scoring
    p.add_argument("--score-rule", default="product", choices=["product", "decay"])
    p.add_argument("--decay", type=float, default=0.5, help="--score-rule decay only")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="drop bound hits below this, and hits whose own set similarity "
                        "never reached it")
    # join
    p.add_argument("--similarity", default="jaccard", choices=["jaccard", "edit"])
    p.add_argument("--tokens", default="names", choices=["names", "ids"])
    # what to write
    p.add_argument("--knowledge", choices=["all", "hyperedges", "chunks"], default="all",
                   help="which sections to write (default: both). generate_answers picks "
                        "which to SHOW the model, so there is rarely a reason to narrow it here")
    p.add_argument("--top-k", type=int, default=None,
                   help="cap the hyperedge list (default: keep everything retrieved)")
    p.add_argument("--chunk-top-k", type=int, default=None,
                   help="cap the chunk list (default: every chunk the hyperedges came from)")
    p.add_argument("--run-name", default="nexus")
    args = p.parse_args()

    if args.rounds < 1:
        raise SystemExit("--rounds must be at least 1")

    t0 = time.time()
    blob = read_json(args.retrieved)
    results = blob["results"]
    edges = load_hyperedges(args.hyperedges, args.postings, args.tokens)
    by_id = {e["canonical_id"]: e for e in edges}
    queries = read_json(args.query_entities)["queries"]
    chunks = load_chunks(args.chunks)
    texts = load_hyperedge_text(args.hyperedges)

    want_edges = args.knowledge in ("all", "hyperedges")
    want_chunks = args.knowledge in ("all", "chunks")
    if want_chunks and not chunks:
        print("!! no chunk texts loaded"
              + (f" from {args.chunks}" if args.chunks else " (--chunks not given)")
              + ": the chunks section will be empty")

    df = build_token_df(edges)
    n_retrieved = sum(len(h) for h in results.values())
    print(f"hyperedges    : {len(edges)}")
    print(f"questions     : {len(results)} with {n_retrieved} retrieved hyperedges")
    print(f"expand-tokens : {args.expand_tokens}, {args.rounds} round(s)")
    if args.max_element_df:
        common = [t for t, n in df.most_common(8)]
        print(f"element df    : dropping elements whose rarest token exceeds "
              f"{args.max_element_df} (most common tokens: {', '.join(common)})")

    # --------------------------------------------------------------- binding
    bind_stats = bind(results, queries, edges, by_id, df, args)

    # ---------------------------------------------------------------- output
    out_results = []
    for qid, hits in results.items():
        query = queries.get(qid, {})
        # qi is the position in questions.json, which generate_answers indexes by.
        qi = query.get("qi", len(out_results))
        hits = [h for h in hits if float(h.get("score", 0.0)) >= args.min_score]

        hyperedges = []
        for rank, hit in enumerate(sorted(hits, key=lambda h: -float(h["score"])), 1):
            node_id = hit.get("hyperedge_id", "")
            description, names = texts.get(node_id, (hit.get("description", ""), []))
            text = hit.get("description") or description
            if not text:
                continue
            chunk_ids = sorted(hit.get("src", []))
            item = {"text": text,
                    "score": round(float(hit["score"]), 4),
                    "rank": rank,
                    "node_id": node_id,
                    "participants": names or hit.get("participants", []),
                    "chunk_ids": chunk_ids}
            if chunks:
                item["chunk_indices"] = sorted({chunks[c][0] for c in chunk_ids
                                                if c in chunks})
            hyperedges.append(item)
            if args.top_k and len(hyperedges) >= args.top_k:
                break

        # The chunks are rolled up from the hyperedges above, after --top-k: a
        # chunk only appears because one of the kept hyperedges came out of it.
        chunk_hits = chunks_of(hyperedges, chunks, args.chunk_top_k) if want_chunks else []

        out_results.append({
            "qi": qi,
            "question": query.get("question", f"(question {qi})"),
            "knowledge": {"hyperedges": hyperedges if want_edges else [],
                          "chunks": chunk_hits},
            "trace": {
                "tokens": query.get("tokens", []),
                "rounds": args.rounds,
                "n_bound": sum(1 for h in hits if h.get("expanded")),
                "n_hyperedges": len(hyperedges) if want_edges else 0,
                "n_chunks": len(chunk_hits),
            },
        })

    out_results.sort(key=lambda r: r["qi"])
    n = [len(r["knowledge"]["hyperedges"]) for r in out_results]
    nc = [len(r["knowledge"]["chunks"]) for r in out_results]
    summary = {
        "questions": len(out_results),
        "mean_knowledge": round(sum(n) / max(len(n), 1), 2),       # hyperedges
        "mean_chunks": round(sum(nc) / max(len(nc), 1), 2),
        "empty": sum(1 for x in n if x == 0),
        "retrieved_before_binding": n_retrieved,
        "mean_top_score": round(sum(r["knowledge"]["hyperedges"][0]["score"]
                                    for r in out_results if r["knowledge"]["hyperedges"])
                                / max(sum(1 for x in n if x), 1), 4)
                          if any(n) else None,
        **bind_stats,
        "seconds": round(time.time() - t0, 1),
    }

    write_json(args.out, {
        "config": {
            "run": args.run_name, "method": "nexus-tokenjoin",
            "retrieved": str(args.retrieved),
            "query_entities": str(args.query_entities),
            "chunks": str(args.chunks) if args.chunks else None,
            "knowledge": args.knowledge,
            "top_k": args.top_k, "chunk_top_k": args.chunk_top_k,
            "min_score": args.min_score,
            "knowledge_sections": ["hyperedges", "chunks"],
            "llm_calls_per_question": 0,
            "binding": {
                "expand_tokens": args.expand_tokens, "rounds": args.rounds,
                "max_element_df": args.max_element_df,
                "seeds": args.seeds, "k": args.k, "max_added": args.max_added,
                "score_rule": args.score_rule, "decay": args.decay,
                "similarity": args.similarity, "tokens": args.tokens,
            },
            "retrieval": blob.get("config", {}),
        },
        "summary": summary,
        "results": out_results,
    })

    print(f"\nhyperedges : {summary['mean_knowledge']} per question, "
          f"{summary['empty']} empty, top score {summary['mean_top_score']}")
    print(f"chunks     : {summary['mean_chunks']} per question "
          f"(the passages those hyperedges were extracted from)")
    print(f"wrote {args.out}  ({summary['seconds']}s)")


if __name__ == "__main__":
    main()
