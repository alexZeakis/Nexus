"""
Nexus - Hyperedge Linking.

A fact with its linked entities is a hyperedge. The same fact is often stated
more than once in a corpus, through repeated definitions, restated guidance, or
the overlap between two chunk windows. Duplicates are not merely redundant
storage: retrieval returns a ranked list, so a fact stored three times can take
three places in it and crowd out facts that would have added something.

Candidates come from the participants, not from the wording. Two hyperedges
that state the same thing name the same entities. Exact set overlap would be
the direct test, but it assumes entity linking collapsed every mention, which
canonicalization makes likely without making it certain: one hyperedge may
carry "Natalie Portman" where another carries "Portman". The join is therefore
fuzzy at the element level, matching participant names by their tokens rather
than by identity, which is the fuzzy set similarity join problem, solved here
with pyTokenJoin.

Adjudication is NOT the MATCH prompt used for entities. Entity linking asks
whether two records refer to the same thing, and two facts about the same
people and the same film refer to the same things while stating different
things, so that question answers yes when it should answer no. Deduplicating
facts is a question about propositions, so each pair is scored by bidirectional
entailment with an NLI cross-encoder over the fact sentences alone. The score
is the weaker direction: one-way entailment means one fact is more specific
than the other, which is not a duplicate.

Inputs:  fact_units.json, entities_linked.json
Outputs: hyperedges.json, hyperedge_linking.json  (written by the runner)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# The union-find used by transitive clustering is shared with entity linking.
from entity_linking import UnionFind

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


# --------------------------------------------------------------------------- #
# pyTokenJoin
# --------------------------------------------------------------------------- #

#: The API:
#:
#:     from pytokenjoin.jaccard.join_delta import TokenJoin
#:     out = TokenJoin().tokenjoin_self(df, id='id', join='elements',
#:                                      delta=0.85, posFilter=True,
#:                                      jointFilter=True)
#:
#: returning a DataFrame with columns l_id, r_id, score, where the id columns
#: are row positions in the input frame.
_MODULES = {
    "jaccard": "pytokenjoin.jaccard.join_delta",
    "edit": "pytokenjoin.edit.join_delta",
}


def _discover_api(similarity: str = "jaccard") -> tuple[Any, str]:
    """Bind TokenJoin.tokenjoin_self from the installed package."""
    import importlib

    module_name = _MODULES[similarity]
    module = importlib.import_module(module_name)
    instance = module.TokenJoin()
    return instance.tokenjoin_self, f"{module_name}.TokenJoin.tokenjoin_self"


def build_frame(sets: Sequence[Sequence[str]]):
    """One row per hyperedge: a row-position id and the participant names.

    The id column holds row positions rather than our own identifiers.
    tokenjoin_self returns positions into the frame and then overwrites the id
    columns through `df.iloc[...]`, so positions are the one interpretation
    that holds either way; the caller maps them back afterwards. Duplicate
    elements are removed, since a name repeated in one set would distort the
    set similarity.
    """
    import pandas as pd

    return pd.DataFrame({
        "id": list(range(len(sets))),
        "elements": [list(dict.fromkeys(elements)) for elements in sets],
    })


def _normalize_result(result: Any) -> list[tuple[int, int, float]]:
    """Read the l_id, r_id and score columns of the returned frame."""
    return [
        (int(row["l_id"]), int(row["r_id"]), float(row["score"]))
        for _, row in result.iterrows()
    ]


def tokenjoin(
    sets: Sequence[Sequence[str]],
    ids: Sequence[Any],
    threshold: float = 0.85,
    similarity: str = "jaccard",
    entry_point: Callable[..., Any] | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> list[tuple[Any, Any, float]]:
    """Self-join the participant sets, returning (id_a, id_b, similarity).

    `entry_point` overrides discovery, for when the installed API differs from
    what this module expects.
    """
    if entry_point is None:
        entry_point, name = _discover_api(similarity)
        if verbose:
            print(f"  pytokenjoin entry point: {name}")

    frame = build_frame(sets)
    call_kwargs: dict[str, Any] = {
        "id": "id",
        "join": "elements",
        # Both default to False in the library. The filters prune candidates
        # without changing the result set, so they are on.
        "posFilter": True,
        "jointFilter": True,
        "delta": threshold,
        **kwargs,
    }
    result = entry_point(frame, **call_kwargs)

    lookup = list(ids)
    return [
        (lookup[int(left)], lookup[int(right)], score)
        for left, right, score in _normalize_result(result)
    ]


# --------------------------------------------------------------------------- #
# Building hyperedges from linked facts
# --------------------------------------------------------------------------- #


def build_hyperedges(facts: list[dict], linked: dict) -> dict:
    """One hyperedge per fact, keyed by fact id, participants = its entities."""
    hyperedges = {}
    for fact in facts:
        fid = str(fact["fact_id"])
        mentions = [m for m in linked.get(fid, []) if m.get("entity_id")]
        n_entities = len({m["entity_id"] for m in mentions})
        hyperedges[fid] = [{
            "hyperedge_id": f"fact-{fid}",
            "description": fact["content"],
            "participants": [
                {"entity_id": m["entity_id"], "official_name": m.get("official_name"),
                 "text": m["text"], "label": m["label"]}
                for m in mentions
            ],
            "confidence": round(float(np.mean([m["confidence"] for m in mentions])), 4) if mentions else 0.0,
            "degenerate": n_entities == 2,
            "src": fact["chunk_ids"],
        }]
    return {"hyperedges": hyperedges}


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass
class HyperedgeRecord:
    """One hyperedge, flattened for linking."""

    index: int
    hyperedge_id: str
    chunk_id: str                          # the fact id this hyperedge came from
    chunk_index: int                       # that fact's position in corpus order
    description: str
    participants: tuple[str, ...]          # canonical entity ids, sorted
    participant_names: tuple[str, ...]
    confidence: float
    degenerate: bool
    src: list[str]                         # source chunk ids

    @property
    def order(self) -> tuple[int, str]:
        """Corpus order, for picking the canonical representative."""
        return (self.chunk_index, self.hyperedge_id)


def load_records(
    payload: dict[str, Any], chunk_order: dict[str, int]
) -> list[HyperedgeRecord]:
    """Flatten hyperedges.json into records with canonical participant ids.

    A fact with fewer than two distinct linked entities relates nothing and is
    dropped here rather than carried through the join.
    """
    records: list[HyperedgeRecord] = []
    for chunk_id, edges in payload["hyperedges"].items():
        for edge in edges:
            ids, names = [], []
            for p in edge["participants"]:
                pid = p.get("entity_id") or p.get("official_name") or p.get("text")
                if pid:
                    ids.append(str(pid))
                    names.append(str(p.get("official_name") or p.get("text")))
            if len(set(ids)) < 2:
                continue
            records.append(HyperedgeRecord(
                index=len(records),
                hyperedge_id=edge["hyperedge_id"],
                chunk_id=chunk_id,
                chunk_index=chunk_order.get(chunk_id, 0),
                description=edge["description"],
                participants=tuple(sorted(set(ids))),
                participant_names=tuple(names),
                confidence=float(edge.get("confidence", 1.0)),
                degenerate=bool(edge.get("degenerate", len(set(ids)) == 2)),
                src=list(edge.get("src", [chunk_id])),
            ))
    return records


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #


def candidate_pairs(
    records: Sequence[HyperedgeRecord],
    threshold: float = 0.85,
    similarity: str = "jaccard",
    verbose: bool = True,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Candidates from a fuzzy set similarity join over the participant sets."""
    stats: dict[str, Any] = {
        "sets": len(records),
        "threshold": threshold,
        "similarity": similarity,
    }
    if len(records) < 2:
        stats["pairs"] = 0
        return [], stats

    result = tokenjoin(
        sets=[list(r.participant_names) for r in records],
        ids=[r.index for r in records],
        threshold=threshold,
        similarity=similarity,
        verbose=verbose,
    )

    pairs: set[tuple[int, int]] = set()
    scores: dict[tuple[int, int], float] = {}
    for left, right, score in result:
        a, b = sorted((int(left), int(right)))
        if a == b:
            continue
        pairs.add((a, b))
        scores[(a, b)] = float(score)

    stats["pairs"] = len(pairs)
    stats["scores"] = scores
    return sorted(pairs), stats


def order_pairs(
    pairs: Sequence[tuple[int, int]],
    scores: dict[tuple[int, int], float],
) -> list[tuple[int, int, float]]:
    """Strongest set similarity first, so a truncated run keeps the best pairs."""
    return sorted(
        ((a, b, scores.get((a, b), 0.0)) for a, b in pairs),
        key=lambda t: -t[2],
    )


# --------------------------------------------------------------------------- #
# Adjudication
# --------------------------------------------------------------------------- #


class NLIMatcher:
    """Bidirectional entailment over pairs of fact sentences."""

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL, batch_size: int = 32,
                 device: str | None = None, dtype: str = "auto") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = (torch.float16 if dtype == "float16" else
                      torch.bfloat16 if dtype == "bfloat16" else
                      torch.float32 if dtype == "float32" else
                      (torch.float16 if "cuda" in self.device else torch.float32))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=self.dtype).to(self.device).eval()
        self.batch_size = batch_size
        self.calls = 0

        labels = self.model.config.id2label
        entail = [i for i, l in labels.items() if str(l).lower().startswith("entail")]
        if not entail:
            raise ValueError(f"no entailment label in {labels}")
        self.entail_index = entail[0]

    def _entailment(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        probs: list[float] = []
        for start in range(0, len(premises), self.batch_size):
            batch_p = premises[start: start + self.batch_size]
            batch_h = hypotheses[start: start + self.batch_size]
            encoded = self.tokenizer(batch_p, batch_h, return_tensors="pt",
                                     padding=True, truncation=True, max_length=256)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with self.torch.no_grad():
                logits = self.model(**encoded).logits.float()
            probs.extend(logits.softmax(-1)[:, self.entail_index].tolist())
            self.calls += 1
        return probs

    def match(self, records: list[tuple[str, str]]) -> list[tuple[float, float, float]]:
        """For each (a, b): (score, P(a entails b), P(b entails a)).

        The score is the weaker direction: one-way entailment means one fact is
        more specific than the other, which is not a duplicate.
        """
        left = [a for a, _ in records]
        right = [b for _, b in records]
        forward = self._entailment(left, right)
        backward = self._entailment(right, left)
        return [(min(f, b), f, b) for f, b in zip(forward, backward)]


# --------------------------------------------------------------------------- #
# Canonicalization
# --------------------------------------------------------------------------- #


def build_components(
    records: Sequence[HyperedgeRecord],
    confirmed: Iterable[tuple[int, int]],
    mode: str = "star",
    scores: dict[tuple[int, int], float] | None = None,
    max_size: int | None = None,
) -> dict[int, list[int]]:
    """Group confirmed matches into components.

    mode="transitive" takes connected components, which is what the literature
    usually does and what over-merges here: A~B and B~C place A with C although
    the two were never compared and would often be rejected. On a corpus where
    many facts share two participants, this chains a whole topic into one
    component.

    mode="star" (default) admits a hyperedge to a component only if it was
    confirmed against that component's representative. Chains cannot form,
    because every member has been adjudicated against the same record. The cost
    is recall: a genuine duplicate that matches a non-representative member but
    not the representative stays separate.
    """
    if mode == "transitive":
        uf = UnionFind()
        for record in records:
            uf.find(record.index)
        for a, b in confirmed:
            uf.union(a, b)
        components: dict[int, list[int]] = defaultdict(list)
        for record in records:
            components[uf.find(record.index)].append(record.index)
        return dict(components)

    scores = scores or {}
    pairs = sorted(
        ((a, b) for a, b in confirmed),
        key=lambda p: -scores.get(p, 1.0),          # strongest evidence first
    )
    confirmed_set = {(min(a, b), max(a, b)) for a, b in confirmed}

    representative_of: dict[int, int] = {}          # member -> representative
    members_of: dict[int, list[int]] = {}           # representative -> members

    def rep(i: int) -> int | None:
        return representative_of.get(i)

    def confirmed_pair(i: int, j: int) -> bool:
        return (min(i, j), max(i, j)) in confirmed_set

    def can_grow(r: int) -> bool:
        return max_size is None or len(members_of[r]) < max_size

    for a, b in pairs:
        ra, rb = rep(a), rep(b)
        if ra is None and rb is None:
            # New component, represented by the earlier of the two.
            r = a if records[a].order <= records[b].order else b
            other = b if r == a else a
            representative_of[r] = r
            representative_of[other] = r
            members_of[r] = [r, other]
        elif ra is not None and rb is None:
            if confirmed_pair(ra, b) and can_grow(ra):
                representative_of[b] = ra
                members_of[ra].append(b)
        elif ra is None and rb is not None:
            if confirmed_pair(rb, a) and can_grow(rb):
                representative_of[a] = rb
                members_of[rb].append(a)
        elif ra != rb:
            # Two components merge only if their representatives were
            # themselves adjudicated and confirmed.
            if confirmed_pair(ra, rb) and (
                max_size is None or len(members_of[ra]) + len(members_of[rb]) <= max_size
            ):
                members_of[ra].extend(members_of[rb])
                for member in members_of[rb]:
                    representative_of[member] = ra
                del members_of[rb]

    components = {r: sorted(m) for r, m in members_of.items()}
    for record in records:                          # singletons
        if record.index not in representative_of:
            components[record.index] = [record.index]
    return components


def canonicalize(
    records: Sequence[HyperedgeRecord],
    components: dict[int, list[int]],
    rule: str = "first",
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Collapse each component into one canonical hyperedge.

    The representative is the first occurrence in corpus order, mirroring the
    official_name of entity linking. With rule="confidence" the highest-
    confidence member is used instead.

    Participants are NOT unioned across the component. Two duplicates rarely
    carry identical participant sets, and unioning them would inflate arity,
    which is a headline statistic. The canonical hyperedge keeps its own
    participants; the variants are recorded for inspection.
    """
    assignment: dict[str, dict[str, Any]] = {}
    canonical: list[dict[str, Any]] = []

    for position, (_, members) in enumerate(sorted(components.items())):
        group = [records[i] for i in members]
        if rule == "confidence":
            representative = max(group, key=lambda r: (r.confidence, -r.order[0]))
        else:
            representative = min(group, key=lambda r: r.order)

        canonical_id = f"hyperedge-{position}"
        for record in group:
            assignment[record.hyperedge_id] = {
                "canonical_id": canonical_id,
                "official_description": representative.description,
            }

        src = sorted({c for record in group for c in record.src})
        variants = [
            {"description": r.description, "participants": list(r.participants)}
            for r in group if r.hyperedge_id != representative.hyperedge_id
        ]
        canonical.append({
            "canonical_id": canonical_id,
            "official_description": representative.description,
            "participants": list(representative.participants),
            "participant_names": list(representative.participant_names),
            "arity": len(representative.participants),
            "degenerate": representative.degenerate,
            "confidence": round(
                float(np.mean([r.confidence for r in group])), 3
            ),
            "src": src,
            "n_merged": len(group),
            "variants": variants,
        })

    return assignment, canonical


def summarize(
    records: Sequence[HyperedgeRecord],
    canonical: Sequence[dict[str, Any]],
    pairs: Sequence[tuple[int, int]],
    confirmed: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    merged = [c for c in canonical if c["n_merged"] > 1]
    sizes = [c["n_merged"] for c in canonical]
    cross_chunk = sum(1 for c in merged if len(c["src"]) > 1)
    return {
        "n_hyperedges": len(records),
        "n_canonical": len(canonical),
        "reduction": round(1 - len(canonical) / max(1, len(records)), 4),
        "n_candidate_pairs": len(pairs),
        "n_confirmed": len(confirmed),
        "confirm_rate": round(len(confirmed) / max(1, len(pairs)), 4),
        "n_merged_groups": len(merged),
        "cross_chunk_merges": cross_chunk,
        "largest_component": max(sizes) if sizes else 0,
        "component_sizes": {
            str(size): sizes.count(size) for size in sorted(set(sizes))
        },
    }
