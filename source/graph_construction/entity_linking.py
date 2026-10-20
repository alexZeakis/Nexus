"""
Nexus - Entity Linking.

Blocking follows the deduplication pipeline of the entity resolution line of
work: embed the records, run an approximate nearest neighbour self-join, keep
pairs above a similarity threshold. Two things are added here, both
consequences of the frozen schema. Candidates are label-constrained, so only
mentions of the same type are ever compared, and an exact-normalized merge runs
first so the matcher never sees pairs that cheap evidence already settles.

Matching is pairwise MATCH. Each candidate pair is therefore one independent 
yes/no decision, scored by flan-t5 in both orders.

A note on naming. MentionRef calls its source a chunk because the same code
would run over chunks. In Nexus the unit is a fact, so a `chunk_id` here is a
fact id and `chunk_index` is the position of that fact in corpus order.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any, Callable, Iterable, Sequence

import numpy as np

#: The same encoder as label extraction. One encoder across the pipeline:
#: entity names here, hyperedge descriptions later, questions at retrieval time.
from embeddings import DEFAULT_MODEL as DEFAULT_EMBED_MODEL

DEFAULT_MATCH_MODEL = "google/flan-t5-large"

ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
PUNCT = re.compile(r"[^\w\s]")
WS = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass
class MentionRef:
    """One occurrence of an entity, pointing back into entities.json."""

    chunk_id: str
    chunk_index: int
    position: int          # index within that unit's mention list
    label: str
    text: str
    start: int
    confidence: float
    context: str = ""      # snippet around the span, evidence for the matcher

    @property
    def key(self) -> tuple[str, int]:
        return (self.chunk_id, self.position)

    @property
    def order(self) -> tuple[int, int]:
        """Corpus order: unit first, then character offset within it."""
        return (self.chunk_index, self.start)


@dataclass
class Group:
    """Mentions already merged by exact normalized name under one label."""

    gid: int
    label: str
    normalized: str
    mentions: list[MentionRef] = field(default_factory=list)

    @property
    def representative(self) -> MentionRef:
        return min(self.mentions, key=lambda m: m.order)

    @property
    def name(self) -> str:
        return self.representative.text


def normalize_name(text: str) -> str:
    """Casing, whitespace, punctuation, leading article, trailing plural."""
    s = ARTICLES.sub("", text.strip().lower())
    s = PUNCT.sub(" ", s)
    s = WS.sub(" ", s).strip()
    if len(s) > 3 and s.endswith("ies"):
        s = s[:-3] + "y"
    elif len(s) > 3 and s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    return s


def build_mentions(
    entities_payload: dict[str, Any],
    chunks: Sequence[dict[str, Any]],
    context_window: int = 120,
) -> list[MentionRef]:
    """Read entities.json into MentionRefs, attaching a context snippet."""
    text_of = {c["chunk_id"]: c["content"] for c in chunks}
    out: list[MentionRef] = []
    for chunk_id, mentions in entities_payload["mentions"].items():
        content = text_of.get(chunk_id, "")
        for position, m in enumerate(mentions):
            start, end = int(m.get("start", 0)), int(m.get("end", 0))
            lo = max(0, start - context_window)
            hi = min(len(content), end + context_window)
            out.append(
                MentionRef(
                    chunk_id=chunk_id,
                    chunk_index=int(m.get("chunk_index", 0)),
                    position=position,
                    label=m["label"],
                    text=m["text"],
                    start=start,
                    confidence=float(m.get("confidence", 1.0)),
                    context=WS.sub(" ", content[lo:hi]).strip(),
                )
            )
    return out


def group_by_exact(mentions: Iterable[MentionRef]) -> list[Group]:
    """Lexical stage: merge mentions whose normalized name and label coincide."""
    buckets: dict[tuple[str, str], Group] = {}
    for m in mentions:
        key = (m.label, normalize_name(m.text))
        if key not in buckets:
            buckets[key] = Group(len(buckets), m.label, key[1])
        buckets[key].mentions.append(m)
    return list(buckets.values())


# --------------------------------------------------------------------------- #
# Serialization for the matcher
# --------------------------------------------------------------------------- #


def serialize(group: Group, mode: str = "context", max_context: int = 200) -> str:
    """Render a group in the [COL]/[VAL] format the matcher was prompted with.

    Blocking uses mode="label", because an embedding of the name and its type
    is what should decide which pairs are worth comparing. Matching uses
    mode="context" by default, where the surrounding snippet is the evidence
    the matcher reads.
    """
    parts = [f"[COL] name [VAL] {group.name}"]
    if mode in ("label", "context"):
        parts.append(f"[COL] type [VAL] {group.label}")
    if mode == "context":
        ctx = group.representative.context[:max_context]
        parts.append(f"[COL] context [VAL] {ctx}")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Blocking
# --------------------------------------------------------------------------- #


def approx_nn_pairs(vectors: np.ndarray, k: int = 5) -> list[tuple[int, int, float]]:
    """Self-join top-k over L2, scored as 1/(1+d), as in the ER pipeline."""
    n, dim = vectors.shape
    k_eff = min(k + 1, n)
    try:
        import hnswlib

        index = hnswlib.Index(space="l2", dim=dim)
        index.init_index(max_elements=n, ef_construction=200, M=16)
        index.add_items(vectors)
        index.set_ef(50)
        labels, distances = index.knn_query(vectors, k=k_eff)
    except ImportError:
        # Brute force. Fine at this scale and keeps the run dependency-light.
        sq = (vectors ** 2).sum(axis=1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * vectors @ vectors.T, 0.0)
        labels = np.argsort(d2, axis=1)[:, :k_eff]
        distances = np.take_along_axis(d2, labels, axis=1)

    pairs: list[tuple[int, int, float]] = []
    for i, row in enumerate(labels):
        for rank, j in enumerate(row):
            if i == int(j):
                continue
            pairs.append((i, int(j), float(1.0 / (1.0 + np.sqrt(max(distances[i][rank], 0.0))))))
    return pairs


def candidate_pairs(
    groups: Sequence[Group],
    embed_fn: Callable[[Sequence[str]], np.ndarray],
    k: int = 5,
    threshold: float = 0.85,
    lexical_threshold: float = 0.85,
    serialize_mode: str = "label",
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Label-constrained ANN candidates, pooled with lexical near-duplicates."""
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, g in enumerate(groups):
        by_label[g.label].append(idx)

    pairs: set[tuple[int, int]] = set()
    stats = {"ann": 0, "lexical": 0, "labels": len(by_label)}

    for label, members in by_label.items():
        if len(members) < 2:
            continue
        texts = [serialize(groups[i], serialize_mode) for i in members]
        vectors = np.asarray(embed_fn(texts), dtype=np.float32)

        for i, j, score in approx_nn_pairs(vectors, k=k):
            if score < threshold:
                continue
            a, b = sorted((members[i], members[j]))
            if a != b and (a, b) not in pairs:
                pairs.add((a, b))
                stats["ann"] += 1

        # Lexical near-duplicates the ANN may have missed, nominated not merged.
        if len(members) <= 4000:
            for i, j in combinations(members, 2):
                if (i, j) in pairs:
                    continue
                a, b = groups[i].normalized, groups[j].normalized
                if abs(len(a) - len(b)) > max(len(a), len(b)) * 0.5:
                    continue
                if SequenceMatcher(None, a, b).ratio() >= lexical_threshold:
                    pairs.add((i, j))
                    stats["lexical"] += 1

    return sorted(pairs), stats


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #

#: The MATCH prompt, in the wording the serialization was designed for.
MATCH_TEMPLATE = (
    "Do these two entity descriptions refer to the same real-world entity?\n"
    "(1) {a}\n"
    "(2) {b}"
)


class FlanT5Matcher:
    """Pairwise MATCH adjudication with flan-t5.

    The decision is read from the logits of the first decoded token rather than
    from generated text, so the answer is always one of Yes/No and a
    probability comes for free. Each pair is scored in both orders and the
    probabilities averaged, mirroring the symmetry of the LLM matcher.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MATCH_MODEL,
        device: str | None = None,
        batch_size: int = 16,
        decision_threshold: float = 0.5,
        max_input_tokens: int = 512,
        dtype: str = "auto",
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        if model is None or tokenizer is None:
            import torch
            from transformers import AutoTokenizer, T5ForConditionalGeneration

            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            # flan-t5-large is 780M parameters, about 3.1GB in fp32, so fp32
            # fits on an ordinary card and is the safe choice. bf16 needs
            # Ampere (sm_80) or newer, and on older cards the only 16-bit
            # option is fp16, which T5 can overflow (see the finite-logit check
            # in _score). The same applies with more force to flan-t5-xl.
            if dtype == "auto":
                if device.startswith("cuda"):
                    major = torch.cuda.get_device_capability(
                        torch.device(device).index or 0
                    )[0]
                    torch_dtype = torch.bfloat16 if major >= 8 else torch.float16
                else:
                    torch_dtype = torch.float32
            else:
                torch_dtype = getattr(torch, dtype)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch_dtype
            ).to(device)
            model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or "cpu"
        self.batch_size = batch_size
        self.decision_threshold = decision_threshold
        self.max_input_tokens = max_input_tokens
        self.dtype = str(getattr(model, "dtype", "unknown"))
        self.yes_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
        self.no_id = tokenizer("No", add_special_tokens=False).input_ids[0]
        self.calls = 0

    def _score(self, prompts: Sequence[str]) -> list[float]:
        """P(Yes) for each prompt, from the first decoder step."""
        import torch

        out: list[float] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = list(prompts[start : start + self.batch_size])
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(self.device)
            decoder_input_ids = torch.full(
                (len(batch), 1), self.model.config.decoder_start_token_id, device=self.device
            )
            with torch.no_grad():
                logits = self.model(**enc, decoder_input_ids=decoder_input_ids).logits[:, 0, :]
            pair = torch.stack([logits[:, self.yes_id], logits[:, self.no_id]], dim=-1)
            # T5 was trained in bf16 and overflows fp16 in some layers. A NaN
            # here would survive the softmax, compare False against the
            # threshold, and silently answer No to every pair, so it has to be
            # an error rather than a result.
            if not torch.isfinite(pair).all():
                raise RuntimeError(
                    "non-finite logits from the matcher. This is the known T5 "
                    "fp16 overflow: every pair would silently be answered No. "
                    "Re-run with --dtype float32 (flan-t5-large is 780M, about "
                    "3.1GB in fp32)."
                )
            probs = torch.softmax(pair, dim=-1)[:, 0]
            out.extend(probs.detach().cpu().tolist())
            self.calls += len(batch)
        return out

    def match(self, records: Sequence[tuple[str, str]]) -> list[tuple[bool, float]]:
        """Adjudicate pairs of serialized records, both orders averaged."""
        prompts: list[str] = []
        for a, b in records:
            prompts.append(MATCH_TEMPLATE.format(a=a, b=b))
            prompts.append(MATCH_TEMPLATE.format(a=b, b=a))
        scores = self._score(prompts)
        results: list[tuple[bool, float]] = []
        for i in range(0, len(scores), 2):
            p = (scores[i] + scores[i + 1]) / 2.0
            results.append((p >= self.decision_threshold, p))
        return results


# --------------------------------------------------------------------------- #
# Clustering and canonicalization
# --------------------------------------------------------------------------- #


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def build_clusters(
    groups: Sequence[Group], confirmed: Iterable[tuple[int, int]]
) -> dict[int, list[int]]:
    """Connected components over confirmed matches, singletons included."""
    uf = UnionFind()
    for idx in range(len(groups)):
        uf.find(idx)
    for a, b in confirmed:
        uf.union(a, b)

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(groups)):
        clusters[uf.find(idx)].append(idx)
    return dict(clusters)


def canonicalize(
    groups: Sequence[Group], clusters: dict[int, list[int]]
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    """Assign every mention its official name and entity id.

    The official name is the surface form of the cluster's FIRST occurrence in
    the corpus, ordered by unit then character offset. This is an assumption,
    not a derivation: the first naming of an entity is usually its fullest
    ("Bill Gates" before "Microsoft's CEO"), but a corpus that opens with an
    abbreviation will canonicalize to the abbreviation.
    """
    assignment: dict[tuple[str, int], dict[str, Any]] = {}
    entities: list[dict[str, Any]] = []

    for entity_index, (_, members) in enumerate(sorted(clusters.items())):
        mentions = [m for gid in members for m in groups[gid].mentions]
        first = min(mentions, key=lambda m: m.order)
        entity_id = f"entity-{entity_index}"
        official = first.text

        for m in mentions:
            assignment[m.key] = {"entity_id": entity_id, "official_name": official}

        surfaces = sorted({m.text for m in mentions})
        entities.append({
            "entity_id": entity_id,
            "official_name": official,
            "label": groups[members[0]].label,
            "aliases": [s for s in surfaces if s != official],
            "n_mentions": len(mentions),
            "n_groups": len(members),
            "context": sorted({m.chunk_id for m in mentions}),
            "confidence": round(float(np.mean([m.confidence for m in mentions])), 4),
        })

    return assignment, entities
