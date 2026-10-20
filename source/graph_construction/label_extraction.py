"""
Nexus - Algorithm 1: diversified traversal with label extraction.

The unit of the traversal is whatever text you pass in. In Nexus that is the
deduplicated facts, not the raw chunks, because a fact is short and states one
thing, so the labels it proposes are cleaner.

You provide three things:

  1. units         : Sequence[str]                    the texts to traverse
  2. extract_fn    : str            -> labels         the expensive LLM call
  3. consolidate_fn: set[str], ...  -> {label: desc}  the single closing call

plus either precomputed `embeddings` or an `embed_fn` to produce them.
Everything else (medoid selection, farthest-point traversal, stopping rule,
label normalization, bookkeeping) is implemented here.

Dependencies: numpy; scikit-learn only for the medoid selection.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# The interfaces you implement
# --------------------------------------------------------------------------- #

#: Embeds a batch of texts. Must return shape (len(texts), dim).
#: Called once on the whole corpus, so batch it properly. Not needed when
#: `embeddings` is passed.
EmbedFn = Callable[[Sequence[str]], np.ndarray]

#: The expensive model call: one unit in, the KINDS of entities it mentions
#: out. Return either a list of labels, or a {label: example_mention} mapping
#: if you want consolidation to see what was actually tagged as each label.
#: Return None, not an empty mapping, when the call itself failed.
ExtractFn = Callable[[str], Iterable[str] | Mapping[str, str] | None]

#: The single closing call: merges synonyms, drops one-off labels, writes one
#: description per label, adds a catch-all. Receives the accumulated labels,
#: how many units each was seen in, and up to `n_examples` example mentions
#: per label. Returns the frozen schema {label: one-line description}.
ConsolidateFn = Callable[
    [set[str], Mapping[str, int], Mapping[str, list[str]]], dict[str, str]
]


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

@dataclass
class TraversalTrace:
    """Per-step bookkeeping. trace.new_labels against trace.label_count is the
    saturation curve that justifies the stopping rule in the paper."""

    visited: list[int] = field(default_factory=list)      # unit indices, in order
    new_labels: list[int] = field(default_factory=list)   # |new labels| per step
    label_count: list[int] = field(default_factory=list)  # |L| after each step
    failed: list[int] = field(default_factory=list)       # units whose call failed
    stopped_early: bool = False

    @property
    def n_calls(self) -> int:
        return len(self.visited)

    @property
    def failure_rate(self) -> float:
        return len(self.failed) / max(1, self.n_calls)


@dataclass
class LabelExtractionResult:
    schema: dict[str, str]                  # frozen: label -> description
    raw_labels: set[str]                    # before consolidation
    label_support: dict[str, int]           # label -> units it appeared in
    label_examples: dict[str, list[str]]    # label -> example mentions
    trace: TraversalTrace


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_WS = re.compile(r"[\s\-]+")
_PUNCT = re.compile(r"[^a-z0-9_ ]")


def normalize_label(label: str) -> str:
    """Fold surface variants onto one string.

    The stopping rule counts NEW label strings, so without this
    'patient group', 'patient_group' and 'patient groups' count as three
    distinct labels and three units of yield, and the traversal keeps running
    on surface noise. Crude singularization on the last token only.
    """
    s = _WS.sub(" ", label.strip().lower())   # hyphens/spaces first, so that
    s = _PUNCT.sub("", s)                     # 'patient-group' does not become
    s = s.strip().replace(" ", "_")           # 'patientgroup'
    if not s:
        return s
    head, _, last = s.rpartition("_")
    if len(last) > 3 and last.endswith("ies"):
        last = last[:-3] + "y"
    elif len(last) > 3 and last.endswith("ses"):
        last = last[:-2]
    elif len(last) > 3 and last.endswith("s") and not last.endswith("ss"):
        last = last[:-1]
    return f"{head}_{last}" if head else last


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Normalize rows so that cosine distance = 1 - dot product."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def kmeans_medoids(X: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    """Index of the unit closest to each k-means centroid.

    The pool is what the traversal walks over. A farthest-point rule applied to
    the whole corpus drifts towards outliers, so it is applied to cluster
    representatives instead.
    """
    from sklearn.cluster import KMeans

    n_clusters = min(n_clusters, X.shape[0])
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed).fit(X)
    medoids = []
    for k in range(n_clusters):
        members = np.flatnonzero(km.labels_ == k)
        if members.size == 0:
            continue
        d = np.linalg.norm(X[members] - km.cluster_centers_[k], axis=1)
        medoids.append(int(members[int(np.argmin(d))]))
    return np.array(sorted(set(medoids)), dtype=int)


def _as_mapping(found: Iterable[str] | Mapping[str, str]) -> dict[str, str]:
    """Accept either a list of labels or {label: example_mention}."""
    if isinstance(found, Mapping):
        return {normalize_label(k): str(v) for k, v in found.items() if normalize_label(k)}
    return {normalize_label(k): "" for k in found if normalize_label(k)}


# --------------------------------------------------------------------------- #
# Algorithm 1
# --------------------------------------------------------------------------- #

def label_extraction(
    units: Sequence[str],
    extract_fn: ExtractFn,
    consolidate_fn: ConsolidateFn,
    *,
    embed_fn: EmbedFn | None = None,
    embeddings: np.ndarray | None = None,
    n_medoids: int = 200,
    window: int = 5,
    tau: float = 1.0,
    max_calls: int | None = None,
    n_examples: int = 3,
    cluster_seed: int = 0,
) -> LabelExtractionResult:
    """Visit medoid units in farthest-point order and collect entity labels.

    Args:
        units: the texts to traverse, in Nexus the deduplicated facts.
        extract_fn: the expensive per-unit LLM call.
        consolidate_fn: the single closing call that freezes the schema.
        embed_fn: batch embedder, required only when `embeddings` is None.
        embeddings: precomputed (n_units, dim) matrix, to avoid re-embedding.
        n_medoids: size of the medoid pool, i.e. how finely the corpus is
            sampled. It is also the ceiling on LLM calls when `max_calls` is
            not set, but it should be chosen for coverage and the budget
            controlled with `max_calls`.
        window: number of recent steps used by the stopping test.
        tau: stop when the mean number of new labels over the last `window`
            steps falls below this. tau=1.0 reads as "fewer than one new label
            per unit on average".
        max_calls: hard budget, independent of the pool size.
        n_examples: example mentions kept per label for consolidation.
        cluster_seed: random_state of the k-means that builds the pool.

    Returns:
        LabelExtractionResult with the frozen schema and the traversal trace.
    """
    if embeddings is None:
        if embed_fn is None:
            raise ValueError("pass either embeddings or embed_fn")
        embeddings = embed_fn(units)
    X = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    if X.shape[0] != len(units):
        raise ValueError("embeddings and units must have the same length")

    pool = kmeans_medoids(X, n_medoids, seed=cluster_seed)
    if pool.size == 0:
        raise ValueError("empty medoid pool")

    M = X[pool]                                   # embeddings of the pool
    budget = min(max_calls or pool.size, pool.size)

    # Seed: medoid closest to the corpus centroid, a stable starting point.
    centroid = X.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) or 1.0)
    current = int(np.argmax(M @ centroid))

    labels: set[str] = set()
    support: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    trace = TraversalTrace()
    yields: deque[int] = deque(maxlen=window)

    # min_dist[i] = distance of pool member i to its nearest visited unit.
    min_dist = 1.0 - (M @ M[current])
    min_dist[current] = -np.inf                   # visited

    while True:
        raw = extract_fn(units[int(pool[current])])
        failed = raw is None                      # the call itself did not work
        found = _as_mapping(raw or {})
        new = set(found) - labels
        labels |= new
        for label, mention in found.items():
            support[label] = support.get(label, 0) + 1
            if mention and len(examples.setdefault(label, [])) < n_examples:
                examples[label].append(mention)

        trace.visited.append(int(pool[current]))
        trace.new_labels.append(len(new))
        trace.label_count.append(len(labels))
        if failed:
            # A failed call is not evidence that the corpus has saturated, so
            # it must not enter the stopping window.
            trace.failed.append(int(pool[current]))
        else:
            yields.append(len(new))

        if trace.n_calls >= budget:
            break
        if len(yields) == window and sum(yields) / window < tau:
            trace.stopped_early = True
            break

        # Farthest-point step: the unvisited medoid least represented by
        # anything extracted so far.
        nxt = int(np.argmax(min_dist))
        if not np.isfinite(min_dist[nxt]):
            break                                 # pool exhausted
        min_dist = np.minimum(min_dist, 1.0 - (M @ M[nxt]))
        min_dist[nxt] = -np.inf
        current = nxt

    schema = consolidate_fn(labels, support, examples)
    return LabelExtractionResult(
        schema=schema,
        raw_labels=labels,
        label_support=support,
        label_examples=examples,
        trace=trace,
    )
