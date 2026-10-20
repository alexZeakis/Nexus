"""
Nexus - Entity Extraction.

Consumes what label extraction froze and nothing else:

    <out_dir>/schema.json      the frozen labels with their descriptions
    <out_dir>/fact_units.json  the deduplicated facts

and produces typed mentions with character spans and confidences. One forward
pass per unit per label batch, no API, no generation.

A note on naming. This module is written against a generic unit of text and
calls it a chunk, because the same code would run over chunks. In Nexus the
unit is a fact: the runner wraps each fact as {"chunk_id": <fact id>,
"content": <fact text>} and renames the field on the way out. So a `chunk_id`
reaching this module is a fact id.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

DEFAULT_MODEL = "fastino/gliner2-base-v1"

#: The catch-all label is a schema device for the reader, not something an
#: encoder can recognise. Asking GLiNER to tag spans as "anything that fits
#: none of the labels above" returns noise, so it is excluded from the label
#: set sent to the model.
CATCH_ALL = "other"

#: Punctuation GLiNER leaves on a span boundary. It is not part of the name, and
#: it is not harmless: the fuzzy set join tokenizes elements, so "Weak and the
#: Wicked." and "Weak and the Wicked" differ in their last token and score 0.6
#: instead of 1.0, which is enough to drop a hyperedge out of a top-k list. It
#: also stops entity linking from merging the two surface forms.
TRIM_CHARS = " \t\n\r\f\v\"'‘’“”«».,;:!?()[]{}<>*_-–—"


@dataclass
class Mention:
    chunk_id: str
    chunk_index: int
    label: str
    text: str
    start: int
    end: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "label": self.label,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class ExtractionStats:
    n_chunks: int = 0
    n_mentions: int = 0
    empty_chunks: list[str] = field(default_factory=list)
    per_label: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def mentions_per_chunk(self) -> float:
        return self.n_mentions / max(1, self.n_chunks)

    @property
    def empty_rate(self) -> float:
        return len(self.empty_chunks) / max(1, self.n_chunks)


# --------------------------------------------------------------------------- #
# Schema handling
# --------------------------------------------------------------------------- #


def load_schema(schema_payload: dict[str, Any]) -> dict[str, str]:
    """Pull {label: description} out of schema.json, dropping the catch-all."""
    schema = schema_payload.get("schema", schema_payload)
    if not isinstance(schema, dict) or not schema:
        raise ValueError("schema.json does not contain a non-empty 'schema' object")
    return {l: d for l, d in schema.items() if l != CATCH_ALL}


def batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


# --------------------------------------------------------------------------- #
# Span normalization
# --------------------------------------------------------------------------- #


def trim_surface(surface: str, start: int, end: int) -> tuple[str, int, int]:
    """Drop boundary punctuation, keeping the offsets aligned with the text."""
    left = len(surface) - len(surface.lstrip(TRIM_CHARS))
    right = len(surface) - len(surface.rstrip(TRIM_CHARS))
    return surface[left: len(surface) - right or None], start + left, end - right


def _normalize_span(raw: dict[str, Any], text: str) -> tuple[str, int, int, float] | None:
    """Read one GLiNER result into (surface, start, end, confidence).

    Key names differ across gliner/gliner2 versions, so every field is looked
    up under its known aliases and offsets are recovered from the text when
    the model does not return them. Boundary punctuation is trimmed here, at
    the one point every mention passes through.
    """
    surface = raw.get("text") or raw.get("span") or raw.get("mention")
    if not surface:
        return None
    surface = str(surface)

    start = raw.get("start", raw.get("start_char", raw.get("char_start")))
    end = raw.get("end", raw.get("end_char", raw.get("char_end")))
    if start is None or end is None:
        found = text.find(surface)
        if found == -1:
            return None
        start, end = found, found + len(surface)

    surface, start, end = trim_surface(surface, int(start), int(end))
    if not surface.strip():
        return None                              # the span was punctuation only

    score = raw.get("confidence", raw.get("score", raw.get("prob", 1.0)))
    return surface, int(start), int(end), float(score)


def resolve_overlaps(mentions: list[Mention]) -> list[Mention]:
    """Keep the strongest mention when two spans overlap under different labels.

    Labels are queried in batches, so the same span can come back under two
    labels with no chance for the model to compare them. Entity linking assumes
    one label per canonical entity, so the ambiguity is settled here, by
    confidence, then by span length.
    """
    ordered = sorted(mentions, key=lambda m: (-m.confidence, -(m.end - m.start), m.start))
    kept: list[Mention] = []
    for mention in ordered:
        if any(mention.start < k.end and k.start < mention.end for k in kept):
            continue
        kept.append(mention)
    return sorted(kept, key=lambda m: (m.start, m.end))


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #


class GLiNERExtractor:
    """Schema-conditioned encoder over the frozen label set."""

    def __init__(
        self,
        schema: dict[str, str],
        model_name: str = DEFAULT_MODEL,
        threshold: float = 0.7,
        label_batch_size: int = 15,
        use_descriptions: bool = True,
        extractor: Any = None,
    ) -> None:
        if extractor is None:
            from gliner2 import GLiNER2

            extractor = GLiNER2.from_pretrained(model_name)
        self.extractor = extractor
        self.schema = schema
        self.labels = list(schema)
        self.threshold = threshold
        self.label_batch_size = label_batch_size
        self.use_descriptions = use_descriptions
        self.descriptions_ok: bool | None = None if use_descriptions else False

    def _label_argument(self, labels: Sequence[str]) -> Any:
        """Labels with descriptions if the installed version accepts them.

        The descriptions written during consolidation exist to condition this
        model, so they are passed when possible. Older versions take a plain
        list, and we fall back once and remember.
        """
        if self.descriptions_ok is False:
            return list(labels)
        return {l: self.schema[l] for l in labels}

    def _call(self, text: str, labels: Sequence[str]) -> dict[str, Any]:
        try:
            return self.extractor.extract_entities(
                text,
                self._label_argument(labels),
                include_confidence=True,
                threshold=self.threshold,
            )
        except (TypeError, ValueError, AttributeError):
            if self.descriptions_ok is False:
                raise
            self.descriptions_ok = False         # retry once, labels only
            return self.extractor.extract_entities(
                text,
                list(labels),
                include_confidence=True,
                threshold=self.threshold,
            )

    def extract_chunk(self, chunk: dict[str, Any], chunk_index: int) -> list[Mention]:
        text = chunk["content"]
        mentions: list[Mention] = []
        for label_batch in batched(self.labels, self.label_batch_size):
            results = self._call(text, label_batch)
            for label, spans in (results.get("entities") or {}).items():
                for raw in spans or []:
                    parsed = _normalize_span(raw, text)
                    if parsed is None:
                        continue
                    surface, start, end, score = parsed
                    mentions.append(
                        Mention(
                            chunk_id=chunk["chunk_id"],
                            chunk_index=chunk_index,
                            label=label,
                            text=surface,
                            start=start,
                            end=end,
                            confidence=score,
                        )
                    )
        if self.descriptions_ok is None:
            self.descriptions_ok = True
        return resolve_overlaps(mentions)


# --------------------------------------------------------------------------- #
# Corpus pass
# --------------------------------------------------------------------------- #


def extract_corpus(
    chunks: Sequence[dict[str, Any]],
    extractor: GLiNERExtractor,
    done: dict[str, list[dict[str, Any]]] | None = None,
    on_progress=None,
    flush_every: int = 25,
    on_flush=None,
) -> tuple[dict[str, list[dict[str, Any]]], ExtractionStats]:
    """Run the extractor over every unit, resuming from `done` if given."""
    per_chunk: dict[str, list[dict[str, Any]]] = dict(done or {})
    stats = ExtractionStats()
    start = time.time()

    for index, chunk in enumerate(chunks):
        if chunk["chunk_id"] in per_chunk:
            continue
        mentions = extractor.extract_chunk(chunk, index)
        per_chunk[chunk["chunk_id"]] = [m.to_dict() for m in mentions]

        stats.n_chunks += 1
        stats.n_mentions += len(mentions)
        if not mentions:
            stats.empty_chunks.append(chunk["chunk_id"])
        for mention in mentions:
            stats.per_label[mention.label] = stats.per_label.get(mention.label, 0) + 1

        if on_progress and index % 10 == 0:
            on_progress(index, len(chunks))
        if on_flush and stats.n_chunks % flush_every == 0:
            on_flush(per_chunk)

    stats.seconds = time.time() - start
    return per_chunk, stats
