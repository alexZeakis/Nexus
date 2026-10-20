"""
Shared utilities for Nexus: corpus IO, token-based chunking and JSON helpers.

The chunker is the boundary the whole pipeline agrees on. Fact extraction and
provenance (src, chunk_ids) all index into the chunk list produced here, so
chunk_id is the identifier that must stay stable.
"""

from __future__ import annotations

import json
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=8)
def _encoder(model_name: str = "gpt-4o"):
    """Cached tiktoken encoder. Building it per call is the usual bottleneck."""
    import tiktoken

    return tiktoken.encoding_for_model(model_name)


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o") -> list[int]:
    return _encoder(model_name).encode(content)


def decode_tokens_by_tiktoken(tokens: Sequence[int], model_name: str = "gpt-4o") -> str:
    return _encoder(model_name).decode(list(tokens))


def chunking_by_token_size(
    content: str,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    """Split one document into overlapping token windows."""
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    step = max_token_size - overlap_token_size
    if step <= 0:
        raise ValueError("overlap_token_size must be smaller than max_token_size")

    results: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(tokens), step)):
        window = tokens[start : start + max_token_size]
        results.append(
            {
                "tokens": len(window),
                "content": decode_tokens_by_tiktoken(window, model_name=tiktoken_model).strip(),
                "chunk_order_index": index,
                "chunk_id": f"chunk-{uuid.uuid4()}",
            }
        )
    return results


def chunk_corpus(
    texts: Sequence[str],
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    """Chunk a corpus of documents, tagging each chunk with its document index."""
    chunks: list[dict[str, Any]] = []
    for doc_index, text in enumerate(texts):
        if not text or not text.strip():
            continue
        for chunk in chunking_by_token_size(
            text, overlap_token_size, max_token_size, tiktoken_model
        ):
            chunk["doc_index"] = doc_index
            chunks.append(chunk)
    return chunks


# --------------------------------------------------------------------------- #
# Corpus IO
# --------------------------------------------------------------------------- #


def load_texts(data_dir: str | Path) -> list[str]:
    """Read <data_dir>/texts.json, an array of strings."""
    path = Path(data_dir) / "texts.json"
    if not path.exists():
        raise FileNotFoundError(f"expected an array of strings at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(t, str) for t in data):
        raise ValueError(f"{path} must contain a JSON array of strings")
    return data


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_json_response(text: str) -> Any:
    """Parse a model response that should be JSON but may not be clean.

    Handles ```json fences and leading commentary. Small open models produce
    both often enough that this is not optional.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"could not parse JSON from response: {text[:200]!r}")
