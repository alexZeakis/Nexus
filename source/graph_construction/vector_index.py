"""
Nexus - the two approximate indexes built at graph assembly.

    entity index      one vector per canonical name AND per alias, so a
                      question phrased with a variant surface form still
                      grounds to the correct vertex
    hyperedge index   one vector per canonical description, for questions that
                      name no entity the posting lists can answer

Both use the same encoder as the rest of the pipeline. The backend is whichever
of hnswlib / faiss is installed, falling back to exact numpy search, which is
fine at the scale of a single corpus and removes a hard dependency.

An index is one self-contained .vdb file: a container holding the backend's
own index, the row map that resolves a hit to a vertex, and the metadata
recording which encoder built it. One file moves, copies and versions cleanly,
and an index cannot be separated from the row map that makes it meaningful.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class VectorIndex:
    """Cosine-similarity index over a fixed set of rows."""

    def __init__(self, backend: str, index: Any, rows: list[dict[str, Any]],
                 dim: int, model: str) -> None:
        self.backend = backend
        self.index = index
        self.rows = rows
        self.dim = dim
        self.model = model

    # ------------------------------------------------------------- building
    @classmethod
    def build(
        cls,
        rows: Sequence[dict[str, Any]],
        vectors: np.ndarray,
        model: str,
        backend: str = "auto",
        ef_construction: int = 200,
        m: int = 16,
    ) -> "VectorIndex":
        vectors = _normalize(np.asarray(vectors, dtype=np.float32))
        dim = int(vectors.shape[1])

        if backend == "auto":
            for candidate in ("hnswlib", "faiss"):
                try:
                    __import__(candidate)
                    backend = candidate
                    break
                except ImportError:
                    continue
            else:
                backend = "numpy"

        if backend == "hnswlib":
            import hnswlib

            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(max_elements=len(rows), ef_construction=ef_construction, M=m)
            index.add_items(vectors, np.arange(len(rows)))
            index.set_ef(max(50, m * 4))
        elif backend == "faiss":
            import faiss

            index = faiss.IndexFlatIP(dim)      # vectors are normalized
            index.add(vectors)
        else:
            index = vectors                     # exact search over the matrix

        return cls(backend, index, list(rows), dim, model)

    # -------------------------------------------------------------- queries
    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[dict[str, Any], float]]:
        """Top-k rows by cosine similarity, most similar first."""
        vector = _normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))
        k = min(k, len(self.rows))
        if self.backend == "hnswlib":
            labels, distances = self.index.knn_query(vector, k=k)
            return [(self.rows[int(i)], 1.0 - float(d))
                    for i, d in zip(labels[0], distances[0])]
        if self.backend == "faiss":
            scores, labels = self.index.search(vector, k)
            return [(self.rows[int(i)], float(s))
                    for i, s in zip(labels[0], scores[0]) if i >= 0]
        scores = (self.index @ vector[0])
        order = np.argsort(-scores)[:k]
        return [(self.rows[int(i)], float(scores[int(i)])) for i in order]

    # ------------------------------------------------------------ on disk
    #: Member names inside the .vdb container.
    _META, _IDS, _INDEX = "meta.json", "ids.json", "index.bin"

    def save(self, path: str | Path) -> Path:
        """Write one self-contained .vdb file. Returns the path written."""
        path = Path(path).with_suffix(".vdb")
        path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            payload = Path(tmp) / self._INDEX
            if self.backend == "hnswlib":
                self.index.save_index(str(payload))
            elif self.backend == "faiss":
                import faiss

                faiss.write_index(self.index, str(payload))
            else:
                np.save(payload, self.index)
                payload = payload.with_suffix(".bin.npy")

            meta = {
                "backend": self.backend,
                "model": self.model,
                "dim": self.dim,
                "count": len(self.rows),
                "format": "nexus-vdb/1",
            }
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(self._META, json.dumps(meta, indent=2))
                archive.writestr(self._IDS, json.dumps(self.rows, ensure_ascii=False))
                archive.write(payload, self._INDEX)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        path = Path(path).with_suffix(".vdb")
        with zipfile.ZipFile(path) as archive:
            meta = json.loads(archive.read(cls._META))
            rows = json.loads(archive.read(cls._IDS))
            backend, dim = meta["backend"], meta["dim"]

            if backend == "numpy":
                with archive.open(cls._INDEX) as handle:
                    index = np.load(handle)
                return cls(backend, index, rows, dim, meta["model"])

            # hnswlib and faiss both read from a path, so the member is
            # extracted to a temporary file and loaded from there.
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / cls._INDEX
                with archive.open(cls._INDEX) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                if backend == "hnswlib":
                    import hnswlib

                    index = hnswlib.Index(space="cosine", dim=dim)
                    index.load_index(str(target), max_elements=len(rows))
                    index.set_ef(64)
                else:
                    import faiss

                    index = faiss.read_index(str(target))
        return cls(backend, index, rows, dim, meta["model"])

    @staticmethod
    def describe(path: str | Path) -> dict[str, Any]:
        """Read a .vdb's metadata without loading the vectors."""
        with zipfile.ZipFile(Path(path).with_suffix(".vdb")) as archive:
            return json.loads(archive.read(VectorIndex._META))


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #


def entity_rows(entities: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per canonical name and per alias, all pointing at the vertex."""
    rows: list[dict[str, Any]] = []
    for entity in entities:
        name = entity.get("official_name", "")
        if not name:
            continue
        rows.append({
            "id": entity["entity_id"], "text": name,
            "label": entity.get("label", ""), "kind": "name",
        })
        for alias in entity.get("aliases", []):
            if alias and alias != name:
                rows.append({
                    "id": entity["entity_id"], "text": alias,
                    "label": entity.get("label", ""), "kind": "alias",
                })
    return rows


def hyperedge_rows(canonical: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per canonical hyperedge description."""
    return [
        {
            "id": edge["canonical_id"],
            "text": edge.get("official_description", ""),
            "arity": edge.get("arity", 0),
            "kind": "description",
        }
        for edge in canonical
        if edge.get("official_description")
    ]


def build_index(
    rows: Sequence[dict[str, Any]],
    embed_fn: Callable[[Sequence[str]], np.ndarray],
    model: str,
    backend: str = "auto",
) -> VectorIndex:
    vectors = embed_fn([row["text"] for row in rows])
    return VectorIndex.build(rows, vectors, model=model, backend=backend)
