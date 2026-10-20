"""
Nexus - Graph Assembly and Indexing.

Assembles the canonical entities E and linked hyperedges H into the hypergraph
G = (E, H), stored as a bipartite graph: entities and hyperedges form the two
vertex sets, incidences form the edges. The encoding is lossless and lets an
ordinary graph database answer the two incidence queries retrieval relies on,
the entities of a hyperedge and the hyperedges of an entity.

GraphML is written directly rather than through a library, because its
attributes must be scalars: lists such as aliases, provenance chunks and
context chunks are serialized with a separator, and the types are declared
explicitly so that readers do not have to guess.

Inputs:  <out_dir>/linking.json, <out_dir>/hyperedge_linking.json
Outputs: <out_dir>/graph.graphml, <out_dir>/postings.json,
         <out_dir>/graph_stats.json
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

LIST_SEP = " | "

GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
SCHEMA_LOCATION = (
    "http://graphml.graphdrawing.org/xmlns "
    "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"
)

#: Node attributes, with their GraphML types.
NODE_KEYS: list[tuple[str, str]] = [
    ("kind", "string"),           # "entity" or "hyperedge"
    ("name", "string"),           # official_name / official_description
    ("label", "string"),          # entity label from L; empty for hyperedges
    ("aliases", "string"),
    ("arity", "int"),
    ("degenerate", "boolean"),
    ("confidence", "double"),
    ("n_mentions", "int"),        # entities
    ("n_merged", "int"),          # hyperedges
    ("chunks", "string"),         # context(e) or src(h)
    ("n_chunks", "int"),
]
EDGE_KEYS: list[tuple[str, str]] = [
    ("label", "string"),          # the label under which the entity participates
]


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


@dataclass
class Graph:
    entities: dict[str, dict[str, Any]] = field(default_factory=dict)
    hyperedges: dict[str, dict[str, Any]] = field(default_factory=dict)
    incidences: list[tuple[str, str, str]] = field(default_factory=list)  # h, e, label
    dangling: Counter = field(default_factory=Counter)

    @property
    def n_incidences(self) -> int:
        return len(self.incidences)


def entities_from_linking(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Canonical entities as written by entity linking."""
    out: dict[str, dict[str, Any]] = {}
    for entity in payload.get("entities", []):
        out[entity["entity_id"]] = {
            "kind": "entity",
            "name": entity.get("official_name", ""),
            "label": entity.get("label", ""),
            "aliases": LIST_SEP.join(entity.get("aliases", [])),
            "confidence": float(entity.get("confidence", 1.0)),
            "n_mentions": int(entity.get("n_mentions", 0)),
            "chunks": LIST_SEP.join(entity.get("context", [])),
            "n_chunks": len(entity.get("context", [])),
        }
    return out


def hyperedges_from_linking(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Canonical hyperedges as written by hyperedge linking."""
    out: dict[str, dict[str, Any]] = {}
    for edge in payload.get("canonical", []):
        out[edge["canonical_id"]] = {
            "kind": "hyperedge",
            "name": edge.get("official_description", ""),
            "arity": int(edge.get("arity", len(edge.get("participants", [])))),
            "degenerate": bool(edge.get("degenerate", False)),
            "confidence": float(edge.get("confidence", 1.0)),
            "n_merged": int(edge.get("n_merged", 1)),
            "chunks": LIST_SEP.join(edge.get("src", [])),
            "n_chunks": len(edge.get("src", [])),
            "_participants": list(edge.get("participants", [])),
        }
    return out


def build_graph(
    entities: dict[str, dict[str, Any]],
    hyperedges: dict[str, dict[str, Any]],
) -> Graph:
    """Wire incidences, dropping participants that name no known entity."""
    graph = Graph(entities=dict(entities))
    for hid, edge in hyperedges.items():
        participants = edge.pop("_participants", [])
        kept = []
        for eid in participants:
            if eid in graph.entities:
                kept.append(eid)
            else:
                graph.dangling[eid] += 1
        # A hyperedge needs two participants to relate anything.
        if len(set(kept)) < 2:
            continue
        graph.hyperedges[hid] = edge
        for eid in dict.fromkeys(kept):
            graph.incidences.append((hid, eid, graph.entities[eid].get("label", "")))
    return graph


def build_postings(graph: Graph) -> dict[str, list[str]]:
    """P(l:e): the hyperedges in which entity e participates under label l.

    Keyed "label:entity_id". Retrieval and binding read this file.
    """
    postings: dict[str, list[str]] = defaultdict(list)
    for hid, eid, label in graph.incidences:
        postings[f"{label}:{eid}"].append(hid)
    return {k: sorted(set(v)) for k, v in sorted(postings.items())}


# --------------------------------------------------------------------------- #
# GraphML
# --------------------------------------------------------------------------- #


#: Characters XML 1.0 forbids. Corpus text carries them (form feeds, stray
#: control bytes from PDF or OCR sources), ElementTree writes them through
#: unescaped, and any conforming parser then rejects the whole document.
_ILLEGAL_XML = re.compile(
    "[^\u0009\u000A\u000D -퟿-�\U00010000-\U0010FFFF]"
)


def clean_xml_text(value: str) -> tuple[str, int]:
    """Drop characters that cannot appear in an XML document."""
    cleaned, n = _ILLEGAL_XML.subn("", value)
    return cleaned, n


def _add_data(parent: ET.Element, key: str, value: Any) -> None:
    """Write one <data> element, skipping values that carry no information.

    None, the empty string and 0 are skipped, and because False == 0 in Python
    a false boolean is skipped too. That is deliberate: an absent attribute
    reads as the default, so a graph of 100k vertices does not carry a few
    hundred thousand elements stating zero.
    """
    if value is None or value == "" or value == 0:
        return
    element = ET.SubElement(parent, "data", {"key": key})
    text = "true" if value is True else "false" if value is False else str(value)
    text, removed = clean_xml_text(text)
    if removed:
        _add_data.removed += removed
    element.text = text


_add_data.removed = 0


def write_graphml(graph: Graph, path: str, graph_attrs: dict[str, Any] | None = None) -> int:
    """Serialize the bipartite encoding of G as GraphML.

    Returns the number of illegal characters stripped from the text.
    """
    _add_data.removed = 0
    root = ET.Element("graphml", {
        "xmlns": GRAPHML_NS,
        "xmlns:xsi": XSI_NS,
        "xsi:schemaLocation": SCHEMA_LOCATION,
    })

    for name, attr_type in NODE_KEYS:
        ET.SubElement(root, "key", {
            "id": f"n_{name}", "for": "node", "attr.name": name, "attr.type": attr_type,
        })
    for name, attr_type in EDGE_KEYS:
        ET.SubElement(root, "key", {
            "id": f"e_{name}", "for": "edge", "attr.name": name, "attr.type": attr_type,
        })

    graph_element = ET.SubElement(root, "graph", {
        "id": "nexus", "edgedefault": "undirected",
    })

    for node_id, attrs in list(graph.entities.items()) + list(graph.hyperedges.items()):
        node = ET.SubElement(graph_element, "node", {"id": node_id})
        for name, _ in NODE_KEYS:
            _add_data(node, f"n_{name}", attrs.get(name))

    for index, (hid, eid, label) in enumerate(graph.incidences):
        edge = ET.SubElement(graph_element, "edge", {
            "id": f"i{index}", "source": hid, "target": eid,
        })
        _add_data(edge, "e_label", label)

    if graph_attrs:
        for key, value in graph_attrs.items():
            ET.SubElement(root, "key", {
                "id": f"g_{key}", "for": "graph", "attr.name": key, "attr.type": "string",
            })
            _add_data(graph_element, f"g_{key}", value)

    # Indent in place rather than reparsing through minidom: the round trip
    # doubles peak memory on a large graph and turns any stray character into
    # an error at the parse rather than at its source.
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return _add_data.removed


# --------------------------------------------------------------------------- #
# Construction metadata
# --------------------------------------------------------------------------- #


def connected_components(graph: Graph) -> list[int]:
    """Component sizes of the bipartite graph, largest first.

    Retrieval can only compose facts that share entities, so the component
    structure bounds which multi-hop chains exist at all.
    """
    adjacency: dict[str, set[str]] = defaultdict(set)
    for hid, eid, _ in graph.incidences:
        adjacency[hid].add(eid)
        adjacency[eid].add(hid)

    seen: set[str] = set()
    sizes: list[int] = []
    for start in list(adjacency):
        if start in seen:
            continue
        stack, size = [start], 0
        seen.add(start)
        while stack:
            current = stack.pop()
            size += 1
            for neighbour in adjacency[current] - seen:
                seen.add(neighbour)
                stack.append(neighbour)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def summarize(graph: Graph, n_chunks: int = 0) -> dict[str, Any]:
    arities = Counter(h.get("arity", 0) for h in graph.hyperedges.values())
    degenerate = sum(1 for h in graph.hyperedges.values() if h.get("degenerate"))
    incident = {eid for _, eid, _ in graph.incidences}
    isolated = [eid for eid in graph.entities if eid not in incident]
    per_entity = Counter(eid for _, eid, _ in graph.incidences)
    labels = Counter(e.get("label", "") for e in graph.entities.values())
    sizes = connected_components(graph)
    total = max(1, len(graph.hyperedges))
    return {
        "n_entities": len(graph.entities),
        "n_hyperedges": len(graph.hyperedges),
        "n_incidences": graph.n_incidences,
        "n_chunks": n_chunks,
        "arity_distribution": dict(sorted(arities.items())),
        "mean_arity": round(
            sum(h.get("arity", 0) for h in graph.hyperedges.values()) / total, 2
        ),
        "degenerate_fraction": round(degenerate / total, 4),
        "isolated_entities": len(isolated),
        "isolated_fraction": round(len(isolated) / max(1, len(graph.entities)), 4),
        "mean_hyperedges_per_entity": round(
            graph.n_incidences / max(1, len(graph.entities)), 2
        ),
        "max_hyperedges_per_entity": max(per_entity.values()) if per_entity else 0,
        "n_components": len(sizes),
        "largest_component": sizes[0] if sizes else 0,
        "largest_component_share": round(
            sizes[0] / max(1, len(graph.entities) + len(graph.hyperedges)), 4
        ) if sizes else 0.0,
        "label_distribution": dict(labels.most_common()),
        "dangling_participants": sum(graph.dangling.values()),
    }
