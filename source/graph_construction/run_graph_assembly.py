"""
Step 7 of 7 - Graph Assembly and Indexing.

Inputs
    <out_dir>/linking.json            canonical entities (run_entity_linking.py)
    <out_dir>/hyperedge_linking.json  canonical hyperedges (run_hyperedge_linking.py)
    <out_dir>/schema.json             the frozen label vocabulary
    <out_dir>/chunks.json             corpus, for provenance counts

Outputs
    <out_dir>/graph.graphml           G, bipartite encoding
    <out_dir>/postings.json           P(l:e), read by retrieval and binding
    <out_dir>/graph_stats.json        construction metadata
    <out_dir>/index/entities.vdb      name and alias vectors   (--indexes)
    <out_dir>/index/hyperedges.vdb    fact vectors             (--indexes)

The two .vdb indexes are not read by the retrieval pipeline, which uses
postings.json and the hyperedge payload directly. They are built only when
--indexes is passed, for anyone who wants to query the released graph by
vector similarity.

Usage
    python run_graph_assembly.py --out-dir ./output/mydataset --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from graph_assembly import (
    build_graph,
    build_postings,
    entities_from_linking,
    hyperedges_from_linking,
    summarize,
    write_graphml,
)
from utils import read_json, write_json


def main() -> None:
    p = argparse.ArgumentParser(description="Nexus Graph Assembly over facts")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--name", default=None, help="dataset name recorded in the graph")
    p.add_argument("--indexes", action="store_true",
                   help="also build the two .vdb vector indexes")
    p.add_argument("--embed-model", default=None, help="defaults to the pipeline encoder")
    p.add_argument("--device", default=None)
    p.add_argument("--backend", default="auto", choices=["auto", "hnswlib", "faiss", "numpy"])
    args = p.parse_args()

    out_dir: Path = args.out_dir
    name = args.name or out_dir.name

    # --------------------------------------------------------------- inputs
    entity_payload = read_json(out_dir / "linking.json")
    # Fact-based linking stores provenance as fact_ids + chunk_ids; the
    # assembler reads chunks from "context", so point it at the source chunks.
    for e in entity_payload["entities"]:
        e["context"] = e.get("chunk_ids", [])
    entities = entities_from_linking(entity_payload)
    print(f"entities     : {len(entities)}")

    hyperedge_payload = read_json(out_dir / "hyperedge_linking.json")
    hyperedges = hyperedges_from_linking(hyperedge_payload)
    print(f"hyperedges   : {len(hyperedges)} canonical facts")

    # ------------------------------------------------------------- assemble
    graph = build_graph(entities, hyperedges)
    postings = build_postings(graph)

    n_chunks = len(read_json(out_dir / "chunks.json"))
    schema = read_json(out_dir / "schema.json").get("schema", {})

    stats = summarize(graph, n_chunks=n_chunks)
    stats["n_postings"] = len(postings)
    stats["n_labels"] = len(schema)
    stats["n_facts"] = hyperedge_payload.get("stats", {}).get("n_facts")

    graphml_path = out_dir / "graph.graphml"
    removed = write_graphml(graph, str(graphml_path), graph_attrs={
        "dataset": name,
        "unit": "fact",
        "n_entities": stats["n_entities"],
        "n_hyperedges": stats["n_hyperedges"],
        "labels": " | ".join(sorted(schema)),
    })
    write_json(out_dir / "postings.json", postings)
    if removed:
        print(f"\n[note] stripped {removed} characters that XML forbids")

    print("\n" + "=" * 66)
    print(f"G = (E, H)        : {stats['n_entities']} entities, "
          f"{stats['n_hyperedges']} hyperedges, {stats['n_incidences']} incidences")
    if stats["n_facts"]:
        print(f"facts covered     : {stats['n_hyperedges']} of {stats['n_facts']} facts "
              f"({stats['n_hyperedges'] / stats['n_facts']:.1%}) after linking")
    print(f"mean arity        : {stats['mean_arity']}  "
          f"(degenerate {stats['degenerate_fraction']:.1%})")
    print(f"posting lists     : {stats['n_postings']} over {stats['n_labels']} labels")
    print(f"components        : {stats['n_components']}, largest holds "
          f"{stats['largest_component_share']:.1%} of all vertices")
    print(f"isolated entities : {stats['isolated_entities']} ({stats['isolated_fraction']:.1%})")
    print("=" * 66)

    # -------------------------------------------------------------- indexes
    written_e = written_h = None
    if args.indexes:
        from embeddings import DEFAULT_MODEL, load_embedder
        from vector_index import build_index, entity_rows, hyperedge_rows

        model = args.embed_model or DEFAULT_MODEL
        index_dir = out_dir / "index"

        rows_e = entity_rows(entity_payload["entities"])
        rows_h = hyperedge_rows([
            edge for edge in hyperedge_payload["canonical"]
            if edge["canonical_id"] in graph.hyperedges
        ])
        n_aliases = sum(1 for r in rows_e if r["kind"] == "alias")
        print(f"\nindex rows        : {len(rows_e)} entity "
              f"({n_aliases} of them aliases), {len(rows_h)} hyperedge")

        embedder = load_embedder(model, device=args.device)
        index_e = build_index(rows_e, embedder, model, args.backend)
        written_e = index_e.save(index_dir / "entities")
        index_h = build_index(rows_h, embedder, model, args.backend)
        written_h = index_h.save(index_dir / "hyperedges")
        print(f"vector indexes    : {index_e.backend}, dim {index_e.dim}")

        stats["index"] = {
            "backend": index_e.backend, "model": model, "dim": index_e.dim,
            "entity_rows": len(rows_e), "alias_rows": n_aliases, "hyperedge_rows": len(rows_h),
        }
        if rows_h:
            probe, _ = index_h.search(embedder([rows_h[0]["text"]])[0], k=1)[0]
            if probe["id"] != rows_h[0]["id"]:
                print("[!] a description does not retrieve itself: index is suspect")

    write_json(out_dir / "graph_stats.json", stats)

    print(f"\nwrote {graphml_path}")
    print(f"      {out_dir / 'postings.json'}")
    if written_e:
        print(f"      {written_e}")
        print(f"      {written_h}")
    print(f"      {out_dir / 'graph_stats.json'}")

    if stats["dangling_participants"]:
        print(f"\n[note] {stats['dangling_participants']} participant references named "
              "no known entity and were dropped.")
    if stats["isolated_fraction"] > 0.3:
        print("\n[!] many entities take part in no hyperedge and are unreachable by retrieval.")
    if stats["largest_component_share"] < 0.5:
        print("\n[!] the graph is fragmented: multi-hop chains cannot cross component boundaries.")


if __name__ == "__main__":
    main()
