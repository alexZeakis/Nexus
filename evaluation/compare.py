"""
Compare methods across datasets (rows: dataset x metric, columns: methods).

Dataset is read from the method tag in the gateway URL as <Method>_<Dataset>,
e.g. http://localhost:8000/Nexus_Mix/retrieval/v1  (split on the first "_").

Usage:
    python compare.py complete                          # chat/responses calls, all stages
    python compare.py complete --stage retrieval        # keep only these stages
    python compare.py embeddings --stage indexing --per-call
    python compare.py complete --files logs_8000.jsonl logs_8001.jsonl
    python compare.py complete --per-question           # stages named retrieve_200, generate_200, ...
    python compare.py complete --graph-stats            # adds graph size rows per dataset
"""
import argparse
import glob
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

GS = ["#nodes", "#edges", "#entity", "#hyperedge"]


def graph_stats(folder):
    """Sum node, edge and role counts over all .graphml files under folder."""
    files = sorted(Path(folder).rglob("*.graphml"))
    if not files:
        return None
    tot = dict.fromkeys(GS, 0)
    for f in files:
        role_keys = set()
        for _, el in ET.iterparse(f, events=("end",)):
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "key" and el.get("attr.name") in ("role", "kind"):
                role_keys.add(el.get("id"))
            elif tag == "node":
                tot["#nodes"] += 1
                for d in el:
                    if d.get("key") in role_keys:
                        role = (d.text or "").strip().lower()
                        if role == "entity":
                            tot["#entity"] += 1
                        elif role == "hyperedge":
                            tot["#hyperedge"] += 1
                el.clear()
            elif tag == "edge":
                tot["#edges"] += 1
                el.clear()
        if not role_keys:
            print(f"warning: no 'role' or 'kind' attribute in {f}", file=sys.stderr)
    return tot

ENDPOINTS = {
    "complete": ("/v1/chat/completions", "/v1/completions", "/v1/responses"),
    "embeddings": ("/v1/embeddings",),
}

p = argparse.ArgumentParser()
p.add_argument("mode", choices=ENDPOINTS)
p.add_argument("--stage", nargs="+", help="keep only these stages")
grp = p.add_mutually_exclusive_group()
grp.add_argument("--per-call", action="store_true", help="input/output per call instead of sums")
grp.add_argument("--per-question", action="store_true",
                 help="input/output per question; N is the trailing number of the stage name (retrieve_200)")
p.add_argument("--files", nargs="+", help="default: all logs_*.jsonl")
p.add_argument("--graph-stats", action="store_true",
               help="add #nodes, #edges, #entity, #hyperedge from .graphml files")
p.add_argument("--graphs-root", default="../output/graphs",
               help="graphs are read from <root>/<method>/<dataset>")
p.add_argument("--sep", default="_", help="separator between method and dataset (first occurrence)")
a = p.parse_args()

files = a.files or sorted(glob.glob("logs_*.jsonl"))
df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
df = df[(df["status"] == 200) & df["endpoint"].isin(ENDPOINTS[a.mode])]
if a.stage:
    df = df[df["stage"].isin(a.stage)]
if df.empty:
    sys.exit("no matching calls")

for c in ("prompt_tokens", "completion_tokens"):
    if c not in df:
        df[c] = 0
df = df.fillna({"prompt_tokens": 0, "completion_tokens": 0})

parts = df["method"].str.split(a.sep, n=1, expand=True)
df["dataset"] = parts[1].fillna("-") if parts.shape[1] > 1 else "-"
df["method"] = parts[0]

AGG = dict(
    calls=("status", "size"),
    input=("prompt_tokens", "sum"),
    output=("completion_tokens", "sum"),
)

if a.per_question:
    df["n_q"] = pd.to_numeric(df["stage"].str.extract(r"_(\d+)$")[0])
    skipped = sorted(df.loc[df["n_q"].isna(), "stage"].unique())
    if skipped:
        print(f"note: stages without a question count skipped: {', '.join(skipped)}", file=sys.stderr)
    df = df.dropna(subset=["n_q"])
    if df.empty:
        sys.exit("no stage name ends with a question count (e.g. retrieve_200)")
    # per-question cost of each stage, summed over the stages of a method
    s = df.groupby(["dataset", "method", "stage", "n_q"]).agg(**AGG).reset_index()
    s["input"] = s["input"] / s["n_q"]
    s["output"] = s["output"] / s["n_q"]
    g = s.groupby(["dataset", "method"]).agg(
        calls=("calls", "sum"), input=("input", "sum"), output=("output", "sum"))
    g = g.rename(columns={"input": "in/question", "output": "out/question"})
else:
    g = df.groupby(["dataset", "method"]).agg(**AGG)
    if a.per_call:
        g["input"] = g["input"] / g["calls"]
        g["output"] = g["output"] / g["calls"]
        g = g.rename(columns={"input": "in/call", "output": "out/call"})

if a.graph_stats:
    rows = [graph_stats(Path(a.graphs_root) / m / d) or {} for d, m in g.index]
    g = g.join(pd.DataFrame(rows, index=g.index).reindex(columns=GS))

metrics = list(g.columns)
if a.mode == "embeddings":
    metrics = [m for m in metrics if m not in ("output", "out/call", "out/question")]

blocks = []
for d, sub in g.groupby(level="dataset"):
    t = sub.droplevel("dataset")[metrics].T
    t.index = pd.MultiIndex.from_product([[d], t.index], names=["dataset", "metric"])
    blocks.append(t)
table = pd.concat(blocks)
table.columns.name = None


def fmt(x):
    return f"{x:,.0f}" if x == int(x) else f"{x:,.1f}"


print(table.to_string(na_rep="-", float_format=fmt))
