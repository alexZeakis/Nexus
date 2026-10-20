"""
Collect the output of score_knowledge.py into one table per question set.

Reads <root>/<method>/<dataset>/<qset>.json, takes the `overall` block of each
and prints it. Each value in those files is already the mean over the questions
of that run, so nothing is averaged again here except the optional Mean row,
which is the unweighted mean over the datasets a method actually has.

Usage
-----
  python show_scores.py                       # root defaults to knowledge_scores
  python show_scores.py --root knowledge_scores --latex
  python show_scores.py --metrics K-R K-P K-F --no-mean
  python show_scores.py --csv scores.csv
"""

import argparse
import csv
import json
import os
import sys

METRICS = ["R-S", "R-S2", "K-R", "K-P", "K-F", "|H|"]
SOURCE = {"R-S": "R-S", "R-S2": "R-S2", "K-R": "K-R", "K-P": "K-P", "K-F": "K-F",
          "|H|": "mean_retrieved"}
PERCENT = {"R-S", "K-R", "K-P", "K-F"}

DATASET_ORDER = ["Agriculture", "CS", "Hypertension", "Medicine", "Legal", "Mix"]
METHOD_ORDER = ["Golden", "SimSearch", "HyperGraphRAG", "HyperRAG",
                "PRoH", "Nexus-Union"]


def rank(name, order):
    for i, o in enumerate(order):
        if name == o:
            return (0, i, name)
    for i, o in enumerate(order):
        if name.startswith(o):          # Nexus-union, PRoH-L-foo
            return (1, i, name)
    return (2, 0, name)


def collect(root):
    """{qset: {dataset: {method: overall}}}"""
    if not os.path.isdir(root):
        sys.exit(f"no such directory: {root}")
    table, bad = {}, []
    for method in sorted(os.listdir(root)):
        mdir = os.path.join(root, method)
        if not os.path.isdir(mdir):
            continue
        for dataset in sorted(os.listdir(mdir)):
            ddir = os.path.join(mdir, dataset)
            if not os.path.isdir(ddir):
                continue
            for fn in sorted(os.listdir(ddir)):
                if not fn.endswith(".json"):
                    continue
                path = os.path.join(ddir, fn)
                try:
                    blob = json.load(open(path))
                except (ValueError, OSError) as e:
                    bad.append(f"{path}: {e}")
                    continue
                ov = blob.get("overall", blob)
                if "K-R" not in ov:
                    bad.append(f"{path}: no overall block")
                    continue
                qset = str(ov.get("qset") or os.path.splitext(fn)[0])
                m = ov.get("method") or method
                d = ov.get("dataset") or dataset
                table.setdefault(qset, {}).setdefault(d, {})[m] = ov
    for b in bad:
        print(f"[warn] {b}", file=sys.stderr)
    return table


def cell(ov, metric, raw):
    if ov is None:
        return None
    v = ov.get(SOURCE[metric])
    if v is None:
        return None
    if metric in PERCENT and not raw:
        return 100.0 * float(v)
    return float(v)


def fmt(v, metric):
    if v is None:
        return "-"
    return f"{v:,.1f}" if metric == "|H|" else f"{v:.2f}"


def build(table, qset, metrics, raw, with_mean):
    """rows of (dataset, metric, {method: value}), plus the method list"""
    datasets = sorted(table[qset], key=lambda d: rank(d, DATASET_ORDER))
    methods = sorted({m for d in table[qset].values() for m in d},
                     key=lambda m: rank(m, METHOD_ORDER))
    rows = []
    for d in datasets:
        for metric in metrics:
            rows.append((d, metric,
                         {m: cell(table[qset][d].get(m), metric, raw) for m in methods}))
    if with_mean and len(datasets) > 1:
        for metric in metrics:
            vals = {}
            for m in methods:
                got = [cell(table[qset][d].get(m), metric, raw) for d in datasets]
                got = [v for v in got if v is not None]
                vals[m] = sum(got) / len(got) if got else None
            rows.append(("Mean", metric, vals))
    return datasets, methods, rows


def print_plain(qset, datasets, methods, rows, metrics, present):
    w_d = max([len("dataset")] + [len(r[0]) for r in rows])
    w_m = max([len("metric")] + [len(x) for x in metrics])
    cols = {m: max(len(m), 8) for m in methods}
    for _, metric, vals in rows:
        for m in methods:
            cols[m] = max(cols[m], len(fmt(vals[m], metric)))
    head = " " * (w_d + 2 + w_m) + "  " + "  ".join(m.rjust(cols[m]) for m in methods)
    print(f"\n=== {qset} questions ===")
    print(head)
    print(f"{'dataset'.ljust(w_d)}  {'metric'.ljust(w_m)}")
    last = None
    for d, metric, vals in rows:
        left = d.ljust(w_d) if d != last else " " * w_d
        last = d
        body = "  ".join(fmt(vals[m], metric).rjust(cols[m]) for m in methods)
        print(f"{left}  {metric.ljust(w_m)}  {body}")
    # completeness note
    for m in methods:
        gone = [d for d in datasets if m not in present[d]]
        if gone:
            print(f"[note] {m} missing on {', '.join(gone)}", file=sys.stderr)


def print_latex(qset, datasets, methods, rows, metrics):
    print(f"\n% ---- {qset} questions ----")
    print("\\begin{tabular}{ll" + "r" * len(methods) + "}")
    print("\\toprule")
    print("Dataset & Metric & " + " & ".join(methods.copy()) + " \\\\")
    print("\\midrule")
    last = None
    for d, metric, vals in rows:
        if last is not None and d != last:
            print("\\midrule")
        left = d if d != last else ""
        last = d
        body = " & ".join(fmt(vals[m], metric) for m in methods)
        metric_tex = metric.replace("|H|", "$|\\mathcal{H}_q|$")
        print(f"{left} & {metric_tex} & {body} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="knowledge_scores")
    ap.add_argument("--metrics", nargs="+", default=METRICS,
                    help=f"subset and order of {METRICS}")
    ap.add_argument("--raw", action="store_true",
                    help="keep similarities in 0..1 instead of scaling to 0..100")
    ap.add_argument("--no-mean", action="store_true", help="drop the Mean row")
    ap.add_argument("--latex", action="store_true", help="also emit a LaTeX body")
    ap.add_argument("--csv", default="", help="also write a long format csv")
    args = ap.parse_args()

    for m in args.metrics:
        if m not in SOURCE:
            sys.exit(f"unknown metric {m}, pick from {METRICS}")

    table = collect(args.root)
    if not table:
        sys.exit(f"{args.root}: no score files found")

    long_rows = []
    for qset in sorted(table, key=lambda q: (len(q), q)):
        datasets, methods, rows = build(table, qset, args.metrics,
                                        args.raw, not args.no_mean)
        print_plain(qset, datasets, methods, rows, args.metrics, table[qset])
        if args.latex:
            print_latex(qset, datasets, methods, rows, args.metrics)
        for d, metric, vals in rows:
            for m in methods:
                if vals[m] is not None:
                    long_rows.append({"qset": qset, "dataset": d, "method": m,
                                      "metric": metric, "value": round(vals[m], 4)})

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["qset", "dataset", "method",
                                              "metric", "value"])
            w.writeheader()
            w.writerows(long_rows)
        print(f"\n[saved] {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()