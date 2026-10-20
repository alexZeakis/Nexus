"""
Collect the output of score_answers.py into one table per question set.

Reads <root>/<method>/<dataset>/<qset>.json and takes the `overall` block of
each. Those values are already the mean over the questions of that run, so
nothing is averaged again here except the Mean row, which is the unweighted
mean over the datasets a method actually has.

Usage
-----
  python show_answer_scores.py
  python show_answer_scores.py --root answer_scores --latex
  python show_answer_scores.py --metrics F1 A-E --no-mean --csv answers.csv
"""

import argparse
import csv
import json
import os
import sys

METRICS = ["F1", "EM", "Cont.", "A-E", "|a|"]
LATEX = {"F1": "F1", "EM": "EM", "Cont.": "Cont.", "A-E": "A-E", "|a|": "$|a|$"}
INTEGERISH = {"|a|"}

DATASET_ORDER = ["Agriculture", "CS", "Hypertension", "Medicine", "Legal", "Mix"]
METHOD_ORDER = ["Golden", "SimSearch", "PlainLLM", "HyperGraphRAG", "HyperRAG",
                "PRoH", "PRoH-L", "Nexus"]


def rank(name, order):
    for i, o in enumerate(order):
        if name == o:
            return (0, i, name)
    for i, o in enumerate(order):
        if name.startswith(o):
            return (1, i, name)
    return (2, 0, name)


def collect(root):
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
            for fn in sorted(f for f in os.listdir(ddir) if f.endswith(".json")):
                path = os.path.join(ddir, fn)
                try:
                    blob = json.load(open(path))
                except (ValueError, OSError) as e:
                    bad.append(f"{path}: {e}")
                    continue
                ov = blob.get("overall", blob)
                if "F1" not in ov:
                    bad.append(f"{path}: no overall block")
                    continue
                qset = str(ov.get("qset") or os.path.splitext(fn)[0])
                table.setdefault(qset, {}) \
                     .setdefault(ov.get("dataset") or dataset, {})[
                         ov.get("method") or method] = ov
                if ov.get("unparsed"):
                    bad.append(f"{path}: {ov['unparsed']} unparsed judgments")
    for b in bad:
        print(f"[warn] {b}", file=sys.stderr)
    return table


def cell(ov, metric):
    if ov is None:
        return None
    v = ov.get(metric)
    return None if v is None else float(v)


def fmt(v, metric):
    if v is None:
        return "-"
    return f"{v:,.1f}" if metric in INTEGERISH else f"{v:.2f}"


def build(table, qset, metrics, with_mean):
    datasets = sorted(table[qset], key=lambda d: rank(d, DATASET_ORDER))
    methods = sorted({m for d in table[qset].values() for m in d},
                     key=lambda m: rank(m, METHOD_ORDER))
    rows = []
    for d in datasets:
        for metric in metrics:
            rows.append((d, metric,
                         {m: cell(table[qset][d].get(m), metric) for m in methods}))
    if with_mean and len(datasets) > 1:
        for metric in metrics:
            vals = {}
            for m in methods:
                got = [cell(table[qset][d].get(m), metric) for d in datasets]
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
    print(f"\n=== {qset} questions ===")
    print(" " * (w_d + 2 + w_m) + "  " +
          "  ".join(m.rjust(cols[m]) for m in methods))
    print(f"{'dataset'.ljust(w_d)}  {'metric'.ljust(w_m)}")
    last = None
    for d, metric, vals in rows:
        left = d.ljust(w_d) if d != last else " " * w_d
        last = d
        print(f"{left}  {metric.ljust(w_m)}  " +
              "  ".join(fmt(vals[m], metric).rjust(cols[m]) for m in methods))
    for m in methods:
        gone = [d for d in datasets if m not in present[d]]
        if gone:
            print(f"[note] {m} missing on {', '.join(gone)}", file=sys.stderr)


def print_latex(qset, methods, rows):
    print(f"\n% ---- {qset} questions ----")
    print("\\begin{tabular}{ll" + "r" * len(methods) + "}")
    print("\\toprule")
    print("Dataset & Metric & " + " & ".join(methods) + " \\\\")
    print("\\midrule")
    last = None
    for d, metric, vals in rows:
        if last is not None and d != last:
            print("\\midrule")
        left = d if d != last else ""
        last = d
        print(f"{left} & {LATEX.get(metric, metric)} & " +
              " & ".join(fmt(vals[m], metric) for m in methods) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="answer_scores")
    ap.add_argument("--metrics", nargs="+", default=METRICS,
                    help=f"subset and order of {METRICS}")
    ap.add_argument("--no-mean", action="store_true")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    for m in args.metrics:
        if m not in METRICS:
            sys.exit(f"unknown metric {m}, pick from {METRICS}")

    table = collect(args.root)
    if not table:
        sys.exit(f"{args.root}: no score files found")

    long_rows = []
    for qset in sorted(table, key=lambda q: (len(q), q)):
        datasets, methods, rows = build(table, qset, args.metrics, not args.no_mean)
        print_plain(qset, datasets, methods, rows, args.metrics, table[qset])
        if args.latex:
            print_latex(qset, methods, rows)
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