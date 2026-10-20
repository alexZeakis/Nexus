"""
Total cost of running one dataset end to end, against answer quality.

Six methods: the three systems we compare against and our three retrievers.
Baselines are left out, they are reference points rather than systems with a
construction bill.

Nothing is hardcoded. Cost comes from the gateway logs and quality from the
output of score_answers.py, so the table always matches the last run.

  Build   tokens spent once to turn the corpus into a graph, before any
          question is asked. HyperRAG reuses the graph HyperGraphRAG builds and
          all three of our retrievers share one graph, so each carries that
          construction cost.
  Serve   tokens spent answering every question of the dataset, retrieval plus
          generation
  Total   Build plus Serve, the whole bill for that dataset
  Calls   LLM calls behind that bill, construction plus retrieval plus generation
  F1      token overlap against the golden answer, weighted over the two
          question sets. --metric AE switches to judged Answer Equivalence.
  Mtok    Total divided by the number of questions answered correctly, so cost
   /corr  and quality are one number

Tokens are input plus output, summed over the log lines themselves rather than
rebuilt from per-call averages. Nothing is amortised, these are the figures the
experiment actually cost.

WHERE THE NUMBERS COME FROM

  --scores DIR    <dir>/<method>/<dataset>/<qset>.json, written by
                  score_answers.py. The `overall` block gives F1, A-E and the
                  number of questions scored.
  --logs GLOB     the gateway jsonl. A line counts when status is 200 and the
                  endpoint is a completion endpoint. `method` is read as
                  <Method>_<Dataset> and `stage` as construct, retrieve_<n> or
                  generate_<n>.

A method with a score but no serve log is reported, so a missing log is never
silently read as a free run.

Usage
-----
  python total_cost.py
  python total_cost.py --metric AE
  python total_cost.py --tex
  python total_cost.py --plot cost.png
  python total_cost.py --show
  python total_cost.py --only Nexus-Hyb PRoH HyperGraphRAG
  python total_cost.py --logs ../eval/logs_merged.jsonl
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

# ------------------------------------------------------------------- the names

DATASETS = ["Agriculture", "CS", "Hypertension", "Legal", "Mix"]
METHODS = ["HyperGraphRAG", "HyperRAG", "PRoH",
           "Nexus-Sem", "Nexus-Str", "Nexus-Hyb"]
QSETS = ["512", "200"]

#: display name -> the name retrieval and generation are logged under, which is
#: also the directory score_answers.py writes into
RUN_NAME = {
    "HyperGraphRAG": "HyperGraphRAG",
    "HyperRAG":      "HyperRAG",
    "PRoH":          "PRoH",
    "Nexus-Sem":     "Nexus-semantic",
    "Nexus-Str":     "Nexus-structured",
    "Nexus-Hyb":     "Nexus-fused",
}

#: display name -> the name the construction run is logged under. HyperRAG runs
#: on the graph HyperGraphRAG builds, and our three retrievers all run on the
#: one graph we build, so each carries that bill.
BUILD_NAME = {
    "HyperGraphRAG": "HyperGraphRAG",
    "HyperRAG":      "HyperGraphRAG",
    "PRoH":          "PRoH",
    "Nexus-Sem":     "Nexus",
    "Nexus-Str":     "Nexus",
    "Nexus-Hyb":     "Nexus",
}

BUILDS_ON = {m: BUILD_NAME[m] for m in METHODS if BUILD_NAME[m] != RUN_NAME[m]}

METRIC_KEY = {"F1": "F1", "AE": "A-E"}
AXIS = {"F1": "F1", "AE": "Answer Equivalence (A-E)"}

COMPLETION_ENDPOINTS = ("/v1/chat/completions", "/v1/completions", "/v1/responses")


# --------------------------------------------------------------------- loading

def load_logs(patterns):
    """(build, serve) token and call totals read off the gateway logs.

    build[(method, dataset)]        = [calls, tokens]
    serve[(method, dataset, qset)]  = [calls, tokens]
    """
    paths = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        sys.exit(f"no log files matched {' '.join(patterns)}")

    build = defaultdict(lambda: [0, 0.0])
    serve = defaultdict(lambda: [0, 0.0])
    seen_stages, bad = set(), 0

    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                if rec.get("status") != 200:
                    continue
                if rec.get("endpoint") not in COMPLETION_ENDPOINTS:
                    continue
                method, _, dataset = str(rec.get("method", "")).partition("_")
                stage = str(rec.get("stage", ""))
                seen_stages.add(stage)
                tokens = ((rec.get("prompt_tokens") or 0)
                          + (rec.get("completion_tokens") or 0))
                if stage.startswith("construct"):
                    entry = build[(method, dataset)]
                elif stage.startswith(("retrieve", "generate")):
                    qset = stage.rsplit("_", 1)[-1]
                    if not qset.isdigit():
                        continue
                    entry = serve[(method, dataset, qset)]
                else:
                    continue
                entry[0] += 1
                entry[1] += tokens

    print(f"[logs]  {len(paths)} file(s), stages {', '.join(sorted(seen_stages))}"
          + (f", {bad} unparseable lines" if bad else ""), file=sys.stderr)
    return build, serve


def load_scores(root):
    """{(method, dataset, qset): overall} from the output of score_answers.py.

    The path wins over the `overall` block. score_answers.py records whatever
    name the run carried when it was scored, and older runs carry names the
    pipeline has since renamed, so the directory the shell loop wrote into is
    the one reliable key.
    """
    if not os.path.isdir(root):
        sys.exit(f"no such directory: {root}")
    out, renamed = {}, []
    for method in sorted(os.listdir(root)):
        mdir = os.path.join(root, method)
        if not os.path.isdir(mdir):
            continue
        for dataset in sorted(os.listdir(mdir)):
            ddir = os.path.join(mdir, dataset)
            if not os.path.isdir(ddir):
                continue
            for fn in sorted(f for f in os.listdir(ddir) if f.endswith(".json")):
                try:
                    blob = json.load(open(os.path.join(ddir, fn)))
                except (ValueError, OSError) as exc:
                    print(f"[warn] {os.path.join(ddir, fn)}: {exc}", file=sys.stderr)
                    continue
                ov = blob.get("overall", blob)
                if "F1" not in ov:
                    continue
                qset = os.path.splitext(fn)[0] or str(ov.get("qset") or "")
                stale = ov.get("method")
                if stale and stale != method:
                    renamed.append(f"{method}/{dataset}/{fn} recorded as {stale}")
                out[(method, dataset, qset)] = ov
    if not out:
        sys.exit(f"{root}: no score files found")
    print(f"[scores] {len(out)} method/dataset/qset blocks", file=sys.stderr)
    if renamed:
        print(f"[scores] {len(renamed)} file(s) carry an older method name, "
              f"the directory is used instead (e.g. {renamed[0]})", file=sys.stderr)
    return out


# ------------------------------------------------------------------- the model

def build_rows(scores, build, serve, metric="F1", methods=None):
    """One row per method and dataset, plus the mean over datasets.

    A method is included only where both a cost and a quality number exist, and
    the mean records how many datasets it covers, so an incomplete mean is never
    mistaken for a complete one.
    """
    key = METRIC_KEY[metric]
    methods = methods or METHODS
    rows, no_log = {}, []

    for ds in DATASETS:
        for m in methods:
            b_calls, b_tok = build.get((BUILD_NAME[m], ds), (0, 0.0))
            serve_tok, calls, correct, asked = 0.0, float(b_calls), 0.0, 0
            for qset in QSETS:
                ov = scores.get((RUN_NAME[m], ds, qset))
                if ov is None:
                    continue
                quality = ov.get(key)
                if quality is None:
                    continue
                n = int(ov.get("questions") or qset)
                s = serve.get((RUN_NAME[m], ds, qset))
                if s is None:
                    no_log.append(f"{m} {ds} {qset}")
                    s = (0, 0.0)
                serve_tok += s[1]
                calls += s[0]
                correct += n * quality / 100.0
                asked += n
            if asked == 0:
                continue
            total = b_tok + serve_tok
            rows[(ds, m)] = {
                "build": b_tok, "serve": serve_tok, "total": total, "calls": calls,
                "questions": asked, "correct": correct,
                "quality": 100.0 * correct / asked,
                "tok_per_correct": total / correct if correct else float("inf"),
                "calls_per_correct": calls / correct if correct else float("inf"),
            }

    for m in methods:
        got = [rows[(ds, m)] for ds in DATASETS if (ds, m) in rows]
        if not got:
            continue
        k = len(got)
        mean = {name: sum(g[name] for g in got) / k for name in
                ("build", "serve", "total", "calls", "questions", "correct")}
        mean["quality"] = 100.0 * mean["correct"] / mean["questions"]
        mean["tok_per_correct"] = mean["total"] / mean["correct"]
        mean["calls_per_correct"] = mean["calls"] / mean["correct"]
        mean["datasets"] = k
        rows[("Mean", m)] = mean

    return rows, no_log


# ------------------------------------------------------------------- reporting

def notes(rows, metric, methods, no_log):
    out = []
    for m in methods:
        gone = [ds for ds in DATASETS if (ds, m) not in rows]
        if gone and len(gone) < len(DATASETS):
            out.append(f"{m} has no {metric} on {', '.join(gone)}, "
                       f"its Mean covers {len(DATASETS) - len(gone)} datasets")
        elif len(gone) == len(DATASETS):
            out.append(f"{m} dropped, no {metric} anywhere")
    for m in methods:
        if m in BUILDS_ON and (("Mean", m) in rows):
            out.append(f"{m} carries {BUILDS_ON[m]}'s construction cost")
    for m in methods:
        if ("Mean", m) in rows and rows[("Mean", m)]["build"] == 0:
            out.append(f"{m} has no construction lines under "
                       f"{BUILD_NAME[m]}, its Build reads 0")
    if no_log:
        out.append("scored but no serve log, Serve reads 0 for: "
                   + ", ".join(no_log))
    return out


def show(rows, metric, methods, no_log):
    print(f"[quality] {metric}")
    print(f"{'dataset':13} {'method':15} {'Build M':>9} {'Serve M':>9} "
          f"{'Total M':>9} {'Calls':>9} {metric:>6} {'Mtok/corr':>10} {'calls/corr':>11}")
    for ds in DATASETS + ["Mean"]:
        if ds == "Mean":
            print()
        for m in methods:
            r = rows.get((ds, m))
            if not r:
                continue
            print(f"{ds:13} {m:15} {r['build']/1e6:9.2f} {r['serve']/1e6:9.2f} "
                  f"{r['total']/1e6:9.2f} {r['calls']:9,.0f} {r['quality']:6.1f} "
                  f"{r['tok_per_correct']/1e6*1000:10.2f} {r['calls_per_correct']:11.1f}")
    for n in notes(rows, metric, methods, no_log):
        print(f"[note] {n}")


TEX_NAME = {"Nexus-Sem": "\\nexus{}-Sem",
            "Nexus-Str": "\\nexus{}-Str",
            "Nexus-Hyb": "\\nexus{}-Hyb"}


def tex(rows, metric, methods):
    print("\n% total cost of one dataset against answer quality")
    print("\\begin{tabular}{llrrrrrr}")
    print("\\toprule")
    print(f"Dataset & Method & Build & Serve & Total & Calls & {metric} & "
          "kTok per correct \\\\")
    print("& & \\multicolumn{3}{c}{million tokens} & & & \\\\")
    for ds in DATASETS + ["Mean"]:
        print("\\midrule")
        for m in methods:
            r = rows.get((ds, m))
            if not r:
                continue
            print(f"{ds} & {TEX_NAME.get(m, m)} & {r['build']/1e6:.2f} & "
                  f"{r['serve']/1e6:.2f} & {r['total']/1e6:.2f} & {r['calls']:,.0f} & "
                  f"{r['quality']:.1f} & {r['tok_per_correct']/1e3:.1f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


# -------------------------------------------------------------------- plotting
# Colour carries the four systems only: that set is validated for all-pairs use
# on a light surface (worst CVD dE 9.2, normal 16.3). No five-hue set passes, so
# the reference points take neutral ink and are told apart by marker and by the
# direct label on every mean.
STYLE = {
    "Nexus-Hyb":     ("#2a78d6", "o", 1.0),
    "Nexus-Sem":     ("#2a78d6", "*", 1.0),
    "Nexus-Str":     ("#2a78d6", "v", 1.0),
    "HyperGraphRAG": ("#eb6834", "s", 1.0),
    "HyperRAG":      ("#1baf7a", "D", 1.0),
    "PRoH":          ("#4a3aa7", "^", 1.0),
}

#: the three retrievers sit close together, so two labels are moved off the
#: default position above the marker to keep them clear of their neighbours
LABEL_OFFSET = {"Nexus-Str": (0, -26), "Nexus-Sem": (-8, 13)}


def plot(rows, metric, methods, path="", show_window=False, label_datasets=False,
         show_calls=False):
    import matplotlib
    if not show_window:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for m in methods:
        pts = [(rows[(ds, m)]["total"] / 1e6, rows[(ds, m)]["quality"], ds)
               for ds in DATASETS if (ds, m) in rows]
        if not pts:
            continue
        color, marker, strength = STYLE.get(m, ("#52514e", "o", 0.8))
        xs, ys, _ = zip(*pts)
        bump = 1.6 if marker == "*" else 1.0
        ax.scatter(xs, ys, s=26 * bump, marker=marker, facecolor=color,
                   edgecolor="none", alpha=0.3 * strength, zorder=2)
        if label_datasets:
            for x, y, nm in pts:
                ax.annotate(nm, (x, y), textcoords="offset points",
                            xytext=(5, -3), fontsize=6, color="#8a8880", zorder=3)
        mean = rows.get(("Mean", m))
        if not mean:
            continue
        mx, my = mean["total"] / 1e6, mean["quality"]
        ax.scatter([mx], [my], s=(150 if strength == 1.0 else 90) * bump, marker=marker,
                   facecolor=color, edgecolor="white", linewidth=1.5, zorder=4)
        label = m + ("*" if mean.get("datasets", 5) < len(DATASETS) else "")
        if show_calls:
            label += f"\n{mean['calls'] / 1000:.1f}K calls"
        ax.annotate(label, (mx, my), textcoords="offset points",
                    xytext=LABEL_OFFSET.get(m, (0, 13)),
                    ha="center", fontsize=8.5,
                    color="#2b2a28" if strength == 1.0 else "#52514e", zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Total cost of one dataset (million tokens)", fontsize=9)
    ax.set_ylabel(AXIS.get(metric, metric), fontsize=9)
    ax.grid(True, which="major", color="#e6e5e1", linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color="#f2f1ee", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(labelsize=8, color="#c3c2b7")
    fig.text(0.5, 0.005, "small markers are the five datasets, large markers "
             "their mean, cheaper is to the left",
             ha="center", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        pdf = path.rsplit(".", 1)[0] + ".pdf"
        fig.savefig(pdf, bbox_inches="tight")
        print(f"[saved] {path} and {pdf}")
    if show_window:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="../output/answer_scores",
                    help="output of score_answers.py "
                         "(<dir>/<method>/<dataset>/<qset>.json)")
    ap.add_argument("--logs", nargs="+", default=["logs_*.jsonl"],
                    metavar="GLOB", help="gateway jsonl logs")
    ap.add_argument("--metric", default="F1", choices=["F1", "AE"],
                    help="quality measure on the y axis (default F1)")
    ap.add_argument("--only", nargs="+", default=None, metavar="METHOD",
                    help="restrict to these methods, in this order")
    ap.add_argument("--tex", action="store_true")
    ap.add_argument("--plot", default="", metavar="PNG")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--label-datasets", action="store_true",
                    help="name each small marker in the plot")
    ap.add_argument("--show-calls", action="store_true",
                    help="put each method's LLM call count under its name")
    args = ap.parse_args()

    methods = args.only or METHODS
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"unknown method(s) {unknown}, pick from {METHODS}")

    scores = load_scores(args.scores)
    build, serve = load_logs(args.logs)

    rows, no_log = build_rows(scores, build, serve, args.metric, methods)
    if not rows:
        raise SystemExit("nothing to report: no method has both a cost and a "
                         "quality number. Check --scores and --logs.")
    show(rows, args.metric, methods, no_log)
    if args.tex:
        tex(rows, args.metric, methods)
    if args.plot or args.show:
        plot(rows, args.metric, methods, args.plot, args.show,
             args.label_datasets, args.show_calls)
