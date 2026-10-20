"""
Answer scoring for Nexus.

Computes, for one method / dataset / question set:

  F1      token overlap against the golden answer, exactly as in the official
          eval.py of HyperGraphRAG, best golden answer when several are given.
          Reported for comparability only, it falls with answer length.
  EM      exact match after the same normalisation, kept as a sanity column.
  Cont.   the `contains` check of the baselines, the golden answer appears in
          the generated answer after normalisation.
  A-E     Answer Equivalence, an LLM judge is shown the question, the golden
          answer and the response, and decides whether the response gives the
          golden answer. OFF by default, since it costs API calls. Pass --judge
          to compute it.
  |a|     average answer length in whitespace tokens of the extracted answer.

The answer is the span inside <answer></answer> when present, otherwise the
whole generation, which is what the official scorer does.

Usage
-----
  python score_answers.py \
      --questions datasets/Mix/questions_200.json \
      --pred      output/generation_200/Nexus-union/Mix/ \
      --out-dir   answer_scores

  python score_answers.py ... --judge         # also compute A-E, costs API calls
  python score_answers.py ... --judge --rejudge   # ignore cached labels

Without --judge only F1, EM, Cont. and |a| are computed, which is free and
instant, so a run can be repeated after every change to the generations.
A-E labels already in the output file are always carried over, judging or not,
so a cheap re-run never destroys judgments that were paid for.

--pred takes a file or the directory holding it. Method, dataset and question
set are read off the path and the question count, overridable with --name,
--dataset and --qset. The result lands at
<out-dir>/<method>/<dataset>/<qset>.json.

The judge, when asked for, needs an OpenAI key, from --api-key-file,
openai_api_key.txt in the working directory, or $OPENAI_API_KEY. Override the
prompt with --judge-prompt file.txt, which must contain {question}, {gold} and
{answer}.
"""

import argparse
import json
import os
import re
import string
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------- normalisation

def normalize_answer(answer: str) -> str:
    """Identical to eval.py in HyperGraphRAG."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(answer.lower())))


def norm_question(q: str) -> str:
    return " ".join(q.lower().split())


def extract_answer(generation: str):
    """The span the official scorer grades, and whether a tag was found."""
    try:
        return generation.split("<answer>")[1].split("</answer>")[0].strip(), True
    except IndexError:
        return generation.strip(), False


# -------------------------------------------------------------- offline metrics

def f1_one(gold: str, predicted: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(predicted).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def em_one(gold: str, predicted: str) -> float:
    return 1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0


def contains_one(gold: str, predicted: str) -> float:
    g = normalize_answer(gold)
    return 1.0 if g and g in normalize_answer(predicted) else 0.0


def best(fn, golds, predicted):
    return max((fn(g, predicted) for g in golds), default=0.0)


# ----------------------------------------------------------------------- judge

DEFAULT_PROMPT = """You are grading a question answering system.

Question:
{question}

Reference answer, any one of which counts as correct:
{gold}

System response:
{answer}

Decide whether the system response gives the reference answer to the question.
The response may be longer than the reference answer and may add context, that
is fine. What matters is the answer it gives to the question.

Reply with exactly one word:
CORRECT    the response gives the reference answer
WRONG      the response gives a different answer
MISSING    the response does not answer the question, or refuses, or says the
           information is not available

One word only, nothing else."""

LABELS = {"CORRECT": 1.0, "WRONG": 0.0, "MISSING": 0.0}


def read_key(path):
    if path and os.path.isfile(path):
        return open(path).read().strip()
    if os.path.isfile("openai_api_key.txt"):
        return open("openai_api_key.txt").read().strip()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    sys.exit("no OpenAI key, use --api-key-file, openai_api_key.txt or "
             "$OPENAI_API_KEY, or drop --judge")


class Judge:
    def __init__(self, model, prompt, api_key, base_url=None, retries=3):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url \
            else OpenAI(api_key=api_key)
        self.model = model
        self.prompt = prompt
        self.retries = retries

    def label(self, question, gold, answer):
        text = self.prompt.format(question=question, gold=gold, answer=answer)
        last = ""
        for _ in range(self.retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_tokens=8,
                    messages=[{"role": "user", "content": text}],
                )
                out = (r.choices[0].message.content or "").strip().upper()
                for lab in LABELS:
                    if lab in out:
                        return lab
                last = out
            except Exception as e:          # rate limit, transient failure
                last = f"ERROR: {e}"
        return f"UNPARSED({last[:60]})"


# ----------------------------------------------------------------- pred loading

_PREFERRED = ("test_generation.json", "generation.json", "answers.json",
              "test_result.json")
_NOT_GENERATION = {"config.json", "summary.json", "chunks.json",
                   "query_entities.json", "questions.json", "test_score.json",
                   "knowledge.json", "test_knowledge.json", "retrieved.json"}


def skip(msg):
    print(f"[skip] {msg}", file=sys.stderr)
    sys.exit(0)


def resolve_pred(path, basename=""):
    if basename:
        cand = path if os.path.isfile(path) else os.path.join(path, basename)
        if not os.path.isfile(cand):
            skip(f"no {basename} at {path}")
        return cand
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        skip(f"no prediction at {path}")
    names = sorted(f for f in os.listdir(path) if f.endswith(".json"))
    for pref in _PREFERRED:
        if pref in names:
            return os.path.join(path, pref)
    rest = [f for f in names if f not in _NOT_GENERATION]
    if len(rest) == 1:
        print(f"[pred]  using {rest[0]}", file=sys.stderr)
        return os.path.join(path, rest[0])
    if not rest:
        skip(f"nothing scorable in {path}")
    skip(f"{path} holds several candidates {rest}, pass --file to pick one")


def _records(path):
    """A list of records, from a bare list or from results / answers, which may
    itself be a dict keyed by question id in arbitrary order."""
    data = json.load(open(path))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "answers", "records", "data"):
            v = data.get(key)
            if isinstance(v, list):
                return v
            if isinstance(v, dict):
                def order(kv):
                    k, r = kv
                    if isinstance(r, dict) and isinstance(r.get("qi"), int):
                        return (0, r["qi"])
                    try:
                        return (0, int(k))
                    except (TypeError, ValueError):
                        return (1, 0)
                return [r for _, r in sorted(v.items(), key=order)]
    sys.exit(f"{path}: expected a list of records, or results / answers holding "
             f"a list or a dict keyed by question id")


# fields that hold the whole model output, so the <answer> span has to be cut out
_FULL = {"generation", "full_output", "output"}


def _generation_of(rec):
    """The text to grade, and the field it came from."""
    for key in ("generation", "answer", "response", "output", "prediction",
                "full_output"):
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            return v, key
        if isinstance(v, dict):
            for k2 in ("text", "content", "answer"):
                if isinstance(v.get(k2), str) and v[k2].strip():
                    return v[k2], key
    return "", ""


def load_pred(path):
    out = []
    for i, rec in enumerate(_records(path)):
        if not isinstance(rec, dict):
            sys.exit(f"{path}: record {i} is {type(rec).__name__}, expected dict")
        text, source = _generation_of(rec)
        golds = rec.get("golden_answers") or rec.get("gold") or []
        if isinstance(golds, str):
            golds = [golds]
        out.append({
            "qid": rec.get("qi", rec.get("id", i)),
            "question": rec.get("question", ""),
            "generation": text,
            "source": source,
            "golden_answers": [g for g in golds if isinstance(g, str) and g.strip()],
        })
    if not any(r["generation"] for r in out):
        sys.exit(f"{path}: no answer field found, looked for generation, answer, "
                 f"response, output, prediction, full_output")
    return out


def load_questions(path):
    out = []
    for i, rec in enumerate(_records(path)):
        golds = rec.get("golden_answers") or rec.get("answers") or []
        if isinstance(golds, str):
            golds = [golds]
        out.append({
            "qid": rec.get("qi", rec.get("id", i)),
            "question": rec.get("question", ""),
            "golden_answers": [g for g in golds if isinstance(g, str) and g.strip()],
        })
    if not any(r["golden_answers"] for r in out):
        sys.exit(f"{path}: no 'golden_answers' found, this is not a questions file")
    return out


def align(gold, pred):
    index = {}
    for i, g in enumerate(gold):
        index.setdefault(norm_question(g["question"]), i)
    pairs, by_text, by_pos, missing = [], 0, 0, 0
    for p_i, p in enumerate(pred):
        g_i = index.get(norm_question(p["question"]))
        if g_i is not None:
            by_text += 1
        elif p_i < len(gold):
            g_i, by_pos = p_i, by_pos + 1
        else:
            missing += 1
            continue
        pairs.append((gold[g_i], p))
    return pairs, {"by_text": by_text, "by_position": by_pos, "unmatched": missing}


# ---------------------------------------------------- naming from the pred path

_KNOWN = {
    "hypergraphrag": "HyperGraphRAG", "hyperrag": "HyperRAG", "proh": "PRoH",
    "proh-l": "PRoH-L", "nexus": "Nexus", "golden": "Golden",
    "simsearch": "SimSearch", "plainllm": "PlainLLM",
}
_DATASETS = {
    "agriculture": "Agriculture", "cs": "CS", "hypertension": "Hypertension",
    "medicine": "Medicine", "legal": "Legal", "mix": "Mix",
}
_SETSIZE = re.compile(r"(?:^|[^0-9])(200|512)(?:[^0-9]|$)")


def infer_tags(pred_path, n_questions):
    parts = [p for p in os.path.normpath(pred_path).split(os.sep)
             if p not in ("", ".", "..")]
    dirs = parts[:-1]
    dataset = ""
    for p in reversed(dirs):
        if p.lower() in _DATASETS:
            dataset = _DATASETS[p.lower()]
            break
    if not dataset and dirs:
        dataset = dirs[-1]
    method = ""
    for p in reversed(dirs):
        if p.lower() in _KNOWN:
            method = _KNOWN[p.lower()]
            break
    if not method:
        rest = [p for p in dirs if p != dataset and not _SETSIZE.search(p)]
        method = rest[-1] if rest else "method"
    qset = str(n_questions)
    if n_questions not in (200, 512):
        for p in reversed(parts):
            m = _SETSIZE.search(p)
            if m:
                qset = m.group(1)
                break
    return method, dataset, qset


# ---------------------------------------------------------------------- scoring

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--pred", required=True,
                    help="generations of one method, a file or its directory")
    ap.add_argument("--file", default="", help="basename inside --pred")
    ap.add_argument("--out-dir", default="answer_scores")
    ap.add_argument("--name", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--qset", default="")
    ap.add_argument("--judge", action="store_true",
                    help="also compute A-E with an LLM judge. Off by default, "
                         "since it costs one API call per question")
    ap.add_argument("--rejudge", action="store_true",
                    help="with --judge, ignore labels already in the output file "
                         "and judge every question again")
    ap.add_argument("--no-judge", action="store_true",
                    help="accepted and ignored, not judging is now the default")
    ap.add_argument("--force", action="store_true",
                    help="accepted and ignored, the output is always rewritten")
    ap.add_argument("--resume", action="store_true",
                    help="accepted and ignored, existing A-E labels are always "
                         "carried over")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--judge-prompt", default="",
                    help="file with {question}, {gold} and {answer}")
    ap.add_argument("--api-key-file", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if not os.path.isfile(args.questions):
        skip(f"no questions file at {args.questions}")
    pred_path = resolve_pred(args.pred, args.file)

    gold = load_questions(args.questions)
    pred = load_pred(pred_path)
    method, dataset, qset = infer_tags(pred_path, len(gold))
    method = args.name or method
    dataset = args.dataset or dataset
    qset = args.qset or qset

    pairs, match = align(gold, pred)
    print(f"[tags]  method={method} dataset={dataset} qset={qset}", file=sys.stderr)
    print(f"[align] {match}", file=sys.stderr)

    out_path = os.path.join(args.out_dir, method, dataset, f"{qset}.json")

    # A-E costs money, so labels already written are carried over whatever this
    # run is doing. Recomputing the free metrics never throws a judgment away.
    cached = {}
    if os.path.isfile(out_path) and not (args.judge and args.rejudge):
        try:
            old = json.load(open(out_path))
            for d in old.get("per_question", []):
                lab = d.get("ae_label") or ""
                if lab and not lab.startswith("UNPARSED"):
                    cached[str(d["qid"])] = (lab, d.get("ae", 0.0))
            if cached:
                print(f"[cache]  {len(cached)} A-E labels carried over",
                      file=sys.stderr)
        except (ValueError, OSError):
            pass

    rows = []
    for g, p in pairs:
        golds = g["golden_answers"] or p["golden_answers"]
        answer, tagged = extract_answer(p["generation"])
        if not tagged and p["source"] not in _FULL:
            tagged = True          # the field already holds the extracted answer
        rows.append({
            "qid": p["qid"],
            "question": g["question"] or p["question"],
            "golds": golds,
            "answer": answer,
            "f1": round(best(f1_one, golds, answer), 6),
            "em": round(best(em_one, golds, answer), 6),
            "contains": round(best(contains_one, golds, answer), 6),
            "alen": len(answer.split()),
            "tagged": tagged,
        })

    if not args.judge:
        for r in rows:
            lab, val = cached.get(str(r["qid"]), ("", None))
            r["ae_label"], r["ae"] = lab, val
        if not cached:
            print("[judge]  skipped, pass --judge for A-E", file=sys.stderr)
    else:
        prompt = open(args.judge_prompt).read() if args.judge_prompt else DEFAULT_PROMPT
        for token in ("{question}", "{gold}", "{answer}"):
            if token not in prompt:
                sys.exit(f"judge prompt is missing {token}")
        todo = [r for r in rows if str(r["qid"]) not in cached]
        print(f"[judge] {len(todo)} calls to {args.model}", file=sys.stderr)
        judge = Judge(args.model, prompt, read_key(args.api_key_file),
                      args.base_url or None) if todo else None

        def work(r):
            gold_text = "\n".join(f"- {g}" for g in r["golds"]) if len(r["golds"]) > 1 \
                else (r["golds"][0] if r["golds"] else "")
            return r, judge.label(r["question"], gold_text, r["answer"])

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r, lab in ex.map(work, todo):
                r["ae_label"] = lab
                r["ae"] = LABELS.get(lab)
        for r in rows:
            if str(r["qid"]) in cached:
                r["ae_label"], r["ae"] = cached[str(r["qid"])]

    n = len(rows) or 1
    judged = [r for r in rows if r.get("ae") is not None]
    unparsed = [r for r in rows if r.get("ae_label", "").startswith("UNPARSED")]
    counts = Counter(r.get("ae_label", "") for r in rows)
    overall = {
        "method": method, "dataset": dataset, "qset": qset,
        "questions": len(rows),
        "F1": round(100 * sum(r["f1"] for r in rows) / n, 4),
        "EM": round(100 * sum(r["em"] for r in rows) / n, 4),
        "Cont.": round(100 * sum(r["contains"] for r in rows) / n, 4),
        "A-E": round(100 * sum(r["ae"] for r in judged) / len(judged), 4) if judged else None,
        "|a|": round(sum(r["alen"] for r in rows) / n, 2),
        "judged": len(judged),
        "untagged": sum(0 if r["tagged"] else 1 for r in rows),
        "answer_field": Counter(p["source"] for _, p in pairs).most_common(1)[0][0]
                        if pairs else "",
        "unparsed": len(unparsed),
        "labels": {k: v for k, v in counts.items() if k},
    }
    print(json.dumps(overall, indent=2))
    if unparsed:
        print(f"[warn] {len(unparsed)} judgments unparsed, rerun with --resume",
              file=sys.stderr)

    per_q = [{"qid": r["qid"], "f1": r["f1"], "em": r["em"],
              "contains": r["contains"], "ae": r["ae"],
              "ae_label": r.get("ae_label", ""), "alen": r["alen"]} for r in rows]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump({"overall": overall, "match": match, "per_question": per_q},
              open(out_path, "w"), indent=2)
    print(f"[saved] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()