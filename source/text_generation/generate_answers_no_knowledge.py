"""
Closed-book answer generation: the question alone, no knowledge.

A copy of generate_answers.py with the knowledge section removed from the
prompt. Everything else is identical on purpose: the same <think>/<answer>
template, the same tag parsing, the same output contract, the same resume
behaviour. Only the Knowledge block is gone, so a difference against any
retrieval run is a difference in what retrieval supplied and not in how the
model was asked.

This is what PlainLLM has to be measured with. Running the normal script on an
empty knowledge file would hand the model "Knowledge: (none)" under a prompt
that tells it to answer from the knowledge alone, and it would decline almost
every question, which measures the prompt rather than the model.

    python -m text_generation.generate_answers_no_knowledge \
        --retrieval ../output/Mix/retrieval/plainllm.json \
        --questions ../datasets/Mix/questions.json \
        --out ../output/Mix/answers/plainllm.json

--retrieval is read only for the question list and the run name, so the file
no_knowledge.py writes is the natural input and the pipeline stays unchanged.
With --questions alone it works without any retrieval file at all.

--knowledge, --top-k and --include-participants are accepted and ignored, so
the same command line serves both scripts.

OUTPUT CONTRACT

As generate_answers.py, with `knowledge` empty and `n_knowledge` zero for
every question, and `config.closed_book` set to true.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_STYLE = "think"
DEFAULT_KNOWLEDGE = "hyperedges"
FLUSH_EVERY = 20
INSUFFICIENT = "INSUFFICIENT EVIDENCE"

Q_FIELDS = ("question", "query", "text", "input")
A_FIELDS = ("answer", "golden_answers", "gold_answer", "ground_truth", "output", "label")

SYSTEM_PROMPT = """\
You answer questions from your own knowledge.

No reference material is supplied. Answer from what you already know.

RULES:
- Answer directly and briefly. A name, a date, a phrase: whatever the question \
asks for, and nothing else. No preamble, no restatement of the question, no \
explanation unless the question asks for one.
- If you do not know the answer, reply exactly: INSUFFICIENT EVIDENCE
- Output the answer as plain text, nothing else."""

USER_PROMPT = """\
Question: {question}
Answer:"""

#: HyperGraphRAG's template with the Knowledge block removed, so the closed-book
#: run differs from every retrieval run in exactly one thing.
THINK_PROMPT = """---Role---

You are a helpful assistant responding to questions.

---Goal---

Answer the given question.
You must first conduct reasoning inside <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>

---Question---

{question}
"""

ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
OPEN_ANSWER_TAG = re.compile(r"<answer>(.*)", re.S | re.I)
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)


# ------------------------------------------------------------------ utils
def first_field(obj, fields, default=""):
    for f in fields:
        if isinstance(obj, dict) and obj.get(f):
            v = obj[f]
            return " ".join(map(str, v)) if isinstance(v, list) else str(v)
    return default


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def extract_answer(raw):
    """(answer, tag_found) from a <think>/<answer> response.

    The model usually closes the tag. When it does not, anything after
    <answer> is taken; when it opens neither, the reasoning is stripped and
    what is left stands in, so a malformed response still scores as whatever
    it actually said rather than as an empty answer.
    """
    match = ANSWER_TAG.search(raw)
    if match:
        return match.group(1).strip(), True
    match = OPEN_ANSWER_TAG.search(raw)
    if match:
        return match.group(1).strip(), True
    stripped = THINK_BLOCK.sub("", raw)
    stripped = re.sub(r"</?(?:think|answer)>", "", stripped, flags=re.I)
    return stripped.strip(), False


def knowledge_sections(result):
    """The retrieval file's knowledge, always as {section: [item, ...]}.

    A retriever that writes one flat list (every baseline, and our own files
    from before the sections existed) is read as a single `hyperedges` section,
    so --knowledge hyperedges and --knowledge all both behave as they did.
    """
    know = result.get("knowledge", [])
    if isinstance(know, dict):
        return OrderedDict((k, v or []) for k, v in know.items())
    return OrderedDict(hyperedges=know or [])


def knowledge_texts(result, top_k=None, include_participants=False,
                    sections=DEFAULT_KNOWLEDGE):
    """The statements to show the model: ranked, deduplicated, text only.

    `sections` is "all" or the name of one section. --top-k applies to EACH
    section, not to the total, so asking for both does not silently crowd the
    hyperedges out of the window. Sections are concatenated in the order they
    appear in the file, hyperedges first, and a chunk whose text repeats a
    hyperedge verbatim is dropped as a duplicate.
    """
    available = knowledge_sections(result)
    wanted = (list(available) if sections == "all"
              else [s for s in [sections] if s in available])

    seen = OrderedDict()
    for name in wanted:
        kept = 0
        for item in available[name]:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            if include_participants and item.get("participants"):
                text = f"{text}  [{', '.join(map(str, item['participants']))}]"
            seen.setdefault(norm(text), text)
            kept += 1
            if top_k and kept >= top_k:
                break
    return list(seen.values())


def make_client(base_url=None, api_key=None):
    from openai import OpenAI
    return OpenAI(base_url=base_url or os.environ.get("NEXUS_LLM_BASE_URL"),
                  api_key=api_key or os.environ.get("NEXUS_LLM_API_KEY", "not-needed"))


def answer_one(client, model, question, knowledge=(), temperature=0.0,
               style=DEFAULT_STYLE, max_tokens=None):
    """(answer, raw response, whether the answer tag was found).

    `knowledge` is accepted and ignored, so the signature matches the script
    this one is a copy of.
    """
    if style == "think":
        messages = [{"role": "user", "content": THINK_PROMPT.format(
            question=question)}]
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(
                        question=question)}]

    kwargs = {"model": model, "temperature": temperature, "messages": messages}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**kwargs)

    raw = (resp.choices[0].message.content or "").strip()
    if style == "think":
        answer, tagged = extract_answer(raw)
        return answer, raw, tagged
    return raw, raw, True


# -------------------------------------------------------------- one run
def generate(retrieval_path, out_path, questions, args, client):
    if retrieval_path:
        blob = json.loads(Path(retrieval_path).read_text())
        results = blob["results"]
        run = blob.get("config", {}).get("run", Path(retrieval_path).stem)
        label = Path(retrieval_path).name
    else:                       # --questions alone, no retrieval file needed
        blob = {}
        results = [{"qi": i} for i in range(len(questions or []))]
        run, label = "plainllm", "(questions only)"
    print(f"\n=== {run} ===  {label}  [{args.prompt_style}, closed book]")

    # no emptiness check here: an empty knowledge file is the point
    supplied = sum(len(items) for r in results
                   for items in knowledge_sections(r).values())
    if supplied:
        print(f"  note: this file carries {supplied} knowledge items, which are "
              f"ignored. Use generate_answers.py to show them to the model.")

    n = min(len(results), args.limit or len(results))
    answers = {}
    if out_path.exists() and not args.restart:
        previous = json.loads(out_path.read_text())
        answers = previous.get("answers", {})
        old_cfg = previous.get("config", {})
        old_style = old_cfg.get("prompt_style", "strict")
        print(f"  resuming: {len(answers)} already answered")
        if answers and not old_cfg.get("closed_book"):
            print("  !! those answers were written WITH knowledge. Mixed runs are "
                  "not comparable: re-run with --restart, or write this variant "
                  "to its own --out.")
        elif answers and old_style != args.prompt_style:
            print(f"  !! those were written with --prompt-style {old_style}, not "
                  f"{args.prompt_style}. Re-run with --restart.")
    todo = [i for i in range(n) if str(i) not in answers]
    print(f"  {n} questions, {len(todo)} to answer")

    def work(i):
        r = results[i]
        qi = r.get("qi", i)
        question = r.get("question") or (
            first_field(questions[qi], Q_FIELDS, f"(question {qi})")
            if questions and qi < len(questions) else f"(question {qi})")
        gold = (first_field(questions[qi], A_FIELDS, "")
                if questions and qi < len(questions) else "")
        know = []                      # closed book, by construction
        t0 = time.time()
        raw, tagged, err = "", True, None
        try:
            ans, raw, tagged = answer_one(client, args.model, question, (),
                                          args.temperature, args.prompt_style,
                                          args.max_tokens)
        except Exception as exc:              # one failure must not end the run
            ans, err = "", f"{type(exc).__name__}: {exc}"
        payload = {"qi": qi, "question": question, "gold": gold,
                   "knowledge": know, "n_knowledge": len(know), "answer": ans,
                   "seconds": round(time.time() - t0, 2),
                   **({"error": err} if err else {})}
        if args.prompt_style == "think":
            payload["full_output"] = raw
            if not tagged and raw:
                payload["no_answer_tag"] = True
        return i, payload

    start, done, failed = time.time(), 0, 0

    def flush():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(
            {"config": {"run": run, "retrieval": str(retrieval_path),
                        "retrieval_method": blob.get("config", {}).get("method"),
                        "model": args.model, "top_k": args.top_k,
                        "temperature": args.temperature,
                        "include_participants": args.include_participants,
                        "prompt_style": args.prompt_style,
                        "knowledge": "none", "closed_book": True,
                        "prompt_version": 2},
             "summary": summarize(answers),
             "answers": answers}, indent=2, ensure_ascii=False))

    def record(i, payload):
        nonlocal done, failed
        answers[str(i)] = payload
        done += 1
        if payload.get("error"):
            failed += 1
            print(f"\n  [error] q{payload['qi']}: {payload['error']}")
        rate = done / max(time.time() - start, 1e-6)
        print(f"  {done}/{len(todo)}  ({rate * 60:.0f}/min, "
              f"eta {(len(todo) - done) / max(rate, 1e-9) / 60:.0f}m)   ",
              end="\r", flush=True)
        if done % FLUSH_EVERY == 0:
            flush()

    if todo:
        if args.workers <= 1:
            for i in todo:
                record(*work(i))
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for fut in as_completed([pool.submit(work, i) for i in todo]):
                    record(*fut.result())
        print()
    flush()

    s = summarize(answers)
    print(f"  answered {s['answered']}  |  failed calls {failed}")
    print("  knowledge per question: none, closed book")
    print(f"  INSUFFICIENT EVIDENCE : {s['insufficient']} "
          f"({100 * s['insufficient'] / max(s['answered'], 1):.1f}%)")
    if args.prompt_style == "think":
        print(f"  no <answer> tag       : {s['no_answer_tag']} "
              "(answer taken from the rest of the response)")
    if s["no_gold"]:
        print(f"  !! {s['no_gold']} answers have no gold answer - check A_FIELDS "
              f"against the keys in questions.json")
    print(f"  wrote {out_path}")
    return out_path


def summarize(answers):
    vals = list(answers.values())
    n = max(len(vals), 1)
    return {
        "answered": len(vals),
        "insufficient": sum(1 for a in vals
                            if a["answer"].strip().upper().startswith("INSUFFICIENT")),
        "empty_answer": sum(1 for a in vals if not a["answer"].strip()),
        "errors": sum(1 for a in vals if a.get("error")),
        "no_answer_tag": sum(1 for a in vals if a.get("no_answer_tag")),
        "no_knowledge": sum(1 for a in vals if a["n_knowledge"] == 0),
        "no_gold": sum(1 for a in vals if not a.get("gold")),
        "mean_knowledge": round(sum(a["n_knowledge"] for a in vals) / n, 2),
        "mean_seconds": round(sum(a.get("seconds", 0) for a in vals) / n, 2),
    }


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Closed-book answer generation")
    ap.add_argument("--retrieval", nargs="+", default=None,
                    help="knowledge file(s), read only for the question list and "
                         "the run name; omit and pass --questions instead")
    ap.add_argument("--out", default=None, help="output file (single --retrieval)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory; each file keeps its input name")
    ap.add_argument("--questions", default=None,
                    help="questions.json, to carry the gold answer through")
    ap.add_argument("--model", default=os.environ.get("NEXUS_LLM_MODEL", DEFAULT_MODEL))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--prompt-style", choices=["think", "strict"], default=DEFAULT_STYLE,
                    help="think: reason in <think>, answer in <answer>, never decline "
                         "(HyperGraphRAG's template). strict: brief answer or "
                         "INSUFFICIENT EVIDENCE. Use one style for every method.")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="cap the response; reasoning makes think longer than strict")
    ap.add_argument("--top-k", type=int, default=None,
                    help="use only the first k statements (default: all retrieved)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N questions only")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--knowledge", default="none",
                    help="accepted and ignored, so one command line serves both "
                         "scripts")
    ap.add_argument("--include-participants", action="store_true",
                    help="accepted and ignored")
    args = ap.parse_args()

    if not args.out and not args.out_dir:
        ap.error("pass --out (one file) or --out-dir (several)")
    if not args.retrieval and not args.questions:
        ap.error("pass --retrieval, or --questions to run from the question list")
    if args.out and args.retrieval and len(args.retrieval) > 1:
        ap.error("--out takes one --retrieval; use --out-dir for several")

    questions = (json.loads(Path(args.questions).read_text())
                 if args.questions else None)
    client = make_client(args.base_url, args.api_key)
    print(f"model     : {args.model}")
    print(f"prompt    : {args.prompt_style}")
    print("knowledge : none (closed book)")
    print(f"retrieval : {len(args.retrieval) if args.retrieval else 0} file(s)")
    if not questions:
        print("            (no --questions: gold answers will be empty)")

    for path in (args.retrieval or [None]):
        out = (Path(args.out) if args.out
               else Path(args.out_dir) / Path(path).name)
        generate(path, out, questions, args, client)


if __name__ == "__main__":
    main()