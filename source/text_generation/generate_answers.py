"""
Answer generation from retrieved knowledge.

Reads the knowledge file any retriever writes (baseline or ours) and asks the
model the question over that knowledge alone. Nothing is scored here: scoring
lives in evaluation, so a re-score never costs an API call.

Only `knowledge[].text` is passed to the model. Participants, chunk ids and
the method's `trace` are deliberately withheld, so every method supplies the
same kind of context and a difference in accuracy is a difference in what was
retrieved, not in how it was formatted. Richer context is an ablation, not the
default (--include-participants exists for exactly that).

WHICH KNOWLEDGE  (--knowledge)

A retrieval file may carry its knowledge in sections. Ours writes two,
`hyperedges` and `chunks`, where a chunk is the passage a retrieved hyperedge
was extracted from.

    hyperedges  (default)  the facts alone: the strictest setting, and the one
                           that measures the hypergraph itself.
    chunks                 the source passages alone, which is roughly what a
                           plain chunk retriever would hand the reader.
    all                    both, hyperedges first, duplicates dropped. This is
                           the fair comparison against HyperGraphRAG, which
                           supplies entities, relations AND chunks.

--top-k applies per section, so `all` does not crowd the facts out with
passages. A file whose knowledge is one flat list (every baseline) is read as a
single `hyperedges` section, so nothing about those runs changes.

Both limits take a whole number or a fraction. 3 keeps the first 3 items; 0.5
keeps the best half of whatever that question retrieved, rounded up, never
fewer than one. A fraction is the honest setting when the section length varies
from question to question, since a fixed cap is a different amount of pruning
for a question that retrieved 3 chunks than for one that retrieved 14.

--chunk-top-k caps the `chunks` section alone, overriding --top-k for it. A
chunk is a whole passage, so it costs far more input tokens than a fact, and the
chunk list our retriever writes is uncapped: a chunk enters it whenever any
retrieved hyperedge was extracted from it, ranked by that hyperedge's score and
never by its own relevance to the question. Capping the chunks is therefore the
one knob that trades generation cost against accuracy directly. The default is
no cap, so every run so far reproduces unchanged.

Answers from different --knowledge settings do not belong in one file: give
each variant its own --out. The resume check refuses to mix them silently.

PROMPT STYLES  (--prompt-style, the same for every method in a comparison)

    think   (default)  the HyperGraphRAG template: reason inside <think>...
                       </think>, then answer inside <answer>...</answer>. The
                       answer is parsed out of the tags; the whole response is
                       kept in `full_output`. There is no way to decline, so
                       the model always answers.
    strict             the previous prompt: answer briefly from the statements
                       alone, or reply INSUFFICIENT EVIDENCE.

The two are not comparable. `strict` refuses when the answer is not literally
stated, which costs questions whose evidence is present but has to be read
("teams began to move out of cities" answering "what is the term for..."), and
`think` never refuses, which wins those back and also lets the model answer
from what it already knows when the evidence is absent. Accuracy over the
questions the gold context can actually answer is the honest number under
`think`.

ANSWER LENGTH  (--answer-length, `think` only)

    long   (default)  the template exactly as HyperGraphRAG publishes it. The
                      model decides how long the <answer> span is, and in
                      practice it writes a sentence.
    short             the same template with one added line and a descriptor
                      inside the answer tag, asking for the name, date or
                      phrase alone.

F1 and EM are token overlap against a golden answer that is usually a single
entity, so they fall with every extra token the answer carries. `long` and
`short` therefore give very different F1 for the same retrieved knowledge, and
a comparison has to hold this fixed: run every method with the same setting, or
the table compares formatting rather than retrieval. The setting is written
into the output config and the resume check refuses to mix the two.

    export NEXUS_LLM_BASE_URL=https://api.openai.com/v1
    export NEXUS_LLM_API_KEY=sk-...
    export NEXUS_LLM_MODEL=gpt-4o-mini

    python -m text_generation.generate_answers \
        --retrieval ../output/Mix/retrieval/simsearch.json \
        --out ../output/Mix/answers/simsearch.json

    # several methods in one go; the output name follows each input name
    python -m text_generation.generate_answers \
        --retrieval ../output/Mix/retrieval/*.json \
        --out-dir ../output/Mix/answers --workers 4

OUTPUT CONTRACT

    {"config": {"run", "retrieval", "model", "top_k", "prompt_style", ...},
     "summary": {"answered", "insufficient", "mean_knowledge", ...},
     "answers": {"0": {"qi", "question", "gold", "answer",
                       "full_output": "<think>...</think><answer>...</answer>",
                       "knowledge": ["..."],     # exactly what the model saw
                       "n_knowledge": 5,
                       "seconds": 0.8}}}

`answer` is what evaluation scores: under `think` it is the contents of the
<answer> tag, with the raw response kept alongside it. `gold` is copied from
questions.json when given, purely so the evaluation step has everything in one
file; it is never shown to the model.

The run is resumable: the output file is re-read on start and only missing
questions are asked again. --restart ignores it. Answers written by an earlier
prompt style are kept as they are, so switching style needs --restart.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from math import ceil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_STYLE = "think"
DEFAULT_KNOWLEDGE = "hyperedges"
DEFAULT_ANSWER_LENGTH = "long"
CHUNK_SECTION = "chunks"
FLUSH_EVERY = 20
INSUFFICIENT = "INSUFFICIENT EVIDENCE"

Q_FIELDS = ("question", "query", "text", "input")
A_FIELDS = ("answer", "golden_answers", "gold_answer", "ground_truth", "output", "label")

SYSTEM_PROMPT = """\
You answer questions from supplied knowledge.

You are given a question and a list of statements retrieved from a corpus. \
Answer the question using ONLY those statements.

RULES:
- Answer directly and briefly. A name, a date, a phrase: whatever the question \
asks for, and nothing else. No preamble, no restatement of the question, no \
explanation unless the question asks for one.
- Use only the statements. Do not use anything you know from elsewhere, and do \
not infer beyond what they say.
- Several statements may be needed together. Combining two statements that \
share a subject is allowed; adding a fact neither of them states is not.
- If the statements do not contain the answer, reply exactly: INSUFFICIENT \
EVIDENCE
- Output the answer as plain text, nothing else."""

USER_PROMPT = """\
Knowledge:
{knowledge}

Question: {question}
Answer:"""

#: HyperGraphRAG's template, reproduced so the comparison is on their terms.
#: One user message, no system prompt, reasoning then answer.
THINK_PROMPT = """---Role---

You are a helpful assistant responding to questions based on given knowledge.

---Knowledge---

{knowledge}

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

#: The same template with the answer length constrained. The line in ---Goal---
#: states the rule and the descriptor inside the answer tag does the work, since
#: the model copies the shape of the example. Nothing else differs, so a
#: difference against `long` is a difference in answer format alone.
THINK_PROMPT_SHORT = """---Role---

You are a helpful assistant responding to questions based on given knowledge.

---Knowledge---

{knowledge}

---Goal---

Answer the given question.
You must first conduct reasoning inside <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.
The answer must be as short as possible: the name, date or phrase that answers the question, with no sentence and no explanation.

Output format for answer:
<think>
...
</think>
<answer>
the shortest phrase that answers the question
</answer>

---Question---

{question}
"""

THINK_PROMPTS = {"long": THINK_PROMPT, "short": THINK_PROMPT_SHORT}

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


def resolve_limit(limit, n):
    """How many items to keep from a section of `n`, given a cap or a fraction.

    None is no limit. A value of 1 or more is a count. A value below 1 is a
    fraction of that question's own section length, rounded up, and never
    rounds down to nothing: 0.5 of 3 chunks keeps 2, not 1.5 and not 0.
    """
    if not limit or n <= 0:
        return None
    if limit >= 1:
        return int(limit)
    return max(1, ceil(limit * n))


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
                    sections=DEFAULT_KNOWLEDGE, chunk_top_k=None):
    """The statements to show the model: ranked, deduplicated, text only.

    `sections` is "all" or the name of one section. --top-k applies to EACH
    section, not to the total, so asking for both does not silently crowd the
    hyperedges out of the window. Sections are concatenated in the order they
    appear in the file, hyperedges first, and a chunk whose text repeats a
    hyperedge verbatim is dropped as a duplicate.

    `chunk_top_k` replaces that limit for the `chunks` section only, since a
    passage costs many times what a fact costs. None leaves the section on
    whatever --top-k says, which is the behaviour every earlier run had.

    Either limit may be a fraction below 1, read as a share of that question's
    own section length; see resolve_limit.
    """
    available = knowledge_sections(result)
    wanted = (list(available) if sections == "all"
              else [s for s in [sections] if s in available])

    seen = OrderedDict()
    for name in wanted:
        items = available[name]
        raw = chunk_top_k if (name == CHUNK_SECTION and chunk_top_k is not None) else top_k
        limit = resolve_limit(raw, sum(1 for it in items
                                       if (it.get("text") or "").strip()))
        kept = 0
        for item in items:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            if include_participants and item.get("participants"):
                text = f"{text}  [{', '.join(map(str, item['participants']))}]"
            seen.setdefault(norm(text), text)
            kept += 1
            if limit and kept >= limit:
                break
    return list(seen.values())


def make_client(base_url=None, api_key=None):
    from openai import OpenAI
    return OpenAI(base_url=base_url or os.environ.get("NEXUS_LLM_BASE_URL"),
                  api_key=api_key or os.environ.get("NEXUS_LLM_API_KEY", "not-needed"))


def answer_one(client, model, question, knowledge, temperature=0.0,
               style=DEFAULT_STYLE, max_tokens=None,
               answer_length=DEFAULT_ANSWER_LENGTH):
    """(answer, raw response, whether the answer tag was found).

    `answer_length` picks the `think` template and is ignored by `strict`,
    whose system prompt already asks for a brief answer.
    """
    listed = "\n".join(f"- {s}" for s in knowledge) or "(none)"
    if style == "think":
        template = THINK_PROMPTS.get(answer_length, THINK_PROMPT)
        messages = [{"role": "user", "content": template.format(
            knowledge=listed, question=question)}]
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(
                        knowledge=listed, question=question)}]

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
    blob = json.loads(Path(retrieval_path).read_text())
    results = blob["results"]
    run = blob.get("config", {}).get("run", Path(retrieval_path).stem)
    print(f"\n=== {run} ===  {Path(retrieval_path).name}  "
          f"[{args.prompt_style}, knowledge={args.knowledge}, "
          f"answer={args.answer_length}"
          + (f", chunks<={args.chunk_top_k:g}" if args.chunk_top_k else "") + "]")

    # a section that exists but is empty everywhere is as useless as a missing
    # one, so the check counts items, not keys
    filled = OrderedDict()
    for r in results:
        for name, items in knowledge_sections(r).items():
            filled[name] = filled.get(name, 0) + len(items)
    have = [s for s, c in filled.items() if c]
    if args.knowledge != "all" and not filled.get(args.knowledge):
        print(f"  !! the '{args.knowledge}' section of this file is empty or absent "
              f"(it has: {', '.join(have) or 'nothing'}). Every question would be "
              f"answered with no knowledge at all, so nothing was asked.")
        return None
    if args.knowledge == "all" and len(have) > 1:
        print(f"  sections: " + ", ".join(f"{s} ({filled[s]})" for s in have))

    n = min(len(results), args.limit or len(results))
    answers = {}
    if out_path.exists() and not args.restart:
        previous = json.loads(out_path.read_text())
        answers = previous.get("answers", {})
        old_cfg = previous.get("config", {})
        old_style = old_cfg.get("prompt_style", "strict")
        old_know = old_cfg.get("knowledge", DEFAULT_KNOWLEDGE)
        old_len = old_cfg.get("answer_length", DEFAULT_ANSWER_LENGTH)
        old_ctk = old_cfg.get("chunk_top_k")
        print(f"  resuming: {len(answers)} already answered")
        if answers and (old_style != args.prompt_style
                        or old_know != args.knowledge
                        or old_len != args.answer_length
                        or old_ctk != args.chunk_top_k):
            print(f"  !! those were written with --prompt-style {old_style} "
                  f"--knowledge {old_know} --answer-length {old_len} "
                  f"--chunk-top-k {old_ctk}, not {args.prompt_style}/"
                  f"{args.knowledge}/{args.answer_length}/{args.chunk_top_k}. "
                  f"Mixed runs are not comparable: re-run with --restart, or write "
                  f"this variant to its own --out.")
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
        know = knowledge_texts(r, args.top_k, args.include_participants,
                               args.knowledge, args.chunk_top_k)
        t0 = time.time()
        raw, tagged, err = "", True, None
        try:
            ans, raw, tagged = answer_one(client, args.model, question, know,
                                          args.temperature, args.prompt_style,
                                          args.max_tokens, args.answer_length)
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
                        "chunk_top_k": args.chunk_top_k,
                        "temperature": args.temperature,
                        "include_participants": args.include_participants,
                        "prompt_style": args.prompt_style,
                        "knowledge": args.knowledge,
                        "answer_length": args.answer_length,
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
    print(f"  knowledge per question: {s['mean_knowledge']}, {s['no_knowledge']} with none")
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
    ap = argparse.ArgumentParser(description="Answer generation from retrieved knowledge")
    ap.add_argument("--retrieval", nargs="+", required=True,
                    help="one or more knowledge files from knowledge_retrieval")
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
    ap.add_argument("--answer-length", choices=["long", "short"],
                    default=DEFAULT_ANSWER_LENGTH,
                    help="think only. long (default) is HyperGraphRAG's template "
                         "unchanged. short adds one line and a descriptor inside "
                         "the answer tag, asking for the name or phrase alone, "
                         "which is what F1 and EM reward. Use one setting for "
                         "every method in a comparison.")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="cap the response; reasoning makes think longer than strict")
    ap.add_argument("--top-k", type=float, default=None, metavar="K",
                    help="use only the first K statements per section (default: all "
                         "retrieved). A whole number is a count, a value below 1 is "
                         "a fraction of what that question retrieved, rounded up "
                         "(0.5 keeps the better half).")
    ap.add_argument("--chunk-top-k", type=float, default=None, metavar="C",
                    help="keep only the first C chunks, overriding --top-k for the "
                         "chunks section alone (default: no cap, as before). Counts "
                         "and fractions as for --top-k, so 3 keeps three chunks and "
                         "0.5 keeps half of whatever that question pulled in. A "
                         "chunk costs many times what a fact costs and the chunk "
                         "list is uncapped, so this is the knob that trades "
                         "generation cost against accuracy. Only applies when "
                         "--knowledge includes chunks.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N questions only")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--knowledge", default=DEFAULT_KNOWLEDGE,
                    help="which section of the retrieval file to show the model: "
                         "hyperedges (default), chunks, all, or any section name the "
                         "file carries. A file with one flat list is read as "
                         "hyperedges, so the baselines are unaffected. --top-k applies "
                         "per section.")
    ap.add_argument("--include-participants", action="store_true",
                    help="ablation: append each statement's entities to its text")
    args = ap.parse_args()

    # a whole number stays a whole number in the config, so a run written as
    # --top-k 3 reads back as 3 and not 3.0
    for name in ("top_k", "chunk_top_k"):
        v = getattr(args, name)
        if v is None:
            continue
        if v <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive, got {v}")
        if float(v).is_integer():
            setattr(args, name, int(v))

    if not args.out and not args.out_dir:
        ap.error("pass --out (one file) or --out-dir (several)")
    if args.out and len(args.retrieval) > 1:
        ap.error("--out takes one --retrieval; use --out-dir for several")

    questions = (json.loads(Path(args.questions).read_text())
                 if args.questions else None)
    client = make_client(args.base_url, args.api_key)
    print(f"model     : {args.model}")
    print(f"prompt    : {args.prompt_style}")
    print(f"knowledge : {args.knowledge}")
    print(f"answer    : {args.answer_length}"
          + ("" if args.prompt_style == "think"
             else "  (ignored, --prompt-style strict is always brief)"))
    if args.chunk_top_k:
        shown = (f"top {args.chunk_top_k:g}" if args.chunk_top_k >= 1
                 else f"top {args.chunk_top_k:g} of each question's own list")
        print(f"chunks    : {shown}"
              + ("" if args.knowledge in ("all", CHUNK_SECTION)
                 else f"  (ignored, --knowledge {args.knowledge} shows no chunks)"))
    print(f"retrieval : {len(args.retrieval)} file(s)")
    if not questions:
        print("            (no --questions: gold answers will be empty)")

    for path in args.retrieval:
        out = (Path(args.out) if args.out
               else Path(args.out_dir) / Path(path).name)
        generate(path, out, questions, args, client)


if __name__ == "__main__":
    main()