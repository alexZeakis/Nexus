"""
Step 2 of 7 - Coreference and Fact Extraction.

    <out_dir>/chunks.json    input: chunks (run_chunking.py)
    <out_dir>/facts.json     output: [{"chunk_id": ..., "facts": [...]}]

Before the LLM call the chunk text is repaired (a missing space after a
sentence end glues two sentences into one token and costs both of them), then
third-person pronouns are resolved with LingMess through fastcoref. The prompt
itself lives in llm.py with the other prompts.

Resumable: chunks already in facts.json are skipped on re-run.

Usage:
    export NEXUS_LLM_BASE_URL=https://api.openai.com/v1
    export NEXUS_LLM_API_KEY=sk-...
    python run_fact_extraction.py --out-dir ./output/mydataset --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from llm import FACT_EXTRACTION_PROMPT, LLMClient
from utils import read_json, write_json

# Sentence boundary: . ! ? followed by space and an uppercase letter, digit or quote.
_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'“A-Z0-9])")
# Do not split after initials or common abbreviations ("J. Smith", "Mr. Sloane").
_ABBR = re.compile(r"\b(?:[A-Z]|Mr|Mrs|Ms|Dr|St|Jr|Sr|Prof|Gen|Col|Lt|Capt|Rev|No|vs|etc)\.$")


def split_facts(lines: list[str]) -> list[str]:
    """Split each line into sentences, keeping the final punctuation."""
    facts = []
    for line in lines:
        parts = []
        for part in _SENT.split(line.strip()):
            if parts and _ABBR.search(parts[-1]):
                parts[-1] += " " + part
            else:
                parts.append(part)
        facts.extend(f for f in parts if f)
    return facts


# Corpus text drops the space after a sentence end ("Thunder of Giants.Instead
# he did..."). spaCy then tokenizes "Giants.Instead" as one token, so coreference
# sees a malformed sentence and the extractor tends to drop both sides of it.
# Fires only where a sentence end is directly followed by a capital with no space.
# The character before the period must be lowercase, a digit, closing punctuation,
# or a capital that follows another letter ("VIII."), so acronyms and initials
# ("U.S.Army", "J.Smith") are left alone and anything already spaced never matches.
_GLUED = re.compile(
    r'((?:[a-z0-9)\]’”"\']|[A-Za-z][A-Z])[.!?]["”\')\]]?)([A-Z])'
)


def repair_spacing(text: str) -> tuple[str, int]:
    """Insert the missing space after a sentence end. Returns (text, n_repairs)."""
    return _GLUED.subn(r"\1 \2", text)


PRONOUNS = {"he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"}
POSSESSIVE = {"his", "hers", "its", "their", "theirs"}


def load_coref(device: str):
    import spacy
    from fastcoref import spacy_component  # noqa: F401  (registers the pipe)

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("fastcoref", config={
        "model_architecture": "LingMessCoref",
        "model_path": "biu-nlp/lingmess-coref",
        "device": device,
    })
    return nlp


def resolve_pronouns(nlp, text: str) -> str:
    """Replace third-person pronouns with the shortest named mention of their cluster."""
    doc = nlp(text)
    edits = {}
    for cluster in doc._.coref_clusters:
        spans = [doc.char_span(s, e, alignment_mode="expand") for s, e in cluster]
        spans = [sp for sp in spans if sp is not None]
        named = [sp for sp in spans if any(t.pos_ == "PROPN" for t in sp)]
        if not named:
            continue                              # e.g. "their baby": leave to the LLM
        rep = re.sub(r"(?:'s|’s|'|’)$", "", min(named, key=len).text).strip()
        for sp in spans:
            tok = sp[0]
            if len(sp) != 1 or tok.lower_ not in PRONOUNS:
                continue
            possessive = tok.lower_ in POSSESSIVE or (tok.lower_ == "her" and tok.tag_ == "PRP$")
            sub = rep + ("'" if rep.endswith("s") else "'s") if possessive else rep
            edits[sp.start_char] = (sp.end_char, sub)
    for start in sorted(edits, reverse=True):
        end, sub = edits[start]
        text = text[:start] + sub + text[end:]
    return text


def extract_facts(client: LLMClient, text: str) -> list[str]:
    resp = client.client.chat.completions.create(
        model=client.model,
        temperature=0,
        messages=[{"role": "user", "content": FACT_EXTRACTION_PROMPT.replace("{text}", text)}],
    )
    out = resp.choices[0].message.content or ""
    return split_facts(out.split("\n"))


def main() -> None:
    p = argparse.ArgumentParser(description="Nexus Fact Extraction")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--model", default=None, help="overrides NEXUS_LLM_MODEL")
    p.add_argument("--coref-device", default="cpu", help="e.g. cuda:0")
    p.add_argument("--no-coref", action="store_true", help="skip the coreference step")
    p.add_argument("--no-repair", action="store_true",
                   help="do not insert missing spaces after sentence ends")
    p.add_argument("--skip", type=int, default=0, help="skip the first N chunks")
    p.add_argument("--limit", type=int, default=None, help="process at most N chunks after --skip")
    args = p.parse_args()

    chunks = read_json(args.out_dir / "chunks.json")
    end = args.skip + args.limit if args.limit else None
    chunks = chunks[args.skip:end]
    out_path = args.out_dir / "facts.json"
    results = read_json(out_path) if out_path.exists() else []
    for r in results:                             # re-split facts saved before the fix
        r["facts"] = split_facts(r["facts"])
    done = {r["chunk_id"] for r in results}

    print(f"chunks      : {len(chunks)} ({len(done)} already done)", flush=True)

    if args.no_coref:
        nlp = None
    else:
        # This is the long silent step. The first run downloads LingMess from
        # HuggingFace, about 1.5GB, with no progress output of its own, then
        # loads it onto the GPU. Say so rather than look hung.
        print(f"coref       : loading LingMess on {args.coref_device}, "
              f"downloading it first if it is not cached", flush=True)
        started = time.time()
        nlp = load_coref(args.coref_device)
        print(f"coref       : ready in {time.time() - started:.0f}s", flush=True)

    client = LLMClient(**({"model": args.model} if args.model else {}))
    print(f"llm         : {client.model} at {client.client.base_url}", flush=True)

    n_repaired = 0
    for i, chunk in enumerate(chunks, args.skip + 1):
        if chunk["chunk_id"] in done:
            continue
        text = chunk["content"]
        repaired = 0
        if not args.no_repair:
            text, repaired = repair_spacing(text)
            n_repaired += repaired
        if nlp is not None:
            text = resolve_pronouns(nlp, text)
        facts = extract_facts(client, text)
        results.append({"chunk_id": chunk["chunk_id"], "facts": facts})
        write_json(out_path, results)
        print(f"  {i:>4}  {len(facts)} facts"
              f"{f'  (repaired {repaired} sentence breaks)' if repaired else ''}", flush=True)

    print(f"\nwrote {sum(len(r['facts']) for r in results)} facts to {out_path}")
    if n_repaired:
        print(f"repaired {n_repaired} missing spaces after a sentence end")


if __name__ == "__main__":
    main()