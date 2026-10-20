"""
Retrieval scoring for Nexus.

Computes four numbers per method / dataset / question set:

  R-S   the official Retrieval Similarity of HyperGraphRAG, reproduced exactly:
        one SimCSE embedding of the whole retrieved knowledge string against one
        embedding of the newline-joined golden fragments, cosine similarity.
        Kept only for comparability with published numbers.

  K-R   knowledge recall.    mean over golden fragments g of max over retrieved
                             hyperedges h of cos(g, h)
  K-P   knowledge precision. mean over retrieved hyperedges h of max over
                             golden fragments g of cos(h, g)
  K-F   harmonic mean of K-R and K-P, per question

  R-S2  the same comparison as R-S under a second encoder, off by default
        (--rs2). See A SECOND ENCODER below.

Same encoder, same normalisation and same truncation length as the official
script, so all four numbers sit on one scale.

A SECOND ENCODER
----------------
R-S is one embedding of the whole retrieved string against one embedding of the
golden fragments, and SimCSE reads at most 128 wordpieces, so whatever a method
put first decides its score and everything past the first paragraph is unread.
R-S2 repeats the comparison with an OpenAI embedding model, text-embedding-3-
small by default, which is the model HyperGraphRAG embeds with and which reads
8191 tokens. The preprocessing, the layer and the pairing are identical, so the
only thing that differs between R-S and R-S2 is how much of the retrieval the
encoder actually sees.

Expect the two to disagree, and expect R-S2 to be kinder to a method that
returns a great deal of text, since R-S never read most of it. That is the
point of reporting both.

R-S2 calls a paid API. Credentials follow the same convention as the rest of
the pipeline, NEXUS_LLM_BASE_URL and NEXUS_LLM_API_KEY, blobs are truncated to
the model window before being sent, and every distinct string is embedded once
and cached on disk so a re-run or a crash does not pay twice.

TWO LAYERS, ON PURPOSE
----------------------
R-S reads everything a method handed its reader, because that is what the
official metric does and what makes the number comparable with published ones.
K-R, K-P and K-F read the hyperedges alone, because they ask whether the fact
layer carries the answer. Both layers must be built the same way for every
method or the metric measures granularity rather than retrieval. --rs-layer
hyperedges puts R-S on the fact layer too, for the comparison.

SUPPORTED PREDICTION SHAPES
---------------------------
  knowledge: "....."                 one context string, HyperGraphRAG style
  knowledge: {hyperedges: [], chunks: []}    Nexus
  knowledge: [ {...}, ... ]          a plain list of items
  retrieved: [ {reasoning_path, entity_descriptions, src_text_chunks}, ... ]
                                     PRoH, one entry per sub-question state

Usage
-----
  python score_knowledge.py \
      --questions datasets/Mix/questions_200.json \
      --pred      output/knowledge_200/Nexus-union/Mix/ \
      --out-dir   knowledge_scores

--pred takes either a file or the directory holding it, since methods name
their output differently (knowledge.json, test_knowledge.json, retrieved.json,
generated_answer.json). Use --file to name the basename yourself.

Method, dataset and question set are read off the prediction path and the
question count, and each can be overridden with --name, --dataset and --qset.
The result lands at <out-dir>/<method>/<dataset>/<qset>.json.

A missing questions file, a missing prediction, or a directory with nothing
scorable in it prints a [skip] line and exits 0, so a shell loop moves on.

Add --stub for a plumbing test with random embeddings and no model download.
It covers R-S2 as well, so --stub --rs2 exercises the plumbing for free.
"""

import argparse
import hashlib
import json
import os
import pickle
import re
import string
import sys
import time

import numpy as np

# ---------------------------------------------------------------- normalisation

def normalize_answer(answer: str) -> str:
    """Identical to eval.py / eval_r.py in HyperGraphRAG."""
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


# ------------------------------------------------------------ hyperedge parsing

# _ROW = re.compile(r"^\s*\d+\s*,\s*<hyperedge>(?P<rest>.*)$")
_ROW = re.compile(r'^\s*\d+\s*,\s*(?P<q>"?)\s*<hyperedge>(?P<rest>.*)$')


def _unquote_scan(rest: str):
    """rest starts with '"'. Return the quoted field, honouring '""' escapes."""
    out, i, n = [], 1, len(rest)
    while i < n:
        ch = rest[i]
        if ch == '"':
            if i + 1 < n and rest[i + 1] == '"':
                out.append('"')
                i += 2
                continue
            return "".join(out), rest[i + 1:]
        out.append(ch)
        i += 1
    return "".join(out), ""


def hyperedges_from_blob(blob: str):
    """Pull the hyperedge sentences out of a HyperGraphRAG style context string.

    The -----Relationships----- block is not valid CSV: the field begins with
    <hyperedge> before the opening quote, so csv.reader splits on commas inside
    the sentence. We scan the quoted field by hand instead.
    """
    i = blob.find("-----Relationships-----")
    if i < 0:
        i = blob.find("-----Hyperedges-----")
    if i < 0:
        return []
    j = blob.find("-----Sources-----", i)
    body = blob[i:j if j > 0 else len(blob)]
    if "```csv" in body:
        body = body.split("```csv")[-1].split("```")[0]
    texts = []
    for line in body.split("\n"):
        m = _ROW.match(line)
        if not m:
            continue
        # rest = m.group("rest").strip()
        rest = m.group("q") + m.group("rest").strip()
        if rest.startswith('"'):
            text, _ = _unquote_scan(rest)
        else:
            text = rest.split(",")[0]
        text = text.strip()
        if text:
            texts.append(text)
    return texts


# ----------------------------------------------------------------- PRoH parsing

#: PRoH keeps its knowledge as three strings on each sub-question state, not as
#: records: the hyperedges of the chosen path joined with " -> ", the entity
#: descriptions, and the source passages. A question's sub-questions overlap
#: heavily, so the same hyperedge and the same passage recur; main() dedups the
#: hyperedges on their normalised text before scoring.
_HYPEREDGE_TAG = re.compile(r"<hyperedge>")
_PASSAGE_TAG = re.compile(r"Passage\s+\d+\s*:")
_TRAILING_ARROW = re.compile(r"\s*->\s*$")


def _is_proh(kn) -> bool:
    return (isinstance(kn, list) and kn and isinstance(kn[0], dict)
            and ("reasoning_path" in kn[0] or "src_text_chunks" in kn[0]))


def _proh_units(entries):
    """(hyperedge sentences, the blob R-S reads) out of a PRoH record.

    The blob is built exactly as PRoH's eval/get_score.py builds it, per entry
    and in this order:

        knowledge += item['reasoning_path'] + "\\n"
        knowledge += item['src_text_chunks'] + "\\n"
        knowledge += item['entity_descriptions'] + "\\n"

    Order is not cosmetic. SimCSE.similarity truncates at max_length=128, so
    R-S only ever sees the first ~128 tokens of the blob and whatever is
    concatenated first decides the score. Reordering these three fields gives a
    different number for identical retrieval.
    """
    hyperedges, blob = [], []
    for entry in entries:
        for part in _HYPEREDGE_TAG.split(entry.get("reasoning_path") or ""):
            text = _TRAILING_ARROW.sub("", part.strip()).strip().strip('"').strip()
            if text:
                hyperedges.append(text)
        blob.append(entry.get("reasoning_path") or "")
        blob.append(entry.get("src_text_chunks") or "")
        blob.append(entry.get("entity_descriptions") or "")
    return hyperedges, "\n".join(blob)


# ----------------------------------------------------------------- pred loading

def _text_of(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "hyperedge", "sentence", "content", "name"):
            if isinstance(item.get(key), str):
                return item[key]
    return None


def _hyperedges_of(kn):
    """kn is whatever sits under the 'knowledge' key, or PRoH's 'retrieved'.

    Returns (hyperedge sentences, the whole retrieved string). The second is
    what R-S reads, so it has to hold every layer the method supplied, not just
    the facts: a blob of hyperedges only, compared against another method's
    blob of hyperedges AND passages, measures the granularity.
    """
    if kn is None:
        return [], ""
    if _is_proh(kn):
        return _proh_units(kn)
    if isinstance(kn, str):
        return hyperedges_from_blob(kn), kn
    if isinstance(kn, dict):
        texts = [t for t in (_text_of(x) for x in kn.get("hyperedges", [])) if t]
        rest = [t for key in kn if key != "hyperedges"
                for t in (_text_of(x) for x in (kn.get(key) or []))
                if isinstance(kn.get(key), list) and t]
        return texts, "\n".join(texts + rest)
    if isinstance(kn, list):
        texts = [t for t in (_text_of(x) for x in kn) if t]
        return texts, "\n".join(texts)
    return [], ""


_PREFERRED = ("knowledge.json", "test_knowledge.json", "retrieved.json",
              "generated_answer.json")
_NOT_KNOWLEDGE = {"config.json", "summary.json", "chunks.json", "scores.json",
                  "query_entities.json", "questions.json", "test_score.json",
                  "test_result.json", "test_generation.json"}


def skip(msg):
    """Report and leave quietly, so a shell loop moves to the next iteration."""
    print(f"[skip] {msg}", file=sys.stderr)
    sys.exit(0)


def resolve_pred(path, basename=""):
    """Accept a file or the directory holding it, methods name theirs differently."""
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
    rest = [f for f in names if f not in _NOT_KNOWLEDGE]
    if len(rest) == 1:
        print(f"[pred]  using {rest[0]}", file=sys.stderr)
        return os.path.join(path, rest[0])
    if not rest:
        skip(f"nothing scorable in {path}")
    skip(f"{path} holds several candidates {rest}, pass --file to pick one")


def _records(path):
    data = json.load(open(path))
    recs = data["results"] if isinstance(data, dict) and "results" in data else data
    if not isinstance(recs, list):
        sys.exit(f"{path}: expected a list of records or a dict with 'results'")
    return recs


def load_pred(path):
    out, shapes = [], set()
    for i, rec in enumerate(_records(path)):
        if not isinstance(rec, dict):
            sys.exit(f"{path}: record {i} is {type(rec).__name__}, expected dict")
        kn = rec.get("knowledge", rec.get("retrieved"))
        shapes.add("proh" if _is_proh(kn) else type(kn).__name__)
        hyps, blob = _hyperedges_of(kn)
        out.append({
            "qid": rec.get("qi", rec.get("id", i)),
            "question": rec.get("question", ""),
            "hyperedges": hyps,
            "blob": blob,
        })
    print(f"[shape] knowledge is {'/'.join(sorted(shapes))}", file=sys.stderr)
    if out and not any(r["hyperedges"] or r["blob"] for r in out):
        # Silence here used to mean every metric was 0 with no explanation.
        skip(f"{path}: parsed {len(out)} records and found no knowledge in any of "
             f"them. The shape is not one this script knows.")
    return out


def load_questions(path):
    out = []
    for i, rec in enumerate(_records(path)):
        ctx, seen = [], set()
        for c in rec.get("context", []):
            if c not in seen:
                seen.add(c)
                ctx.append(c)
        out.append({
            "qid": rec.get("qi", rec.get("id", i)),
            "question": rec.get("question", ""),
            "context": ctx,
        })
    if not any(r["context"] for r in out):
        sys.exit(f"{path}: no 'context' field found, this is not a questions file")
    return out


def align(gold, pred):
    """Match predictions to questions by text, fall back to position."""
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
    "hypergraphrag": "HyperGraphRAG",
    "hyperrag": "HyperRAG",
    "proh": "PRoH",
    "proh-l": "PRoH-L",
    "nexus": "Nexus",
    "golden": "Golden",
    "simsearch": "SimSearch",
    "plainllm": "PlainLLM",
}
_DATASETS = {
    "agriculture": "Agriculture",
    "cs": "CS",
    "hypertension": "Hypertension",
    "medicine": "Medicine",
    "legal": "Legal",
    "mix": "Mix",
}
_SETSIZE = re.compile(r"(?:^|[^0-9])(200|512)(?:[^0-9]|$)")


def infer_tags(pred_path, n_questions):
    """method, dataset and question set, from the path and the question count."""
    parts = [p for p in os.path.normpath(pred_path).split(os.sep) if p not in ("", ".", "..")]
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


# ---------------------------------------------------------------------- encoder

class StubEncoder:
    """Deterministic random unit vectors, for plumbing tests only."""

    def encode_many(self, texts, max_length=128):
        out = np.empty((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=4).digest(), "big")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(64).astype(np.float32)
            out[i] = v / (np.linalg.norm(v) + 1e-12)
        return out


class SimCSEEncoder:
    def __init__(self, model_name, batch_size=64, device=None):
        from simcse import SimCSE
        self.model = SimCSE(model_name, device=device)
        self.batch_size = batch_size

    def encode_many(self, texts, max_length=128):
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=max_length,     # the package default, kept explicit
            normalize_to_unit=True,    # so cosine is a dot product
            return_numpy=True,
            keepdim=True,
        )
        return np.asarray(emb, dtype=np.float32)


#: The model HyperGraphRAG embeds with. 8191 tokens of window against SimCSE's
#: 128 wordpieces, which is the whole reason for a second number.
DEFAULT_RS2_MODEL = "text-embedding-3-small"
RS2_WINDOW = 8191


def _truncate(text, max_tokens, model):
    """Cut to the model window. Falls back to characters without tiktoken."""
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(text, disallowed_special=())
        return enc.decode(ids[:max_tokens]) if len(ids) > max_tokens else text
    except ImportError:
        limit = max_tokens * 4
        return text[:limit] if len(text) > limit else text


class OpenAIEncoder:
    """Embeddings over an OpenAI compatible endpoint, with a disk cache.

    Batches are bounded by both count and tokens, since the endpoint limits the
    tokens of one request and a blob here can be the whole window. Every
    distinct string is embedded once and written to the cache, so a re-run or a
    crash part way through does not pay for the same text twice.
    """

    def __init__(self, model=DEFAULT_RS2_MODEL, base_url=None, api_key=None,
                 cache_path=None, batch_tokens=100_000, batch_items=64,
                 retries=4):
        from openai import OpenAI
        self.client = OpenAI(
            base_url=base_url or os.environ.get("NEXUS_LLM_BASE_URL"),
            api_key=api_key or os.environ.get("NEXUS_LLM_API_KEY", "not-needed"))
        self.model = model
        self.batch_tokens = batch_tokens
        self.batch_items = batch_items
        self.retries = retries
        self.cache_path = cache_path
        self.cache = {}
        self.calls = self.hits = self.sent = 0
        if cache_path and os.path.isfile(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    blob = pickle.load(f)
                if blob.get("model") == model:
                    self.cache = blob.get("vec", {})
            except Exception:
                self.cache = {}            # unreadable cache: rebuild, do not die
            print(f"[rs2]   cache {len(self.cache)} vectors from {cache_path}",
                  file=sys.stderr)

    @staticmethod
    def _key(text):
        return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()

    def _save(self):
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump({"model": self.model, "vec": self.cache}, f)

    def _embed(self, batch):
        for attempt in range(self.retries):
            try:
                resp = self.client.embeddings.create(model=self.model, input=batch)
                self.calls += 1
                self.sent += len(batch)
                return [d.embedding for d in resp.data]
            except Exception as exc:
                if attempt == self.retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"[rs2]   {type(exc).__name__}, retrying in {wait}s",
                      file=sys.stderr)
                time.sleep(wait)

    def encode_many(self, texts, max_length=RS2_WINDOW):
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        cut = [_truncate(t, max_length, self.model) for t in texts]
        todo, batch, used = [], [], 0
        for text in cut:
            key = self._key(text)
            if key in self.cache:
                self.hits += 1
                continue
            if any(key == self._key(t) for t in batch):
                continue
            size = min(len(text) // 4 + 1, max_length)
            if batch and (used + size > self.batch_tokens
                          or len(batch) >= self.batch_items):
                todo.append(batch)
                batch, used = [], 0
            batch.append(text)
            used += size
        if batch:
            todo.append(batch)

        done = 0
        for group in todo:
            for text, vec in zip(group, self._embed(group)):
                v = np.asarray(vec, dtype=np.float32)
                self.cache[self._key(text)] = v / (np.linalg.norm(v) + 1e-12)
            done += len(group)
            print(f"[rs2]   {done}/{sum(len(g) for g in todo)} embedded",
                  end="\r", file=sys.stderr, flush=True)
        if todo:
            print(file=sys.stderr)
            self._save()
        return np.stack([self.cache[self._key(t)] for t in cut])


class Cache:
    """Encode every distinct string once."""

    def __init__(self, encoder, max_length=128):
        self.encoder = encoder
        self.max_length = max_length
        self.vec = {}

    def add(self, texts):
        for t in texts:
            self.vec.setdefault(t, None)

    def run(self):
        todo = [t for t, v in self.vec.items() if v is None]
        if not todo:
            return
        for t, v in zip(todo, self.encoder.encode_many(todo, max_length=self.max_length)):
            self.vec[t] = v

    def matrix(self, texts):
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack([self.vec[t] for t in texts])


# ----------------------------------------------------------------------- scoring

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True,
                    help="questions.json of the dataset, holds the golden context")
    ap.add_argument("--pred", required=True,
                    help="retrieved knowledge of one method, a file or its directory")
    ap.add_argument("--file", default="",
                    help="basename inside --pred, when the directory is ambiguous")
    ap.add_argument("--out-dir", default="scores",
                    help="result goes to <out-dir>/<method>/<dataset>/<qset>.json")
    ap.add_argument("--name", default="", help="override the inferred method name")
    ap.add_argument("--dataset", default="", help="override the inferred dataset")
    ap.add_argument("--qset", default="", help="override the inferred question set")
    ap.add_argument("--model", default="princeton-nlp/sup-simcse-roberta-large")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-length", type=int, default=128,
                    help="SimCSE truncation length, 128 is the package default "
                         "and what the official R-S uses")
    ap.add_argument("--top-k", type=int, default=0,
                    help="keep only the first K retrieved hyperedges, 0 keeps all")
    ap.add_argument("--rs-layer", default="all", choices=["all", "hyperedges"],
                    help="what R-S reads. 'all' is every layer the method supplied, "
                         "which is the official metric. 'hyperedges' puts R-S on the "
                         "same fact layer as K-R, K-P and K-F")
    ap.add_argument("--no-rs", action="store_true",
                    help="skip the official R-S, which needs the long blobs")
    ap.add_argument("--rs2", action="store_true",
                    help="also report R-S2, the same comparison under an OpenAI "
                         "embedding model that reads the whole blob rather than "
                         "the first 128 wordpieces. Costs API tokens.")
    ap.add_argument("--rs2-model", default=DEFAULT_RS2_MODEL)
    ap.add_argument("--rs2-max-tokens", type=int, default=RS2_WINDOW,
                    help=f"model window, blobs are cut to it (default {RS2_WINDOW})")
    ap.add_argument("--rs2-cache", default="",
                    help="embedding cache (default: <out-dir>/.rs2_<model>.pkl)")
    ap.add_argument("--base-url", default=None, help="defaults to NEXUS_LLM_BASE_URL")
    ap.add_argument("--api-key", default=None, help="defaults to NEXUS_LLM_API_KEY")
    ap.add_argument("--stub", action="store_true",
                    help="random embeddings, no model, for plumbing tests")
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

    out_path = os.path.join(args.out_dir, method, dataset, f"{qset}.json")
    if os.path.isfile(out_path):
        print(f'Skipping existing {out_path}')
        exit(0)

    pairs, match = align(gold, pred)
    print(f"[tags]  method={method} dataset={dataset} qset={qset}", file=sys.stderr)
    print(f"[align] {match}", file=sys.stderr)

    encoder = StubEncoder() if args.stub else SimCSEEncoder(
        args.model, batch_size=args.batch_size, device=args.device)

    units = Cache(encoder, max_length=args.max_length)
    blobs = Cache(encoder, max_length=args.max_length)

    blobs2 = None
    if args.rs2:
        cache_path = args.rs2_cache or os.path.join(
            args.out_dir, f".rs2_{args.rs2_model.replace('/', '_')}.pkl")
        encoder2 = StubEncoder() if args.stub else OpenAIEncoder(
            model=args.rs2_model, base_url=args.base_url, api_key=args.api_key,
            cache_path=cache_path)
        blobs2 = Cache(encoder2, max_length=args.rs2_max_tokens)
        print(f"[rs2]   {args.rs2_model}, window {args.rs2_max_tokens}",
              file=sys.stderr)

    prepared = []
    n_raw_hyps = 0
    for g, p in pairs:
        gold_units = [u for u in (normalize_answer(c) for c in g["context"]) if u]
        hyps = p["hyperedges"][:args.top_k] if args.top_k else p["hyperedges"]
        n_raw_hyps += len(hyps)
        # Sub-questions overlap, so the same hyperedge arrives several times.
        # Dedup before scoring or K-P is an average over repeats.
        pred_units, seen = [], set()
        for h in hyps:
            u = normalize_answer(h)
            if u and u not in seen:
                seen.add(u)
                pred_units.append(u)
        gold_blob = normalize_answer("\n".join(g["context"]))
        pred_source = p["blob"] if args.rs_layer == "all" else "\n".join(hyps)
        pred_blob = normalize_answer(pred_source) if pred_source else ""
        units.add(gold_units)
        units.add(pred_units)
        if not args.no_rs:
            blobs.add([gold_blob])
            if pred_blob:
                blobs.add([pred_blob])
        if blobs2 is not None:
            blobs2.add([gold_blob])
            if pred_blob:
                blobs2.add([pred_blob])
        prepared.append({
            "qid": p["qid"],
            "gold_units": gold_units,
            "pred_units": pred_units,
            "gold_blob": gold_blob,
            "pred_blob": pred_blob,
        })

    print(f"[encode] {len(units.vec)} distinct units, {len(blobs.vec)} blobs",
          file=sys.stderr)
    units.run()
    if not args.no_rs:
        blobs.run()
    if blobs2 is not None:
        print(f"[rs2]   {len(blobs2.vec)} blobs to embed", file=sys.stderr)
        blobs2.run()

    per_q, empty, n_pred_tot, n_gold_tot = [], 0, 0, 0
    for rec in prepared:
        G = units.matrix(rec["gold_units"])
        H = units.matrix(rec["pred_units"])
        n_gold_tot += len(rec["gold_units"])
        n_pred_tot += len(rec["pred_units"])
        if len(rec["pred_units"]) == 0 or len(rec["gold_units"]) == 0:
            empty += 1
            kr = kp = kf = 0.0
        else:
            S = G @ H.T                       # |G| x |H| cosine matrix
            kr = float(S.max(axis=1).mean())  # each gold takes its best hyperedge
            kp = float(S.max(axis=0).mean())  # each hyperedge takes its best gold
            kf = 0.0 if kr + kp <= 0 else 2 * kr * kp / (kr + kp)
        if args.no_rs or not rec["pred_blob"]:
            rs = 0.0
        else:
            rs = float(blobs.vec[rec["gold_blob"]] @ blobs.vec[rec["pred_blob"]])
        entry = {
            "qid": rec["qid"],
            "rs": round(rs, 6),
            "kr": round(kr, 6),
            "kp": round(kp, 6),
            "kf": round(kf, 6),
        }
        if blobs2 is not None:
            rs2 = (0.0 if not rec["pred_blob"] else
                   float(blobs2.vec[rec["gold_blob"]] @ blobs2.vec[rec["pred_blob"]]))
            entry["rs2"] = round(rs2, 6)
        per_q.append(entry)

    n = len(per_q) or 1
    overall = {
        "method": method,
        "dataset": dataset,
        "qset": qset,
        "questions": len(per_q),
        "empty_or_no_gold": empty,
        "rs_layer": args.rs_layer,
        "mean_retrieved": round(n_pred_tot / n, 3),
        "mean_retrieved_raw": round(n_raw_hyps / n, 3),
        "mean_golden": round(n_gold_tot / n, 3),
        "R-S": round(sum(d["rs"] for d in per_q) / n, 6),
        "K-R": round(sum(d["kr"] for d in per_q) / n, 6),
        "K-P": round(sum(d["kp"] for d in per_q) / n, 6),
        "K-F": round(sum(d["kf"] for d in per_q) / n, 6),
    }
    if blobs2 is not None:
        overall["R-S2"] = round(sum(d["rs2"] for d in per_q) / n, 6)
        overall["rs2_model"] = args.rs2_model
    print(json.dumps(overall, indent=2))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump({"overall": overall, "match": match, "per_question": per_q},
              open(out_path, "w"), indent=2)
    print(f"[saved] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()