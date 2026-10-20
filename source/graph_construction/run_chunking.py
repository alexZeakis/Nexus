"""
Step 1 of 7 - Chunking.

    <data_dir>/texts.json    input: a JSON array of strings
    <out_dir>/chunks.json    output: the chunks

Usage:
    python run_chunking.py --data-dir ./data/mydataset --out-dir ./output/mydataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import chunk_corpus, load_texts, write_json


def main() -> None:
    p = argparse.ArgumentParser(description="Nexus Chunking")
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--overlap-tokens", type=int, default=128)
    p.add_argument("--tiktoken-model", default="gpt-4o")
    args = p.parse_args()

    texts = load_texts(args.data_dir)
    chunks = chunk_corpus(
        texts,
        overlap_token_size=args.overlap_tokens,
        max_token_size=args.max_tokens,
        tiktoken_model=args.tiktoken_model,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "chunks.json", chunks)
    print(f"wrote {len(chunks)} chunks from {len(texts)} documents "
          f"to {args.out_dir / 'chunks.json'}")


if __name__ == "__main__":
    main()
