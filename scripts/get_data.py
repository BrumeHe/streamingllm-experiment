"""Fetch a long text for the PPL experiment, writing experiment/data/long_text.txt.

Cascade (first success wins):
  1. LongBench narrativeqa (parquet revision, first document)
  2. Project Gutenberg plain text (Pride and Prejudice), repeated if needed
The chosen source is recorded in experiment/data/SOURCE.txt.
"""
import os
import sys

from common import ROOT

DATA_DIR = os.path.join(ROOT, "data")
OUT = os.path.join(DATA_DIR, "long_text.txt")
SRC = os.path.join(DATA_DIR, "SOURCE.txt")
MIN_CHARS = 200_000  # ~50K tokens, comfortably above the 12K we actually use


def save(text, source):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    with open(SRC, "w", encoding="utf-8") as f:
        f.write(source + "\n")
    print(f"[get_data] source: {source}")
    print(f"[get_data] saved {len(text)} chars to {OUT}")


def try_longbench():
    from datasets import load_dataset

    ds = load_dataset(
        "THUDM/LongBench", "narrativeqa", revision="refs/convert/parquet",
        split="test",
    )
    text = ds[0]["context"]
    if len(text) < MIN_CHARS:
        raise RuntimeError(f"LongBench doc too short: {len(text)} chars")
    return text, "LongBench narrativeqa doc 0"


def try_gutenberg():
    import urllib.request

    url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
    with urllib.request.urlopen(url, timeout=60) as r:
        text = r.read().decode("utf-8")
    while len(text) < MIN_CHARS:
        text = text + "\n\n" + text
    return text, "Project Gutenberg #1342 (Pride and Prejudice), concatenated"


def main():
    if os.path.exists(OUT) and os.path.getsize(OUT) > MIN_CHARS:
        print(f"[get_data] {OUT} already exists, skipping download")
        return
    for name, fn in [("LongBench", try_longbench), ("Gutenberg", try_gutenberg)]:
        try:
            text, source = fn()
            save(text, source)
            return
        except Exception as e:  # noqa: BLE001 - cascade by design
            print(f"[get_data] {name} failed: {e}", file=sys.stderr)
    raise SystemExit("[get_data] all sources failed")


if __name__ == "__main__":
    main()
