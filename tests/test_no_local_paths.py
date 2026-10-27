"""The released code must not contain machine-specific paths or secrets.

What it does
------------
Scans every .py / .sh / .md / .json / .txt file of the codebase for absolute
home or cluster paths, personal environment names and access tokens.

Paper
-----
not described in the paper (release hygiene)

How to run
----------
  python -m pytest tests/test_no_local_paths.py
"""
import re
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    r"(^|[\s\"'`(=])/(home|Users|scratch|nobackup|data|mnt|gpfs|tmp)/",   # absolute paths
    r"\.conda/envs", r"\bstructtok\b", r"\bswminer\b",
    r"hf_[A-Za-z0-9]{20,}",                       # Hugging Face access token
]
SUFFIXES = {".py", ".sh", ".md", ".json", ".txt", ".yaml", ".yml", ".toml"}


def test_no_local_paths():
    rx = re.compile("|".join(PATTERNS))
    hits = []
    for p in CODE_ROOT.rglob("*"):
        if p.suffix not in SUFFIXES or "__pycache__" in p.parts or p.name == Path(__file__).name:
            continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{p.relative_to(CODE_ROOT)}:{i}: {line.strip()[:100]}")
    assert not hits, "machine-specific references:\n" + "\n".join(hits)
