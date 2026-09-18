"""
src/llm/checkpoint.py
---------------------
Resume-safe JSONL checkpoint I/O for Ollama batch inference.

Each line in a checkpoint file is a JSON object:
    {"idx": <int>, "hate_speech": <0|1>, "toxic": <0|1>,
     "threat": <0|1>, "insult": <0|1>, "success": <bool>}

Design:
- Appends one line per completed row immediately after inference.
- On resume, load_checkpoint() returns a dict of already-done rows.
- Lines that fail to parse are silently skipped (robust to interrupted writes).
- Thread-safe for single-writer use (each process owns its own file).
"""

import json
from pathlib import Path
from typing import Dict

LABELS = ["hate_speech", "toxic", "threat", "insult"]


def load_checkpoint(path: Path) -> Dict[int, Dict[str, int]]:
    """
    Load an existing checkpoint file.

    Returns:
        {row_idx: {label: 0|1, …}} for every successfully parsed line.
        Empty dict if the file does not exist.
    """
    done: Dict[int, Dict[str, int]] = {}
    if not path.exists():
        return done

    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                idx = int(entry["idx"])
                done[idx] = {label: int(entry[label]) for label in LABELS}
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue  # skip corrupt lines

    return done


def append_checkpoint(
    path: Path,
    idx: int,
    pred: Dict[str, int],
    success: bool,
) -> None:
    """Append one prediction to the checkpoint file (atomic line append)."""
    entry = {"idx": idx, **pred, "success": success}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
