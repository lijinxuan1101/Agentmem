#!/usr/bin/env python3
"""
Normalize the official LoCoMo release into VALM-friendly JSONL rows.

Each JSONL row corresponds to a (conversation, QA) pair and keeps the
full dialog transcript plus the annotated metadata used by VALM judges.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

SESSION_PATTERN = re.compile(r"session_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LoCoMo raw JSON into JSONL format.")
    parser.add_argument(
        "--input",
        default="data/external/locomo/raw/locomo10.json",
        help="Path to the official LoCoMo JSON bundle.",
    )
    parser.add_argument(
        "--output",
        default="data/external/locomo/processed/locomo10.jsonl",
        help="Destination JSONL file.",
    )
    return parser.parse_args()


def flatten_conversation(sample: Dict[str, Any]) -> str:
    conv = sample["conversation"]
    session_indices = sorted(
        int(match.group(1))
        for key in conv.keys()
        if (match := SESSION_PATTERN.match(key))
    )
    blocks: List[str] = []
    for idx in session_indices:
        turns: Iterable[Dict[str, Any]] = conv.get(f"session_{idx}", [])
        timestamp = conv.get(f"session_{idx}_date_time", "")
        blocks.append(f"SESSION_{idx} ({timestamp})")
        for turn in turns:
            dia_id = turn.get("dia_id", f"D{idx}:?")
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            blocks.append(f"{dia_id} {speaker}: {text}")
        blocks.append("")  # blank line between sessions
    return "\n".join(blocks).strip()


def convert(input_path: Path, output_path: Path) -> None:
    data = json.loads(input_path.read_text())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as sink:
        for sample in data:
            context = flatten_conversation(sample)
            sample_id = sample["sample_id"]
            for qa_idx, qa in enumerate(sample.get("qa", [])):
                record = {
                    "id": f"{sample_id}_{qa_idx}",
                    "topic": sample_id,
                    "context": context,
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "type": qa.get("category"),
                    "metadata": {
                        "evidence": qa.get("evidence", []),
                        "event_summary": sample.get("event_summary", {}),
                        "session_summary": sample.get("session_summary", {}),
                        "observation": sample.get("observation", {}),
                    },
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    convert(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
