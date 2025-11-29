#!/usr/bin/env python3
"""
Batch runner for VALM benchmarks.

Usage:
  python scripts/run_benchmark_suite.py --config configs/benchmarks.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from valm.bench.runner import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple benchmarks sequentially.")
    parser.add_argument("--config", required=True, help="YAML file describing datasets.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/benchmarks",
        help="Directory to save individual benchmark summaries.",
    )
    return parser.parse_args()


def load_config(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if isinstance(data, dict) and "datasets" in data:
        return data["datasets"]
    if isinstance(data, list):
        return data
    raise ValueError("Invalid config format. Expect list or {datasets: [...]}.")


def main() -> None:
    args = parse_args()
    datasets = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    summary: List[Dict[str, Any]] = []

    for entry in datasets:
        name = entry.get("name", Path(entry["data"]).stem)
        print(f"\n=== Running benchmark: {name} ===")
        results = run_benchmark(
            dataset_path=entry["data"],
            backend_mode=entry.get("mode", "synthetic"),
            max_samples=entry.get("max_samples"),
            benchmark_name=entry.get("benchmark", "locomo"),
        )
        summary.append(
            {
                "name": name,
                "path": entry["data"],
                "benchmark": entry.get("benchmark", "locomo"),
                "mode": entry.get("mode", "synthetic"),
                "max_samples": entry.get("max_samples"),
                "results": results,
            }
        )

    summary_path = out_dir / f"{timestamp}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\nBenchmark suite finished. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
