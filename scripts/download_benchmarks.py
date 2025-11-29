#!/usr/bin/env python3
"""Download benchmark datasets via Hugging Face (Tsinghua mirror friendly)."""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from typing import Iterable, Optional

from huggingface_hub import snapshot_download
from urllib.parse import quote
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download VALM benchmark datasets.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        help="Hugging Face dataset repo id, e.g., deepseek-ai/locomo",
    )
    group.add_argument(
        "--github-repo",
        help="GitHub repo in the form owner/repo (e.g., anthropics/locomo).",
    )
    parser.add_argument(
        "--target",
        default="data/external",
        help="Local directory to place the downloaded dataset.",
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=None,
        help="Optional allow_patterns passed to snapshot_download.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="HF_ENDPOINT to use (defaults to Tsinghua mirror).",
    )
    parser.add_argument(
        "--github-branch",
        default="main",
        help="Branch/tag to download when using --github-repo.",
    )
    parser.add_argument(
        "--github-mirror",
        default="https://ghproxy.com/",
        help="Proxy prefix for GitHub downloads (e.g., https://ghproxy.com/).",
    )
    return parser.parse_args()


def ensure_endpoint(endpoint: str) -> None:
    if endpoint and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = endpoint


def download_from_hf(args: argparse.Namespace, output_dir: Path) -> None:
    ensure_endpoint(args.hf_endpoint)
    snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=args.patterns,
        resume_download=True,
    )


def download_from_github(args: argparse.Namespace, output_dir: Path) -> None:
    repo = args.github_repo
    branch = args.github_branch
    mirror = args.github_mirror.rstrip("/")
    zip_url = f"{mirror}/https://github.com/{repo}/archive/refs/heads/{branch}.zip"
    tmp_zip = output_dir / f"{repo.replace('/', '_')}_{branch}.zip"

    print(f"Downloading {repo}@{branch} via {mirror} ...")
    with requests.get(zip_url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(tmp_zip, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)

    print(f"Extracting {tmp_zip} ...")
    with zipfile.ZipFile(tmp_zip, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    tmp_zip.unlink()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.target)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        download_from_hf(args, output_dir)
    else:
        download_from_github(args, output_dir)


if __name__ == "__main__":
    main()
