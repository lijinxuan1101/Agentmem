import argparse
import difflib
import json
import time
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from valm.bench.judges import get_judge
from valm.config import OCRConfig, VisualizerConfig
from valm.pipeline import VALMPipeline
from valm.types import QueryContext
from valm.bench.judges import LoCoMoJudge


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def calculate_match_score(prediction: str, ground_truth: str) -> float:
    """Calculates a fuzzy match score between 0.0 and 1.0."""
    if not prediction or not ground_truth:
        return 0.0
    pred = str(prediction).lower().strip()
    gt = str(ground_truth).lower().strip()
    if gt in pred:
        return 1.0
    matcher = difflib.SequenceMatcher(None, pred, gt)
    ratio = matcher.ratio()
    pred_tokens = set(pred.split())
    gt_tokens = set(gt.split())
    if not gt_tokens:
        return ratio
    overlap = len(pred_tokens & gt_tokens) / len(gt_tokens)
    return max(ratio, overlap)


def _log_ingest_stats(topic: str, raw_tokens: int, hyper_tokens: List[Any]) -> None:
    if not hyper_tokens:
        return
    importance_values = [ht.metadata.get("importance", 0.0) for ht in hyper_tokens if ht.metadata]
    if not importance_values:
        return
    avg_importance = sum(importance_values) / len(importance_values)
    min_importance = min(importance_values)
    max_importance = max(importance_values)
    approx_drop = 1.0 - (len(hyper_tokens) / max(1, raw_tokens))
    print(
        f"\n[Topic {topic}] raw≈{raw_tokens}, hyper={len(hyper_tokens)}, "
        f"importance(avg/min/max)={avg_importance:.3f}/{min_importance:.3f}/{max_importance:.3f}, "
        f"approx_drop={approx_drop:.3f}"
    )


def _derive_time_hint(pipeline: VALMPipeline, topic: str, question: str) -> Tuple[float | None, float | None]:
    if not LoCoMoJudge._is_temporal_question(question):
        return None, None
    event, score = pipeline.store.match_event(topic, question)
    if event and score > 0.15 and event.get("timestamp"):
        return event["timestamp"], 30 * 24 * 3600.0
    return None, None


def run_benchmark(
    dataset_path: str,
    backend_mode: str = "synthetic",
    max_samples: int | None = 10,
    benchmark_name: str = "locomo",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if backend_mode == "real":
        viz_cfg = VisualizerConfig(
            render_mode="vist",
            render_backend="hf_vit",
            vist_model_name="models/vit-base",
            vist_device=device,
        )
        ocr_cfg = OCRConfig(backend="deepseek", model_name="models/deepseek-ocr", device=device)
    else:
        viz_cfg = VisualizerConfig(render_mode="vist", render_backend="synthetic")
        ocr_cfg = OCRConfig(backend="regex")

    print(f"Initializing VALM Pipeline (Backend: {backend_mode})...")
    pipeline = VALMPipeline.build(visualizer_config=viz_cfg, ocr_config=ocr_cfg)
    judge = get_judge(benchmark_name)

    data = load_jsonl(dataset_path)
    if max_samples:
        data = data[:max_samples]

    results = []
    start_time = time.time()

    print(f"Starting benchmark on {len(data)} samples...")
    topic_cache: Dict[str, Dict[str, Any]] = {}

    for sample_idx, sample in enumerate(tqdm(data, desc="Processing")):
        topic = sample.get("topic") or f"bench_{sample_idx}"
        try:
            cache_entry = topic_cache.get(topic)
            if cache_entry is None:
                print(f"\n[Sample {sample_idx+1}/{len(data)}] Ingesting...", end="", flush=True)
                hyper_tokens = pipeline.ingest(sample["context"], topic=topic)
                compressed_len = len(hyper_tokens)
                raw_tokens_est = len(sample["context"].split())
                topic_cache[topic] = {
                    "raw_tokens": raw_tokens_est,
                    "hyper_tokens": compressed_len,
                }
                event_summary = (sample.get("metadata") or {}).get("event_summary")
                pipeline.store.add_event_summary(topic, event_summary)
                _log_ingest_stats(topic, raw_tokens_est, hyper_tokens)
                print(f" Done ({compressed_len} hyper_tokens). Retrieving...", end="", flush=True)
            else:
                compressed_len = cache_entry["hyper_tokens"]
                raw_tokens_est = cache_entry["raw_tokens"]
                print(
                    f"\n[Sample {sample_idx+1}/{len(data)}] Reusing cached ingest for topic '{topic}'. Retrieving...",
                    end="",
                    flush=True,
                )

            time_hint, time_window = _derive_time_hint(pipeline, topic, sample.get("question", ""))
            query = QueryContext(text=sample["question"], topic=topic, time_hint=time_hint, time_window=time_window)
            retrieved = pipeline.retrieve(query)
            print(" Done.", flush=True)

            hit = False
            score = 0.0
            top_text = ""

            structured_pred = judge.structured_prediction(pipeline, sample, topic)
            if structured_pred is not None:
                top_text = structured_pred
                hit = structured_pred == sample["answer"]
                score = 1.0 if hit else calculate_match_score(structured_pred, sample["answer"])
            elif retrieved:
                anchors = pipeline.store.anchors_for_token(retrieved[0].token_id)
                if anchors:
                    top_text = anchors[0].text
                    score = calculate_match_score(top_text, sample["answer"])
                    hit = score > 0.4

            results.append(
                {
                    "id": sample.get("id", sample_idx),
                    "question": sample.get("question", ""),
                    "raw_tokens": raw_tokens_est,
                    "hyper_tokens": compressed_len,
                    "compression_ratio": raw_tokens_est / max(1, compressed_len),
                    "hit": hit,
                    "score": score,
                    "retrieved_snippet": top_text[:200],
                    "ground_truth": sample.get("answer", ""),
                }
            )
        except Exception as exc:
            print(f"Error processing sample {sample.get('id', sample_idx)}: {exc}")
            continue

    duration = time.time() - start_time
    if not results:
        print("No results generated.")
        return []

    avg_compression = sum(r["compression_ratio"] for r in results) / len(results)
    accuracy = sum(1 for r in results if r["hit"]) / len(results)
    avg_score = sum(r["score"] for r in results) / len(results)

    print("\n" + "=" * 40)
    print(f"Benchmark Complete: {dataset_path}")
    print(f"Backend: {backend_mode}")
    print(f"Benchmark: {benchmark_name}")
    print(f"Samples: {len(results)}")
    print(f"Time: {duration:.2f}s ({len(results)/duration:.2f} samples/s)")
    print(f"Avg Compression Ratio: {avg_compression:.2f}x")
    print(f"Retrieval Accuracy (Hit@1): {accuracy*100:.1f}%")
    print(f"Avg Match Score: {avg_score:.3f}")
    print("=" * 40 + "\n")

    failures = [r for r in results if not r["hit"]]
    if failures:
        f = failures[0]
        print("Sample Failure Case:")
        print(f"Question: {f['question']}")
        print(f"Expected: {f['ground_truth']}")
        print(f"Got: {f['retrieved_snippet']}")
        print("-" * 20)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--mode", type=str, default="synthetic", choices=["real", "synthetic"])
    parser.add_argument("--max", type=int, default=10)
    parser.add_argument("--benchmark", type=str, default="locomo", choices=["locomo", "infinitebench", "agentbench"])
    args = parser.parse_args()

    run_benchmark(args.data, args.mode, args.max, benchmark_name=args.benchmark)
