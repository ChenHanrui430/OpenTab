"""Run inference on the released TMWF and/or MTWF closed-world test split."""

from __future__ import annotations

import argparse
import bisect
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import Architecture, TrafficRetriever


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("tmwf", "mtwf", "all"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke-test on a prefix.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


class ArrowTestDataset(Dataset):
    def __init__(self, path: Path, trace_length: int, label_to_index: dict[str, int]):
        self.path = path
        self.trace_length = trace_length
        self.label_to_index = label_to_index
        self.source = pa.memory_map(str(path), "r")
        self.reader = ipc.open_file(self.source)
        self.ends: list[int] = []
        total = 0
        for index in range(self.reader.num_record_batches):
            total += self.reader.get_batch(index).num_rows
            self.ends.append(total)
        self.length = total
        schema = self.reader.schema
        self.time_index = schema.get_field_index("time")
        self.data_index = schema.get_field_index("data")
        self.text_index = schema.get_field_index("label_text")
        if min(self.time_index, self.data_index, self.text_index) < 0:
            raise ValueError(f"Unexpected Arrow schema in {path}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
        batch_index = bisect.bisect_right(self.ends, index)
        start = 0 if batch_index == 0 else self.ends[batch_index - 1]
        row = index - start
        batch = self.reader.get_batch(batch_index)
        times = np.asarray(batch.column(self.time_index)[row].as_py(), dtype=np.float32)
        data = np.asarray(batch.column(self.data_index)[row].as_py(), dtype=np.float32)
        labels = [str(value) for value in (batch.column(self.text_index)[row].as_py() or [])]
        indices = [self.label_to_index[label] for label in labels]
        return data, times, indices, labels


class Collator:
    def __init__(self, trace_length: int, candidate_count: int):
        self.trace_length = trace_length
        self.candidate_count = candidate_count

    def __call__(self, rows):
        traffic = torch.zeros(len(rows), self.trace_length, 2, dtype=torch.float32)
        targets = torch.zeros(len(rows), self.candidate_count, dtype=torch.bool)
        labels: list[list[str]] = []
        for row, (data, times, indices, texts) in enumerate(rows):
            length = min(self.trace_length, len(data), len(times))
            traffic[row, :length, 0] = torch.from_numpy(data[:length])
            traffic[row, :length, 1] = torch.from_numpy(times[:length])
            targets[row, indices] = True
            labels.append(texts)
        return traffic, targets, labels


def metrics_for(scores: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict:
    probabilities = scores.sigmoid()
    predictions = probabilities >= threshold
    true_positive = (predictions & targets).sum(dim=1).float()
    false_positive = (predictions & ~targets).sum(dim=1).float()
    false_negative = (~predictions & targets).sum(dim=1).float()
    precision = true_positive / (true_positive + false_positive).clamp_min(1)
    recall = true_positive / (true_positive + false_negative).clamp_min(1)
    sample_f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    micro_tp = true_positive.sum()
    micro_fp = false_positive.sum()
    micro_fn = false_negative.sum()
    micro_precision = micro_tp / (micro_tp + micro_fp).clamp_min(1)
    micro_recall = micro_tp / (micro_tp + micro_fn).clamp_min(1)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall).clamp_min(1e-8)
    pred_count = predictions.sum(dim=1)
    true_count = targets.sum(dim=1)
    return {
        "samples": int(scores.shape[0]),
        "exact_set_accuracy": float((predictions == targets).all(dim=1).float().mean()),
        "sample_precision": float(precision.mean()),
        "sample_recall": float(recall.mean()),
        "sample_f1": float(sample_f1.mean()),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "predicted_cardinality_mean": float(pred_count.float().mean()),
        "true_cardinality_mean": float(true_count.float().mean()),
        "cardinality_mae": float((pred_count - true_count).abs().float().mean()),
        "exact_cardinality_rate": float((pred_count == true_count).float().mean()),
    }


def grouped_metrics(scores: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict:
    result = {"overall": metrics_for(scores, targets, threshold)}
    counts = targets.sum(dim=1)
    for tabs in range(1, 6):
        mask = counts == tabs
        if mask.any():
            result[f"{tabs}_tab"] = metrics_for(scores[mask], targets[mask], threshold)
    return result


def evaluate_one(
    name: str,
    path: Path,
    model: TrafficRetriever,
    candidates: torch.Tensor,
    labels: list[str],
    config: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    dataset = ArrowTestDataset(path, int(config["trace_length"]), {v: i for i, v in enumerate(labels)})
    if args.max_samples is not None:
        dataset.length = min(dataset.length, args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=Collator(int(config["trace_length"]), len(labels)),
    )
    all_scores: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    true_texts: list[list[str]] = []
    autocast = (
        lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode():
        for traffic, targets, text_rows in tqdm(loader, desc=f"Evaluating {name}"):
            with autocast():
                scores = model(traffic.to(device, non_blocking=True), candidates)
            all_scores.append(scores.float().cpu())
            all_targets.append(targets)
            true_texts.extend(text_rows)
    scores = torch.cat(all_scores)
    targets = torch.cat(all_targets)
    threshold = float(config["threshold"] if args.threshold is None else args.threshold)
    report = {
        "dataset": name,
        "protocol": "closed_world_unknown_cardinality_global_threshold",
        "candidate_count": len(labels),
        "classification_threshold": threshold,
        "metrics": grouped_metrics(scores, targets, threshold),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{name}_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"]["overall"], indent=2))
    print(f"Saved metrics to {report_path}")

    if args.save_predictions:
        probabilities = scores.sigmoid()
        predictions = probabilities >= threshold
        prediction_path = args.output_dir / f"{name}_predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for row in range(scores.shape[0]):
                indices = predictions[row].nonzero().flatten().tolist()
                record = {
                    "sample_index": row,
                    "true_labels": true_texts[row],
                    "predicted_labels": [labels[index] for index in indices],
                    "predicted_probabilities": [float(probabilities[row, index]) for index in indices],
                }
                handle.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    manifest_path = ROOT / config["embedding_manifest"]
    embedding_path = ROOT / config["embedding_weights"]
    if not manifest_path.is_file() or not embedding_path.is_file():
        raise FileNotFoundError(
            "Candidate embeddings are missing. Run `uv run python prepare_embeddings.py "
            "--model models/Qwen3-Embedding-4B` first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = [str(value) for value in manifest["labels"]]
    candidates = load_file(str(embedding_path), device=str(device))["embeddings"].float()
    if tuple(candidates.shape) != (len(labels), int(config["architecture"]["text_dim"])):
        raise RuntimeError("Candidate embedding cache is inconsistent with its manifest.")

    architecture = Architecture(**config["architecture"])
    model = TrafficRetriever(architecture)
    state = load_file(str(ROOT / config["checkpoint"]), device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    names = list(config["datasets"]) if args.dataset == "all" else [args.dataset]
    for name in names:
        evaluate_one(
            name,
            ROOT / config["datasets"][name],
            model,
            candidates,
            labels,
            config,
            args,
            device,
        )


if __name__ == "__main__":
    main()

