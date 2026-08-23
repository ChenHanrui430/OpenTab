"""Build the fixed closed-world candidate bank with the public text encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="models/Qwen3-Embedding-4B",
        help="Local Qwen3-Embedding-4B directory or Hugging Face model ID.",
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or a CUDA device such as cuda:0"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    return parser.parse_args()


def read_labels(path: Path) -> set[str]:
    labels: set[str] = set()
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        for batch_index in range(reader.num_record_batches):
            column = reader.get_batch(batch_index).column("label_text")
            for row in column.to_pylist():
                labels.update(str(value) for value in (row or []) if value)
    return labels


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    indices = (mask.sum(dim=1) - 1).clamp(min=0)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, indices]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_paths = [ROOT / value for value in config["datasets"].values()]
    for path in data_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    labels = sorted(set().union(*(read_labels(path) for path in data_paths)))
    prompts = [f"Tor with {label}." for label in labels]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(prompts), args.batch_size), desc="Embedding labels"):
            batch = tokenizer(
                prompts[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(device)
            output = model(**batch, return_dict=True)
            pooled = last_token_pool(output.last_hidden_state, batch.attention_mask)
            rows.append(F.normalize(pooled.float(), dim=-1).cpu())

    embeddings = torch.cat(rows, dim=0).contiguous()
    expected_dim = int(config["architecture"]["text_dim"])
    if tuple(embeddings.shape) != (len(labels), expected_dim):
        raise RuntimeError(
            f"Unexpected candidate-bank shape {tuple(embeddings.shape)}; "
            f"expected {(len(labels), expected_dim)}"
        )

    weights_path = ROOT / config["embedding_weights"]
    manifest_path = ROOT / config["embedding_manifest"]
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"embeddings": embeddings.to(torch.bfloat16)}, str(weights_path))
    manifest = {
        "embedding_model": config["embedding_model"],
        "prompt_template": "Tor with {label}.",
        "candidate_count": len(labels),
        "embedding_dim": expected_dim,
        "labels": labels,
        "data_sha256": {path.name: sha256(path) for path in data_paths},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(labels)} candidate embeddings to {weights_path}")


if __name__ == "__main__":
    main()

