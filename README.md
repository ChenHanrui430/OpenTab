# Anonymous Closed-World Inference Artifact

This anonymous repository provides the inference-only artifact for our
multi-tab website fingerprinting method. It contains one traffic-side
checkpoint and the processed closed-world test splits of TMWF and MTWF. The
checkpoint was trained with the joint label space and evaluates both datasets
against the same 116-candidate semantic bank.


> Upon acceptance, we will release the complete source code and all datasets
> used in the paper.

## Repository contents

```text
.
├── checkpoints/model.safetensors       # proposed method; inference weights only
├── data/
│   ├── tmwf_closed_world_test.arrow    # 3,750 samples, 50 classes
│   └── mtwf_closed_world_test.arrow    # 3,750 samples, 100 classes
├── embeddings/                         # generated locally; not committed
├── config.json                         # paths, architecture, and threshold
├── model.py                            # inference-only traffic encoder
├── prepare_embeddings.py               # builds the semantic candidate bank
├── evaluate.py                         # closed-world inference and metrics
├── pyproject.toml
└── uv.lock
```

The two label sets overlap, so their union contains 116 candidates rather than
150. The released checkpoint contains only the approximately 50.2M trainable
traffic-side parameters. The frozen text encoder is downloaded separately.

## Requirements

- Linux with Python 3.11 or 3.12.
- [`uv`](https://docs.astral.sh/uv/) for dependency and environment management.
- A CUDA GPU is strongly recommended. We recommend at least 16 GB of GPU
  memory for preparing Qwen embeddings and at least 12 GB for inference with
  the default batch size. CPU execution is supported but slow.

No virtual environment is included in this repository. `uv` creates and
manages one locally.

## 1. Clone and create the environment
Install the locked dependencies:

```bash
uv sync --frozen
```

If `uv` is not installed, follow its official installation instructions and
rerun the command above. Do not install this project with `pip`.

## 2. Download the public embedding model

Download the official `Qwen/Qwen3-Embedding-4B` model from Hugging Face into
the ignored `models/` directory:

```bash
uv run hf download Qwen/Qwen3-Embedding-4B \
  --local-dir models/Qwen3-Embedding-4B
```

The model is approximately 8 GB. It is not modified or redistributed by this
artifact. A pre-existing local copy can be used by passing its path in the
next command.

## 3. Build the candidate embedding bank

Generate embeddings for the 116 label texts found in the two released test
sets:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python prepare_embeddings.py \
  --model models/Qwen3-Embedding-4B \
  --device cuda
```

This creates `embeddings/candidate_embeddings.safetensors` and its JSON
manifest. Both are ignored by Git because they are deterministically
reproducible from the public embedding model and the released labels.

For a CPU-only machine, replace `--device cuda` with `--device cpu`.

## 4. Run inference

First run a short end-to-end smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python evaluate.py \
  --dataset tmwf \
  --batch-size 1 \
  --max-samples 8
```

Evaluate both complete test sets:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python evaluate.py \
  --dataset all \
  --batch-size 4
```

Evaluate only one dataset when needed:

```bash
uv run python evaluate.py --dataset tmwf --batch-size 4
uv run python evaluate.py --dataset mtwf --batch-size 4
```

Metrics are written to `results/tmwf_metrics.json` and
`results/mtwf_metrics.json`. Add `--save-predictions` to also write one JSONL
record per trace. Results and generated embeddings are excluded from version
control.

The default protocol predicts an unknown number of tabs using the fixed
global probability threshold `0.75` in `config.json`; it does not use the
ground-truth number of tabs at inference time. For the full 116-candidate bank,
the reference checkpoint gives approximately the following overall results:

| Test split | Exact-set accuracy | Micro F1 |
| --- | ---: | ---: |
| TMWF closed world | 0.686 | 0.901 |
| MTWF closed world | 0.460 | 0.836 |

Minor numerical differences can occur across CUDA, PyTorch, and attention
kernel versions.

## Useful options

```text
--device cpu|cuda|cuda:N   execution device (default: auto)
--batch-size N             reduce this if GPU memory is insufficient
--num-workers N            Arrow data-loader workers (default: 0)
--threshold FLOAT          override the released global threshold
--max-samples N            evaluate only a prefix for debugging
--save-predictions         save predicted labels and probabilities
```

If CUDA runs out of memory, use `--batch-size 1`. If the candidate bank is
missing, rerun Step 3. If the large files are only small text pointers, run
`git lfs pull` and verify their sizes with `ls -lh checkpoints data`.

## Release scope

The source code is licensed under Apache-2.0. The checkpoint is supplied for
non-commercial academic artifact evaluation during anonymous review. The code
license does not override the terms of the original TMWF or MTWF datasets.
