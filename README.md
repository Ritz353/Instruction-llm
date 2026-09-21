# Instruction LLM

A standalone PyTorch toolkit for instruction-finetuning GPT-2, saving portable
checkpoints, and generating responses from the command line. Built from the
Chapter 7 implementation in Sebastian Raschka's
[Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch).

**Status:** research and learning software, packaged for GitHub. No trained
weights or benchmark claims are included. The tiny model verifies the pipeline;
it is randomly initialized and does not produce useful answers. Repository
organization alone does not make a model production-ready.

## What is included

- GPT transformer implementation in PyTorch, with causal attention.
- Instruction dataset validation, padding masks, and reproducible seeded splits.
- GPT-2 124M and 355M pretrained weight loading, plus a tiny smoke-test model.
- CPU, NVIDIA CUDA, and Apple MPS device selection.
- Training and validation losses, gradient clipping, and experiment metadata.
- Versioned checkpoints containing architecture, weights, and training metadata.
- CLI inference, automated tests, GitHub Actions, and model documentation.

## Install

Use Python 3.10 or newer in a virtual environment. Run these commands inside
this folder (the intended repository root):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, activate with `.venv\Scripts\activate`. PyTorch installation may
need platform-specific accelerator support. TensorFlow is only required to
load the original pretrained GPT-2 checkpoints:

```bash
python -m pip install -e '.[pretrained]'
```

Dependency ranges express compatibility intent; they are not a locked,
fully reproducible environment. Record `python -m pip freeze` for experiments.

## Verify the pipeline

```bash
python -m unittest discover -s tests -v
instruction-llm train --model tiny --data examples/instructions.json --output outputs/smoke --epochs 1 --device cpu
instruction-llm generate --checkpoint outputs/smoke/model.pt --prompt "Say hello." --max-new-tokens 20 --device cpu
```

The first tokenizer use may download its vocabulary. Unit tests require no
model downloads. Use a new output directory for each training run; nonempty
output directories are rejected to protect earlier experiments.

## Finetune pretrained GPT-2

Prepare a local JSON array following [the data guide](docs/DATA.md), then run:

```bash
instruction-llm train --model 124M --data data/train.json --output outputs/run-001 --epochs 2 --batch-size 2
instruction-llm generate --checkpoint outputs/run-001/model.pt --prompt "Summarize this text." --input "Your text here."
```

Training downloads GPT-2 weights into `.cache/gpt2`. Allow disk space for both
weights and training checkpoints. Training memory includes gradients and
optimizer state and is substantially greater than the weight file size. Start
with 124M and batch size 1 if memory is limited; CPU training can be slow.
`--device auto` chooses CUDA, then MPS, then CPU. Use `--device cpu` if your
accelerator cannot execute an operation. Run `instruction-llm train --help`
for seed, learning rate, and cache options.

Each run saves `model.pt` and `metrics.json`. Metadata records the dataset
SHA-256, seed, split sizes, hyperparameters, device, PyTorch version, and epoch
losses. Checkpoints support inference; optimizer state and interrupted-training
resume are not implemented. They use a project-specific wrapper and do not
directly load the original chapter's raw `.pth` files.

Training uses a seeded 90/10 train/validation split and next-token loss over
the full prompt plus response. Sequences are truncated to model context length
(1024 for GPT-2, 128 for tiny). Longer examples may lose response tokens; prepare
shorter records before training. The tiny demonstration dataset is not suitable
for quality evaluation. Maintain a separate untouched test set.

## Repository layout

```text
src/instruction_llm/  Model, data processing, downloader, CLI
examples/            Small synthetic dataset for pipeline verification
tests/               Causal attention, loss masks, updates, checkpoint tests
docs/                Dataset guide and model card
.github/workflows/   CI tests, smoke training, inference, and wheel build
```

See [MODEL_CARD.md](docs/MODEL_CARD.md) for limitations and evaluation guidance,
[CONTRIBUTING.md](CONTRIBUTING.md) for development, and
[SECURITY.md](SECURITY.md) for security reporting.

## Publish to GitHub

Run these commands **inside this folder** after reviewing the files:

```bash
git init -b main
git add .
git status
git commit -m "Initial instruction LLM toolkit"
```

Create an empty repository on GitHub, then use its URL:

```bash
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Weights, local datasets, caches, and secrets are ignored. Publish model weights
separately only after reviewing their distribution terms and completing the
model card with measured results.

## Attribution and license

Distributed under [Apache License 2.0](LICENSE.txt). See [NOTICE](NOTICE) for
upstream attribution and the files adapted from the book. This project does
not claim the upstream implementation or pretrained GPT-2 architecture as
original work. External model weights and datasets retain their own terms.
