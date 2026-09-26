"""Local training and inference commands. No downloads occur at import time."""
import argparse
import hashlib
import json
import math
import random
from functools import partial
from pathlib import Path

import tiktoken
import torch
from torch.utils.data import DataLoader

from .data import InstructionDataset, custom_collate_fn, format_input, load_data
from .model import GPTModel, calc_loss_batch, calc_loss_loader, generate, load_weights_into_gpt

SIZES = {
    "124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "355M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
}


def config_for(size):
    base = dict(vocab_size=50257, context_length=1024, drop_rate=0.0, qkv_bias=True)
    base.update(SIZES[size] if size != "tiny" else
                dict(emb_dim=32, n_layers=2, n_heads=4, context_length=128))
    return base


def device_for(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or auto.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu or auto.")
    return name


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def train(args):
    if args.learning_rate <= 0 or not float('-inf') < args.learning_rate < float('inf'):
        raise ValueError("Learning rate must be finite and positive.")
    records = load_data(args.data)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty to avoid overwriting an experiment.")
    device = device_for(args.device)
    random.Random(args.seed).shuffle(records)
    torch.manual_seed(args.seed)
    cfg = config_for(args.model)
    tokenizer = tiktoken.get_encoding("gpt2")
    split = max(1, min(len(records) - 1, int(len(records) * 0.9)))
    collate = partial(custom_collate_fn, allowed_max_length=cfg["context_length"])
    loaders = [DataLoader(InstructionDataset(rows, tokenizer), batch_size=args.batch_size,
                          shuffle=i == 0, collate_fn=collate)
               for i, rows in enumerate((records[:split], records[split:]))]
    model = GPTModel(cfg)
    if args.model != "tiny":
        try:
            from .gpt_download import download_and_load_gpt2
        except ImportError as error:
            raise ValueError('Pretrained training needs: pip install -e ".[pretrained]"') from error
        _, params = download_and_load_gpt2(args.model, args.cache_dir)
        load_weights_into_gpt(model, params)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.1,
    )

    # Several small batches now act like one larger batch without requiring
    # the full effective batch to fit in GPU memory at once.
    accumulation_steps = args.gradient_accumulation_steps
    updates_per_epoch = math.ceil(len(loaders[0]) / accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = max(1, int(total_updates * 0.03))

    def learning_rate_multiplier(step):
        if step < warmup_updates:
            return (step + 1) / warmup_updates
        decay_steps = max(1, total_updates - warmup_updates)
        progress = min(1.0, (step - warmup_updates) / decay_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=learning_rate_multiplier,
    )
    history = []
    best_validation = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_index, (inputs, targets) in enumerate(loaders[0]):
            loss = calc_loss_batch(inputs, targets, model, device)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is non-finite; reduce the learning rate.")

            # Scaling keeps accumulated gradients comparable to a normal
            # larger-batch average.
            (loss / accumulation_steps).backward()
            total += loss.item()

            update_due = (batch_index + 1) % accumulation_steps == 0
            final_batch = batch_index + 1 == len(loaders[0])
            if update_due or final_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        with torch.inference_mode():
            validation = calc_loss_loader(loaders[1], model, device)

        metrics = {
            "epoch": epoch + 1,
            "train_loss": total / len(loaders[0]),
            "val_loss": validation,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(metrics)
        print(json.dumps(metrics), flush=True)

        # Keep a CPU copy of the epoch that generalizes best instead of
        # automatically saving the possibly overfit final epoch.
        if validation < best_validation:
            best_validation = validation
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": args.batch_size * accumulation_steps,
        "learning_rate": args.learning_rate,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "device": device,
        "torch_version": str(torch.__version__),
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "train_records": split,
        "validation_records": len(records) - split,
        "history": history,
    }
    torch.save({"format_version": 1, "config": cfg,
                "state_dict": best_state, "metadata": metadata}, output / "model.pt")
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved checkpoint: {output / 'model.pt'}")


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Expected an instruction-llm version 1 checkpoint.")
    model = GPTModel(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint["config"]


def respond(args):
    device = device_for(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = format_input({"instruction": args.prompt, "input": args.input}) + "\n\n### Response:\n"
    ids = tokenizer.encode(prompt, disallowed_special=())
    if len(ids) > cfg["context_length"]:
        raise ValueError("Prompt exceeds this model's context length; shorten the prompt or input.")
    with torch.inference_mode():
        tokens = generate(model, torch.tensor([ids], device=device), args.max_new_tokens,
                          cfg["context_length"], eos_id=50256,
                          temperature=args.temperature, top_k=args.top_k,
                          repetition_penalty=args.repetition_penalty,
                          no_repeat_ngram_size=args.no_repeat_ngram_size)
    print(tokenizer.decode(tokens[0, len(ids):].tolist()).strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Finetune pretrained GPT-2 or smoke-test a tiny random model")
    training.add_argument("--data", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--model", choices=["tiny", *SIZES], default="124M")
    training.add_argument("--epochs", type=positive, default=2)
    training.add_argument("--batch-size", type=positive, default=2)
    training.add_argument(
        "--gradient-accumulation-steps", type=positive, default=16,
        help="Accumulate this many batches before each optimizer update (default: 16).",
    )
    training.add_argument("--learning-rate", type=float, default=5e-5)
    training.add_argument("--seed", type=int, default=123)
    training.add_argument("--cache-dir", default=".cache/gpt2")
    training.set_defaults(func=train)
    inference = commands.add_parser("generate", help="Generate from a saved checkpoint")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--prompt", required=True)
    inference.add_argument("--input", default="")
    inference.add_argument("--max-new-tokens", type=positive, default=128)
    inference.add_argument("--temperature", type=float, default=0.0,
                           help="Sampling temperature; 0 uses deterministic decoding (default).")
    inference.add_argument("--top-k", type=positive, default=50,
                           help="Sample from the top K tokens with positive temperature (default: 50).")
    inference.add_argument("--repetition-penalty", type=float, default=1.15,
                           help="Penalize response tokens already used; 1 disables (default: 1.15).")
    inference.add_argument("--no-repeat-ngram-size", type=int, default=4,
                           help="Block repeated response token sequences; 0 disables (default: 4).")
    inference.set_defaults(func=respond)
    for sub in (training, inference):
        sub.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
