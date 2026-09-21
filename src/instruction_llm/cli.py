"""Local training and inference commands. No downloads occur at import time."""
import argparse
import hashlib
import json
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for inputs, targets in loaders[0]:
            optimizer.zero_grad(set_to_none=True)
            loss = calc_loss_batch(inputs, targets, model, device)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is non-finite; reduce the learning rate.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        model.eval()
        with torch.inference_mode():
            validation = calc_loss_loader(loaders[1], model, device)
        metrics = dict(epoch=epoch + 1, train_loss=total / len(loaders[0]), val_loss=validation)
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"model": args.model, "seed": args.seed, "epochs": args.epochs,
                "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                "device": device, "torch_version": str(torch.__version__),
                "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                "train_records": split, "validation_records": len(records) - split,
                "history": history}
    torch.save({"format_version": 1, "config": cfg,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "metadata": metadata}, output / "model.pt")
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
                          cfg["context_length"], eos_id=50256)
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
    training.add_argument("--learning-rate", type=float, default=5e-5)
    training.add_argument("--seed", type=int, default=123)
    training.add_argument("--cache-dir", default=".cache/gpt2")
    training.set_defaults(func=train)
    inference = commands.add_parser("generate", help="Generate from a saved checkpoint")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--prompt", required=True)
    inference.add_argument("--input", default="")
    inference.add_argument("--max-new-tokens", type=positive, default=128)
    inference.set_defaults(func=respond)
    for sub in (training, inference):
        sub.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
