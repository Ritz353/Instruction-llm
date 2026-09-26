# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# A minimal instruction finetuning file based on the code in chapter 7

# Modified: extracted data helpers; added local dataset validation.
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            # The prompt is context for the model, not a prediction target.
            prompt_text = (
                format_input(entry)
                + "\n\n### Response:\n"
            )
            response_text = entry["output"]

            prompt_ids = tokenizer.encode(prompt_text)
            response_ids = tokenizer.encode(response_text)

            # Keeping the sections separate lets the collator mask the prompt.
            self.encoded_texts.append((prompt_ids, response_ids))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

def _shorten_prompt(prompt_ids, maximum_length):
    """Keep the start and end of an oversized prompt."""
    if len(prompt_ids) <= maximum_length:
        return prompt_ids

    beginning_length = maximum_length // 2
    ending_length = maximum_length - beginning_length

    return (
        prompt_ids[:beginning_length]
        + prompt_ids[-ending_length:]
    )

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu",
):
    prepared_examples = []

    for prompt_ids, response_ids in batch:
        prompt_ids = prompt_ids.copy()
        response_ids = response_ids.copy()

        if allowed_max_length is not None:
            if allowed_max_length < 2:
                raise ValueError(
                    "allowed_max_length must be at least 2."
                )

            # Leave room for at least one prompt token. Very long
            # responses must still be shortened to fit the model.
            maximum_response_length = allowed_max_length - 1
            response_ids = response_ids[:maximum_response_length]

            # Give the remaining context space to the prompt.
            prompt_budget = allowed_max_length - len(response_ids)
            prompt_ids = _shorten_prompt(
                prompt_ids,
                max(1, prompt_budget),
            )

        response_start = len(prompt_ids)
        token_ids = prompt_ids + response_ids

        prepared_examples.append(
            (token_ids, response_start)
        )

    # Add one position for the end-of-text token.
    batch_max_length = max(
        len(token_ids) + 1
        for token_ids, _ in prepared_examples
    )

    inputs_list = []
    targets_list = []

    for token_ids, response_start in prepared_examples:
        # End every response with GPT-2's end-of-text token.
        token_ids = token_ids + [pad_token_id]

        padded = token_ids + [pad_token_id] * (
            batch_max_length - len(token_ids)
        )

        # Input token N is trained to predict target token N + 1.
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # Ignore prompt targets. The first response token appears at
        # target position response_start - 1.
        prompt_target_count = max(0, response_start - 1)
        targets[:prompt_target_count] = ignore_index

        # Keep the real end-of-text target but ignore padding after it.
        padding_indices = torch.nonzero(
            targets == pad_token_id
        ).flatten()

        if padding_indices.numel() > 1:
            targets[padding_indices[1:]] = ignore_index

        inputs_list.append(inputs)
        targets_list.append(targets)

    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)

    return inputs_tensor, targets_tensor

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def load_data(path):
    """Load a local JSON array of instruction/input/output records."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) < 3:
        raise ValueError("Dataset must be a JSON array with at least 3 records.")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be an object.")
        record.setdefault("input", "")
        for key in ("instruction", "input", "output"):
            if not isinstance(record.get(key), str):
                raise ValueError(f"Record {index}: {key} must be a string.")
        if not record["instruction"].strip() or not record["output"].strip():
            raise ValueError(f"Record {index}: instruction and output cannot be empty.")
    return records
