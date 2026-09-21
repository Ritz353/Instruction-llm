# Dataset format

Use a UTF-8 JSON array with at least three objects:

```json
[
  {"instruction": "Say hello.", "input": "", "output": "Hello!"},
  {"instruction": "Translate to Spanish.", "input": "Hello", "output": "Hola"},
  {"instruction": "Name a color.", "input": "", "output": "Blue"}
]
```

`instruction` and `output` must be nonempty strings. `input` is optional and
must be a string when present. Extra fields are ignored. Records are tokenized
in memory, so dataset size is limited by available RAM. Literal GPT-2 special
token strings in training records are rejected by the tokenizer.

Use only data you are permitted to use. Remove secrets and personal information,
deduplicate related records before splitting, and retain source and permission
records outside the model's training text. For serious evaluation, reserve a
separate test set before supplying the training file. The CLI's random split
does not protect against near-duplicate or semantic leakage.

Loss includes instruction, input, and response tokens. Padding is ignored after
the first end-of-text target. The CLI truncates long sequences rather than
packing them. Check token lengths and retain complete responses when preparing
data. The six included examples only exercise the code path.
