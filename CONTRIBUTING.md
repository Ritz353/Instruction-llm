# Contributing

Install with `python -m pip install -e .`, make a focused change, and run:

```bash
python -m unittest discover -s tests -v
instruction-llm train --model tiny --data examples/instructions.json --output outputs/contribution-test --epochs 1 --device cpu
```

Include a test when changing model, data, or checkpoint behavior. Document any
CLI or checkpoint format changes. Describe the issue, implementation, and
validation in your pull request. Preserve attribution on upstream-derived code.
Do not commit model weights, private datasets, access tokens, or generated caches.
Contributions are distributed under the repository's Apache 2.0 license.
