# Local validation

Validated on September 21, 2026, with Python 3.11 and PyTorch 2.13.0 on CPU:

- Six unit tests passed: dataset validation, prompt formatting, padding masks,
  truncation, causal attention/training update, and checkpoint/generation roundtrip.
- One tiny-model epoch completed on the six synthetic example records.
- Saved checkpoint reloaded successfully and generated five tokens.
- A distributable Python wheel built successfully, including license and NOTICE.

The tokenizer files were downloaded for the smoke test. No pretrained GPT-2
weights were downloaded or finetuned. CUDA, MPS, TensorFlow weight conversion,
and the GitHub-hosted CI matrix have not been validated locally. Tiny-model
output is arbitrary and does not demonstrate instruction-following quality.
