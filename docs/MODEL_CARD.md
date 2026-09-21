# Model card: Instruction LLM toolkit

## Release status

This release distributes training code, not a trained model. No task-quality,
safety, fairness, latency, or throughput benchmark has been measured. The
synthetic tiny-model run is a software smoke test only.

## Architecture and intended use

Decoder-only GPT transformer with the GPT-2 tokenizer (50,257 tokens), learned
position embeddings, and causal self-attention. Pretrained configurations:
GPT-2 124M and 355M, with a 1024-token context. Intended for learning, experiments,
and reproducible small-model instruction-finetuning research.

## Training and evaluation

Supervised next-token cross-entropy, AdamW, gradient clipping at norm 1.0,
seeded 90/10 record split. Defaults: two epochs, learning rate 0.00005, batch
size two. See each run's metrics.json for actual settings. Epoch losses are
batch averages, not token-weighted corpus perplexity. Equal seeds do not imply
bitwise reproducibility across devices or library versions.

Before distributing a trained checkpoint, document the exact base weights,
data provenance and permissions, filtering, hardware, dependency versions,
training settings, held-out task scores, failure examples, and generation
settings. Compare against the unchanged base model on the same test set.

## Limitations

Small GPT-2 models can hallucinate, repeat, reproduce training text, follow
malicious instructions, and generate biased or offensive content. Finetuning
does not establish factual reliability or safety. There is no moderation,
authentication, network API, retrieval system, or multi-turn conversation state.
Do not treat outputs as verified facts or use this release for autonomous
high-stakes decisions. Production deployment needs application-specific
validation, monitoring, access controls, and resource limits.
