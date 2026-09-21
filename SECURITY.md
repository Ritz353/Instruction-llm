# Security

Load checkpoints only from trusted sources. The loader uses PyTorch's
`weights_only=True` mode, but model files can still cause excessive memory or
compute use. Do not expose this local research CLI directly to untrusted users.

Keep dependencies patched, inspect datasets for private information, and avoid
committing secrets. Pretrained downloads use HTTPS; this version does not pin
upstream checkpoint hashes. Validate downloaded artifacts independently when
reproducible or high-assurance provenance is required.

Report vulnerabilities using GitHub's private vulnerability reporting if the
repository owner has enabled it. Otherwise ask the maintainer for a private
reporting channel without posting exploit details or credentials publicly.
