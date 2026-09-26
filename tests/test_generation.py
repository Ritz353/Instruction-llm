import unittest
from unittest.mock import patch

import torch

from instruction_llm.cli import main
from instruction_llm.model import generate


class ConstantModel(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = torch.tensor(scores, dtype=torch.float)

    def forward(self, ids):
        return self.scores.to(ids.device).expand(ids.shape[0], ids.shape[1], -1)


class GenerationTests(unittest.TestCase):
    def test_penalty_handles_positive_and_negative_scores(self):
        for scores in ([0, 10, 9], [-10, -1, -1.1]):
            with self.subTest(scores=scores):
                model = ConstantModel(scores)
                prompt = torch.tensor([[1]])
                baseline = generate(model, prompt, 2, 8)
                controlled = generate(model, prompt, 2, 8, repetition_penalty=1.2)
                self.assertEqual(baseline.tolist(), [[1, 1, 1]])
                self.assertEqual(controlled.tolist(), [[1, 1, 2]])

    def test_blocks_phrases_beyond_context_window_with_sampling(self):
        for temperature in (0, 0.8):
            result = generate(ConstantModel([0, 30, 29, 28, 27, 26, 25, 24, 23, 22]),
                              torch.tensor([[1, 1, 1, 1]]), 30, 2,
                              no_repeat_ngram_size=4, temperature=temperature, top_k=1)
            response = result[0, 4:].tolist()
            self.assertEqual(response[:4], [1, 1, 1, 1])
            grams = [tuple(response[i:i+4]) for i in range(len(response)-3)]
            self.assertEqual(len(grams), len(set(grams)))

    def test_unigram_guard_and_eos(self):
        result = generate(ConstantModel([0, 10, 9]), torch.tensor([[0]]),
                          10, 8, no_repeat_ngram_size=1, eos_id=0)
        self.assertEqual(result.tolist(), [[0, 1, 2]])

    def test_all_tokens_blocked_reports_error(self):
        with self.assertRaisesRegex(ValueError, "blocked every token"):
            generate(ConstantModel([1]), torch.tensor([[0]]), 2, 8,
                     no_repeat_ngram_size=1)

    def test_eos_remains_unpenalized(self):
        result = generate(ConstantModel([10, 9]), torch.tensor([[0]]),
                          10, 8, eos_id=0, repetition_penalty=100)
        self.assertEqual(result.tolist(), [[0]])

    def test_batch_top_k_and_independent_eos(self):
        class BatchModel(torch.nn.Module):
            def forward(self, ids):
                scores = [[10, 0, 0], [0, 20, 0] if ids.shape[1] == 1 else [20, 0, 0]]
                return torch.tensor(scores, dtype=torch.float)[:, None, :].expand(-1, ids.shape[1], -1)
        result = generate(BatchModel(), torch.tensor([[2], [2]]), 5, 8,
                          top_k=50, eos_id=0)
        self.assertEqual(result.tolist(), [[2, 0], [2, 1]])

    def test_invalid_controls(self):
        for kwargs in (dict(repetition_penalty=0.9), dict(repetition_penalty=float("nan")),
                       dict(no_repeat_ngram_size=-1), dict(temperature=-1),
                       dict(temperature=float("inf")), dict(top_k=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate(ConstantModel([1, 2]), torch.tensor([[0]]), 1, 8, **kwargs)

    def test_cli_defaults_and_overrides(self):
        for extra, expected in (([], (1.15, 4, 0, 50)),
                                (["--repetition-penalty", "1", "--no-repeat-ngram-size", "0",
                                  "--temperature", "0.8", "--top-k", "20"], (1, 0, 0.8, 20))):
            with patch("sys.argv", ["instruction-llm", "generate", "--checkpoint", "unused",
                                    "--prompt", "Hello"] + extra), patch("instruction_llm.cli.respond") as respond:
                main()
                args = respond.call_args.args[0]
                self.assertEqual((args.repetition_penalty, args.no_repeat_ngram_size,
                                  args.temperature, args.top_k), expected)


if __name__ == "__main__":
    unittest.main()
