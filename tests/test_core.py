import json
import tempfile
import unittest
from pathlib import Path

import torch

from instruction_llm.cli import config_for, load_checkpoint
from instruction_llm.data import custom_collate_fn, format_input, load_data
from instruction_llm.model import GPTModel, generate


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)

    def test_padding_preserves_one_eos_target(self):
        inputs, targets = custom_collate_fn([[7, 8], [9]])
        self.assertEqual(inputs.tolist(), [[7, 8], [9, 50256]])
        self.assertEqual(targets.tolist(), [[8, 50256], [50256, -100]])

    def test_truncation(self):
        inputs, targets = custom_collate_fn([[1, 2, 3]], allowed_max_length=2)
        self.assertEqual(inputs.tolist(), [[1, 2]])
        self.assertEqual(targets.tolist(), [[2, 3]])

    def test_data_validation_and_optional_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps([{"instruction": "Say hi", "output": "Hi"}] * 3))
            self.assertEqual(load_data(path)[0]["input"], "")
            path.write_text(json.dumps([{"instruction": "", "output": "Hi"}] * 3))
            with self.assertRaises(ValueError):
                load_data(path)
            path.write_text('{}')
            with self.assertRaises(ValueError):
                load_data(path)

    def test_prompt(self):
        prompt = format_input({"instruction": "Summarize", "input": "hello"})
        self.assertIn("### Instruction:\nSummarize", prompt)
        self.assertIn("### Input:\nhello", prompt)

    def test_causal_attention_and_training_step(self):
        cfg = config_for("tiny") | {"vocab_size": 32}
        model = GPTModel(cfg).eval()
        first = torch.tensor([[1, 2, 3, 4]])
        changed = torch.tensor([[1, 2, 8, 9]])
        torch.testing.assert_close(model(first)[:, :2], model(changed)[:, :2])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = model.tok_emb.weight.detach().clone()
        loss = torch.nn.functional.cross_entropy(model(first).flatten(0, 1), changed.flatten())
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, model.tok_emb.weight))

    def test_checkpoint_roundtrip_and_generation(self):
        cfg = config_for("tiny") | {"vocab_size": 32}
        model = GPTModel(cfg).eval()
        inputs = torch.tensor([[1, 2, 3]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"format_version": 1, "config": cfg, "state_dict": model.state_dict()}, path)
            restored, restored_cfg = load_checkpoint(path, "cpu")
            self.assertEqual(cfg, restored_cfg)
            torch.testing.assert_close(model(inputs), restored(inputs))
            self.assertEqual(generate(restored, inputs, 3, cfg["context_length"]).shape, (1, 6))
            torch.save(model.state_dict(), path)
            with self.assertRaises(ValueError):
                load_checkpoint(path, "cpu")


if __name__ == "__main__":
    unittest.main()
