from __future__ import annotations

import unittest

import torch

from src.evidence_transformer import EvidenceTransformer, RESPONSIBLE_USE_NOTE


class WordTokenizer:
    """Minimal reversible tokenizer for chunk-orchestration tests."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(1, len(text.split()) + 1))

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(f"token-{token_id}" for token_id in ids)


class EvidenceTransformerTests(unittest.TestCase):
    def model_stub(self) -> EvidenceTransformer:
        model = EvidenceTransformer.__new__(EvidenceTransformer)
        model.model_id = "test/model"
        model.device = torch.device("cpu")
        model.max_input_tokens = 160
        model.max_chunks = 2
        model.tokenizer = WordTokenizer()
        model._generate_text = lambda text, max_new_tokens: "Generated observation."
        return model

    def test_generation_reports_truncation_and_safety_note(self) -> None:
        model = self.model_stub()
        result = model.generate(" ".join(["evidence"] * 200), "Summarize facts.", 64)
        self.assertTrue(result.truncated)
        self.assertEqual(result.chunks_processed, 2)
        self.assertIn(RESPONSIBLE_USE_NOTE, result.output)
        self.assertEqual(result.model_id, "test/model")

    def test_empty_input_is_rejected(self) -> None:
        model = self.model_stub()
        with self.assertRaises(ValueError):
            model.generate("", "Summarize facts.")

    def test_compact_model_output_gets_source_grounded_section_scaffold(self) -> None:
        model = self.model_stub()
        result = model.generate(
            "Mira Patel collected device EX-17 at 09:15, but a later note says 10:15. "
            "The timezone is missing.",
            "Analyze evidence.",
            64,
        )
        self.assertIn("### Evidence Summary", result.output)
        self.assertIn("### Important Entities", result.output)
        self.assertIn("EX-17", result.output)
        self.assertIn("09:15", result.output)
        self.assertIn("10:15", result.output)
        self.assertIn("time-zone context", result.output)
        self.assertIn("conservative pattern matches", result.output)


if __name__ == "__main__":
    unittest.main()
