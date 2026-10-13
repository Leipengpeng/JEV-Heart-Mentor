import sys
from pathlib import Path
import unittest
from unittest import mock
import json
import os
import tempfile

try:
    import mlx.core
except ImportError:
    raise unittest.SkipTest("MLX structured decoding runs on Apple Silicon only")

from chatlens.structured_output import make_json_processor
from chatlens import structured_output

SCHEMA = {
    "type": "object",
    "required": ["a"],
    "properties": {"a": {"type": "string", "minLength": 1}},
    "additionalProperties": False,
}


class ByteTokenizer:
    """Synthetic UTF-8 fragments: a complete character is absent from the vocabulary."""

    def __init__(self):
        self.parts = [
            "0",
            "{",
            "}",
            '"',
            ":",
            "a",
            "#",
            "\t",
            "\n",
            b"\xf0\xa0",
            b"\xae\xb7",
            b"\xf0\x9f",
            b"\xa7\xa0",
            "<end>",
            "<chat-end>",
        ]
        self.all_special_ids = []
        self.eos_token_ids = {len(self.parts) - 2, len(self.parts) - 1}

    def get_vocab(self):
        return {str(part): index for index, part in enumerate(self.parts)}

    def encode(self, text, **kwargs):
        ids = []
        for character in text:
            if character == "𠮷":
                ids.extend([9, 10])
            elif character == "🧠":
                ids.extend([11, 12])
            else:
                ids.append(self.parts.index(character))
        return ids

    def decode(self, ids):
        raw = b"".join(
            self.parts[index] if isinstance(self.parts[index], bytes) else self.parts[index].encode("utf-8")
            for index in ids
        )
        return raw.decode("utf-8", errors="replace")


class JSONSamplingTests(unittest.TestCase):
    def setUp(self):
        # Tiny synthetic tokenizer identities must not reuse an earlier cache entry.
        structured_output._TOKEN_DATA.clear()

    def test_chat_eos_missing_from_special_ids_cannot_end_an_open_string(self):
        import mlx.core as mx

        class Tokenizer:
            chars = ["0", "{", "}", '"', ":", "a", "中", "<end>"]
            all_special_ids = []
            eos_token_ids = {7}

            def get_vocab(self):
                return {c: i for i, c in enumerate(self.chars)}

            def encode(self, text, **kwargs):
                return [self.chars.index(c) for c in text]

            def decode(self, ids):
                return "".join(self.chars[i] for i in ids)

        tokenizer = Tokenizer()
        processor, _ = make_json_processor(
            tokenizer,
            {
                "type": "object",
                "required": ["a"],
                "properties": {"a": {"type": "string"}},
                "additionalProperties": False,
            },
        )
        tokens = [0]
        processor(mx.array(tokens), mx.zeros((1, 8)))
        for token in tokenizer.encode('{"a":"'):
            tokens.append(token)
            logits = processor(mx.array(tokens), mx.zeros((1, 8)))
            self.assertTrue(bool(mx.isneginf(logits[0, 7])))
        self.assertFalse(bool(mx.isneginf(logits[0, 6])))
        tokens.append(tokenizer.encode("中")[0])
        logits = processor(mx.array(tokens), mx.zeros((1, 8)))
        self.assertTrue(bool(mx.isneginf(logits[0, 7])))
        self.assertFalse(bool(mx.isneginf(logits[0, 3])))
        for token in tokenizer.encode('"}'):
            tokens.append(token)
            logits = processor(mx.array(tokens), mx.zeros((1, 8)))
        self.assertFalse(bool(mx.isneginf(logits[0, 7])))

    def test_byte_fragments_can_complete_unicode_without_silently_finishing_json(self):
        import mlx.core as mx

        tokenizer = ByteTokenizer()
        processor, stats = make_json_processor(tokenizer, SCHEMA)
        tokens = [0]
        logits = processor(mx.array(tokens), mx.zeros((1, len(tokenizer.parts))))
        for token in tokenizer.encode('{"a":"𠮷🧠'):
            self.assertFalse(bool(mx.isneginf(logits[0, token])))
            tokens.append(token)
            logits = processor(mx.array(tokens), mx.zeros((1, len(tokenizer.parts))))
            for eos in tokenizer.eos_token_ids:
                self.assertTrue(bool(mx.isneginf(logits[0, eos])))
        self.assertGreaterEqual(stats["unicode_characters_added"], 2)
        for token in tokenizer.encode('"}'):
            tokens.append(token)
            logits = processor(mx.array(tokens), mx.zeros((1, len(tokenizer.parts))))
        for eos in tokenizer.eos_token_ids:
            self.assertFalse(bool(mx.isneginf(logits[0, eos])))
        self.assertEqual(stats["parser_failures"], 0)

    def test_disallowed_token_is_rejected_instead_of_turning_into_eos(self):
        import mlx.core as mx

        tokenizer = ByteTokenizer()
        processor, stats = make_json_processor(tokenizer, SCHEMA)
        processor(mx.array([0]), mx.zeros((1, len(tokenizer.parts))))
        with self.assertRaisesRegex(RuntimeError, "disallowed token"):
            processor(mx.array([0, tokenizer.parts.index("#")]), mx.zeros((1, len(tokenizer.parts))))
        self.assertEqual(stats["parser_failures"], 1)

    def test_completed_json_survives_mlx_prefetch_after_both_eos_markers(self):
        import mlx.core as mx

        tokenizer = ByteTokenizer()
        processor, stats = make_json_processor(tokenizer, SCHEMA)
        tokens = [0]
        processor(mx.array(tokens), mx.zeros((1, len(tokenizer.parts))))
        for token in tokenizer.encode('{"a":"a"}'):
            tokens.append(token)
            processor(mx.array(tokens), mx.zeros((1, len(tokenizer.parts))))
        for eos in tokenizer.eos_token_ids:
            logits = processor(mx.array(tokens + [eos]), mx.zeros((1, len(tokenizer.parts))))
            for allowed_eos in tokenizer.eos_token_ids:
                self.assertFalse(bool(mx.isneginf(logits[0, allowed_eos])))
            self.assertTrue(bool(mx.isneginf(logits[0, tokenizer.parts.index("a")])))
        self.assertEqual(stats["parser_failures"], 0)

    def test_eos_prefetch_exception_does_not_allow_either_marker_in_an_open_string(self):
        import mlx.core as mx

        tokenizer = ByteTokenizer()
        processor, _ = make_json_processor(tokenizer, SCHEMA)
        tokens = [0]
        processor(mx.array(tokens), mx.zeros((1, len(tokenizer.parts))))
        for token in tokenizer.encode('{"a":"a'):
            tokens.append(token)
            processor(mx.array(tokens), mx.zeros((1, len(tokenizer.parts))))
        for eos in tokenizer.eos_token_ids:
            with self.assertRaisesRegex(RuntimeError, "disallowed token"):
                processor(mx.array(tokens + [eos]), mx.zeros((1, len(tokenizer.parts))))

    def test_unescaped_control_character_cannot_empty_the_parser_stack(self):
        import mlx.core as mx

        tokenizer = ByteTokenizer()
        processor, stats = make_json_processor(tokenizer, SCHEMA)
        tokens = [0]
        processor(mx.array(tokens), mx.zeros((1, len(tokenizer.parts))))
        for token in tokenizer.encode('{"a":"'):
            tokens.append(token)
            processor(mx.array(tokens), mx.zeros((1, len(tokenizer.parts))))
        tokens.append(tokenizer.parts.index("\t"))
        with self.assertRaisesRegex(RuntimeError, "disallowed (character|token)"):
            processor(mx.array(tokens), mx.zeros((1, len(tokenizer.parts))))
        self.assertEqual(stats["parser_failures"], 1)

    def test_internal_constraint_error_does_not_log_prefix_or_fall_back_to_eos(self):
        import mlx.core as mx
        from lmformatenforcer import TokenEnforcer

        tokenizer = ByteTokenizer()
        processor, stats = make_json_processor(tokenizer, SCHEMA)
        with mock.patch.object(
            TokenEnforcer, "_collect_allowed_tokens", side_effect=ValueError("synthetic-private-marker")
        ):
            with mock.patch("lmformatenforcer.tokenenforcer.logging.exception") as log:
                with self.assertRaisesRegex(RuntimeError, r"computation failed \(ValueError\)") as error:
                    processor(mx.array([0]), mx.zeros((1, len(tokenizer.parts))))
                log.assert_not_called()
                self.assertNotIn("synthetic-private-marker", str(error.exception))
        self.assertEqual(stats["parser_failures"], 1)

    def test_eos_requires_complete_decoded_json_even_if_parser_reports_done(self):
        import mlx.core as mx
        from lmformatenforcer import JsonSchemaParser

        tokenizer = ByteTokenizer()
        processor, stats = make_json_processor(tokenizer, SCHEMA)
        with mock.patch.object(JsonSchemaParser, "can_end", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "EOS before a complete value"):
                processor(mx.array([0]), mx.zeros((1, len(tokenizer.parts))))
        self.assertEqual(stats["parser_failures"], 1)

    def test_opt_in_diagnostics_keep_only_ids_and_static_error_in_a_private_file(self):
        import mlx.core as mx

        tokenizer = ByteTokenizer()
        processor, _ = make_json_processor(tokenizer, SCHEMA)
        processor(mx.array([0]), mx.zeros((1, len(tokenizer.parts))))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "synthetic-diagnostic.jsonl"
            with mock.patch.dict(os.environ, {"CHATLENS_JSON_DIAGNOSTICS": str(destination)}):
                with self.assertRaisesRegex(RuntimeError, "disallowed token"):
                    processor(mx.array([0, 6]), mx.zeros((1, len(tokenizer.parts))))
            record = json.loads(destination.read_text())
            self.assertEqual(record["generated_token_ids"], [6])
            self.assertEqual(record["error_type"], "RuntimeError")
            self.assertEqual(record["error_reason"], "JSON generation received a disallowed token")
            self.assertEqual(set(record), {"generated_token_ids", "error_type", "error_reason"})
            self.assertEqual(destination.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
