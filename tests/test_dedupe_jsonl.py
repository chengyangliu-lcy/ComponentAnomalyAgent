from __future__ import annotations

import unittest

from scripts.dedupe_jsonl import dedupe_rows


class DedupeJsonlTests(unittest.TestCase):
    def test_dedupe_rows_keeps_last_by_default(self) -> None:
        rows = [
            {"sample_id": "a", "value": 1},
            {"sample_id": "b", "value": 2},
            {"sample_id": "a", "value": 3},
        ]

        self.assertEqual(
            dedupe_rows(rows),
            [{"sample_id": "a", "value": 3}, {"sample_id": "b", "value": 2}],
        )

    def test_dedupe_rows_can_keep_first(self) -> None:
        rows = [
            {"sample_id": "a", "value": 1},
            {"sample_id": "a", "value": 3},
        ]

        self.assertEqual(dedupe_rows(rows, keep="first"), [{"sample_id": "a", "value": 1}])


if __name__ == "__main__":
    unittest.main()
