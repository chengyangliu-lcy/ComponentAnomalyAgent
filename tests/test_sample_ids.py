from __future__ import annotations

from pathlib import Path
import unittest
from uuid import uuid4

from tools.sample_ids import filter_items_by_sample_ids, read_sample_ids_file


class SampleIdTests(unittest.TestCase):
    def test_read_sample_ids_file_dedupes_and_skips_comments(self) -> None:
        path = Path("outputs") / "test_sample_ids" / f"{uuid4().hex}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# comment\nb\n\na\nb\n", encoding="utf-8")

        self.assertEqual(read_sample_ids_file(path), ["b", "a"])

    def test_filter_items_by_sample_ids_uses_file_order(self) -> None:
        items = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

        filtered = filter_items_by_sample_ids(items, ["c", "a"], lambda item: item["id"])

        self.assertEqual(filtered, [{"id": "c"}, {"id": "a"}])


if __name__ == "__main__":
    unittest.main()
