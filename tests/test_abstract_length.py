import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_abstract_length import abstract_body, word_count  # noqa: E402


class AbstractLengthTests(unittest.TestCase):
    def test_title_and_repository_notice_are_not_counted(self):
        markdown = "# Title\n\n> Status notice\n> continuation\n\nOne two three.\n"
        self.assertEqual(abstract_body(markdown), "One two three.")
        self.assertEqual(word_count(markdown), 3)

    def test_repository_abstract_is_within_aan_limit(self):
        text = (ROOT / "ABSTRACT.md").read_text(encoding="utf-8")
        self.assertLessEqual(word_count(text), 300)


if __name__ == "__main__":
    unittest.main()
