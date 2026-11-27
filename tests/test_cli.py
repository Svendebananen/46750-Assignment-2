import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAIN = REPOSITORY_ROOT / "main.py"


class EntryPointTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(MAIN), *args],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_help_lists_all_question_and_test_entry_points(self):
        result = self.run_cli("--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--q1", "--DP", "--q3", "--unit-tests", "--test"):
            self.assertIn(option, result.stdout)

    def test_question_three_run_list_does_not_require_a_solve(self):
        result = self.run_cli("--q3-list")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DP_BB", result.stdout)
        self.assertIn("SDPx_N5", result.stdout)


if __name__ == "__main__":
    unittest.main()
