from __future__ import annotations

import unittest

from src.eval_framework import (
    is_better_checkpoint,
    magnitude_pair_bucket,
    milestone_gate_summary,
    parse_patch_text,
    parse_patch_text_with_bounds,
    pilot_bar_summary,
    sign_pattern,
)
from train import deterministic_subset


class TestEvalFramework(unittest.TestCase):
    def test_parse_patch_text_accepts_valid(self) -> None:
        self.assertEqual(parse_patch_text("A=17;B=-3\n"), (17, -3))
        self.assertEqual(parse_patch_text("  A = 0 ; B = 42  \n"), (0, 42))

    def test_parse_patch_text_rejects_invalid(self) -> None:
        self.assertIsNone(parse_patch_text("17+3"))
        self.assertIsNone(parse_patch_text("A=1 B=2"))
        self.assertIsNone(parse_patch_text("A=foo;B=2"))
        self.assertIsNone(parse_patch_text("A=9223372036854775808;B=1"))
        self.assertIsNone(parse_patch_text("A=1;B=-9223372036854775809"))

    def test_parse_patch_text_with_bounds(self) -> None:
        self.assertEqual(
            parse_patch_text_with_bounds("A=-10;B=42\n", min_int=-100, max_int=100),
            (-10, 42),
        )
        self.assertIsNone(parse_patch_text_with_bounds("A=-101;B=42\n", min_int=-100, max_int=100))
        self.assertIsNone(parse_patch_text_with_bounds("A=-10;B=101\n", min_int=-100, max_int=100))

    def test_magnitude_pair_bucket(self) -> None:
        self.assertEqual(magnitude_pair_bucket("core_magnitude", "core_magnitude"), "core/core")
        self.assertEqual(magnitude_pair_bucket("core_magnitude", "heldout_magnitude"), "core/heldout")
        self.assertEqual(magnitude_pair_bucket("heldout_magnitude", "core_magnitude"), "heldout/core")
        self.assertEqual(magnitude_pair_bucket("heldout_magnitude", "heldout_magnitude"), "heldout/heldout")

    def test_sign_pattern(self) -> None:
        self.assertEqual(sign_pattern(1, 2), "++")
        self.assertEqual(sign_pattern(-1, 2), "-+")
        self.assertEqual(sign_pattern(1, -2), "+-")
        self.assertEqual(sign_pattern(-1, -2), "--")

    def test_checkpoint_tie_break(self) -> None:
        best = {"answer_accuracy": 0.50, "exec_success_rate": 0.70, "val_loss": 1.0}

        better_answer = {"answer_accuracy": 0.51, "exec_success_rate": 0.60, "val_loss": 10.0}
        self.assertTrue(is_better_checkpoint(better_answer, best))

        better_exec = {"answer_accuracy": 0.50, "exec_success_rate": 0.71, "val_loss": 10.0}
        self.assertTrue(is_better_checkpoint(better_exec, best))

        better_loss = {"answer_accuracy": 0.50, "exec_success_rate": 0.70, "val_loss": 0.9}
        self.assertTrue(is_better_checkpoint(better_loss, best))

        worse = {"answer_accuracy": 0.49, "exec_success_rate": 1.0, "val_loss": 0.0}
        self.assertFalse(is_better_checkpoint(worse, best))

    def test_pilot_bar_summary(self) -> None:
        metrics = {
            "answer_accuracy": 0.6,
            "exec_success_rate": 0.75,
        }
        slices = {
            "template": {
                "t1": {"count": 10, "answer_accuracy": 0.7},
                "t2": {"count": 5, "answer_accuracy": 0.2},
            },
            "magnitude_pair": {
                "core/core": {"count": 10, "answer_accuracy": 0.6},
            },
        }
        summary = pilot_bar_summary(metrics, slices)
        self.assertTrue(summary["pilot_bar_pass"])

        collapse_slices = {
            "template": {
                "t1": {"count": 10, "answer_accuracy": 0.7},
                "t2": {"count": 3, "answer_accuracy": 0.0},
            }
        }
        summary2 = pilot_bar_summary(metrics, collapse_slices)
        self.assertFalse(summary2["pilot_bar_pass"])
        self.assertTrue(summary2["catastrophic_slice_collapse"])

    def test_deterministic_subset_is_seed_stable(self) -> None:
        rows = [{"id": i} for i in range(100)]
        s1 = deterministic_subset(rows, max_samples=10, seed=123)
        s2 = deterministic_subset(rows, max_samples=10, seed=123)
        self.assertEqual([r["id"] for r in s1], [r["id"] for r in s2])

    def test_milestone_gate_summary(self) -> None:
        metrics = {
            "answer_accuracy": 0.8,
            "exec_success_rate": 0.75,
            "parse_success_rate": 0.97,
        }
        slices = {
            "template": {
                "t1": {"count": 10, "answer_accuracy": 0.8},
            },
            "magnitude_pair": {
                "heldout/heldout": {"count": 3, "answer_accuracy": 0.6},
            },
        }
        gates = milestone_gate_summary(
            metrics,
            slices,
            train_loss_start=2.0,
            train_loss_end=1.0,
        )
        self.assertFalse(gates["g0_loop_sanity"]["pass"])
        self.assertTrue(gates["g1_pilot_success"]["pass"])
        self.assertTrue(gates["g2_rl_ready"]["pass"])


if __name__ == "__main__":
    unittest.main()
