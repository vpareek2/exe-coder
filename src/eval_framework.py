from __future__ import annotations

import re
from typing import Any

PATCH_REGEX = re.compile(r"\s*A\s*=\s*(-?\d+)\s*;\s*B\s*=\s*(-?\d+)\s*\n?\s*$")


def parse_patch_text(text: str) -> tuple[int, int] | None:
    match = PATCH_REGEX.match(text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def magnitude_pair_bucket(a_bucket: str | None, b_bucket: str | None) -> str:
    a = str(a_bucket or "unknown")
    b = str(b_bucket or "unknown")
    if a not in {"core_magnitude", "heldout_magnitude"}:
        a = "unknown"
    if b not in {"core_magnitude", "heldout_magnitude"}:
        b = "unknown"

    if a == "core_magnitude" and b == "core_magnitude":
        return "core/core"
    if a == "core_magnitude" and b == "heldout_magnitude":
        return "core/heldout"
    if a == "heldout_magnitude" and b == "core_magnitude":
        return "heldout/core"
    if a == "heldout_magnitude" and b == "heldout_magnitude":
        return "heldout/heldout"
    return "unknown"


def sign_pattern(a: int | None, b: int | None) -> str:
    if a is None or b is None:
        return "unknown"

    def s(x: int) -> str:
        return "+" if x >= 0 else "-"

    return f"{s(int(a))}{s(int(b))}"


def is_better_checkpoint(candidate: dict[str, float], best: dict[str, float] | None) -> bool:
    """Returns True if candidate beats best by answer_accuracy, then exec_success_rate, then lower val_loss."""
    if best is None:
        return True

    cand_ans = float(candidate["answer_accuracy"])
    best_ans = float(best["answer_accuracy"])
    if cand_ans != best_ans:
        return cand_ans > best_ans

    cand_exec = float(candidate["exec_success_rate"])
    best_exec = float(best["exec_success_rate"])
    if cand_exec != best_exec:
        return cand_exec > best_exec

    cand_loss = float(candidate["val_loss"])
    best_loss = float(best["val_loss"])
    return cand_loss < best_loss


def pilot_bar_summary(metrics: dict[str, Any], slices: dict[str, Any]) -> dict[str, Any]:
    answer_accuracy = float(metrics.get("answer_accuracy", 0.0))
    exec_success_rate = float(metrics.get("exec_success_rate", 0.0))

    catastrophic_slice_collapse = False
    if answer_accuracy > 0.0:
        for family in slices.values():
            if not isinstance(family, dict):
                continue
            for entry in family.values():
                if not isinstance(entry, dict):
                    continue
                count = int(entry.get("count", 0))
                acc = float(entry.get("answer_accuracy", 0.0))
                if count > 0 and acc == 0.0:
                    catastrophic_slice_collapse = True
                    break
            if catastrophic_slice_collapse:
                break

    pilot_bar_pass = (
        answer_accuracy >= 0.50
        and exec_success_rate >= 0.70
        and not catastrophic_slice_collapse
    )

    return {
        "pilot_thresholds": {
            "answer_accuracy": 0.50,
            "exec_success_rate": 0.70,
            "require_no_catastrophic_slice_collapse": True,
        },
        "answer_accuracy": answer_accuracy,
        "exec_success_rate": exec_success_rate,
        "catastrophic_slice_collapse": catastrophic_slice_collapse,
        "pilot_bar_pass": pilot_bar_pass,
    }
