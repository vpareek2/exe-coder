from __future__ import annotations

import re
from typing import Any

PATCH_REGEX = re.compile(r"\s*A\s*=\s*(-?\d+)\s*;\s*B\s*=\s*(-?\d+)\s*\n?\s*$")
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def parse_patch_text(text: str) -> tuple[int, int] | None:
    match = PATCH_REGEX.match(text)
    if not match:
        return None
    a = int(match.group(1))
    b = int(match.group(2))
    if not (INT64_MIN <= a <= INT64_MAX):
        return None
    if not (INT64_MIN <= b <= INT64_MAX):
        return None
    return a, b


def parse_patch_text_with_bounds(text: str, *, min_int: int, max_int: int) -> tuple[int, int] | None:
    parsed = parse_patch_text(text)
    if parsed is None:
        return None
    a, b = parsed
    if not (int(min_int) <= a <= int(max_int)):
        return None
    if not (int(min_int) <= b <= int(max_int)):
        return None
    return a, b


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


def _has_catastrophic_slice_collapse(slices: dict[str, Any], *, answer_accuracy: float) -> bool:
    if answer_accuracy <= 0.0:
        return False
    for family in slices.values():
        if not isinstance(family, dict):
            continue
        for entry in family.values():
            if not isinstance(entry, dict):
                continue
            count = int(entry.get("count", 0))
            acc = float(entry.get("answer_accuracy", 0.0))
            if count > 0 and acc == 0.0:
                return True
    return False


def pilot_bar_summary(metrics: dict[str, Any], slices: dict[str, Any]) -> dict[str, Any]:
    answer_accuracy = float(metrics.get("answer_accuracy", 0.0))
    exec_success_rate = float(metrics.get("exec_success_rate", 0.0))

    catastrophic_slice_collapse = _has_catastrophic_slice_collapse(slices, answer_accuracy=answer_accuracy)

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


def milestone_gate_summary(
    metrics: dict[str, Any],
    slices: dict[str, Any],
    *,
    train_loss_start: float | None = None,
    train_loss_end: float | None = None,
) -> dict[str, Any]:
    answer_accuracy = float(metrics.get("answer_accuracy", 0.0))
    exec_success_rate = float(metrics.get("exec_success_rate", 0.0))
    parse_success_rate = float(metrics.get("parse_success_rate", 0.0))
    catastrophic_slice_collapse = _has_catastrophic_slice_collapse(slices, answer_accuracy=answer_accuracy)

    g0_loss_decreased = (
        train_loss_start is not None
        and train_loss_end is not None
        and float(train_loss_end) < float(train_loss_start)
    )
    g0_parse_ok = parse_success_rate >= 0.95
    g0_answer_ok = answer_accuracy >= 0.90
    g0_pass = bool(g0_loss_decreased and g0_parse_ok and g0_answer_ok)

    g1_answer_ok = answer_accuracy >= 0.50
    g1_exec_ok = exec_success_rate >= 0.70
    g1_pass = bool(g1_answer_ok and g1_exec_ok and not catastrophic_slice_collapse)

    heldout_heldout = (
        slices.get("magnitude_pair", {}).get("heldout/heldout", {})
        if isinstance(slices.get("magnitude_pair"), dict)
        else {}
    )
    heldout_acc = (
        float(heldout_heldout.get("answer_accuracy"))
        if isinstance(heldout_heldout, dict) and heldout_heldout.get("answer_accuracy") is not None
        else None
    )

    g2_answer_ok = answer_accuracy >= 0.75
    g2_parse_ok = parse_success_rate >= 0.95
    g2_heldout_ok = heldout_acc is not None and heldout_acc >= 0.55
    g2_pass = bool(g2_answer_ok and g2_parse_ok and g2_heldout_ok)

    return {
        "g0_loop_sanity": {
            "thresholds": {
                "require_train_loss_decrease": True,
                "parse_success_rate_min": 0.95,
                "answer_accuracy_min": 0.90,
            },
            "observed": {
                "train_loss_start": train_loss_start,
                "train_loss_end": train_loss_end,
                "parse_success_rate": parse_success_rate,
                "answer_accuracy": answer_accuracy,
            },
            "checks": {
                "train_loss_decreased": bool(g0_loss_decreased),
                "parse_success_rate_ok": bool(g0_parse_ok),
                "answer_accuracy_ok": bool(g0_answer_ok),
            },
            "pass": g0_pass,
        },
        "g1_pilot_success": {
            "thresholds": {
                "answer_accuracy_min": 0.50,
                "exec_success_rate_min": 0.70,
                "require_no_catastrophic_slice_collapse": True,
            },
            "observed": {
                "answer_accuracy": answer_accuracy,
                "exec_success_rate": exec_success_rate,
                "catastrophic_slice_collapse": catastrophic_slice_collapse,
            },
            "checks": {
                "answer_accuracy_ok": bool(g1_answer_ok),
                "exec_success_rate_ok": bool(g1_exec_ok),
                "no_catastrophic_slice_collapse": bool(not catastrophic_slice_collapse),
            },
            "pass": g1_pass,
        },
        "g2_rl_ready": {
            "thresholds": {
                "answer_accuracy_min": 0.75,
                "parse_success_rate_min": 0.95,
                "heldout_heldout_answer_accuracy_min": 0.55,
            },
            "observed": {
                "answer_accuracy": answer_accuracy,
                "parse_success_rate": parse_success_rate,
                "heldout_heldout_answer_accuracy": heldout_acc,
            },
            "checks": {
                "answer_accuracy_ok": bool(g2_answer_ok),
                "parse_success_rate_ok": bool(g2_parse_ok),
                "heldout_heldout_answer_accuracy_ok": bool(g2_heldout_ok),
            },
            "pass": g2_pass,
        },
    }
