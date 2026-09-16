"""Descriptive summaries of the IG path curves, for comparing baselines.

These are **descriptions of the diagnostic plots**, not an evaluation metric for
attribution quality. They exist so "did the artefact improve?" can be answered
by looking at a table instead of squinting at eight line charts. Nothing here
scores whether an attribution map is *correct* — there is no ground truth for
that on real photographs, which is exactly why the time-series side of this repo
carries the quantitative comparisons.

Two artefacts get summarised, matching the two failure modes described in
`baselines.py`:

- **non-monotonicity** of ``P(target class)`` along the path
- **saturation** of the pixel gradients along the path
"""
from __future__ import annotations

import numpy as np


def decreasing_step_fraction(probs: np.ndarray) -> float:
    """Fraction of path steps where ``P(target)`` moves downward — how *often*.

    Counts steps without regard to size, so it is sensitive to numerical jitter:
    a curve that is flat with tiny noise registers ~50% here while going nowhere.
    Read it alongside `backtrack_fraction`, which measures how *much*. Large
    values in both means genuine wandering; large here but small there means the
    curve is merely noisy.
    """
    d = np.diff(probs)
    if len(d) == 0:
        return 0.0
    return float((d < 0).mean())


def backtrack_fraction(probs: np.ndarray) -> float:
    """Total downward movement as a fraction of the curve's net rise — how *much*.

    0.0 means perfectly monotone. 0.2 means the curve gave back a fifth of its
    net rise along the way. Above ~0.5 the path is overshooting badly: the model
    became more confident somewhere in the middle of the interpolation than it is
    about the actual input, which means the attribution is integrating gradients
    through states that are not on any sensible route to the input.

    This is the magnitude-based counterpart to `decreasing_step_fraction`, and it
    is the one to trust when the two disagree. (The total-variation ratio
    ``sum|step| / net`` is just ``1 + 2 * backtrack_fraction``, so it is not
    reported separately.)
    """
    d = np.diff(probs)
    net = float(probs[-1] - probs[0])
    if abs(net) < 1e-8:
        return float("inf")
    # abs() rather than negation: with no decreasing steps the sum is 0.0 and
    # plain negation yields -0.0, which prints as "-0.0%"
    return float(abs(d[d < 0].sum()) / net)


def saturation_fraction(mean_abs_grads: np.ndarray, rel_threshold: float = 0.1) -> float:
    """Fraction of the path whose gradient magnitude is below ``rel_threshold``
    of the path maximum.

    This is the "gradients die" symptom: if most of the alpha range contributes
    almost nothing to the integral, then the attribution is being decided by a
    narrow slice of the path and the rest of the computation is wasted.
    """
    m = np.asarray(mean_abs_grads, dtype=float)
    peak = m.max()
    if peak <= 0:
        return 1.0
    return float((m / peak < rel_threshold).mean())


def summarize_path(diag, label: str = "") -> dict:
    """Roll the two curves of one `PathDiagnostics` into a row of numbers."""
    return {
        "baseline": label,
        "decreasing_steps": decreasing_step_fraction(diag.probs),
        "backtrack": backtrack_fraction(diag.probs),
        "saturation_frac": saturation_fraction(diag.mean_abs_grads),
        "prob_start": float(diag.probs[0]),
        "prob_end": float(diag.probs[-1]),
    }


def format_table(rows: list[dict]) -> str:
    """Plain-text table of summary rows, in the order given."""
    if not rows:
        return "(no rows)"
    header = (
        f"{'baseline':<20} {'backtrack':>10} {'saturated':>10} "
        f"{'dec.steps':>10} {'P(start)':>9} {'P(end)':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        bt = "inf" if not np.isfinite(r["backtrack"]) else f"{r['backtrack']:.1%}"
        lines.append(
            f"{r['baseline']:<20} {bt:>10} {r['saturation_frac']:>9.1%} "
            f"{r['decreasing_steps']:>9.1%} {r['prob_start']:>9.3f} {r['prob_end']:>8.3f}"
        )
    return "\n".join(lines)
