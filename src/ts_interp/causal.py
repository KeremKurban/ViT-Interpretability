"""Causal discovery over multivariate time series.

This module is a different axis from everything else in the package, and the
distinction is the whole point:

- Attribution (``attribution.py``, ``dynamask.py``) explains **a model**. It
  answers "which inputs did this network weigh?" If the model learned a
  confounded shortcut, a faithful attribution will faithfully report the
  shortcut. Attribution cannot tell you the shortcut is not causal.
- Causal discovery explains **the data-generating process**. It asks which
  variables actually drive which others, independent of any predictor.

For wearables this is the difference between "the model looked at the
accelerometer channel" and "movement drives heart-rate change". Only the second
survives a change in deployment conditions, and only the second supports an
intervention.

Two methods are implemented, both linear and lag-based:

- ``granger_causality`` — pairwise Granger F-test. Does adding X's past improve
  prediction of Y's future beyond Y's own past?
- ``conditional_granger`` — the same test but conditioning on *all* other
  variables' pasts. This is the key idea PCMCI builds on: it is what
  distinguishes a direct link from one induced by a common driver.

Both are linear-VAR based, so they detect linear lagged dependence only, and
both assume causal sufficiency (no unobserved confounders). For real work use
``tigramite`` (the authors' PCMCI implementation), which adds nonlinear
independence tests and proper multiple-testing control across lags.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CausalDataset:
    x: np.ndarray  # (n_vars, n_timesteps)
    var_names: tuple[str, ...]
    true_edges: tuple[tuple[str, str, int], ...]  # (cause, effect, lag)


def make_causal_dataset(
    n_timesteps: int = 2000,
    noise_std: float = 0.3,
    seed: int = 0,
) -> CausalDataset:
    """Simulated wearable variables with a known lagged causal graph.

    Ground-truth structure::

        activity ->(lag 2) heart_rate
        activity ->(lag 4) eda
        heart_rate ->(lag 1) hrv          (negative coefficient)
        sleep_debt: an independent slow driver of activity (lag 3)

    Note the trap deliberately built in: ``heart_rate`` and ``eda`` are both
    driven by ``activity``, so they are strongly correlated with each other with
    no direct link between them. Pairwise Granger will report a spurious
    ``heart_rate -> eda`` edge; conditioning on ``activity`` is what removes it.
    That contrast is the reason this dataset exists.
    """
    rng = np.random.default_rng(seed)
    n_vars = 5
    names = ("sleep_debt", "activity", "heart_rate", "hrv", "eda")
    burn_in = 50
    total = n_timesteps + burn_in
    x = np.zeros((n_vars, total))

    # index: 0 sleep_debt, 1 activity, 2 heart_rate, 3 hrv, 4 eda
    x[0] = np.cumsum(rng.normal(0, 0.05, total))  # slow random-walk drift
    x[0] = (x[0] - x[0].mean()) / (x[0].std() + 1e-8)

    for t in range(5, total):
        x[1, t] = 0.5 * x[1, t - 1] + 0.6 * x[0, t - 3] + rng.normal(0, noise_std)
        x[2, t] = 0.4 * x[2, t - 1] + 0.8 * x[1, t - 2] + rng.normal(0, noise_std)
        x[3, t] = 0.3 * x[3, t - 1] - 0.7 * x[2, t - 1] + rng.normal(0, noise_std)
        x[4, t] = 0.5 * x[4, t - 1] + 0.5 * x[1, t - 4] + rng.normal(0, noise_std)

    true_edges = (
        ("sleep_debt", "activity", 3),
        ("activity", "heart_rate", 2),
        ("heart_rate", "hrv", 1),
        ("activity", "eda", 4),
    )
    return CausalDataset(x=x[:, burn_in:], var_names=names, true_edges=true_edges)


def _lag_matrix(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Stack lags 1..max_lag of a 1D series as columns, aligned to t."""
    n = len(series)
    return np.column_stack([series[max_lag - k : n - k] for k in range(1, max_lag + 1)])


def _ols_rss(design: np.ndarray, y: np.ndarray) -> float:
    """Residual sum of squares of an OLS fit with intercept."""
    design = np.column_stack([np.ones(len(design)), design])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return float(resid @ resid)


def _f_test(rss_restricted: float, rss_full: float, n: int, p_full: int, p_diff: int):
    """F statistic and an approximate p-value for nested OLS models."""
    from scipy import stats

    df_resid = n - p_full - 1
    if df_resid <= 0 or rss_full <= 0 or p_diff <= 0:
        return float("nan"), float("nan")
    f = ((rss_restricted - rss_full) / p_diff) / (rss_full / df_resid)
    p = 1.0 - stats.f.cdf(f, p_diff, df_resid)
    return float(f), float(p)


def granger_causality(
    x: np.ndarray, cause: int, effect: int, max_lag: int = 5
) -> tuple[float, float]:
    """Pairwise Granger test: does ``cause``'s past improve prediction of ``effect``?

    Returns ``(F statistic, p-value)``. Small p => evidence of a directed link,
    *conditional on nothing else* — which is exactly why it reports edges that
    are really common-cause artefacts.
    """
    max_lag = int(max_lag)
    y = x[effect, max_lag:]
    own = _lag_matrix(x[effect], max_lag)
    other = _lag_matrix(x[cause], max_lag)

    rss_r = _ols_rss(own, y)
    rss_f = _ols_rss(np.column_stack([own, other]), y)
    return _f_test(rss_r, rss_f, len(y), own.shape[1] + other.shape[1], other.shape[1])


def conditional_granger(
    x: np.ndarray, cause: int, effect: int, max_lag: int = 5
) -> tuple[float, float]:
    """Granger test conditioning on every other variable's past (PCMCI-style).

    The restricted model already contains the lagged history of all variables
    except ``cause``; the full model adds ``cause``. A link that survives this
    is not explained by any other observed variable's past, which is what makes
    it a candidate *direct* edge rather than a common-cause artefact.
    """
    max_lag = int(max_lag)
    n_vars = x.shape[0]
    y = x[effect, max_lag:]

    conditioning = [
        _lag_matrix(x[v], max_lag) for v in range(n_vars) if v != cause
    ]
    restricted = np.column_stack(conditioning)
    cause_lags = _lag_matrix(x[cause], max_lag)
    full = np.column_stack([restricted, cause_lags])

    rss_r = _ols_rss(restricted, y)
    rss_f = _ols_rss(full, y)
    return _f_test(rss_r, rss_f, len(y), full.shape[1], cause_lags.shape[1])


def discover_graph(
    ds: CausalDataset, max_lag: int = 5, alpha: float = 0.01, conditional: bool = True
) -> np.ndarray:
    """Run the chosen test over all ordered variable pairs.

    Returns a ``(n_vars, n_vars)`` boolean adjacency where ``[i, j]`` means
    "i causes j" at the given significance level.
    """
    test = conditional_granger if conditional else granger_causality
    n_vars = ds.x.shape[0]
    adj = np.zeros((n_vars, n_vars), dtype=bool)

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            _, p = test(ds.x, i, j, max_lag=max_lag)
            adj[i, j] = bool(p < alpha)
    return adj


def score_graph(adj: np.ndarray, ds: CausalDataset) -> dict:
    """Precision / recall of a discovered adjacency against the true edge set."""
    name_to_idx = {n: i for i, n in enumerate(ds.var_names)}
    truth = np.zeros_like(adj, dtype=bool)
    for cause, effect, _lag in ds.true_edges:
        truth[name_to_idx[cause], name_to_idx[effect]] = True

    tp = int((adj & truth).sum())
    fp = int((adj & ~truth).sum())
    fn = int((~adj & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
