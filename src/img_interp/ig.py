"""Integrated gradients with a parameterised baseline, plus Expected Gradients.

The core maths is unchanged from the TensorFlow tutorial this repo started from
(`notebooks/images/integrated_gradients_vit.ipynb`): interpolate along a
straight line, take gradients of the target-class probability at each step,
average them with the trapezoidal rule, and scale by ``(image - baseline)``.

What changed is that the baseline is an argument rather than a hardcoded black
image, and the per-path diagnostics are returned as data instead of being
plotted inline — so several baselines can be compared on the same axes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .baselines import make_baseline


@dataclass
class PathDiagnostics:
    """The two curves the tutorial plots, kept as arrays so they can be compared.

    ``probs`` is ``P(target class)`` at each alpha; ``mean_abs_grads`` is the
    mean absolute pixel gradient at each alpha. Saturation shows up as
    ``mean_abs_grads`` collapsing to ~0; path artefacts show up as ``probs``
    failing to rise monotonically.
    """

    alphas: np.ndarray
    probs: np.ndarray
    mean_abs_grads: np.ndarray


def interpolate_images(baseline: tf.Tensor, image: tf.Tensor, alphas: tf.Tensor) -> tf.Tensor:
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    return baseline_x + alphas_x * (input_x - baseline_x)


def compute_gradients(model, images: tf.Tensor, target_class_idx: int) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


def integral_approximation(gradients: tf.Tensor) -> tf.Tensor:
    """Trapezoidal Riemann sum over the path."""
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    return tf.math.reduce_mean(grads, axis=0)


def _path_batches(model, baseline, image, alphas, target_class_idx, batch_size):
    """Yield gradients and probabilities along the path, batched over alpha."""
    for start in tf.range(0, len(alphas), batch_size):
        stop = tf.minimum(start + batch_size, len(alphas))
        alpha_batch = alphas[start:stop]
        interpolated = interpolate_images(baseline, image, alpha_batch)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            logits = model(interpolated)
            probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
        yield tape.gradient(probs, interpolated), probs


def integrated_gradients(
    model,
    image: tf.Tensor,
    target_class_idx: int,
    baseline: tf.Tensor,
    m_steps: int = 50,
    batch_size: int = 32,
    return_diagnostics: bool = False,
):
    """IG attribution for one image against one baseline.

    Returns the ``(H, W, 3)`` attribution, or ``(attribution, PathDiagnostics)``
    when ``return_diagnostics`` is set. Computing the diagnostics is free — the
    probabilities come from the same forward passes the gradients need — so the
    only reason not to ask for them is to keep the return type simple.
    """
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)

    grad_batches, prob_batches = [], []
    for grads, probs in _path_batches(
        model, baseline, image, alphas, target_class_idx, batch_size
    ):
        grad_batches.append(grads)
        prob_batches.append(probs)

    total_gradients = tf.concat(grad_batches, axis=0)
    avg_gradients = integral_approximation(total_gradients)
    attribution = (image - baseline) * avg_gradients

    if not return_diagnostics:
        return attribution

    diagnostics = PathDiagnostics(
        alphas=alphas.numpy(),
        probs=tf.concat(prob_batches, axis=0).numpy(),
        mean_abs_grads=tf.reduce_mean(
            tf.abs(total_gradients), axis=[1, 2, 3]
        ).numpy(),
    )
    return attribution, diagnostics


def completeness_error(
    model, image: tf.Tensor, baseline: tf.Tensor, target_class_idx: int, attribution: tf.Tensor
) -> float:
    """How far the attribution is from satisfying IG's completeness axiom.

    IG is supposed to satisfy ``sum(attributions) == F(image) - F(baseline)``.
    The gap is a direct measure of whether ``m_steps`` was large enough to
    approximate the integral — a large gap means the Riemann sum has not
    converged, which is itself a symptom of a badly-behaved path.

    Returned as an absolute difference in probability units.
    """
    batch = tf.stack([image, baseline], axis=0)
    probs = tf.nn.softmax(model(batch), axis=-1)[:, target_class_idx]
    delta = float(probs[0] - probs[1])
    return abs(float(tf.reduce_sum(attribution)) - delta)


def expected_gradients(
    model,
    image: tf.Tensor,
    target_class_idx: int,
    pool: tf.Tensor,
    n_baselines: int = 25,
    m_steps: int = 50,
    batch_size: int = 32,
    seed: int = 0,
    progress: bool = False,
):
    """Expected Gradients: average IG over many real-image baselines.

    Rather than trusting one arbitrary baseline, sample ``n_baselines`` real
    images and average the resulting attributions. The motivation is that any
    single baseline injects its own structure into the explanation — a black
    baseline cannot assign importance to genuinely black pixels, because
    ``(image - baseline)`` is zero there. Averaging over many baselines makes
    that bias wash out instead of accumulating.

    Returns ``(mean_attribution, diagnostics_list)`` where the list holds one
    `PathDiagnostics` per baseline, so the spread across runs can be plotted as
    a band rather than a single curve.

    **Fidelity note.** This runs a *full* IG path per baseline and averages the
    results. Erion et al.'s Expected Gradients instead treats the baseline and
    the interpolation coefficient as a joint expectation and samples one
    ``(baseline, alpha)`` pair at a time, which converges to the same quantity
    far more cheaply. The variant here costs ``n_baselines`` complete paths, but
    it keeps every individual path's diagnostics intact — which is the point of
    this notebook, since the comparison is about path behaviour, not just the
    final attribution map.
    """
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pool), size=n_baselines)

    attributions, diagnostics = [], []
    for n, idx in enumerate(indices):
        if progress:
            print(f"  baseline {n + 1}/{n_baselines}", end="\r")
        baseline = make_baseline("real_image", image, pool=pool, pool_index=int(idx))
        attr, diag = integrated_gradients(
            model,
            image,
            target_class_idx,
            baseline=baseline,
            m_steps=m_steps,
            batch_size=batch_size,
            return_diagnostics=True,
        )
        attributions.append(attr)
        diagnostics.append(diag)

    mean_attribution = tf.reduce_mean(tf.stack(attributions, axis=0), axis=0)
    return mean_attribution, diagnostics


def stack_diagnostics(diagnostics: list[PathDiagnostics]):
    """Collapse many `PathDiagnostics` into mean/std bands for plotting."""
    probs = np.stack([d.probs for d in diagnostics], axis=0)
    grads = np.stack([d.mean_abs_grads for d in diagnostics], axis=0)
    return {
        "alphas": diagnostics[0].alphas,
        "probs_mean": probs.mean(axis=0),
        "probs_std": probs.std(axis=0),
        "grads_mean": grads.mean(axis=0),
        "grads_std": grads.std(axis=0),
    }
