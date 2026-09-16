"""Baseline construction for integrated gradients on images.

Integrated gradients attributes a prediction by integrating gradients along a
straight line from a **baseline** to the input. The baseline is supposed to
represent "absence of signal", and the usual default is an all-black image.

That default causes the artefacts this module exists to fix. A straight line
from black to a real photograph passes through a sequence of images that are
uniformly dark, washed out, and unlike anything in the training distribution.
The model's behaviour on those intermediate images is not really evidence about
the input — it is evidence about how the network responds to off-manifold input.
Two symptoms show up in the diagnostic curves:

- **non-monotonicity** in ``P(target class)`` along the path: the probability
  wanders up and down instead of rising cleanly toward the true prediction.
- **saturation**: gradients collapse to ~0 over much of the path, so most of the
  integration range contributes nothing and the attribution is decided by a
  narrow band of alpha.

Every baseline here is an attempt to keep the path closer to plausible images.

| kind | path start | keeps path in-distribution? |
| --- | --- | --- |
| ``black`` | all zeros | no — the original artefact |
| ``random_noise`` | uniform noise | no, but breaks black's structural bias |
| ``real_image`` | another real photo | partly — endpoints are real, midpoints are blends |
| ``blurred`` | Gaussian-blurred *input* | yes — every step is a plausible photo |

``blurred`` is the strongest of the four precisely because interpolating from a
blurred version of the same image to its sharp original only ever varies detail,
never content. Nothing along that path is an image the world could not produce.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

BASELINE_KINDS = ("black", "random_noise", "real_image", "blurred")


def gaussian_kernel_1d(sigma: float, radius: int | None = None) -> tf.Tensor:
    """1D Gaussian kernel, normalised to sum to 1."""
    if radius is None:
        radius = max(1, int(3.0 * sigma))
    x = tf.range(-radius, radius + 1, dtype=tf.float32)
    k = tf.exp(-(x**2) / (2.0 * sigma**2))
    return k / tf.reduce_sum(k)


def gaussian_blur_image(image: tf.Tensor, sigma: float = 12.0) -> tf.Tensor:
    """Blur an ``(H, W, 3)`` image with a separable Gaussian.

    Implemented as two depthwise 1D convolutions rather than one 2D pass — same
    result, far cheaper. ``tensorflow_addons`` would have provided this but is
    deprecated, so it is written out here.

    ``sigma`` is deliberately large by default: the point of the blurred baseline
    is to destroy *object detail* while preserving colour layout, so the model
    genuinely cannot identify the class at alpha=0.
    """
    k = gaussian_kernel_1d(sigma)
    n = tf.size(k)
    channels = image.shape[-1]

    k_h = tf.reshape(k, [1, n, 1, 1])
    k_h = tf.tile(k_h, [1, 1, channels, 1])
    k_v = tf.reshape(k, [n, 1, 1, 1])
    k_v = tf.tile(k_v, [1, 1, channels, 1])

    x = tf.expand_dims(image, 0)
    x = tf.nn.depthwise_conv2d(x, k_h, strides=[1, 1, 1, 1], padding="SAME")
    x = tf.nn.depthwise_conv2d(x, k_v, strides=[1, 1, 1, 1], padding="SAME")
    return tf.squeeze(x, 0)


def make_baseline(
    kind: str,
    image: tf.Tensor,
    pool: tf.Tensor | None = None,
    seed: int | None = None,
    blur_sigma: float = 12.0,
    pool_index: int | None = None,
) -> tf.Tensor:
    """Build a baseline of the given ``kind``, shaped like ``image``.

    ``pool`` is a ``(N, H, W, 3)`` tensor of real images, required for
    ``real_image``. ``pool_index`` picks a specific one; otherwise one is drawn
    at random using ``seed``.
    """
    if kind == "black":
        return tf.zeros_like(image)

    if kind == "random_noise":
        return tf.random.uniform(tf.shape(image), minval=0.0, maxval=1.0, seed=seed)

    if kind == "real_image":
        if pool is None or len(pool) == 0:
            raise ValueError("kind='real_image' needs a non-empty `pool`")
        if pool_index is None:
            rng = np.random.default_rng(seed)
            pool_index = int(rng.integers(0, len(pool)))
        return tf.convert_to_tensor(pool[pool_index])

    if kind == "blurred":
        return gaussian_blur_image(image, sigma=blur_sigma)

    raise ValueError(f"unknown baseline kind {kind!r}; expected one of {BASELINE_KINDS}")
