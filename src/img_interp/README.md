# `img_interp` — developer notes

Backs the image notebooks in `notebooks/images/`, the same way
[`ts_interp`](../ts_interp/README.md) backs the time-series ones. The notebooks
call each function in a line or two; this file is the reference for what they do.

**Framework note.** This package is TensorFlow, because the notebook it grew out
of (`integrated_gradients_vit.ipynb`, adapted from the TensorFlow tutorial) uses
a TF-Hub Inception V1. `ts_interp` is PyTorch. The two sides of the repo do not
share code and are not meant to — porting either to match the other would be
churn for no gain, since nothing crosses the boundary.

```
loading.py      model, example images, and the real-image baseline pool
baselines.py    the four baseline types + a separable Gaussian blur
ig.py           integrated gradients with a parameterised baseline, expected gradients
diagnostics.py  descriptive summaries of the path curves
```

Import style used by the notebooks (two directories deep, package not installed):

```python
import sys; sys.path.insert(0, "../../src")
from img_interp import loading, baselines, ig, diagnostics
```

---

## The problem this package exists for

Integrated Gradients integrates gradients along a straight line from a
**baseline** to the input. The baseline is meant to represent "absence of
signal", and the near-universal default is an all-black image.

That default distorts the result in two ways.

**It puts the path off the data manifold.** Interpolating from black to a
photograph produces uniformly dark, washed-out frames that resemble nothing the
network was trained on. Gradients measured there describe how the model reacts to
implausible input, which is not what the explanation claims to be about.

**It makes some pixels unattributable by construction.** IG scales its result by
`(image - baseline)`. Where the image is genuinely black, that factor is exactly
zero, so those pixels receive zero attribution no matter how much the model
relies on them. This is a property of the arithmetic, not a finding about the
image, and it is invisible in the output.

---

## `loading.py`

- `load_model()` — Inception V1 from TF-Hub wrapped as a `tf_keras` model over
  `(224, 224, 3)`. Outputs 1001 logits (ImageNet + background at index 0).
- `load_imagenet_labels()` — the label array.
- `load_example_images()` — the tutorial's Fireboat and Giant Panda photos.
  `EXAMPLE_TARGETS` holds their class indices (555, 389).
- `read_image(path)` — decode, cast to float in [0, 1], pad-resize to 224x224.
- `load_baseline_pool(n_images=40, seed=0)` — `(N, 224, 224, 3)` of real photos,
  from TensorFlow's `flower_photos` archive (~220 MB, cached by
  `tf_keras.utils.get_file` after the first call).

⚠️ **The pool is deliberately not drawn from the class being explained.** A
baseline is a neutral reference point; sampling it from the target class leaks
the answer into the reference and inflates the apparent attribution.

---

## `baselines.py`

`make_baseline(kind, image, pool=None, seed=None, blur_sigma=12.0, pool_index=None)`
returns a baseline shaped like `image`. Four kinds:

| kind | what it is | notes |
| --- | --- | --- |
| `black` | `zeros_like(image)` | the original default, and the source of both problems above |
| `random_noise` | uniform [0,1] noise | still off-manifold, but has no structural bias toward dark pixels, so the `(image - baseline)` blind spot disappears |
| `real_image` | a photo from `pool` | endpoints are real; the **midpoints are double-exposures** and are the most off-manifold of all four |
| `blurred` | Gaussian-blurred copy of the input | the only one where every interpolation step is a plausible photograph |

`gaussian_blur_image(image, sigma=12.0)` — separable Gaussian via two depthwise
1-D convolutions (cheaper than one 2-D pass, same result).
`tensorflow_addons` would have supplied this but is deprecated, hence the
hand-rolled version.

`blur_sigma` defaults high on purpose: the baseline has to actually destroy
object detail, or alpha=0 is already classifiable and there is no path to
integrate along.

---

## `ig.py`

### `integrated_gradients(model, image, target_class_idx, baseline, m_steps=50, batch_size=32, return_diagnostics=False)`

The tutorial's algorithm with the baseline lifted to an argument: interpolate,
take gradients of the target-class **probability** (not the logit) at each step,
average with the trapezoidal rule, scale by `(image - baseline)`.

With `return_diagnostics=True` it also returns a `PathDiagnostics` holding
`alphas`, `probs`, and `mean_abs_grads` — the two curves the tutorial plots.
Collecting them is free, since the probabilities come from the same forward
passes the gradients already require.

### `expected_gradients(model, image, target_class_idx, pool, n_baselines=25, ...)`

Samples `n_baselines` real images, runs the full IG path for each, and averages
the attributions. Returns `(mean_attribution, [PathDiagnostics, ...])` — one
diagnostics object per run, so the spread can be drawn as a band with
`stack_diagnostics`.

Cost is `n_baselines` x the cost of one IG run. With `m_steps=50` and `N=25`
that is ~1275 forward+backward passes at 224x224, a few minutes on CPU.

⚠️ **Fidelity note.** This runs a *full* IG path per baseline and averages.
Erion et al.'s Expected Gradients instead treats the baseline and interpolation
coefficient as a joint expectation, sampling one `(baseline, alpha)` pair at a
time — same quantity in the limit, far cheaper. The variant here is the one the
task called for, and it has a concrete advantage for this notebook: it preserves
each individual path's diagnostics, which is what gets plotted as the ±1 std
band. The cheap sampler would give the attribution map but not the per-path
curves.

### `completeness_error(...)`

IG's completeness axiom says `sum(attributions) == F(image) - F(baseline)`. The
gap is a convergence check on the Riemann sum: a large value means `m_steps` was
too small. Useful as a guard, since an under-converged IG map looks perfectly
plausible and is simply wrong.

---

## `diagnostics.py`

These functions **describe the diagnostic plots**. They are not a quality metric
for attributions, and a better number here does not mean a better explanation.
There is no ground truth on a real photograph — which is precisely why this
repo's quantitative method comparisons live on the time-series side, where the
important region is known by construction.

| Function | Measures |
| --- | --- |
| `backtrack_fraction(probs)` | total downward movement of `P(target)` as a fraction of net rise. 0 = monotone. **The one to trust.** |
| `decreasing_step_fraction(probs)` | fraction of steps that decrease, ignoring size |
| `saturation_fraction(grads, rel_threshold=0.1)` | fraction of the path with gradient magnitude under 10% of peak |
| `summarize_path(diag, label)` / `format_table(rows)` | one row per baseline, rendered as text |

⚠️ **`decreasing_step_fraction` is noise-sensitive and was actively misleading
during development.** A curve that is flat with floating-point jitter registers
~50% decreasing steps while going nowhere, which made the `blurred` baseline look
badly non-monotonic when its real problem was different (it overshoots early,
then descends). `backtrack_fraction` weights by magnitude and does not have this
failure. Both are reported because disagreement between them is itself
informative: high count with low magnitude means noise, high in both means
genuine wandering.

The total-variation ratio `sum|step| / net` is exactly
`1 + 2 * backtrack_fraction`, so it is not reported as a separate column.

---

## Reproducibility

`load_baseline_pool` and `expected_gradients` both take `seed`. The notebook is
committed with outputs, so the numbers in its markdown match the committed cells.
Results are specific to this image/model pair — see the notebook's takeaways for
why the ranking should not be generalised.
