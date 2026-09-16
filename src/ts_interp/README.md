# `ts_interp` — developer notes

The notebooks import from here and use each function in a line or two, which
keeps them readable but hides what the code actually does. This file is the
reference for that.

Nothing here is a published reference implementation. These are compact,
readable reimplementations written to make the *ideas* legible — see
[Fidelity to the papers](#fidelity-to-the-papers) below for exactly where they
diverge, and use the real libraries for real work.

```
data.py           synthetic wearable signals with ground-truth events
model.py          the 1D CNN under interpretation
attribution.py    occlusion, integrated gradients, TSR, scoring metrics
dynamask.py       learned perturbation masks
counterfactual.py nearest-unlike-neighbour counterfactuals
causal.py         Granger / conditional-Granger causal discovery
```

Import style used by the notebooks (they are two directories deep, and the
package is not installed):

```python
import sys; sys.path.insert(0, "../../src")
from ts_interp import data, model, attribution
```

---

## Shape and label conventions

| Thing | Shape | Notes |
| --- | --- | --- |
| Dataset tensor `x` | `(n_samples, n_channels, length)` | float32, PyTorch conv1d layout |
| One sample passed to attribution | `(n_channels, length)` | **no batch dim** — the functions add it |
| Attribution map returned | `(n_channels, length)` | same shape as the sample |
| `*_1d` wrapper output | `(length,)` | single-channel convenience only |

Every attribution function takes **one sample at a time**, not a batch. This is
deliberate — occlusion and TSR are internally loops over perturbations, and
per-sample calls keep the code obvious at the cost of speed.

---

## `data.py`

Two generators, both returning a `TimeSeriesDataset` dataclass
(`x`, `y`, `event_windows`, `event_channels`, `channel_names`).

The signals are PPG-like: a sine fundamental plus a first harmonic, which
approximates the dicrotic-notch shape of a real optical pulse waveform, with
Gaussian noise on top. An "event" is a short high-rate, high-amplitude burst
spliced in — a stand-in for a motion artifact or an arrhythmia-like irregularity.

### `make_dataset(...)` — single channel, binary label

The original generator. `y ∈ {0, 1}`, one channel. Used by
`occlusion_vs_integrated_gradients.ipynb`.

### `make_multivariate_dataset(...)` — three channels, 4-class label

Channels are `(ppg, accel, eda)`. Three parameters carry design decisions worth
understanding before you change them:

**`crosstalk` (default 0.45).** Leaks an attenuated copy of the burst into the
non-primary channels. Real wearables do this — a wrist motion artifact appears
in the accelerometer *and* pushes the optical PPG trace around. Set it to 0 and
the channel axis becomes trivially separable, so every attribution method scores
100% and the metric tells you nothing.

**The label is 4-class and channel-dependent**: `0` = no event, `1 + c` = event
whose primary source is channel `c`. This is not cosmetic. Under a binary
event/no-event label the model never needs to know *which* sensor fired, so
scoring an attribution on the channel axis would punish it for failing to
surface information the model had no reason to learn. Making the label depend on
the channel means a correct model provably must use channel identity, which is
what makes `channel_hit` a fair measurement.

**`event_sec` (default 0.8).** Burst duration. Short bursts suit fixed-window
occlusion fine. Long bursts (≥2.0s on a 192-sample window) create **redundant
evidence**: blanking any one small window leaves the rest of the burst visible,
so the prediction barely moves. That regime is the subject of `dynamask.ipynb`.

`train_test_split(ds, test_frac, seed)` returns two `TimeSeriesDataset`s and
carries all ground-truth fields through the split.

---

## `model.py`

`TSClassifier(n_channels, n_classes, hidden)` — two `Conv1d` + ReLU blocks, then
global average pooling, then a linear head. Small on purpose: fast to train in a
notebook, and simple enough that its behaviour is not itself a mystery.

`train(model, x, y, epochs, lr, batch_size, seed)` returns the per-epoch loss
list. `accuracy(model, x, y)` is a plain top-1 score. Both take numpy arrays and
handle the tensor conversion internally.

Note the classifier reaches ~100% test accuracy on these datasets. That is
intentional: interpretability results are hard to reason about when the model is
also wrong, so the synthetic tasks are easy by design.

---

## `attribution.py`

### Base methods

**`occlusion(model, x, target, window=8, stride=4, baseline_value=0.0, per_channel=True)`**

Slides a window over time, replaces it with `baseline_value`, and records the
drop in target-class probability. Overlapping windows are averaged.

`per_channel=True` occludes each channel independently, so the result
distinguishes *which* channel a time window mattered in. `per_channel=False`
occludes all channels together — cheaper, but channel-blind.

⚠️ **Occlusion magnitudes are not effect sizes.** Under redundant evidence they
collapse by more than 10x while the *ranking* stays correct (demonstrated in
`dynamask.ipynb`). Anything that thresholds them absolutely will silently break
when event duration changes.

**`integrated_gradients(model, x, target, baseline_value=0.0, steps=64)`**

Thin wrapper over Captum's `IntegratedGradients`. Captum is shape-agnostic, so
the image-domain class works on `(batch, channel, time)` with no adaptation.

**`occlusion_1d` / `integrated_gradients_1d`** — single-channel wrappers
returning a flat `(length,)` vector. They exist so the first notebook keeps
working after the API was generalised to multivariate; prefer the base functions
in new code.

### `temporal_saliency_rescaling(...)`

TSR, from Ismail et al. (NeurIPS 2020). Three steps:

1. **Time relevance** `dt` — mask all channels in a time block, recompute the
   base saliency, take the total absolute change.
2. **Feature relevance** `d_ct` — *only for blocks clearing a percentile
   threshold*, mask one channel in the block and measure the change again.
3. **Combine** — the map is the product `dt * d_ct`, so a point must score on
   both axes to survive.

**Cost.** This is O(T/stride × C) *full saliency recomputations*. With
`time_stride=8` and `ig_steps=16` a 192-sample 3-channel sample takes ~0.25s;
dropping `time_stride` to 1 makes it ~8x slower. The threshold in step 2 is a
real optimisation, not just filtering — sub-threshold blocks skip the
per-channel loop entirely.

**`time_stride` is an accuracy/compute trade-off, not an implementation detail.**
Exact per-timestep TSR on a long recording is expensive, and that is a property
of the method itself.

**TSR is vacuous on single-channel data.** With one channel there is no feature
axis to separate from the time axis, and TSR collapses to its time term. Use it
only on multivariate input.

### Scoring

**`top_k_overlap(attr, event_window, top_frac=None)`** — IoU between the
top-attributed timesteps and the true event window. Collapses multi-channel maps
over channels (it scores the time axis only).

⚠️ **`k` defaults to the true event length.** A fixed `top_frac` silently caps
the achievable IoU whenever the event is longer than `top_frac × length` —
select 38 timesteps against a 128-sample event and the best possible score is
38/128 ≈ 0.30. That artefact cost real debugging time here: two methods both
scored ~0.29 and looked like joint failures when they were in fact both at the
metric's ceiling. Pass `top_frac` explicitly only when comparing events of equal
length.

**`channel_hit(attr, event_channel)`** — did the method put the most attribution
mass on the channel that carried the event? Returns 1.0/0.0, or NaN for
event-free samples. On the current datasets this metric **saturates near 1.0 for
every method**, so it does not discriminate; it is kept because it is the axis
TSR is designed to fix and it would discriminate on a harder generator.

---

## `dynamask.py`

Follows Crabbé & van der Schaar (ICML 2021). Independent implementation, not a
copy of the [reference repo](https://github.com/JonathanCrabbe/Dynamask).

⚠️ **Shape convention differs from the paper.** The paper and reference code use
`(T, C)` — time first. This repo uses `(C, T)` everywhere, so the public API here
takes and returns `(C, T)` and transposes internally. The internal maths is kept
in the reference's layout so it can be read against the paper.

### Perturbation operators

**`GaussianBlur(sigma_max=2.0)`** — the paper's operator. The mask sets the width
of a Gaussian at each point, `sigma(t,c) = sigma_max * (1 + eps - M[t,c])`, so
`M=1` gives a delta (original value kept) and `M=0` gives the widest blur.
Perturbed values are a convex combination of real values from the same signal, so
they stay in a plausible range.

This is the piece a naive implementation gets wrong. **Zeroing** a masked region
creates a discontinuity no sensor produces, and the model's reaction to that
cliff edge is not evidence about the signal.

⚠️ **`sigma_max` is load-bearing and must match the signal's timescale.** Too
small and the blur does not perturb: fidelity error sits at ~0 for every area and
the mask is driven entirely by the size term, reporting the area constraint back
at you. Too large and the background is destroyed too, so no mask of reasonable
area can preserve the prediction. On the bundled dataset (burst period ~15–29
samples) the paper's default `sigma_max=2` is in the first failure mode; ~6 works.

**`FadeMovingAverage()`** — fades toward each channel's global mean. Same
"perturb toward something plausible" idea with no locality; useful as a contrast
that isolates how much the *local* smoothing matters.

### Fitting

**`Mask(perturbation, device).fit(x, black_box, loss_function=mse, keep_ratio=0.1, n_epoch=500, size_reg_factor_init=0.01, ...)`**

Loss is `fidelity + size_reg_factor * size_reg + time_reg_factor * time_reg`.

The **size regularisation** follows the paper: the *sorted* mask values are pushed
toward a reference vector of `(1 - keep_ratio)` zeros followed by `keep_ratio`
ones. That constrains the mask's **area** without dictating which entries are
kept — fidelity decides where the retained mass lands. Without this term the
trivial solution wins: an all-ones mask preserves the prediction and explains
nothing.

`size_reg_factor` is **annealed geometrically** from `size_reg_factor_init` to
`init * dilation` across training. Early on the mask explores with the area
barely constrained; by the end the constraint dominates. Starting at full
strength collapses the mask before fidelity has shaped it.

`.mask` returns `(C, T)`; `.get_error()` returns the final fidelity error.

**`make_black_box(model, softmax=True)`** adapts a repo classifier
(`(batch, C, T)` → logits) to the `(T, C)` → probabilities interface Dynamask
expects. It keeps the graph intact — gradients must flow back through the model
to the mask.

### Areas and the extremal mask

**`MaskGroup(perturbation).fit(x, black_box, area_list=(.1,.15,.2,.25), ...)`**
fits one `Mask` per area. **`get_extremal_mask(threshold)`** returns the
smallest-area mask whose fidelity error is below `threshold` (the paper's
epsilon) — the point past which more area stops buying fidelity.

⚠️ **The extremal criterion degenerates when the perturbation is weak.** If every
area's error is already far below `threshold`, every area passes and the smallest
is returned by default — the selection is not choosing, it is defaulting. On the
bundled single-channel dataset this happens for 48/48 samples at `epsilon=0.01`,
because errors are ~1e-4 to ~4e-3. Check the scale of `get_errors()` before
reading the selected area as meaningful; `epsilon=0.01` does not transplant
across datasets.

**`fit_dynamask(...)`** is a one-line convenience wrapper over a single `Mask`,
kept for notebooks that call Dynamask alongside occlusion and IG. Its `target`
argument is accepted for signature compatibility but unused — Dynamask preserves
the model's *whole* output, not one class's score. For the paper's full procedure
use `MaskGroup`.

---

## `counterfactual.py`

**`nun_counterfactual(model, x, x_ref, y_ref, target_class=None, min_len=4, step=4)`**

Native-Guide / NUN-CF (Delaney et al., 2021):

1. `nearest_unlike_neighbour` finds the closest reference instance with a
   different label (plain Euclidean distance over the flattened signal — **DTW
   would be better** for phase-shifted signals, and is the obvious upgrade).
2. Segment lengths are tried in increasing order from `min_len`; for each length
   every start position is tried. The first splice that flips the prediction is
   returned, so it is the shortest on this grid.

Returns a `Counterfactual` dataclass: `x_cf`, `segment`, `nun_index`,
`original_class`, `cf_class`, `n_changed`, `success`. **Check `success`** — on
failure it returns the full-donor substitution as a record rather than raising.

`step` controls the search grid: `step=4` is ~4x faster than `step=1` but
recovers a coarser (slightly longer) minimal segment.

The point of splicing from a real donor rather than optimising a perturbation is
that the result stays **on the data manifold** — every value came from a genuine
signal of the target class. An adversarial perturbation would also flip the
label while explaining nothing.

---

## `causal.py`

This module answers a **different question** from everything above. Attribution
explains a *model*; if that model learned a confounded shortcut, a faithful
attribution faithfully reports the shortcut. Causal discovery explains the
*data-generating process* instead.

**`make_causal_dataset(n_timesteps, noise_std, seed)`** — a 5-variable linear VAR
system (`sleep_debt`, `activity`, `heart_rate`, `hrv`, `eda`) with a known
lagged graph, returned as a `CausalDataset` with `true_edges`.

The trap is deliberate: `heart_rate` and `eda` are both driven by `activity`,
so they correlate strongly with **no direct link between them**. Recovering that
correctly is the whole test.

**`granger_causality(x, cause, effect, max_lag=5)`** — pairwise F-test. Does
`cause`'s past improve prediction of `effect` beyond `effect`'s own past?
Conditions on nothing else, so it reports common-cause artefacts as edges. On
the bundled dataset it achieves precision 0.25 (16 edges for a 4-edge graph).

**`conditional_granger(x, cause, effect, max_lag=5)`** — the same test with every
*other* variable's lagged history already in the restricted model. This is the
core idea PCMCI builds on, and it recovers the bundled graph exactly
(precision 1.0, recall 1.0).

**`discover_graph(ds, max_lag, alpha, conditional=True)`** → boolean adjacency
where `[i, j]` means "i causes j". **`score_graph(adj, ds)`** → precision /
recall / F1 against `true_edges`.

### Assumptions — these are load-bearing

- **Linear** VAR only. Nonlinear drivers are invisible to both tests.
- **Causal sufficiency**: no unobserved confounders. Routinely false in wearable
  data — sleep, stress, medication, ambient temperature are rarely all measured
  — and conditioning on the variables you *do* have cannot fix a confounder you
  never recorded.
- **No multiple-testing correction** across the 20 ordered pairs × 5 lags.
- `discover_graph` tests each ordered pair independently rather than running
  PCMCI's iterative condition selection, which is what lets real PCMCI scale
  without the conditioning set exploding.

---

## Fidelity to the papers

| Module | Divergence from the reference method |
| --- | --- |
| `attribution.temporal_saliency_rescaling` | Follows Algorithm 1, but operates on **time blocks** (`time_stride`) rather than individual timesteps, for tractability. |
| `dynamask` | Implements the paper's `GaussianBlur` operator, sorted-mask area regularisation, annealed `size_reg_factor`, `MaskGroup` and extremal-mask selection. Omits the authors' *adaptive* area constraint (areas are supplied explicitly via `area_list`) and their other perturbation operators beyond `FadeMovingAverage`. |
| `counterfactual` | Euclidean NUN rather than DTW; grid search over segments rather than the authors' guided search. |
| `causal` | Linear VAR F-tests. Not PCMCI — no iterative condition selection, no nonlinear independence tests, no FDR control. |

For real work: [`captum`](https://captum.ai/) (IG, occlusion, GradientSHAP),
[`tsinterpret`](https://github.com/fzi-forschungszentrum-informatik/TSInterpret)
(TSR, Dynamask, LEFTIST, NUN-CF under one API), and
[`tigramite`](https://github.com/jakobrunge/tigramite) (PCMCI/PCMCI+).

---

## Reproducibility

Every generator, trainer, and mask fit takes a `seed`. The notebooks are
committed **with outputs**, so the numbers in the markdown match what the
committed cells produced. If you re-run and numbers shift slightly, check that
you did not change `n_samples`, `length`, or `crosstalk` — the metrics are
sensitive to all three.
