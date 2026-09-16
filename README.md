# ViT-Interpretability

Notes and runnable experiments on neural-network interpretability, working
from the well-established image/ViT methods toward **time-series** signals of
the kind wearable devices produce (PPG, ECG, accelerometer, EDA).

## Layout

```
notebooks/
  images/        interpretability on image models
  time_series/   the same method families applied to 1D signals
src/img_interp/  baselines, integrated gradients, expected gradients  (TensorFlow)
                 -> see src/img_interp/README.md for developer notes
src/ts_interp/   data / model / attribution / dynamask / causal        (PyTorch)
                 -> see src/ts_interp/README.md for developer notes
```

The notebooks call `src/` in a line or two per function, which keeps them
readable but hides the details. The two `README.md` files under `src/` are the
reference for what each function does, the shape conventions, the metric
pitfalls, and exactly where these compact reimplementations diverge from the
published methods.

**On the two frameworks.** `img_interp` is TensorFlow because the image notebook
grew out of the TF tutorial and its TF-Hub Inception V1; `ts_interp` is PyTorch.
No code crosses between them, and porting either to match the other would be
churn for no gain.

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

Notebooks add `src/` to `sys.path`, so no install step is needed. The image
notebooks additionally download ~220 MB of real photographs the first time they
run (cached afterwards), used as Expected Gradients baselines.

## Notebooks

Read in this order — each builds on the previous one's result.

**Time series** — read in order; each builds on the previous one's result.

| Notebook | What it covers |
| --- | --- |
| [`occlusion_vs_integrated_gradients.ipynb`](notebooks/time_series/occlusion_vs_integrated_gradients.ipynb) | The main harness: occlusion vs. Integrated Gradients vs. Dynamask on a synthetic wearable-style PPG signal, all scored by IoU against known ground-truth event windows |
| [`temporal_saliency_rescaling.ipynb`](notebooks/time_series/temporal_saliency_rescaling.ipynb) | Why IG underperformed, and whether TSR fixes it. Moves to 3-channel data, since TSR is vacuous on a single channel |
| [`dynamask.ipynb`](notebooks/time_series/dynamask.ipynb) | Dynamask internals — the Gaussian-blur perturbation operator, the area/size trade-off, and extremal-mask selection |
| [`causal_and_counterfactual.ipynb`](notebooks/time_series/causal_and_counterfactual.ipynb) | Counterfactual explanations, then causal discovery — the shift from explaining a *model* to explaining the *data* |

**Images**

| Notebook | What it covers |
| --- | --- |
| [`baseline_diversification.ipynb`](notebooks/images/baseline_diversification.ipynb) | The IG baseline as a parameter: black vs. noise vs. real-image vs. blurred, plus Expected Gradients averaged over 25 real baselines |
| [`integrated_gradients_vit.ipynb`](notebooks/images/integrated_gradients_vit.ipynb) | The starting point — IG with a single black baseline (adapted from the TensorFlow tutorial) |

### Results so far

**Time-series attribution.** IoU between the top-attributed timesteps and the
known event window (`k` = true event length). Single-channel dataset, 48 event
samples:

| Method | IoU |
| --- | --- |
| **Occlusion** | **0.784** |
| Dynamask (extremal, `sigma_max=6`) | 0.481 |
| Integrated Gradients | 0.395 |

Occlusion wins, but this dataset is its best case by construction — the injected
event is compact, contiguous and roughly fixed-width, exactly what a fixed
rectangular window assumes. Two caveats on the Dynamask row: `sigma_max=6` was
picked by a sweep using this same IoU, so it is mildly optimistic; and the
**extremal-mask criterion degenerates here**, returning the smallest area for
47 of 48 samples because every area's fidelity error sits orders of magnitude
below the `epsilon=0.01` threshold. A criterion that always fires is
indistinguishable from no criterion.

`sigma_max` is coupled to the signal's timescale and cannot be inherited from
the paper. On this data the sweep plateaus between 4 and 8 (IoU ~0.51) and falls
off hard either side — 0.412 at the paper's default of 2, where the blur is too
weak to perturb the burst at all, and 0.309 at 12, where it destroys the
background too. **A near-zero fidelity error at every area is the tell that
`sigma_max` is too small**, and it is visible without ground truth.

A separate check confirms the method is doing real work rather than the area
constraint alone: a fitted mask scores 0.552 IoU against 0.148 for a random mask
of identical area, with fidelity error 0.006 vs 0.248
(`scripts/validate_dynamask.py`).

On the 3-channel dataset (75 event samples), TSR recovers much of IG's gap —
IG 0.547 → **TSR(IG) 0.706**, against occlusion's 0.861. Caveats worth carrying:

- **Channel accuracy saturates** near 1.0 for every method on this data, so that
  axis does not discriminate. Reported rather than tuned away.
- **Occlusion's magnitudes collapse >10x under redundant evidence** while its
  *ranking* holds — so rank-based metrics like IoU never reveal the failure.
  Occlusion scores are not effect sizes.
- **Metric choice is load-bearing.** A fixed `top_frac` caps achievable IoU when
  the event is long; two methods can both sit at the ceiling and look like joint
  failures. See the developer notes.

On causal discovery, pairwise Granger scores precision 0.25 on a 5-variable
system (it fires on nearly every pair, since everything is downstream of
`activity`); conditioning on the other variables' histories recovers the true
graph exactly.

**Image IG baselines.** Descriptive summaries of the two diagnostic curves —
*backtrack* is how much `P(target)` gives back along the path, *saturated* is the
fraction of the path where gradients are under 10% of peak. Fireboat image,
Inception V1, `m_steps=50`:

| Baseline | Backtrack | Saturated | Completeness err |
| --- | --- | --- | --- |
| black | 18.5% | 7.8% | 0.0108 |
| random_noise | 39.4% | 56.9% | 0.0472 |
| real_image | 7.6% | 76.5% | 0.0426 |
| blurred | 70.7% | 11.8% | **0.0068** |
| **expected_grads (N=25)** | **0.0%** | 54.9% | n/a |

**No single baseline fixes both artefacts**, which was the surprise here.
Expected Gradients makes the averaged path perfectly monotone — averaging over 25
baselines cancels the individual paths' wiggles — but inherits the heavy
saturation of the real-image baselines it averages over. The blurred baseline is
the reverse: gradients stay alive and the integral converges best, but it
overshoots badly, because the model is *more* confident about a partly-blurred
fireboat than the sharp original.

Also worth flagging against the usual story: the black baseline is **not** the
worst offender on this image. It has the lowest saturation of all four. Its real
defect is structural rather than visible in these curves — IG scales by
`(image - baseline)`, so genuinely black pixels can never receive attribution no
matter how much the model relies on them.

All notebooks are committed **with outputs**, so the numbers are readable
without running anything.

## Method survey

Image-derived interpretability methods do not transfer to time series for free.
This is the map I'm working from.

### Perturbation-based

| Method | Idea | Notes for 1D signals |
| --- | --- | --- |
| **Occlusion** | Mask a window, measure the probability drop | Simple and surprisingly strong, but a fixed rectangular window assumes the important region is contiguous and window-sized |
| **Dynamask** ([Crabbé & van der Schaar, 2021](https://arxiv.org/abs/2106.05303)) | Learn a smooth, temporally-extended perturbation mask | Handles shape-over-interval importance, which fixed windows miss |
| **WinIT** | Extend the perturbation window to capture lagged effects | Wearable signals are autocorrelated and effects are often delayed |
| **Frequency-domain occlusion** | Mask frequency bands rather than time windows | Often more physiologically interpretable (e.g. knock out the respiratory band of a PPG trace) |

### Gradient-based

- **Integrated Gradients / GradientSHAP / DeepLIFT** — transfer directly to 1D;
  Captum is shape-agnostic, so the same class works on `(batch, channels, time)`.
- **Temporal Saliency Rescaling (TSR)** — Ismail et al.,
  [*Benchmarking Deep Learning Interpretability in Time Series Predictions*](https://arxiv.org/abs/2010.13924)
  (NeurIPS 2020), show that raw saliency conflates **time importance** with
  **feature importance**; TSR separates the two before combining them. The key
  correction to apply on top of vanilla IG.
- **1D Grad-CAM** — standard in ECG/PPG CNN classifiers, gives per-timestep
  class activation.

### Attention-based

- **Temporal Fusion Transformer** — interpretable by design; exposes per-timestep
  attention and per-feature variable-selection weights.
- **Attention rollout** — the same math as in ViTs, with time patches instead of
  image patches; applies to PatchTST / Informer / Autoformer.
- **Time-series foundation models** (Chronos, TimesFM, Moment, Lag-Llama) —
  attention-based interpretability for these is largely unpublished. An open problem.

### Interpretable-by-design

- **Shapelets** — learn short discriminative subsequences; the decision *is* the
  explanation. Common in wearable / human-activity-recognition work.

### Counterfactual and causal

Attribution says *what the model used*, not *what caused the outcome*. These
methods are the bridge to the causal side:

- **CXPlain** — auxiliary model estimating causal attribution of inputs to output.
- **NUN-CF** and related instance-based counterfactuals — "what is the minimal
  edit to this segment that flips the prediction?"
- **Granger causality**, **PCMCI**, **TCDF** — causal *discovery* over multivariate
  time series (does step count causally precede heart-rate change, or merely
  correlate). A different axis from post-hoc attribution, not a competitor to it.

## Libraries

- [`captum`](https://captum.ai/) — IG, occlusion, GradientSHAP on any PyTorch model
- [`tsinterpret`](https://github.com/fzi-forschungszentrum-informatik/TSInterpret) —
  TSR, Dynamask/dCAM, LEFTIST, NUN-CF counterfactuals under one API

## License

MIT — see [LICENSE](LICENSE).
