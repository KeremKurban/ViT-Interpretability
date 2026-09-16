# ViT-Interpretability

Notes and runnable experiments on neural-network interpretability, working
from the well-established image/ViT methods toward **time-series** signals of
the kind wearable devices produce (PPG, ECG, accelerometer, EDA).

## Layout

```
notebooks/
  images/        interpretability on image models (ViT / CNN)
  time_series/   the same method families applied to 1D signals
src/ts_interp/   reusable data / model / attribution code for the TS notebooks
                 -> see src/ts_interp/README.md for developer notes
```

The notebooks use `src/ts_interp` in a line or two per call, which keeps them
readable but hides the details. **[`src/ts_interp/README.md`](src/ts_interp/README.md)**
is the reference for what each function does, the shape conventions, the metric
pitfalls, and exactly where these compact reimplementations diverge from the
published methods.

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

The time-series notebooks add `src/` to `sys.path`, so no install step is needed.

## Notebooks

Read in this order — each builds on the previous one's result.

| Notebook | What it covers |
| --- | --- |
| [`time_series/occlusion_vs_integrated_gradients.ipynb`](notebooks/time_series/occlusion_vs_integrated_gradients.ipynb) | Occlusion vs. Integrated Gradients on a synthetic wearable-style PPG signal, scored against a known ground-truth event window |
| [`time_series/temporal_saliency_rescaling.ipynb`](notebooks/time_series/temporal_saliency_rescaling.ipynb) | Why IG underperformed, and whether TSR fixes it. Moves to 3-channel data, since TSR is vacuous on a single channel |
| [`time_series/dynamask.ipynb`](notebooks/time_series/dynamask.ipynb) | Learned masks vs. fixed windows, and the regime where occlusion's attribution *magnitudes* stop meaning anything |
| [`time_series/causal_and_counterfactual.ipynb`](notebooks/time_series/causal_and_counterfactual.ipynb) | Counterfactual explanations, then causal discovery — the shift from explaining a *model* to explaining the *data* |
| [`images/integrated_gradients_vit.ipynb`](notebooks/images/integrated_gradients_vit.ipynb) | Integrated Gradients on an image classifier (adapted from the TensorFlow tutorial) |

### Results so far

Scored as IoU between the top-attributed timesteps and the known event window
(`k` = true event length), on the 3-channel dataset, 75 event samples:

| Method | Time IoU |
| --- | --- |
| Integrated Gradients | 0.547 |
| **TSR(IG)** | **0.706** |
| Occlusion | 0.861 |

TSR recovers a substantial part of IG's gap, which is what it was designed to
do. Three caveats worth carrying:

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
