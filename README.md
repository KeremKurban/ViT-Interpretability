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
```

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

The time-series notebooks add `src/` to `sys.path`, so no install step is needed.

## Notebooks

| Notebook | What it covers |
| --- | --- |
| `notebooks/time_series/occlusion_vs_integrated_gradients.ipynb` | Occlusion vs. Integrated Gradients on a synthetic wearable-style PPG signal, scored against a known ground-truth event window |
| `notebooks/images/integrated_gradients_vit.ipynb` | Integrated Gradients on an image classifier (adapted from the TensorFlow tutorial) |

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
