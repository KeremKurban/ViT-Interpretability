# Blog posts

Write-ups of findings from this repo, in the order they were written. Each one
is backed by a committed notebook, so every number is reproducible.

| # | Post | Backed by |
| --- | --- | --- |
| 01 | [Your Integrated Gradients baseline is doing more work than you think](01-integrated-gradients-baselines.md) | [`notebooks/images/baseline_diversification.ipynb`](../notebooks/images/baseline_diversification.ipynb) |

## Queued

Findings that are already measured and sitting in the repo, waiting to be
written up:

- **When your interpretability metric hides the failure.** Three separate times
  in this repo the evaluation was the problem, not the method: a fixed top-k
  fraction capped achievable IoU so two methods sat at the ceiling looking like
  joint failures; occlusion's attribution magnitudes collapsed ~18x under
  redundant evidence while its ranking stayed correct, so a rank-based metric
  scored it 0.93 and never noticed; and a count-based non-monotonicity measure
  flagged a flat noisy curve as pathological. Backed by the time-series
  notebooks.
- **Porting image interpretability to 1D signals: what breaks.** TSR is vacuous
  on single-channel data by construction. Dynamask's blur width does not
  transplant from the paper's data. An extremal-mask criterion that always fires
  is not a criterion. Backed by `temporal_saliency_rescaling.ipynb`,
  `dynamask.ipynb`, `occlusion_vs_integrated_gradients.ipynb`.
- **Attribution is not causation, demonstrated.** Pairwise Granger scores 0.25
  precision on a system where everything is downstream of one driver;
  conditioning recovers the true graph exactly. Backed by
  `causal_and_counterfactual.ipynb`.

## House rules

- Every claim traces to a committed notebook with outputs.
- State the scope. One image and one model is one image and one model.
- Report what was measured, including when it contradicts the motivating
  premise. The black-baseline result in post 01 is an example.
- Separate "this is what the paper says" from "this is what I measured here".
