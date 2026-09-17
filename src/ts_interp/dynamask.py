"""Dynamask — learned dynamic perturbation masks for time series.

Reference: Crabbé & van der Schaar, *Explaining Time Series Predictions with
Dynamic Masks* (ICML 2021), https://arxiv.org/abs/2106.05303
Reference implementation: https://github.com/JonathanCrabbe/Dynamask (MIT).

This is an independent implementation following the paper's formulation, not a
vendored copy. Where it departs from the reference, the docstrings say so.

## The idea

Occlusion asks "what happens if I blank out this fixed rectangular window?",
which bakes in two assumptions: that the important region is contiguous, and
that it is exactly ``window`` samples wide. Dynamask drops both. It learns a
continuous mask ``M ∈ [0,1]^(T×C)`` by gradient descent, where a mask value
near 1 *keeps* the original value at that timestep and a value near 0 *perturbs*
it.

Three pieces make that work:

1. A **differentiable perturbation operator** (`GaussianBlur`) that replaces
   masked-out values with a locally smoothed version of the signal rather than
   zeroing them. Blurring toward a neighbourhood average keeps the perturbed
   input far closer to the data manifold than a hard blank does — zeroing a
   physiological trace creates a discontinuity no real sensor would produce, and
   the model's response to that artefact is not evidence about the signal.
2. A **fidelity loss** — the perturbed input should still produce roughly the
   original prediction, so the mask has to retain whatever actually drives it.
3. A **size regularisation** term pinning the mask to a target area. Without it
   the trivial solution is to keep everything: a 100%-area mask preserves the
   prediction perfectly and explains nothing.

## Shape convention

The paper and reference implementation use ``(T, C)`` — time first. This repo
uses ``(C, T)`` everywhere else (see ``src/ts_interp/README.md``), so the public
API here takes and returns ``(C, T)`` and transposes internally. The internal
maths is kept in the reference's layout so it can be read against the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Loss functions between the original and perturbed black-box outputs
# --------------------------------------------------------------------------


def mse(y_pert: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between perturbed and original outputs."""
    return ((y_pert - y_target) ** 2).mean()


def cross_entropy(y_pert: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
    """Cross entropy treating the original output distribution as the target.

    Expects both arguments to be probability vectors. Usually the better choice
    than ``mse`` for a classifier, since it scores the whole distribution rather
    than its raw coordinates.
    """
    return -(y_target * torch.log(y_pert + 1e-12)).sum(dim=-1).mean()


# --------------------------------------------------------------------------
# Perturbation operators
# --------------------------------------------------------------------------


class Perturbation:
    """Base class. Subclasses implement ``apply(x, mask) -> perturbed x``.

    Both arguments are ``(T, C)`` tensors and the operator must be
    differentiable with respect to ``mask`` — that is what lets the fidelity
    loss backpropagate all the way to the mask values.
    """

    def __init__(self, device: torch.device | str = "cpu", eps: float = 1e-7):
        self.device = device
        self.eps = eps

    def apply(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GaussianBlur(Perturbation):
    """Blur each timestep toward a Gaussian-weighted average of its neighbours.

    The mask controls the width of the Gaussian at each point:

        sigma(t, c) = sigma_max * (1 + eps - M[t, c])

    so ``M = 1`` gives ``sigma -> 0``, a delta function that returns the original
    value untouched, while ``M = 0`` gives the widest blur. Every intermediate
    value interpolates smoothly between the two, which is what makes the operator
    differentiable in the mask.

    The perturbed signal is then

        x_pert[t, c] = sum_s w[s, t, c] * x[s, c]

    with ``w`` the normalised Gaussian weights. Note the perturbation is a convex
    combination of *real values from the same signal*, so it stays in a plausible
    range no matter what the mask does.
    """

    def __init__(self, device="cpu", eps: float = 1e-7, sigma_max: float = 2.0):
        super().__init__(device=device, eps=eps)
        self.sigma_max = sigma_max

    def apply(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        # width of the Gaussian at each (t, c); mask 1 -> ~0 width, mask 0 -> sigma_max
        sigma = self.sigma_max * ((1 + self.eps) - mask)  # (T, C)
        sigma = sigma.unsqueeze(0)  # (1, T, C) — indexed by the *output* time

        t_axis = torch.arange(1, T + 1, dtype=x.dtype, device=x.device)
        t_source = t_axis.unsqueeze(1).unsqueeze(2)  # (T, 1, 1)
        t_out = t_axis.unsqueeze(0).unsqueeze(2)  # (1, T, 1)

        # (T_source, T_out, C) Gaussian weights
        weights = torch.exp(-((t_source - t_out) ** 2) / (2 * sigma**2))
        weights = weights / (weights.sum(dim=0, keepdim=True) + self.eps)

        return torch.einsum("stc,sc->tc", weights, x)


class FadeMovingAverage(Perturbation):
    """Fade masked-out regions toward each channel's global mean.

    Included as a cheap contrast to `GaussianBlur`: it is the same "perturb
    toward something plausible" idea with no locality, so comparing the two
    isolates how much the *local* smoothing matters.
    """

    def apply(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)  # (1, C)
        return mask * x + (1 - mask) * mean


# --------------------------------------------------------------------------
# Single fixed-area mask
# --------------------------------------------------------------------------


@dataclass
class MaskHistory:
    total: list = field(default_factory=list)
    fidelity: list = field(default_factory=list)
    size: list = field(default_factory=list)
    time_reg: list = field(default_factory=list)


class Mask:
    """A single mask fitted at one target area (``keep_ratio``).

    Usage mirrors the reference implementation::

        mask = Mask(GaussianBlur(), device)
        mask.fit(x, black_box, loss_function=mse, keep_ratio=0.1)
        m = mask.mask  # (C, T) in [0, 1]
    """

    def __init__(self, perturbation: Perturbation, device: torch.device | str = "cpu"):
        self.perturbation = perturbation
        self.device = device
        self.mask_tensor: torch.Tensor | None = None
        self.keep_ratio: float | None = None
        self.history = MaskHistory()
        self._error: float | None = None

    def fit(
        self,
        x: np.ndarray | torch.Tensor,
        black_box,
        loss_function=mse,
        keep_ratio: float = 0.1,
        n_epoch: int = 500,
        learning_rate: float = 1.0e-1,
        momentum: float = 0.9,
        size_reg_factor_init: float = 0.01,
        size_reg_factor_dilation: float = 100.0,
        time_reg_factor: float = 0.0,
        initial_mask_value: float = 0.5,
        verbose: bool = False,
    ) -> "Mask":
        """Fit the mask for one input.

        ``x`` is ``(C, T)``. ``black_box`` maps a ``(T, C)`` tensor to an output
        tensor (for a classifier, a probability vector).

        The size regularisation follows the paper: the *sorted* mask values are
        pushed toward a reference vector holding ``(1 - keep_ratio)`` zeros
        followed by ``keep_ratio`` ones. That constrains the mask's **area**
        without dictating *which* entries are kept, which is the point — the
        fidelity term decides where the retained mass goes.

        ``size_reg_factor`` is annealed geometrically from
        ``size_reg_factor_init`` to ``init * dilation`` across training, matching
        the reference implementation. The paper's rationale is that the mask
        should be free to explore while the area is barely constrained, and be
        forced to commit only once fidelity has shaped it — so starting at the
        final strength would collapse the mask prematurely.

        That rationale is the authors', not something measured here: the schedule
        is used as published and no ablation of it was run in this repo.
        """
        x_t = _to_tensor(x, self.device)  # (C, T)
        x_tc = x_t.transpose(0, 1).contiguous()  # (T, C) — reference layout
        T, C = x_tc.shape
        self.keep_ratio = keep_ratio

        y_target = black_box(x_tc).detach()

        # reference vector for the area constraint: (1-a)*T*C zeros, then ones
        n_entries = T * C
        n_zeros = int((1 - keep_ratio) * n_entries)
        reg_ref = torch.zeros(n_entries, dtype=x_tc.dtype, device=self.device)
        reg_ref[n_zeros:] = 1.0

        mask_tensor = torch.full(
            (T, C), initial_mask_value, dtype=x_tc.dtype, device=self.device
        ).requires_grad_(True)
        optimizer = torch.optim.SGD([mask_tensor], lr=learning_rate, momentum=momentum)

        # geometric annealing of the size-regularisation strength
        reg_multiplicator = np.exp(np.log(size_reg_factor_dilation) / n_epoch)
        size_reg_factor = size_reg_factor_init

        self.history = MaskHistory()
        for epoch in range(n_epoch):
            optimizer.zero_grad()

            x_pert = self.perturbation.apply(x_tc, mask_tensor)
            y_pert = black_box(x_pert)

            error = loss_function(y_pert, y_target)

            mask_sorted = mask_tensor.reshape(n_entries).sort()[0]
            size_reg = ((reg_ref - mask_sorted) ** 2).mean()

            if time_reg_factor > 0:
                time_reg = (mask_tensor[1:] - mask_tensor[:-1]).abs().mean()
            else:
                time_reg = torch.zeros((), dtype=x_tc.dtype, device=self.device)

            loss = error + size_reg_factor * size_reg + time_reg_factor * time_reg
            loss.backward()
            optimizer.step()

            # the mask is a probability per entry; keep it in range
            with torch.no_grad():
                mask_tensor.clamp_(0, 1)

            size_reg_factor *= reg_multiplicator

            self.history.total.append(loss.item())
            self.history.fidelity.append(error.item())
            self.history.size.append(size_reg.item())
            self.history.time_reg.append(float(time_reg))

            if verbose and epoch % 100 == 0:
                print(f"  epoch {epoch:4d}  loss {loss.item():.5f}  err {error.item():.5f}")

        self.mask_tensor = mask_tensor.detach()
        self._error = self.history.fidelity[-1]
        return self

    @property
    def mask(self) -> np.ndarray:
        """The fitted mask as ``(C, T)``, matching the repo's convention."""
        if self.mask_tensor is None:
            raise RuntimeError("call fit() first")
        return self.mask_tensor.transpose(0, 1).cpu().numpy()

    def get_error(self) -> float:
        """Final fidelity error — how much the prediction moved under the mask."""
        if self._error is None:
            raise RuntimeError("call fit() first")
        return self._error


# --------------------------------------------------------------------------
# Group of masks across areas, plus extremal-mask extraction
# --------------------------------------------------------------------------


class MaskGroup:
    """Fit one `Mask` per target area and extract the extremal mask.

    The motivation is that the "right" mask area is not known in advance, and
    picking one arbitrarily makes the explanation a function of that arbitrary
    choice. Fitting a range of areas and taking the smallest one that still
    preserves the prediction turns the choice into something measured.
    """

    def __init__(self, perturbation: Perturbation, device: torch.device | str = "cpu"):
        self.perturbation = perturbation
        self.device = device
        self.mask_list: list[Mask] = []
        self.area_list: list[float] = []

    def fit(
        self,
        x: np.ndarray | torch.Tensor,
        black_box,
        loss_function=mse,
        area_list=(0.1, 0.15, 0.2, 0.25),
        verbose: bool = False,
        **fit_kwargs,
    ) -> "MaskGroup":
        self.area_list = sorted(float(a) for a in area_list)
        self.mask_list = []
        for area in self.area_list:
            if verbose:
                print(f"fitting mask at area {area}")
            m = Mask(self.perturbation, self.device)
            m.fit(x, black_box, loss_function=loss_function, keep_ratio=area, **fit_kwargs)
            self.mask_list.append(m)
        return self

    def get_errors(self) -> list[float]:
        return [m.get_error() for m in self.mask_list]

    def get_extremal_mask(self, threshold: float = 0.01) -> Mask:
        """Smallest-area mask whose fidelity error is below ``threshold``.

        This is the paper's extremal mask (epsilon = ``threshold``): the point
        past which granting the mask more area stops buying meaningful fidelity.
        Because `fit` sorts `area_list` ascending, the first mask under the
        threshold is by construction the smallest acceptable one.

        If no mask clears the threshold, the lowest-error mask is returned rather
        than raising — an explanation is still wanted, and `get_errors()` shows
        the threshold was not met.
        """
        errors = self.get_errors()
        for mask, error in zip(self.mask_list, errors):
            if error < threshold:
                return mask
        return self.mask_list[int(np.argmin(errors))]


# --------------------------------------------------------------------------
# Adapters and back-compatible helper
# --------------------------------------------------------------------------


def _to_tensor(x, device) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device).float()
    return x.to(device).float()


def make_black_box(model: torch.nn.Module, softmax: bool = True):
    """Wrap a repo classifier as a Dynamask black box.

    The classifiers in `model.py` take ``(batch, C, T)`` and return logits;
    Dynamask wants a function of a single ``(T, C)`` tensor. This bridges the
    two and (by default) returns probabilities, so fidelity is measured on the
    same scale the other attribution methods use.

    The returned callable keeps the graph intact — gradients must flow back
    through the model to the mask.
    """
    model.eval()

    def black_box(x_tc: torch.Tensor) -> torch.Tensor:
        x_bct = x_tc.transpose(0, 1).unsqueeze(0)  # (1, C, T)
        logits = model(x_bct)
        out = F.softmax(logits, dim=-1) if softmax else logits
        return out.squeeze(0)

    return black_box


def fit_dynamask(
    model: torch.nn.Module,
    x: np.ndarray,
    target: int | None = None,
    baseline: str = "gaussian_blur",
    n_epochs: int = 300,
    lr: float = 0.1,
    keep_ratio: float = 0.15,
    size_reg_factor_init: float = 0.01,
    tv_reg: float = 0.0,
    sigma_max: float = 2.0,
    seed: int = 0,
    return_history: bool = False,
):
    """Convenience wrapper fitting a single mask and returning it as ``(C, T)``.

    Kept for the notebooks that call Dynamask as a one-liner alongside occlusion
    and integrated gradients. For the paper's full procedure — several areas plus
    extremal-mask selection — use `MaskGroup` directly.

    ``baseline`` selects the perturbation operator: ``"gaussian_blur"`` (the
    paper's, and the default) or ``"fade_moving_average"``. ``target`` is
    accepted for signature compatibility with the other attribution functions
    but is unused: Dynamask preserves the model's *whole* output rather than one
    class's score.
    """
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)

    if baseline in ("gaussian_blur", "blur"):
        pert = GaussianBlur(sigma_max=sigma_max)
    elif baseline in ("fade_moving_average", "mean"):
        pert = FadeMovingAverage()
    else:
        raise ValueError(f"unknown perturbation: {baseline}")

    m = Mask(pert)
    m.fit(
        x,
        make_black_box(model),
        loss_function=mse,
        keep_ratio=keep_ratio,
        n_epoch=n_epochs,
        learning_rate=lr,
        size_reg_factor_init=size_reg_factor_init,
        time_reg_factor=tv_reg,
    )

    for p in model.parameters():
        p.requires_grad_(True)

    if return_history:
        hist = {
            "fidelity": m.history.fidelity,
            "size": m.history.size,
            "tv": m.history.time_reg,
        }
        return m.mask, hist
    return m.mask
