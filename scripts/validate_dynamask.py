"""Does a fitted Dynamask actually beat a random mask of the same area?

Run with: python3 scripts/validate_dynamask.py

If it does not, the size regulariser is doing all the work and the fidelity
term is not earning its place — which would mean the Dynamask IoU numbers say
little about the method.

Deliberately small so it runs in about a minute: short signals, few
samples, fewer epochs.
"""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import torch
from ts_interp import data, model, attribution, dynamask

LENGTH, N_SAMPLES, N_EVAL, EPOCHS, AREA, SIGMA = 128, 150, 6, 200, 0.15, 6.0

ds = data.make_dataset(n_samples=N_SAMPLES, length=LENGTH, seed=1)
tr, te = data.train_test_split(ds, test_frac=0.3, seed=1)
clf = model.TSClassifier()
model.train(clf, tr.x, tr.y, epochs=25)
print(f"test acc {model.accuracy(clf, te.x, te.y):.3f}", flush=True)

bb = dynamask.make_black_box(clf)
gb = dynamask.GaussianBlur(sigma_max=SIGMA)
rng = np.random.default_rng(0)

fit_iou, rnd_iou, fit_err, rnd_err = [], [], [], []

for n, i in enumerate(np.where(te.y == 1)[0][:N_EVAL]):
    x, ew = te.x[i], tuple(te.event_windows[i])
    xt = torch.from_numpy(x).T.contiguous()
    with torch.no_grad():
        y0 = bb(xt)

    m = dynamask.Mask(gb)
    m.fit(x, bb, keep_ratio=AREA, n_epoch=EPOCHS, size_reg_factor_init=0.01)
    fit_iou.append(attribution.top_k_overlap(m.mask, ew))
    fit_err.append(m.get_error())

    # random mask, identical area
    r = np.zeros_like(x)
    r.flat[rng.choice(x.size, int(AREA * x.size), replace=False)] = 1.0
    rnd_iou.append(attribution.top_k_overlap(r, ew))
    with torch.no_grad():
        yr = bb(gb.apply(xt, torch.from_numpy(r).T.contiguous()))
        rnd_err.append(float(((yr - y0) ** 2).mean()))
    print(f"  sample {n+1}/{N_EVAL}", flush=True)

print("\nFitted mask vs. random mask of identical area (%.2f), n=%d" % (AREA, len(fit_iou)))
print(f"  fitted : IoU {np.mean(fit_iou):.3f}   fidelity err {np.mean(fit_err):.6f}")
print(f"  random : IoU {np.mean(rnd_iou):.3f}   fidelity err {np.mean(rnd_err):.6f}")
print()
print(f"  fitted localizes better        : {np.mean(fit_iou) > np.mean(rnd_iou)}")
print(f"  fitted preserves prediction better: {np.mean(fit_err) < np.mean(rnd_err)}")
print("VALIDATION_DONE")
