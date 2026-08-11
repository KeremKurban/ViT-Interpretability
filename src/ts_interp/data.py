"""Synthetic wearable-sensor signals for interpretability experiments.

We simulate a PPG-like waveform (as you'd get from a wrist/finger optical
heart-rate sensor) and inject a short "event" segment — a burst that mimics
a motion artifact / arrhythmia-like irregularity — into half the samples.
The task is binary classification: does the window contain an event?

Because we control exactly where the event is injected, we can score an
attribution method by how well its saliency overlaps the true event window
— a stand-in for ground truth that real wearable data rarely gives you.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TimeSeriesDataset:
    x: np.ndarray  # (n_samples, 1, length) float32
    y: np.ndarray  # (n_samples,) int64, 0 = normal, 1 = event
    event_windows: np.ndarray  # (n_samples, 2) start/end index, -1 if none


def _ppg_wave(length: int, fs: float, heart_rate_hz: float, phase: float) -> np.ndarray:
    t = np.arange(length) / fs
    # fundamental + first harmonic approximates the dicrotic-notch shape of a PPG pulse
    return np.sin(2 * np.pi * heart_rate_hz * t + phase) + 0.3 * np.sin(
        4 * np.pi * heart_rate_hz * t + phase
    )


def make_dataset(
    n_samples: int = 400,
    length: int = 256,
    fs: float = 64.0,
    noise_std: float = 0.08,
    event_frac: float = 0.5,
    seed: int = 0,
) -> TimeSeriesDataset:
    rng = np.random.default_rng(seed)
    x = np.zeros((n_samples, 1, length), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.int64)
    event_windows = np.full((n_samples, 2), -1, dtype=np.int64)

    n_events = int(n_samples * event_frac)
    event_idx = rng.choice(n_samples, size=n_events, replace=False)

    for i in range(n_samples):
        hr = rng.uniform(1.0, 1.4)  # ~60-84 bpm
        phase = rng.uniform(0, 2 * np.pi)
        sig = _ppg_wave(length, fs, hr, phase)
        sig += rng.normal(0, noise_std, size=length)

        if i in event_idx:
            win = int(fs * 0.8)  # ~0.8s burst, e.g. motion artifact
            start = rng.integers(length // 4, length - win - length // 4)
            end = start + win
            burst_hr = hr * rng.uniform(2.2, 3.0)  # sudden rate spike
            burst = _ppg_wave(win, fs, burst_hr, phase) * rng.uniform(1.3, 1.8)
            sig[start:end] = burst
            y[i] = 1
            event_windows[i] = [start, end]

        x[i, 0] = sig

    return TimeSeriesDataset(x=x, y=y, event_windows=event_windows)


def train_test_split(ds: TimeSeriesDataset, test_frac: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(ds.y)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    def subset(indices):
        return TimeSeriesDataset(
            x=ds.x[indices], y=ds.y[indices], event_windows=ds.event_windows[indices]
        )

    return subset(train_idx), subset(test_idx)
