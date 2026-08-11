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
    x: np.ndarray  # (n_samples, n_channels, length) float32
    y: np.ndarray  # (n_samples,) int64, 0 = normal, 1 = event
    event_windows: np.ndarray  # (n_samples, 2) start/end index, -1 if none
    # Which channel carries the event, -1 if none. Only set by the multivariate
    # generator; the single-channel generator leaves it at 0 for event samples.
    event_channels: np.ndarray | None = None
    channel_names: tuple[str, ...] = ("ppg",)


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

    event_channels = np.where(y == 1, 0, -1).astype(np.int64)
    return TimeSeriesDataset(
        x=x,
        y=y,
        event_windows=event_windows,
        event_channels=event_channels,
        channel_names=("ppg",),
    )


def make_multivariate_dataset(
    n_samples: int = 400,
    length: int = 256,
    fs: float = 64.0,
    noise_std: float = 0.08,
    event_frac: float = 0.5,
    crosstalk: float = 0.45,
    event_sec: float = 0.8,
    seed: int = 0,
) -> TimeSeriesDataset:
    """Three-channel wearable-style dataset: (ppg, accel, eda).

    The event burst is injected into one randomly chosen *primary* channel per
    positive sample. This gives ground truth along both axes an attribution can
    get wrong:

    - **when** the event happened (`event_windows`)
    - **which channel** carried it (`event_channels`)

    That two-axis ground truth is what makes Temporal Saliency Rescaling
    measurable. On single-channel data TSR has no feature axis to separate,
    so its decomposition is vacuous by construction.

    ``crosstalk`` leaks an attenuated copy of the burst into the other channels,
    which is what real wearables do — a wrist motion artifact shows up in the
    accelerometer *and* pushes the optical PPG trace around. Without it the
    channel axis is trivially separable and every method scores 100%, so the
    metric tells you nothing. With it, a method has to distinguish the primary
    source from its echoes.

    The label is **4-class and channel-dependent**: ``0`` = no event, and
    ``1 + c`` = event whose primary source is channel ``c``. This matters for
    the experiment's validity. Under a binary event/no-event label the model
    never needs to know *which* sensor fired, so scoring an attribution on the
    channel axis would punish it for not surfacing information the model had
    no reason to learn. Making the label channel-dependent means a correct
    model provably must use channel identity, so channel attribution becomes a
    fair thing to measure.

    ``event_sec`` sets the burst duration. Short bursts (the 0.8s default) suit
    fixed-window occlusion fine. Long bursts create **redundant evidence**:
    blanking any one small window leaves the rest of the burst visible, the
    prediction barely moves, and occlusion reports near-zero importance
    everywhere. That regime is the point of the Dynamask comparison.
    """
    rng = np.random.default_rng(seed)
    n_channels = 3
    x = np.zeros((n_samples, n_channels, length), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.int64)
    event_windows = np.full((n_samples, 2), -1, dtype=np.int64)
    event_channels = np.full((n_samples,), -1, dtype=np.int64)

    n_events = int(n_samples * event_frac)
    event_idx = set(rng.choice(n_samples, size=n_events, replace=False).tolist())

    t = np.arange(length) / fs
    for i in range(n_samples):
        hr = rng.uniform(1.0, 1.4)
        phase = rng.uniform(0, 2 * np.pi)

        # ch0: PPG pulse waveform
        x[i, 0] = _ppg_wave(length, fs, hr, phase)
        # ch1: accelerometer — slower, lower amplitude sway
        x[i, 1] = 0.6 * np.sin(2 * np.pi * rng.uniform(0.2, 0.5) * t + rng.uniform(0, 6.28))
        # ch2: EDA — very slow drift
        x[i, 2] = 0.4 * np.sin(2 * np.pi * rng.uniform(0.05, 0.15) * t + rng.uniform(0, 6.28))
        x[i] += rng.normal(0, noise_std, size=(n_channels, length)).astype(np.float32)

        if i in event_idx:
            win = int(fs * event_sec)
            margin = max(1, (length - win) // 4)
            start = int(rng.integers(margin, max(margin + 1, length - win - margin)))
            end = min(start + win, length)
            win = end - start
            ch = int(rng.integers(0, n_channels))

            burst = _ppg_wave(win, fs, hr * rng.uniform(2.2, 3.0), phase)
            x[i, ch, start:end] = burst * rng.uniform(1.3, 1.8)

            # attenuated echoes of the same burst in the other sensors
            for other in range(n_channels):
                if other == ch:
                    continue
                leak = crosstalk * rng.uniform(0.7, 1.0)
                x[i, other, start:end] += burst * leak

            y[i] = 1 + ch  # class encodes which sensor was the primary source
            event_windows[i] = [start, end]
            event_channels[i] = ch

    return TimeSeriesDataset(
        x=x,
        y=y,
        event_windows=event_windows,
        event_channels=event_channels,
        channel_names=("ppg", "accel", "eda"),
    )


def train_test_split(ds: TimeSeriesDataset, test_frac: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(ds.y)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    def subset(indices):
        return TimeSeriesDataset(
            x=ds.x[indices],
            y=ds.y[indices],
            event_windows=ds.event_windows[indices],
            event_channels=(
                None if ds.event_channels is None else ds.event_channels[indices]
            ),
            channel_names=ds.channel_names,
        )

    return subset(train_idx), subset(test_idx)
