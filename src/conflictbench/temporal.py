"""Synthetic temporal conflict data and sequence-aware learned baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .core import ActionOutput, Dataset


TEMPORAL_LEARNED_METHODS = ("temporal_fusion", "temporal_gated")
TEMPORAL_MECHANISMS = (
    "clean",
    "invert",
    "swap",
    "dropout",
    "mixed",
    "burst",
    "ambiguity",
    "segment_invert",
    "occlusion",
    "drift",
)
TEMPORAL_PROTOCOL = "zero_mean_marker_v2"
TEMPORAL_DIAGNOSTIC_TRANSFORMS = (
    "identity",
    "shared_permutation",
    "independent_channel_permutation",
    "per_example_circular_shift",
)


@dataclass(frozen=True)
class TemporalDataset:
    """A sequence split with three modality traces per example."""

    x: np.ndarray
    y: np.ndarray
    ambiguous: np.ndarray
    mechanism: np.ndarray

    def __post_init__(self) -> None:
        if self.x.ndim != 3 or self.x.shape[1] != 3 or self.x.shape[2] < 2:
            raise ValueError("x must have shape [n, 3, time] with time >= 2")
        n = self.x.shape[0]
        if self.y.shape != (n,) or self.ambiguous.shape != (n,) or self.mechanism.shape != (n,):
            raise ValueError("dataset fields have inconsistent lengths")
        if not np.isfinite(self.x).all():
            raise ValueError("x contains non-finite values")
        if not np.isin(self.y, (0, 1)).all():
            raise ValueError("y must contain binary labels")


def _normalise_probabilities(
    mechanisms: tuple[str, ...], mechanism_probs: Mapping[str, float] | Iterable[float] | None
) -> np.ndarray:
    if mechanism_probs is None:
        return np.full(len(mechanisms), 1.0 / len(mechanisms), dtype=np.float64)
    if isinstance(mechanism_probs, Mapping):
        try:
            values = np.asarray([mechanism_probs[name] for name in mechanisms], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(f"mechanism_probs is missing {exc.args[0]!r}") from exc
    else:
        values = np.asarray(list(mechanism_probs), dtype=np.float64)
        if values.shape != (len(mechanisms),):
            raise ValueError("mechanism_probs must match the mechanisms list")
    if not np.isfinite(values).all() or np.any(values < 0) or float(values.sum()) <= 0:
        raise ValueError("mechanism_probs must be finite, non-negative, and non-zero")
    return values / values.sum()


def generate_temporal_dataset(
    seed: int,
    n: int,
    mechanisms: Iterable[str],
    *,
    seq_len: int = 64,
    strength: float = 1.0,
    noise: float = 0.22,
    positive_rate: float = 0.5,
    mechanism_probs: Mapping[str, float] | Iterable[float] | None = None,
) -> TemporalDataset:
    """Generate modality traces with explicit temporal conflict mechanisms.

    The hidden label controls a smooth event trace in each channel. Corruption
    changes are applied to complete channels or contiguous intervals, so a
    sequence model can use temporal structure while scalar controls receive
    only the mean trace. Mechanism names and labels remain evaluation metadata.
    """

    mechanisms = tuple(mechanisms)
    valid = set(TEMPORAL_MECHANISMS)
    if n <= 0:
        raise ValueError("n must be positive")
    if not mechanisms or any(name not in valid for name in mechanisms):
        raise ValueError(f"mechanisms must be drawn from {tuple(sorted(valid))}")
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if strength <= 0 or noise < 0:
        raise ValueError("strength must be positive and noise cannot be negative")
    if not 0.0 < positive_rate < 1.0:
        raise ValueError("positive_rate must be strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    probabilities = _normalise_probabilities(mechanisms, mechanism_probs)
    y = (rng.random(n) < positive_rate).astype(np.int64)
    signs = (2.0 * y - 1.0).astype(np.float32)
    mechanism = rng.choice(mechanisms, size=n, p=probabilities).astype("U16")

    # Encode the event in a zero-mean temporal marker.  Removing the channel
    # means is important: the scalar controls receive only this mean and must
    # not recover the label from a DC offset.  The two harmonics make a cyclic
    # lag observable without making the marker a single sinusoid shortcut.
    time = np.arange(seq_len, dtype=np.float32) / float(seq_len)
    phase = np.asarray((0.0, 0.73, 1.41), dtype=np.float32)
    frequency = np.asarray((1.0, 1.0, 1.0), dtype=np.float32)
    scale = np.asarray((1.00, 0.94, 1.06), dtype=np.float32)
    pattern = scale[:, None] * (
        np.sin(2.0 * np.pi * frequency[:, None] * time[None, :] + phase[:, None])
        + 0.35 * np.cos(4.0 * np.pi * time[None, :] - 0.5 * phase[:, None])
    )
    pattern = pattern - pattern.mean(axis=1, keepdims=True)
    x = signs[:, None, None] * float(strength) * pattern[None, :, :]
    x = x + rng.normal(0.0, noise, size=x.shape).astype(np.float32)
    ambiguous = np.zeros(n, dtype=bool)

    rows = np.arange(n)
    for name in mechanisms:
        mask = mechanism == name
        if not np.any(mask):
            continue
        count = int(mask.sum())
        indices = rows[mask]
        channels = rng.integers(0, 3, size=count)
        opposite = -signs[indices]
        if name == "invert":
            replacement = (
                opposite[:, None] * float(strength) * pattern[channels]
                + rng.normal(0.0, noise, size=(count, seq_len))
            ).astype(np.float32)
            x[indices, channels, :] = replacement
        elif name == "swap":
            # A source swap is an opposite event observed with a fixed,
            # nonzero temporal lag.  It is intentionally distinct from a
            # whole-sequence inversion while remaining observation-only.
            lags = rng.integers(max(1, seq_len // 16), max(2, seq_len // 8 + 1), size=count)
            for position, (row, channel, lag) in enumerate(zip(indices, channels, lags)):
                shifted = np.roll(pattern[channel], int(lag))
                x[row, channel, :] = (
                    opposite[position] * float(strength) * shifted
                    + rng.normal(0.0, noise, size=seq_len)
                ).astype(np.float32)
        elif name == "dropout":
            replacement = rng.normal(0.0, max(noise * 1.8, 0.05), size=(count, seq_len)).astype(np.float32)
            x[indices, channels, :] = replacement
        elif name == "segment_invert":
            starts = rng.integers(0, max(1, seq_len // 2), size=count)
            widths = rng.integers(max(2, seq_len // 8), max(3, seq_len // 3 + 1), size=count)
            for position, (row, channel, start, width_value) in enumerate(
                zip(indices, channels, starts, widths)
            ):
                stop = min(seq_len, int(start + width_value))
                x[row, channel, start:stop] = (
                    opposite[position] * float(strength) * pattern[channel, start:stop]
                    + rng.normal(0.0, noise, size=stop - start)
                ).astype(np.float32)
        elif name == "occlusion":
            starts = rng.integers(0, max(1, seq_len // 2), size=count)
            widths = rng.integers(max(2, seq_len // 8), max(3, seq_len // 3 + 1), size=count)
            for row, channel, start, width_value in zip(indices, channels, starts, widths):
                stop = min(seq_len, int(start + width_value))
                x[row, channel, start:stop] = rng.normal(
                    0.0, max(noise * 1.8, 0.05), size=stop - start
                ).astype(np.float32)
        elif name == "drift":
            # Add a channel-local baseline drift with zero net area.  This
            # stresses temporal encoders while preserving the scalar-control
            # no-DC-shortcut invariant.
            drift = np.linspace(-1.0, 1.0, seq_len, dtype=np.float32)
            drift -= drift.mean()
            amplitudes = rng.uniform(0.55, 1.05, size=count).astype(np.float32)
            x[indices, channels, :] += amplitudes[:, None] * drift[None, :]
        elif name == "mixed":
            other = (channels + rng.integers(1, 3, size=count)) % 3
            replacement = (
                opposite[:, None] * float(strength) * pattern[channels]
                + rng.normal(0.0, noise, size=(count, seq_len))
            ).astype(np.float32)
            x[indices, channels, :] = replacement
            dropout = rng.normal(0.0, max(noise * 1.8, 0.05), size=(count, seq_len)).astype(np.float32)
            x[indices, other, :] = dropout
        elif name == "burst":
            starts = rng.integers(0, max(1, seq_len // 2), size=count)
            widths = rng.integers(max(2, seq_len // 8), max(3, seq_len // 3 + 1), size=count)
            # The wrong channel is only high-confidence inside its burst.
            for position, (row, channel, start, width) in enumerate(zip(indices, channels, starts, widths)):
                stop = min(seq_len, int(start + width))
                x[row, channel, start:stop] = (
                    opposite[position] * float(strength) * 2.5 * pattern[channel, start:stop]
                    + rng.normal(0.0, noise, size=stop - start)
                )
        elif name == "ambiguity":
            order = rng.permutation(3) if count == 1 else np.asarray([rng.permutation(3) for _ in range(count)])
            if count == 1:
                order = order.reshape(1, 3)
            ambiguous[indices] = True
            x[indices] = rng.normal(0.0, noise, size=(count, 3, seq_len)).astype(np.float32)
            first = order[:, 0]
            second = order[:, 1]
            first_values = signs[indices, None] * float(strength) * pattern[first]
            second_values = opposite[:, None] * float(strength) * pattern[second]
            x[indices, first, :] = (first_values + rng.normal(0.0, noise, size=(count, seq_len))).astype(np.float32)
            x[indices, second, :] = (second_values + rng.normal(0.0, noise, size=(count, seq_len))).astype(np.float32)

    return TemporalDataset(x=x.astype(np.float32), y=y, ambiguous=ambiguous, mechanism=mechanism)


def temporal_summary(dataset: TemporalDataset) -> Dataset:
    """Collapse traces to the scalar observation used by fixed controls."""

    return Dataset(
        x=np.mean(dataset.x, axis=2, dtype=np.float64).astype(np.float32),
        y=dataset.y,
        ambiguous=dataset.ambiguous,
        mechanism=dataset.mechanism,
    )


def transform_temporal_dataset(dataset: TemporalDataset, *, mode: str, seed: int) -> TemporalDataset:
    """Apply a deterministic time transformation for shortcut diagnostics.

    ``shared_permutation`` uses one permutation for every example and may be
    reused across train and evaluation splits. ``independent_channel_permutation``
    destroys within-channel time order independently for each example. A
    circular shift is shared across channels within an example, preserving
    relative lags while removing the generator's global phase reference.
    """

    if mode not in TEMPORAL_DIAGNOSTIC_TRANSFORMS:
        raise ValueError(f"mode must be one of {TEMPORAL_DIAGNOSTIC_TRANSFORMS}")
    if mode == "identity":
        return dataset
    rng = np.random.default_rng(seed)
    values = np.asarray(dataset.x, dtype=np.float32).copy()
    n, channels, steps = values.shape
    if mode == "shared_permutation":
        permutation = rng.permutation(steps)
        values = values[:, :, permutation]
    elif mode == "independent_channel_permutation":
        for sample in range(n):
            for channel in range(channels):
                values[sample, channel, :] = values[sample, channel, rng.permutation(steps)]
    else:  # per_example_circular_shift
        shifts = rng.integers(0, steps, size=n)
        for sample, shift in enumerate(shifts):
            values[sample] = np.roll(values[sample], int(shift), axis=-1)
    return TemporalDataset(x=values, y=dataset.y, ambiguous=dataset.ambiguous, mechanism=dataset.mechanism)


def temporal_template_actions(train: TemporalDataset, evaluation: TemporalDataset) -> ActionOutput:
    """Classify full traces with train-only nearest class templates.

    This simple baseline tests whether a fixed class-level waveform is enough
    to explain sequence-model results. It always answers and does not perform
    acquisition or abstention.
    """

    observed = ~train.ambiguous
    templates = []
    for label in (0, 1):
        mask = observed & (train.y == label)
        if not np.any(mask):
            raise ValueError(f"train split has no non-ambiguous examples for label {label}")
        templates.append(np.mean(train.x[mask], axis=0, dtype=np.float64))
    template_array = np.asarray(templates, dtype=np.float64)
    difference = np.asarray(evaluation.x, dtype=np.float64)[:, None, :, :] - template_array[None, :, :, :]
    squared_error = np.mean(difference * difference, axis=(2, 3))
    action = np.argmin(squared_error, axis=1).astype(np.int64)
    return ActionOutput(action=action, query=np.zeros(len(action), dtype=bool))


def temporal_linear_actions(
    train: TemporalDataset, evaluation: TemporalDataset, *, ridge: float = 1e-3
) -> ActionOutput:
    """Fit a train-only ridge classifier over flattened full traces."""

    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    observed = ~train.ambiguous
    values = np.asarray(train.x[observed], dtype=np.float64).reshape(int(np.sum(observed)), -1)
    labels = np.asarray(train.y[observed], dtype=np.int64)
    if not np.isin(labels, (0, 1)).all() or len(np.unique(labels)) != 2:
        raise ValueError("train split must contain both non-ambiguous labels")
    design = np.concatenate([values, np.ones((len(values), 1), dtype=np.float64)], axis=1)
    targets = np.eye(2, dtype=np.float64)[labels]
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    evaluation_design = np.concatenate(
        [
            np.asarray(evaluation.x, dtype=np.float64).reshape(len(evaluation.y), -1),
            np.ones((len(evaluation.y), 1), dtype=np.float64),
        ],
        axis=1,
    )
    action = np.argmax(evaluation_design @ weights, axis=1).astype(np.int64)
    return ActionOutput(action=action, query=np.zeros(len(action), dtype=bool))


def _validate_temporal_method(method: str) -> None:
    if method not in TEMPORAL_LEARNED_METHODS:
        raise ValueError(f"unknown temporal method: {method}")


def fit_and_predict_temporal_torch(
    method: str,
    train: TemporalDataset,
    evaluation_splits: dict[str, TemporalDataset],
    *,
    seed: int,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    width: int = 96,
    depth: int = 4,
    device: str,
    torch_threads: int = 2,
    eval_batch_size: int = 1024,
) -> tuple[dict[str, ActionOutput], dict[str, Any]]:
    """Fit a sequence model for a fixed number of optimizer steps."""

    _validate_temporal_method(method)
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"PyTorch unavailable: {exc}") from exc
    if max_steps <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("max_steps, batch_size, and learning_rate must be positive")
    if width <= 0 or depth <= 0 or eval_batch_size <= 0:
        raise ValueError("width, depth, and eval_batch_size must be positive")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device == "cpu":
        torch.set_num_threads(max(1, int(torch_threads)))
    device_obj = torch.device(device)
    train_values = torch.from_numpy(np.asarray(train.x, dtype=np.float32)).to(device_obj)
    train_targets = torch.from_numpy(np.where(train.ambiguous, 2, train.y).astype(np.int64)).to(device_obj)

    groups = min(8, width)
    while width % groups:
        groups -= 1

    class Encoder(nn.Module):
        def __init__(self, in_channels: int) -> None:
            super().__init__()
            self.stem = nn.Conv1d(in_channels, width, kernel_size=5, padding=2)
            blocks = []
            for _ in range(depth):
                blocks.extend(
                    [
                        nn.Conv1d(width, width, kernel_size=3, padding=1),
                        nn.GroupNorm(groups, width),
                        nn.GELU(),
                        nn.Conv1d(width, width, kernel_size=3, padding=1),
                        nn.GroupNorm(groups, width),
                    ]
                )
            self.blocks = nn.ModuleList(blocks)
            self.projection = nn.Sequential(nn.Linear(width * 2, width), nn.GELU())

        def forward(self, values: Any) -> Any:
            hidden = self.stem(values)
            for index in range(0, len(self.blocks), 5):
                residual = hidden
                hidden = self.blocks[index](hidden)
                hidden = self.blocks[index + 1](hidden)
                hidden = self.blocks[index + 2](hidden)
                hidden = self.blocks[index + 3](hidden)
                hidden = self.blocks[index + 4](hidden)
                hidden = torch.nn.functional.gelu(hidden + residual)
            pooled = torch.cat([hidden.mean(dim=-1), hidden.amax(dim=-1)], dim=1)
            return self.projection(pooled)

    class Fusion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder(3)
            self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))

        def forward(self, values: Any) -> Any:
            return self.head(self.encoder(values))

    class Gated(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder(1)
            self.gate = nn.Sequential(nn.Linear(width, max(8, width // 2)), nn.GELU(), nn.Linear(max(8, width // 2), 1))
            self.head = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 3))

        def forward(self, values: Any) -> Any:
            batch, modalities, steps = values.shape
            embeddings = self.encoder(values.reshape(batch * modalities, 1, steps)).reshape(batch, modalities, width)
            weights = torch.softmax(self.gate(embeddings).squeeze(-1), dim=1)
            pooled = torch.sum(weights.unsqueeze(-1) * embeddings, dim=1)
            spread = embeddings.std(dim=1, unbiased=False)
            return self.head(torch.cat([pooled, spread], dim=1))

    model: Any = Fusion() if method == "temporal_fusion" else Gated()
    model.to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator(device=device_obj)
    generator.manual_seed(seed + 991)
    losses: list[float] = []
    model.train()
    for _ in range(max_steps):
        indices = torch.randint(0, len(train_values), (batch_size,), generator=generator, device=device_obj)
        batch_values = train_values.index_select(0, indices)
        batch_targets = train_targets.index_select(0, indices)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch_values), batch_targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    predictions: dict[str, ActionOutput] = {}
    model.eval()
    with torch.no_grad():
        for split_name, split in evaluation_splits.items():
            values = torch.from_numpy(np.asarray(split.x, dtype=np.float32))
            actions: list[np.ndarray] = []
            for start in range(0, len(values), eval_batch_size):
                batch = values[start : start + eval_batch_size].to(device_obj)
                actions.append(torch.argmax(model(batch), dim=1).cpu().numpy().astype(np.int64))
            action = np.concatenate(actions) if actions else np.empty(0, dtype=np.int64)
            predictions[split_name] = ActionOutput(action=action, query=np.zeros(len(action), dtype=bool))
    return predictions, {
        "status": "verified",
        "max_steps": int(max_steps),
        "batch_size": int(batch_size),
        "width": int(width),
        "depth": int(depth),
        "final_train_loss": float(np.mean(losses[-min(100, len(losses)) :])),
        "device": str(device_obj),
        "input_shape": [int(train.x.shape[1]), int(train.x.shape[2])],
    }
