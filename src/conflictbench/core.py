"""Synthetic data, transparent policies, metrics, and optional torch models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Optional

import numpy as np


MODALITIES = ("audio", "video", "text")
ALL_MECHANISMS = ("clean", "invert", "swap", "dropout", "mixed", "burst", "ambiguity")
HEURISTIC_METHODS = ("majority", "weighted", "median", "active_diagnostic")
LEARNED_METHODS = ("mlp", "gated", "learned_acquisition")
DEFAULT_ACQUISITION_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


@dataclass(frozen=True)
class Dataset:
    """A generated split. Policies receive x only; the other fields are evaluation metadata."""

    x: np.ndarray
    y: np.ndarray
    ambiguous: np.ndarray
    mechanism: np.ndarray

    def __post_init__(self) -> None:
        if self.x.ndim != 2 or self.x.shape[1] != 3:
            raise ValueError("x must have shape [n, 3]")
        n = self.x.shape[0]
        if self.y.shape != (n,) or self.ambiguous.shape != (n,) or self.mechanism.shape != (n,):
            raise ValueError("dataset fields have inconsistent lengths")
        if not np.isfinite(self.x).all():
            raise ValueError("x contains non-finite values")


def _latent_score(label: int, strength: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    return (2.0 * label - 1.0) * strength + rng.normal(0.0, noise, size=3)


def generate_dataset(
    seed: int,
    n: int,
    mechanisms: Iterable[str],
    *,
    strength: float = 1.0,
    noise: float = 0.22,
) -> Dataset:
    """Generate balanced latent events with explicit but hidden corruption mechanisms."""

    mechanisms = tuple(mechanisms)
    if n <= 0:
        raise ValueError("n must be positive")
    if not mechanisms or any(m not in ALL_MECHANISMS for m in mechanisms):
        raise ValueError(f"mechanisms must be drawn from {ALL_MECHANISMS}")
    if strength <= 0 or noise < 0:
        raise ValueError("strength must be positive and noise cannot be negative")

    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n, dtype=np.int64)
    mechanism = rng.choice(mechanisms, size=n).astype("U16")
    x = np.zeros((n, 3), dtype=np.float32)
    ambiguous = np.zeros(n, dtype=bool)

    for i, name in enumerate(mechanism.tolist()):
        label = int(y[i])
        clean = _latent_score(label, strength, noise, rng)
        index = int(rng.integers(0, 3))
        if name == "clean":
            values = clean
        elif name == "invert":
            values = clean.copy()
            values[index] = (1.0 - 2.0 * label) * strength + rng.normal(0.0, noise)
        elif name == "swap":
            values = clean.copy()
            values[index] = (1.0 - 2.0 * label) * strength + rng.normal(0.0, noise)
        elif name == "dropout":
            values = clean.copy()
            values[index] = rng.normal(0.0, max(noise * 1.8, 0.05))
        elif name == "mixed":
            values = clean.copy()
            other = (index + int(rng.integers(1, 3))) % 3
            values[index] = (1.0 - 2.0 * label) * strength + rng.normal(0.0, noise)
            values[other] = rng.normal(0.0, max(noise * 1.8, 0.05))
        elif name == "burst":
            values = clean * 0.7
            values[index] = (1.0 - 2.0 * label) * strength * 2.5 + rng.normal(0.0, noise)
        elif name == "ambiguity":
            # Two strong, opposite observations and one weak observation do not identify a source.
            order = rng.permutation(3)
            values = np.zeros(3, dtype=np.float64)
            values[order[0]] = (2.0 * label - 1.0) * strength + rng.normal(0.0, noise)
            values[order[1]] = (1.0 - 2.0 * label) * strength + rng.normal(0.0, noise)
            values[order[2]] = rng.normal(0.0, max(noise * 0.6, 0.04))
            ambiguous[i] = True
        else:  # guarded above; keeps static checkers honest
            raise AssertionError(name)
        x[i] = values

    return Dataset(x=x, y=y, ambiguous=ambiguous, mechanism=mechanism)


@dataclass(frozen=True)
class ActionOutput:
    """Predicted action: 0/1 answer or 2 for abstention, plus request indicators."""

    action: np.ndarray
    query: np.ndarray


def _label_from_score(score: np.ndarray) -> np.ndarray:
    return (score >= 0.0).astype(np.int64)


def policy_actions(name: str, x: np.ndarray, *, threshold: float = 0.35) -> ActionOutput:
    """Run a policy using observations only."""

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape [n, 3]")
    if name == "majority":
        action = (np.sum(x >= 0.0, axis=1) >= 2).astype(np.int64)
        return ActionOutput(action=action, query=np.zeros(len(x), dtype=bool))
    if name == "weighted":
        action = _label_from_score(np.sum(x * np.abs(x), axis=1))
        return ActionOutput(action=action, query=np.zeros(len(x), dtype=bool))
    if name == "median":
        action = _label_from_score(np.median(x, axis=1))
        return ActionOutput(action=action, query=np.zeros(len(x), dtype=bool))
    if name != "active_diagnostic":
        raise ValueError(f"unknown observation-only policy: {name}")

    first = x[:, :2]
    first_sign = first >= 0.0
    weak = np.any(np.abs(first) < threshold, axis=1)
    disagreement = first_sign[:, 0] != first_sign[:, 1]
    query = disagreement | weak
    action = np.full(len(x), 2, dtype=np.int64)
    no_query = ~query
    action[no_query] = first_sign[no_query, 0].astype(np.int64)

    # After a request, answer only if at least two sufficiently strong channels agree.
    strong = np.abs(x) >= threshold
    positive = np.sum(strong & (x >= 0.0), axis=1)
    negative = np.sum(strong & (x < 0.0), axis=1)
    enough_positive = query & (positive >= 2)
    enough_negative = query & (negative >= 2)
    action[enough_positive] = 1
    action[enough_negative] = 0
    return ActionOutput(action=action, query=query)


def _pair_action(x: np.ndarray) -> np.ndarray:
    """Answer from the initially observed pair, abstaining on disagreement."""

    signs = x[:, :2] >= 0.0
    agree = signs[:, 0] == signs[:, 1]
    action = np.full(len(x), 2, dtype=np.int64)
    action[agree] = signs[agree, 0].astype(np.int64)
    return action


def _post_query_action(x: np.ndarray, *, threshold: float = 0.35) -> np.ndarray:
    """Answer after acquiring the third score using strong-score majority."""

    strong = np.abs(x) >= threshold
    positive = np.sum(strong & (x >= 0.0), axis=1)
    negative = np.sum(strong & (x < 0.0), axis=1)
    action = np.full(len(x), 2, dtype=np.int64)
    action[positive >= 2] = 1
    action[negative >= 2] = 0
    return action


def acquisition_actions(
    x: np.ndarray,
    query_score: np.ndarray,
    *,
    threshold: float,
    initial_threshold: float = 0.35,
) -> ActionOutput:
    """Apply a learned query threshold while keeping decisions observation-only."""

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape [n, 3]")
    score = np.asarray(query_score, dtype=float)
    if score.shape != (len(x),) or not np.isfinite(score).all():
        raise ValueError("query_score must be a finite vector matching x")
    query = score >= float(threshold)
    action = _pair_action(x)
    queried_action = _post_query_action(x, threshold=initial_threshold)
    action[query] = queried_action[query]
    return ActionOutput(action=action, query=query)


def acquisition_training_targets(
    train: Dataset,
    *,
    query_cost: float = 0.10,
    abstain_cost: float = 0.20,
    initial_threshold: float = 0.35,
) -> np.ndarray:
    """Label when querying improves train utility, using observations plus labels only."""

    target = np.where(train.ambiguous, 2, train.y)
    pair = _pair_action(train.x)
    queried = _post_query_action(train.x, threshold=initial_threshold)

    def reward(action: np.ndarray) -> np.ndarray:
        value = np.where(action == target, 1.0, -1.0)
        value -= abstain_cost * ((action == 2) & ~train.ambiguous)
        return value

    # The acquisition model learns the cost-sensitive decision boundary; it
    # never receives mechanism names or labels at inference time.
    return ((reward(queried) - reward(pair)) > query_cost).astype(np.float32)


def _safe_mean(values: np.ndarray) -> Optional[float]:
    """Return a finite mean, or ``None`` when the subgroup is empty.

    Empty subgroups are expected for split-specific metrics (for example, an
    ambiguity rate on a split with no ambiguity cases). ``None`` keeps the
    result valid JSON and prevents an undefined quantity from entering
    summary statistics as a NaN.
    """

    return float(np.mean(values)) if len(values) else None


def evaluate_actions(
    output: ActionOutput,
    dataset: Dataset,
    *,
    query_cost: float = 0.10,
    abstain_cost: float = 0.20,
    initial_threshold: float = 0.35,
) -> dict[str, Any]:
    """Compute decision, selectivity, request, and per-mechanism metrics."""

    action = np.asarray(output.action, dtype=np.int64)
    query = np.asarray(output.query, dtype=bool)
    if action.shape != dataset.y.shape or query.shape != dataset.y.shape:
        raise ValueError("action and query lengths must match dataset")
    if np.any((action < 0) | (action > 2)):
        raise ValueError("actions must be 0, 1, or 2")

    target = np.where(dataset.ambiguous, 2, dataset.y)
    correct = action == target
    covered = action != 2
    nonambiguous = ~dataset.ambiguous
    initial_disagreement = (dataset.x[:, 0] >= 0.0) != (dataset.x[:, 1] >= 0.0)
    initial_disagreement |= np.any(np.abs(dataset.x[:, :2]) < initial_threshold, axis=1)
    utility = np.where(correct, 1.0, -1.0)
    utility = utility - query_cost * query.astype(float)
    utility = utility - abstain_cost * ((action == 2) & nonambiguous).astype(float)

    metrics: dict[str, Any] = {
        "n": int(len(action)),
        "decision_accuracy": _safe_mean(correct),
        "coverage": _safe_mean(covered),
        "selective_accuracy": _safe_mean(correct[covered]),
        "nonambiguous_accuracy": _safe_mean(correct[nonambiguous]),
        "overall_error_rate": _safe_mean(~correct),
        "abstain_rate": _safe_mean(action == 2),
        "ambiguous_abstain_rate": _safe_mean((action == 2)[dataset.ambiguous]),
        "nonambiguous_abstain_rate": _safe_mean((action == 2)[nonambiguous]),
        "query_rate": _safe_mean(query),
        "query_rate_on_initial_disagreement": _safe_mean(query[initial_disagreement]),
        "utility": _safe_mean(utility),
    }

    by_mechanism: dict[str, Any] = {}
    for mechanism in sorted(set(dataset.mechanism.tolist())):
        mask = dataset.mechanism == mechanism
        by_mechanism[mechanism] = {
            "n": int(np.sum(mask)),
            "decision_accuracy": _safe_mean(correct[mask]),
            "coverage": _safe_mean(covered[mask]),
            "selective_accuracy": _safe_mean(correct[mask & covered]),
            "abstain_rate": _safe_mean((action == 2)[mask]),
            "query_rate": _safe_mean(query[mask]),
            "utility": _safe_mean(utility[mask]),
        }
    return {"metrics": metrics, "by_mechanism": by_mechanism}


def feature_matrix(dataset: Dataset) -> np.ndarray:
    """Observation-only features used by learned baselines."""

    return np.concatenate([dataset.x, np.abs(dataset.x)], axis=1).astype(np.float32)


def acquisition_feature_matrix(dataset: Dataset) -> np.ndarray:
    """Return the first-pair features available before requesting text.

    The acquisition scorer must not inspect the third modality before it
    decides whether to request it.  Keep this transformation separate from
    ``feature_matrix`` so a positional slice cannot accidentally expose text.
    """

    pair = dataset.x[:, :2]
    return np.concatenate([pair, np.abs(pair)], axis=1).astype(np.float32)


class _UnavailableTorch:
    pass


def torch_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment
        return {"available": False, "reason": str(exc)}
    return {
        "available": True,
        "version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def fit_and_predict_torch(
    method: str,
    train: Dataset,
    evaluation_splits: dict[str, Dataset],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    torch_threads: int = 2,
) -> tuple[dict[str, ActionOutput], dict[str, Any]]:
    """Fit a small observation-only model and return actions for each split."""

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"PyTorch unavailable: {exc}") from exc
    if method not in LEARNED_METHODS:
        raise ValueError(f"unknown learned method: {method}")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device == "cpu":
        torch.set_num_threads(max(1, int(torch_threads)))
    device_obj = torch.device(device)

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 3))

        def forward(self, values: Any) -> Any:
            return self.net(values)

    class Gated(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Sequential(nn.Linear(6, 24), nn.Tanh(), nn.Linear(24, 3))
            self.head = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3))

        def forward(self, values: Any) -> Any:
            raw = values[:, :3]
            weights = torch.softmax(self.gate(values), dim=-1)
            pooled = torch.sum(weights * raw, dim=1, keepdim=True)
            spread = torch.std(raw, dim=1, keepdim=True, unbiased=False)
            return self.head(torch.cat([pooled, spread], dim=1))

    model: Any = MLP() if method == "mlp" else Gated()
    model.to(device_obj)
    features = torch.from_numpy(feature_matrix(train))
    target = torch.from_numpy(np.where(train.ambiguous, 2, train.y).astype(np.int64))
    loader = DataLoader(TensorDataset(features, target), batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    last_loss = math.nan
    model.train()
    for _ in range(epochs):
        losses = []
        for batch_features, batch_target in loader:
            batch_features = batch_features.to(device_obj)
            batch_target = batch_target.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses)) if losses else math.nan

    model.eval()
    predictions: dict[str, ActionOutput] = {}
    with torch.no_grad():
        for split_name, split in evaluation_splits.items():
            values = torch.from_numpy(feature_matrix(split)).to(device_obj)
            action = torch.argmax(model(values), dim=1).detach().cpu().numpy().astype(np.int64)
            predictions[split_name] = ActionOutput(action=action, query=np.zeros(len(action), dtype=bool))
    return predictions, {"status": "verified", "epochs": epochs, "final_train_loss": last_loss, "device": str(device_obj)}


def fit_and_predict_acquisition_torch(
    train: Dataset,
    calibration: Dataset,
    evaluation_splits: dict[str, Dataset],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    query_cost: float = 0.10,
    abstain_cost: float = 0.20,
    initial_threshold: float = 0.35,
    threshold_grid: Iterable[float] = DEFAULT_ACQUISITION_THRESHOLDS,
    torch_threads: int = 2,
) -> tuple[dict[str, ActionOutput], dict[str, Any]]:
    """Fit an observation-only query scorer and calibrate it on train-only data."""

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"PyTorch unavailable: {exc}") from exc
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if len(calibration.y) == 0:
        raise ValueError("acquisition calibration split cannot be empty")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device == "cpu":
        torch.set_num_threads(max(1, int(torch_threads)))
    device_obj = torch.device(device)

    class QueryScorer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 24),
                nn.ReLU(),
                nn.Linear(24, 12),
                nn.ReLU(),
                nn.Linear(12, 1),
            )

        def forward(self, values: Any) -> Any:
            return self.net(values).squeeze(-1)

    model: Any = QueryScorer().to(device_obj)
    pair_features = acquisition_feature_matrix(train)
    query_target = acquisition_training_targets(
        train,
        query_cost=query_cost,
        abstain_cost=abstain_cost,
        initial_threshold=initial_threshold,
    )
    features = torch.from_numpy(pair_features)
    targets = torch.from_numpy(query_target)
    loader = DataLoader(TensorDataset(features, targets), batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    last_loss = math.nan
    model.train()
    for _ in range(epochs):
        losses = []
        for batch_features, batch_target in loader:
            batch_features = batch_features.to(device_obj)
            batch_target = batch_target.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses)) if losses else math.nan

    def scores(dataset: Dataset) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            values = torch.from_numpy(acquisition_feature_matrix(dataset)).to(device_obj)
            return torch.sigmoid(model(values)).detach().cpu().numpy().astype(float)

    calibration_scores = scores(calibration)
    candidates = tuple(sorted({float(value) for value in threshold_grid if 0.0 <= float(value) <= 1.0}))
    if not candidates:
        raise ValueError("threshold_grid must contain at least one value in [0, 1]")
    calibration_records = []
    for candidate in candidates:
        output = acquisition_actions(
            calibration.x,
            calibration_scores,
            threshold=candidate,
            initial_threshold=initial_threshold,
        )
        metrics = evaluate_actions(
            output,
            calibration,
            query_cost=query_cost,
            abstain_cost=abstain_cost,
            initial_threshold=initial_threshold,
        )["metrics"]
        calibration_records.append({"threshold": candidate, "utility": metrics["utility"], "query_rate": metrics["query_rate"]})
    chosen = max(calibration_records, key=lambda row: (float(row["utility"]), -float(row["threshold"])))
    chosen_threshold = float(chosen["threshold"])

    predictions: dict[str, ActionOutput] = {}
    for split_name, split in evaluation_splits.items():
        predictions[split_name] = acquisition_actions(
            split.x,
            scores(split),
            threshold=chosen_threshold,
            initial_threshold=initial_threshold,
        )
    return predictions, {
        "status": "verified",
        "epochs": epochs,
        "final_train_loss": last_loss,
        "device": str(device_obj),
        "calibration_size": int(len(calibration.y)),
        "calibration_source": "train_only",
        "input_features": ["audio", "video", "abs_audio", "abs_video"],
        "unavailable_modalities": ["text"],
        "threshold_grid": list(candidates),
        "chosen_threshold": chosen_threshold,
        "calibration_utility": float(chosen["utility"]),
        "calibration_query_rate": float(chosen["query_rate"]),
    }
