"""Dataset loading, deterministic partitioning, and tabular federated training.

Supported datasets: covertype, pendigits, can_mirgu, ciciov2024.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SUPPORTED_DATASETS = frozenset({"covertype", "pendigits", "can_mirgu", "ciciov2024"})


def _normalize_dataset_name(dataset_name: str) -> str:
    name = str(dataset_name).strip().lower().replace("-", "_")
    if name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset {dataset_name!r}. "
            f"Supported datasets: {', '.join(sorted(SUPPORTED_DATASETS))}"
        )
    return name


def resolved_attack_config(run_config: dict) -> dict:
    """Return effective run settings without mutating Flower's shared configuration.

    No-attack is a FedAvg run with zero malicious clients, independently
    of any configured attack_type or malicious-percentage values.
    """
    config = dict(run_config)
    if str(config.get("aggregation_method", "")).strip().lower() == "no-attack":
        for key in (
            "Malicious-Client-Percentage",
            "Malicious-Client-Percentage-min",
            "Malicious-Client-Percentage-max",
        ):
            config[key] = 0.0
    return config


def _deterministic_tabular_client_indices(
    n: int,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    """Select the fixed tabular client pool using the legacy pandas sampling order.

    This reproduces the membership and row order previously produced by
    ``df.sample(n=dataset_max_samples, random_state=seed).reset_index(drop=True)``.
    Keeping this order preserves the existing deterministic Dirichlet partitions,
    while allowing FLTrust data to be selected later from unused rows.
    """
    n = int(n)
    k = int(max_samples)
    if n <= 0:
        return np.empty((0,), dtype=np.int64)
    if k <= 0 or k >= n:
        return np.arange(n, dtype=np.int64)
    sampled = pd.Series(np.arange(n, dtype=np.int64)).sample(
        n=k,
        random_state=int(seed),
        replace=False,
    )
    return sampled.to_numpy(dtype=np.int64, copy=False)


def _transform_features_for_loader(
    X: np.ndarray,
    *,
    artifacts: "DatasetArtifacts",
    use_global_scaler: bool,
    fit_local_scaler: bool,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)

    if use_global_scaler:
        return ((X - artifacts.feature_mean) / artifacts.feature_scale).astype(np.float32)

    if fit_local_scaler:
        scaler = StandardScaler()
        return scaler.fit_transform(X).astype(np.float32)

    return X.astype(np.float32, copy=False)


def _transform_train_test_for_loader(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    artifacts: "DatasetArtifacts",
    use_global_scaler: bool,
) -> tuple[np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    if use_global_scaler:
        X_train = ((X_train - artifacts.feature_mean) / artifacts.feature_scale).astype(np.float32)
        X_test = ((X_test - artifacts.feature_mean) / artifacts.feature_scale).astype(np.float32)
        return X_train, X_test

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test


# --------------------------------------------------------------------------------------
# Malicious client selection (kept deterministic and stable across the project)
# --------------------------------------------------------------------------------------

def round_malicious_percentage(
    server_round: int,
    mal_pct_min: float,
    mal_pct_max: float,
    *,
    base_seed: int = 42,
    seed_offset: int = 0,
) -> float:
    """Deterministically choose the malicious percentage for this round.

    If min==max, this returns that exact value.
    """
    lo = float(min(mal_pct_min, mal_pct_max))
    hi = float(max(mal_pct_min, mal_pct_max))
    lo = max(0.0, min(1.0, lo))
    hi = max(0.0, min(1.0, hi))
    if hi <= lo:
        return lo
    rng = random.Random(int(base_seed) + 777_777 + int(seed_offset) + int(server_round))
    return lo + (hi - lo) * float(rng.random())


def sampled_partition_ids_for_round(
    server_round: int,
    num_clients: int,
    fraction_fit: float,
    *,
    base_seed: int = 42,
    seed_offset: int = 0,
) -> List[str]:
    """Reproduce the deterministic sampled partition-ids for this round.

    This mirrors the server-side deterministic sampling which seeds Python's RNG with
    (base_seed + 100_000 * round) and samples from clients ordered by partition-id.
    """
    n = int(num_clients)
    if n <= 0:
        return []
    k = max(1, int(math.ceil(float(fraction_fit) * n)))
    k = min(k, n)
    round_seed = int(base_seed) + 100_000 * int(server_round) + int(seed_offset)
    rng = random.Random(round_seed)
    chosen = rng.sample(list(range(n)), k=k)
    try:
        chosen_sorted = sorted(chosen)
    except Exception:
        chosen_sorted = chosen
    return [str(i) for i in chosen_sorted]


def malicious_ids_for_round(
    server_round: int,
    num_clients: int,
    mal_pct: float,
    *,
    seed_offset: int = 0,
    # New (optional) behavior: choose malicious as a deterministic SUBSET of selected clients
    selected_ids: Optional[List[str]] = None,
    mal_pct_min: Optional[float] = None,
    mal_pct_max: Optional[float] = None,
    base_seed: int = 42,
) -> Set[str]:
    """Return a deterministic set of malicious client IDs for the given round.

    - If `selected_ids` is provided, malicious clients are chosen as a *subset of that
      selected set*.
    - If `mal_pct_min/max` are provided, the per-round percentage is deterministically
      chosen in [min,max]; otherwise `mal_pct` is used.

    IDs are strings, matching how partition-id is stored in context.node_config.
    """
    # Determine per-round percentage (deterministic if min/max provided)
    if mal_pct_min is not None or mal_pct_max is not None:
        lo = float(mal_pct if mal_pct_min is None else mal_pct_min)
        hi = float(mal_pct if mal_pct_max is None else mal_pct_max)
        mal_pct_round = round_malicious_percentage(
            server_round,
            lo,
            hi,
            base_seed=base_seed,
            seed_offset=seed_offset,
        )
    else:
        mal_pct_round = float(mal_pct)

    if mal_pct_round <= 0.0:
        return set()

    if selected_ids is not None:
        # Sort pool for stability even if the caller's selected_ids order differs
        pool = [str(x) for x in selected_ids]
        try:
            pool = sorted(pool, key=lambda x: int(x) if str(x).isdigit() else str(x))
        except Exception:
            pool = sorted(pool)
        m = len(pool)
        if m <= 0:
            return set()
        k = int(math.floor(m * mal_pct_round))
        if k <= 0:
            return set()
        k = min(k, m)
        rng = random.Random(int(base_seed) + 1337 + int(seed_offset) + int(server_round))
        chosen = rng.sample(pool, k=k)
        return set(map(str, chosen))

    # Backward-compatible behavior: malicious set is chosen from ALL clients
    k = int(math.floor(int(num_clients) * mal_pct_round))
    if k <= 0:
        return set()
    rng = random.Random(int(base_seed) + 1337 + int(seed_offset) + int(server_round))
    ids = list(range(int(num_clients)))
    rng.shuffle(ids)
    chosen = ids[:k]
    return {str(i) for i in chosen}


# --------------------------------------------------------------------------------------
# Dataset + preprocessing cache
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetKey:
    dataset_name: str
    max_samples: int
    alpha: float
    num_partitions: int
    seed: int
    label_column: str
    num_classes_override: int
    test_fraction: float
    min_partition_size: int
    fltrust_root_size: int
    fltrust_root_seed_offset: int
    fltrust_root_mode: str
    fltrust_root_alpha: float
    fltrust_root_min_per_class: int


@dataclass
class DatasetArtifacts:
    X_all: np.ndarray  # shape [N, D], float32
    y_all: np.ndarray  # shape [N], int64 in [0..C-1]
    input_dim: int
    num_classes: int
    partitions: List[np.ndarray]  # len = num_partitions

    # Global standardization params (shared across all partitions)
    feature_mean: np.ndarray  # shape [D], float32
    feature_scale: np.ndarray  # shape [D], float32 (std; zeros -> 1)

    # FLTrust root subset indices. Client partitions are fixed first; these root
    # indices are then selected only from samples unused by every client.
    root_indices: np.ndarray  # shape [R], int64 indices into X_all/y_all


_DATASET_CACHE: Dict[DatasetKey, DatasetArtifacts] = {}


def _project_root() -> Path:
    # task.py is inside fltabular/
    return Path(__file__).resolve().parent


def _resolve_dataset_file(dataset_name: str, dataset_dir: Optional[str] = None) -> Path:
    """Expected layout:
    fltabular/data/<dataset_name>.csv   (or <dataset_name>.csv.gz)

    You can override dataset_dir via toml, but the default is fltabular/data.
    """
    base = Path(dataset_dir) if dataset_dir else (_project_root() / "data")
    cand1 = base / f"{dataset_name}.csv"
    cand2 = base / f"{dataset_name}.csv.gz"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    raise FileNotFoundError(
        f"Dataset file not found. Expected one of:\n  {cand1}\n  {cand2}"
    )


def _encode_labels(y_raw: pd.Series, num_classes_override: int) -> Tuple[np.ndarray, int]:
    y_cat = pd.Categorical(y_raw)
    y = y_cat.codes.astype(np.int64)

    if (y < 0).any():
        raise ValueError("Label column contains missing values; please clean the dataset.")

    inferred = int(y_cat.categories.size)
    if num_classes_override > 0:
        if inferred > num_classes_override:
            raise ValueError(
                f"num_classes={num_classes_override} < inferred={inferred}."
            )
        return y, int(num_classes_override)

    return y, inferred


def _dirichlet_partition(
    y: np.ndarray,
    num_partitions: int,
    alpha: float,
    seed: int,
    min_partition_size: int,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    classes = np.unique(y)
    idx_by_class = {c: np.where(y == c)[0] for c in classes}

    for c in classes:
        rng.shuffle(idx_by_class[c])

    parts: List[List[int]] = [[] for _ in range(num_partitions)]

    if alpha <= 0.0:
        all_idx = np.arange(n)
        rng.shuffle(all_idx)
        splits = np.array_split(all_idx, num_partitions)
        return [s.astype(np.int64) for s in splits]

    for c in classes:
        idx_c = idx_by_class[c]
        if len(idx_c) == 0:
            continue
        props = rng.dirichlet(alpha * np.ones(num_partitions))
        counts = (props * len(idx_c)).astype(int)

        diff = len(idx_c) - int(counts.sum())
        if diff > 0:
            counts[rng.choice(num_partitions, size=diff, replace=True)] += 1
        elif diff < 0:
            for _ in range(-diff):
                j = int(rng.integers(0, num_partitions))
                tries = 0
                while counts[j] <= 0 and tries < 10 * num_partitions:
                    j = int(rng.integers(0, num_partitions))
                    tries += 1
                if counts[j] > 0:
                    counts[j] -= 1

        start = 0
        for j in range(num_partitions):
            take = int(counts[j])
            if take <= 0:
                continue
            sel = idx_c[start : start + take]
            parts[j].extend(sel.tolist())
            start += take

        leftover = idx_c[start:]
        for idx in leftover:
            parts[int(rng.integers(0, num_partitions))].append(int(idx))

    sizes = np.array([len(p) for p in parts], dtype=int)
    while sizes.min() < min_partition_size:
        small = int(sizes.argmin())
        large = int(sizes.argmax())
        if sizes[large] <= min_partition_size:
            break
        moved = parts[large].pop()
        parts[small].append(moved)
        sizes[small] += 1
        sizes[large] -= 1

    out = []
    for p in parts:
        arr = np.array(p, dtype=np.int64)
        rng.shuffle(arr)
        out.append(arr)
    return out


def _get(run_config: dict, key: str, default):
    return run_config.get(key, default)


def _select_fltrust_root_indices(
    y: np.ndarray,
    *,
    root_n: int,
    rng: np.random.Generator,
    mode: str,
    alpha_root: float,
    min_per_class: int,
) -> np.ndarray:
    """Select exactly ``root_n`` unique FLTrust root samples.

    ``mode='random'`` samples uniformly from the full training pool.

    ``mode='dirichlet'`` performs class-aware sampling. ``min_per_class`` is
    treated as a best-effort lower bound: when ``root_n`` is too small to give
    every present class that many samples, the available root budget is shared
    across classes without ever exceeding ``root_n``.
    """
    y = np.asarray(y, dtype=np.int64)
    n = int(y.shape[0])
    k = int(max(0, min(int(root_n), n)))
    if k <= 0:
        return np.empty((0,), dtype=np.int64)

    mode_norm = str(mode or "random").strip().lower()
    if mode_norm != "dirichlet":
        return rng.choice(n, size=k, replace=False).astype(np.int64)

    classes = np.unique(y)
    if int(classes.size) <= 0:
        return rng.choice(n, size=k, replace=False).astype(np.int64)

    class_indices: list[np.ndarray] = []
    for cls in classes.tolist():
        idx = np.where(y == int(cls))[0].astype(np.int64)
        rng.shuffle(idx)
        class_indices.append(idx)

    num_present = len(class_indices)
    alpha_safe = max(float(alpha_root), 1e-6)
    class_probs = rng.dirichlet(np.full(num_present, alpha_safe, dtype=np.float64))
    counts = np.zeros(num_present, dtype=np.int64)
    available = np.asarray([len(idx) for idx in class_indices], dtype=np.int64)

    # Best-effort minimum allocation. Allocate one level at a time so an
    # infeasible minimum cannot make the selected root set larger than root_n.
    requested_min = max(0, int(min_per_class))
    selected = 0
    for _level in range(requested_min):
        eligible = np.where(counts < available)[0]
        if selected >= k or int(eligible.size) <= 0:
            break
        order = eligible.copy()
        rng.shuffle(order)
        for pos in order.tolist():
            if selected >= k:
                break
            counts[int(pos)] += 1
            selected += 1

    # Allocate the remaining budget according to the requested Dirichlet class
    # distribution, while respecting the number of available samples per class.
    while selected < k:
        capacity = available - counts
        eligible = capacity > 0
        if not bool(np.any(eligible)):
            break
        probs = class_probs * eligible.astype(np.float64)
        prob_sum = float(probs.sum())
        if not np.isfinite(prob_sum) or prob_sum <= 0.0:
            probs = eligible.astype(np.float64)
            prob_sum = float(probs.sum())
        probs = probs / prob_sum
        pos = int(rng.choice(num_present, p=probs))
        counts[pos] += 1
        selected += 1

    chosen_parts: list[np.ndarray] = []
    for pos, take in enumerate(counts.tolist()):
        if int(take) > 0:
            chosen_parts.append(class_indices[pos][: int(take)])

    if chosen_parts:
        chosen = np.concatenate(chosen_parts).astype(np.int64, copy=False)
    else:
        chosen = np.empty((0,), dtype=np.int64)

    # Defensive exact-size fallback. This should only be needed if class
    # capacities were unexpectedly inconsistent with y.
    if int(chosen.size) < k:
        remaining = np.setdiff1d(np.arange(n, dtype=np.int64), chosen, assume_unique=False)
        extra = rng.choice(remaining, size=k - int(chosen.size), replace=False).astype(np.int64)
        chosen = np.concatenate([chosen, extra])
    elif int(chosen.size) > k:
        chosen = rng.choice(chosen, size=k, replace=False).astype(np.int64)

    rng.shuffle(chosen)
    return chosen.astype(np.int64, copy=False)


def _get_or_create_artifacts(run_config: dict, num_partitions: int) -> DatasetArtifacts:
    dataset_name = _normalize_dataset_name(_get(run_config, "dataset_name", "can_mirgu"))
    dataset_dir = _get(run_config, "dataset_dir", None)

    # dataset_max_samples is the fixed number of samples assigned to clients.
    # FLTrust root samples are additional samples selected from the unused remainder.
    max_samples = int(_get(run_config, "dataset_max_samples", 0))
    alpha = float(_get(run_config, "dirichlet_alpha", 0.5))
    seed = int(_get(run_config, "seed", 42))
    min_partition_size = int(_get(run_config, "min_partition_size", 10))
    label_column = str(_get(run_config, "label_column", "label"))
    num_classes_override = int(_get(run_config, "num_classes", 0))
    test_fraction = float(_get(run_config, "test_fraction", 0.2))

    root_size = int(_get(run_config, "fltrust_root_size", 50))
    root_seed_offset = int(_get(run_config, "fltrust_root_seed_offset", 12345))
    root_mode = str(_get(run_config, "fltrust_root_mode", "random")).strip().lower()
    alpha_root = float(_get(run_config, "fltrust_root_alpha", 5.0))
    root_min_per_class = int(_get(run_config, "fltrust_root_min_per_class", 1))

    key = DatasetKey(
        dataset_name=dataset_name,
        max_samples=max_samples,
        alpha=alpha,
        num_partitions=int(num_partitions),
        seed=seed,
        label_column=label_column,
        num_classes_override=num_classes_override,
        test_fraction=test_fraction,
        min_partition_size=min_partition_size,
        fltrust_root_size=root_size,
        fltrust_root_seed_offset=root_seed_offset,
        fltrust_root_mode=root_mode,
        fltrust_root_alpha=alpha_root,
        fltrust_root_min_per_class=root_min_per_class,
    )
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]

    # Load numeric tabular CSV data, then deterministically assign the client pool.
    data_file = _resolve_dataset_file(dataset_name, dataset_dir=dataset_dir)
    df = pd.read_csv(data_file)

    if label_column not in df.columns:
        raise ValueError(
            f"label_column='{label_column}' not found in {data_file}. "
            f"Columns are: {list(df.columns)[:20]}..."
        )

    y_raw = df[label_column]
    X_raw = df.drop(columns=[label_column])

    # Option B: CSV must already contain numeric-only features.
    non_numeric = X_raw.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Non-numeric feature columns found in Option-B CSV: {non_numeric}. "
            "Fix preprocessing to make all features numeric BEFORE saving CSV."
        )

    # Remove invalid rows once, before selecting the fixed client pool.
    X_clean = X_raw.replace([np.inf, -np.inf], np.nan).dropna()
    y_clean = y_raw.loc[X_clean.index].reset_index(drop=True)
    X_all = X_clean.reset_index(drop=True).to_numpy(dtype=np.float32)
    y_all, num_classes = _encode_labels(
        y_clean,
        num_classes_override=num_classes_override,
    )

    # Preserve the previous deterministic client sample membership/order.
    client_pool_indices = _deterministic_tabular_client_indices(
        len(y_all),
        max_samples,
        seed,
    )


    client_pool_indices = np.asarray(client_pool_indices, dtype=np.int64)
    if int(client_pool_indices.size) < int(num_partitions) * int(max(0, min_partition_size)):
        raise ValueError(
            "The fixed client sample pool is too small for the configured number "
            f"of clients and min_partition_size. Client samples={int(client_pool_indices.size)}, "
            f"num_partitions={num_partitions}, min_partition_size={min_partition_size}."
        )
    if int(np.unique(client_pool_indices).size) != int(client_pool_indices.size):
        raise RuntimeError("Client sample selection produced duplicate indices.")

    # Partition the fixed client pool before FLTrust root selection. Therefore,
    # these partitions depend only on the dataset/client settings, never root size.
    client_partitions_local = _dirichlet_partition(
        y=y_all[client_pool_indices],
        num_partitions=int(num_partitions),
        alpha=alpha,
        seed=seed,
        min_partition_size=min_partition_size,
    )
    partitions = [
        client_pool_indices[p].astype(np.int64, copy=False)
        for p in client_partitions_local
    ]

    # Ensure client partitioning used the entire fixed client pool exactly once.
    client_used_indices = np.concatenate(partitions).astype(np.int64, copy=False)
    if int(client_used_indices.size) != int(client_pool_indices.size):
        raise RuntimeError(
            "Client partitioning did not preserve the fixed client sample count."
        )
    if int(np.unique(client_used_indices).size) != int(client_pool_indices.size):
        raise RuntimeError(
            "Client partitioning produced duplicate or missing client samples."
        )

    # Remove all client samples from the root candidate pool. This is the
    # functional equivalent of deleting them from X_all/y_all while retaining
    # original indices for efficient client/root loaders.
    all_indices = np.arange(len(y_all), dtype=np.int64)
    remaining_indices = np.setdiff1d(
        all_indices,
        client_used_indices,
        assume_unique=False,
    ).astype(np.int64, copy=False)

    requested_root_n = int(root_size)
    if requested_root_n <= 0:
        raise ValueError(
            f"fltrust_root_size must be a positive integer, got {requested_root_n}."
        )
    if requested_root_n > int(remaining_indices.size):
        raise ValueError(
            f"fltrust_root_size={requested_root_n} exceeds the {int(remaining_indices.size)} "
            "samples left after the fixed client sample pool was selected. Reduce "
            "dataset_max_samples or fltrust_root_size so root data remains disjoint."
        )

    rng_root = np.random.default_rng(seed + root_seed_offset)
    root_local_indices = _select_fltrust_root_indices(
        y_all[remaining_indices],
        root_n=requested_root_n,
        rng=rng_root,
        mode=root_mode,
        alpha_root=alpha_root,
        min_per_class=root_min_per_class,
    )
    root_indices = remaining_indices[root_local_indices].astype(np.int64, copy=False)

    if int(root_indices.size) != requested_root_n:
        raise RuntimeError(
            f"FLTrust root selection returned {int(root_indices.size)} samples; "
            f"expected exactly {requested_root_n}."
        )
    if int(np.unique(root_indices).size) != requested_root_n:
        raise RuntimeError("FLTrust root selection produced duplicate sample indices.")

    # Fit shared preprocessing statistics only on the fixed client pool. Thus,
    # client features and partitions remain identical when root size changes.
    X_scaler_fit = X_all[client_pool_indices]
    mu = X_scaler_fit.mean(axis=0, dtype=np.float64)
    sigma = X_scaler_fit.std(axis=0, dtype=np.float64)
    sigma[sigma < 1e-12] = 1.0

    feature_mean = mu.astype(np.float32)
    feature_scale = sigma.astype(np.float32)

    # Fail fast if a future change reintroduces any client/root overlap.
    overlap = np.intersect1d(client_used_indices, root_indices, assume_unique=False)
    if int(overlap.size) != 0:
        raise RuntimeError(
            f"FLTrust data leakage detected: {int(overlap.size)} root sample(s) "
            "also appear in client partitions."
        )

    artifacts = DatasetArtifacts(
        X_all=X_all,
        y_all=y_all,
        input_dim=int(X_all.shape[1]),
        num_classes=int(num_classes),
        partitions=partitions,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        root_indices=root_indices,
    )
    _DATASET_CACHE[key] = artifacts
    return artifacts


def load_data(*, partition_id: int, num_partitions: int, run_config: dict, server_round: int = 0):
    artifacts = _get_or_create_artifacts(run_config, num_partitions=num_partitions)

    batch_size = int(_get(run_config, "batch_size", 32))
    seed = int(_get(run_config, "seed", 42)) + int(partition_id)

    idx = artifacts.partitions[int(partition_id)]
    X = artifacts.X_all[idx]
    y = artifacts.y_all[idx]

    test_fraction = float(_get(run_config, "test_fraction", 0.2))

    # Decide if stratify is safe
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    min_count = int(counts.min()) if n_classes > 0 else 0

    n = len(y)
    n_test = int(np.ceil(test_fraction * n))
    n_train = n - n_test

    use_stratify = (
        n_classes > 1
        and min_count >= 2
        and n_test >= n_classes
        and n_train >= n_classes
    )

    strat = y if use_stratify else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=seed, stratify=strat
        )
    except ValueError:
        # fallback for rare edge cases
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=seed, stratify=None
        )


    use_global_scaler = bool(_get(run_config, "use_global_scaler", True))
    X_train, X_test = _transform_train_test_for_loader(
        X_train,
        X_test,
        artifacts=artifacts,
        use_global_scaler=use_global_scaler,
    )


    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )


    base_seed = int(_get(run_config, "seed", 42))
    dl_seed = base_seed + int(partition_id) + 100_000 * int(server_round)

    g = torch.Generator()
    g.manual_seed(dl_seed)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def load_root_data(*, num_partitions: int, run_config: dict, server_round: int = 0):
    """Return a DataLoader for the server-only FLTrust root data.

    Client partitions are selected first. ``artifacts.root_indices`` is then
    selected only from the samples left unused by all client shards.
    """
    artifacts = _get_or_create_artifacts(run_config, num_partitions=num_partitions)
    idx = artifacts.root_indices
    X = artifacts.X_all[idx]
    y = artifacts.y_all[idx]

    batch_size = int(_get(run_config, "batch_size", 32))
    seed = int(_get(run_config, "seed", 42))

    use_global_scaler = bool(_get(run_config, "use_global_scaler", True))
    X = _transform_features_for_loader(
        X,
        artifacts=artifacts,
        use_global_scaler=use_global_scaler,
        fit_local_scaler=not use_global_scaler,
    )

    root_ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )

    g = torch.Generator()
    g.manual_seed(seed + 777_777 + 100_000 * int(server_round))
    return DataLoader(root_ds, batch_size=batch_size, shuffle=True, generator=g)


# ---------------------------------------------------------------------------
# SDEA-only unlabeled public-input loader. No client or test split is modified.
# ---------------------------------------------------------------------------
def load_sdea_public_data(*, num_partitions: int, run_config: dict) -> torch.Tensor:
    """Return fixed unlabeled inputs in the same feature space as the FL model.

    source='unused': reproducibly sample previously UNUSED training-pool rows,
    disjoint from ALL client partitions and FLTrust root samples. Their labels
    are deliberately not read. This is a *related-data adaptation* of SDEA.

    source='external_csv': read an independently provided unlabeled public CSV.
    Features must have precisely the model input columns/order; set
    sdea_public_label_column if the CSV contains a column to drop. This mode
    is closest to SDEA's original unrelated-real-public-data protocol.

    Source 'noise' is deliberately unsupported: SDEA's paper explicitly warns
    meaningless random noise is not an appropriate replacement for public data.
    """
    artifacts = _get_or_create_artifacts(run_config, num_partitions)
    source = str(_get(run_config, "sdea_public_source", "unused")).strip().lower()
    size = int(_get(run_config, "sdea_public_size", 512))
    if size <= 0:
        raise ValueError("sdea_public_size must be positive.")
    seed = int(_get(run_config, "seed", 42)) + 9_738_133
    rng = np.random.default_rng(seed)

    if source == "unused":
        client_ids = np.concatenate(artifacts.partitions).astype(np.int64)
        used = np.concatenate((client_ids, artifacts.root_indices)).astype(np.int64)
        available = np.setdiff1d(
            np.arange(len(artifacts.y_all), dtype=np.int64), used,
            assume_unique=False,
        )
        if len(available) < size:
            raise ValueError(
                f"SDEA requested {size} unlabeled unused examples, but only "
                f"{len(available)} are available after reserving client and "
                "FLTrust root data. Reduce sdea_public_size or use "
                "sdea_public_source='external_csv'."
            )
        chosen = np.sort(rng.choice(available, size=size, replace=False))
        X = artifacts.X_all[chosen].copy()
    elif source == "external_csv":
        path = Path(str(_get(run_config, "sdea_public_path", "") or "")).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"SDEA external unlabeled public CSV not found: {path}. "
                "Set sdea_public_path to a compatible public dataset."
            )
        frame = pd.read_csv(path)
        label_col = str(_get(run_config, "sdea_public_label_column", "") or "").strip()
        if label_col:
            if label_col not in frame.columns:
                raise ValueError(f"sdea_public_label_column={label_col!r} is absent from {path}")
            frame = frame.drop(columns=[label_col])
        if frame.shape[1] != artifacts.input_dim:
            raise ValueError(
                f"SDEA public CSV has {frame.shape[1]} columns but model expects "
                f"{artifacts.input_dim}; columns must be in training feature order."
            )
        if frame.select_dtypes(exclude=["number", "bool"]).shape[1]:
            raise ValueError("SDEA public CSV must contain numeric-only features.")
        X = frame.to_numpy(dtype=np.float32)
        X = X[np.isfinite(X).all(axis=1)]
        if len(X) < size:
            raise ValueError(f"SDEA public CSV has {len(X)} valid rows, need {size}.")
        X = X[np.sort(rng.choice(len(X), size=size, replace=False))]
    else:
        raise ValueError("sdea_public_source must be 'unused' or 'external_csv'.")

    if X.ndim != 2 or X.shape[1] != artifacts.input_dim:
        raise ValueError("SDEA public examples have an incompatible feature shape.")
    if not np.isfinite(X).all():
        raise ValueError("SDEA public examples must be finite.")

    # External inputs can already be transformed; explicitly opt in to bypass.
    transformed = bool(_get(run_config, "sdea_public_pretransformed", False))
    if source == "unused" and transformed:
        raise ValueError("sdea_public_pretransformed=true only applies to external_csv.")
    if not transformed:
        X = _transform_features_for_loader(
            X, artifacts=artifacts,
            use_global_scaler=bool(_get(run_config, "use_global_scaler", True)),
            fit_local_scaler=not bool(_get(run_config, "use_global_scaler", True)),
        )
    result = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32)
    print(f"[SDEA] public source={source}; unlabeled examples={len(result)}; "
          f"features={result.shape[1]}; seed={seed}", flush=True)
    return result


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, h1: int = 128, h2: int = 64, dropout: float = 0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(*, num_partitions: int, run_config: dict) -> nn.Module:
    artifacts = _get_or_create_artifacts(run_config, num_partitions=num_partitions)
    h1 = int(_get(run_config, "mlp_h1", 128))
    h2 = int(_get(run_config, "mlp_h2", 64))
    dropout = float(_get(run_config, "mlp_dropout", 0.0))
    return TabularMLP(artifacts.input_dim, artifacts.num_classes, h1=h1, h2=h2, dropout=dropout)


def get_standardized_feature_bounds(
    *,
    num_partitions: int,
    run_config: dict,
    clip: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-feature (low, high) bounds **in standardized space**.

    This is optional utility for server-side probe generation.

    Notes:
      - Bounds are computed from X_all (no labels used).
      - Standardization matches the global scaler (feature_mean/feature_scale).
      - clip prevents extreme outliers from creating overly-wide ranges.
    """
    artifacts = _get_or_create_artifacts(run_config, num_partitions=num_partitions)

    X = artifacts.X_all.astype(np.float32, copy=False)
    low_raw = np.nanmin(X, axis=0)
    high_raw = np.nanmax(X, axis=0)

    mu = artifacts.feature_mean.astype(np.float32, copy=False)
    sigma = artifacts.feature_scale.astype(np.float32, copy=False)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    low = (low_raw - mu) / sigma
    high = (high_raw - mu) / sigma

    # Guard against inverted ranges and extreme values
    lo = np.minimum(low, high)
    hi = np.maximum(low, high)

    c = float(clip)
    if np.isfinite(c) and c > 0:
        lo = np.clip(lo, -c, c)
        hi = np.clip(hi, -c, c)

    return lo.astype(np.float32, copy=False), hi.astype(np.float32, copy=False)


def _parse_feature_index_groups(raw: object, *, num_features: int) -> list[list[int]] | None:
    """Parse semicolon-separated feature-index groups.

    Examples:
      - ``"10-13;14-53"``
      - ``"10,11,12,13;14:53"``

    ``None`` means auto-detect groups; an empty list means explicitly disabled.
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in {"", "auto", "detect", "automatic"}:
        return None
    if s in {"none", "off", "disable", "disabled"}:
        return []

    groups: list[list[int]] = []
    for group_text in str(raw).split(";"):
        idxs: list[int] = []
        for token in group_text.split(","):
            token = token.strip()
            if not token:
                continue
            sep = None
            if "-" in token:
                sep = "-"
            elif ":" in token:
                sep = ":"
            if sep is not None:
                try:
                    a_raw, b_raw = token.split(sep, 1)
                    a, b = int(a_raw.strip()), int(b_raw.strip())
                except Exception:
                    continue
                step = 1 if b >= a else -1
                idxs.extend(range(a, b + step, step))
            else:
                try:
                    idxs.append(int(token))
                except Exception:
                    continue
        clean = sorted({i for i in idxs if 0 <= int(i) < int(num_features)})
        if len(clean) >= 2:
            groups.append(clean)
    return groups


def _infer_contiguous_onehot_groups(
    X_raw: np.ndarray,
    binary_indices: list[int],
    *,
    min_group_size: int = 3,
) -> list[list[int]]:
    """Infer contiguous one-hot groups from globally observed raw features.

    The detector only accepts a group when every row has exactly one active value
    in that contiguous block. This keeps ordinary independent binary features from
    being grouped together merely because they are adjacent.
    """
    if X_raw.ndim != 2 or not binary_indices:
        return []

    binary_sorted = sorted({int(i) for i in binary_indices})
    runs: list[list[int]] = []
    run: list[int] = []
    for idx in binary_sorted:
        if not run or idx == run[-1] + 1:
            run.append(idx)
        else:
            runs.append(run)
            run = [idx]
    if run:
        runs.append(run)

    active_by_feature: dict[int, np.ndarray] = {}
    for j in binary_sorted:
        col = np.asarray(X_raw[:, j])
        finite = col[np.isfinite(col)]
        vals = np.unique(finite)
        if vals.size != 2:
            continue
        hi = float(np.max(vals))
        active_by_feature[j] = np.isclose(col, hi, rtol=0.0, atol=1e-8)

    out: list[list[int]] = []
    min_size = max(2, int(min_group_size))
    for contiguous in runs:
        pos = 0
        while pos < len(contiguous):
            found_end = None
            for end in range(pos + min_size - 1, len(contiguous)):
                cols = contiguous[pos : end + 1]
                if any(c not in active_by_feature for c in cols):
                    continue
                active_count = np.zeros((X_raw.shape[0],), dtype=np.int16)
                for c in cols:
                    active_count += active_by_feature[c].astype(np.int16, copy=False)
                if bool(np.all(active_count == 1)):
                    found_end = end
                    break
            if found_end is None:
                pos += 1
            else:
                out.append(contiguous[pos : found_end + 1])
                pos = found_end + 1
    return out


def get_feature_poison_metadata(
    *,
    num_partitions: int,
    run_config: dict,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    clip: float = 8.0,
    onehot_groups: object = "auto",
    onehot_min_group_size: int = 3,
) -> dict:
    """Return global metadata used by the feature-poisoning attack.

    The returned values are aligned with the tensors produced by ``load_data`` when
    ``use_global_scaler=true``:
      - robust global lower/upper quantile bounds;
      - valid low/high levels for binary features;
      - contiguous one-hot feature groups, auto-detected or explicitly configured.

    When local scaling is used, globally transformed numeric levels are not
    compatible with each client's private scaler. In that case the client attack
    safely derives numeric levels from its local tensors while still using the
    globally inferred one-hot index groups.
    """
    artifacts = _get_or_create_artifacts(run_config, num_partitions=num_partitions)
    X_raw = np.asarray(artifacts.X_all, dtype=np.float32)
    d = int(X_raw.shape[1]) if X_raw.ndim == 2 else 0
    if d <= 0:
        return {
            "robust_low": None,
            "robust_high": None,
            "binary_indices": [],
            "binary_low": None,
            "binary_high": None,
            "onehot_groups": [],
            "global_model_space": False,
        }

    q_lo = float(np.clip(float(lower_quantile), 0.0, 1.0))
    q_hi = float(np.clip(float(upper_quantile), 0.0, 1.0))
    if not q_lo < q_hi:
        q_lo, q_hi = 0.05, 0.95

    low_raw = np.nanquantile(X_raw, q_lo, axis=0).astype(np.float32)
    high_raw = np.nanquantile(X_raw, q_hi, axis=0).astype(np.float32)

    binary_indices: list[int] = []
    binary_low_raw = np.full((d,), np.nan, dtype=np.float32)
    binary_high_raw = np.full((d,), np.nan, dtype=np.float32)
    for j in range(d):
        col = X_raw[:, j]
        vals = np.unique(col[np.isfinite(col)])
        if vals.size == 2:
            binary_indices.append(j)
            binary_low_raw[j] = float(np.min(vals))
            binary_high_raw[j] = float(np.max(vals))

    configured_groups = _parse_feature_index_groups(onehot_groups, num_features=d)
    if configured_groups is None:
        detected_groups = _infer_contiguous_onehot_groups(
            X_raw,
            binary_indices,
            min_group_size=int(onehot_min_group_size),
        )
    else:
        binary_set = set(binary_indices)
        detected_groups = [
            [int(j) for j in group if int(j) in binary_set]
            for group in configured_groups
        ]
        detected_groups = [g for g in detected_groups if len(g) >= 2]

    use_global_scaler = bool(_get(run_config, "use_global_scaler", True))
    if use_global_scaler:
        mu = artifacts.feature_mean.astype(np.float32, copy=False)
        sigma = artifacts.feature_scale.astype(np.float32, copy=False)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)

        robust_low = (low_raw - mu) / sigma
        robust_high = (high_raw - mu) / sigma
        binary_low = (binary_low_raw - mu) / sigma
        binary_high = (binary_high_raw - mu) / sigma

        lo = np.minimum(robust_low, robust_high)
        hi = np.maximum(robust_low, robust_high)
        c = float(clip)
        if np.isfinite(c) and c > 0.0:
            lo = np.clip(lo, -c, c)
            hi = np.clip(hi, -c, c)
        robust_low = lo.astype(np.float32, copy=False)
        robust_high = hi.astype(np.float32, copy=False)
        binary_low = binary_low.astype(np.float32, copy=False)
        binary_high = binary_high.astype(np.float32, copy=False)
        global_model_space = True
    else:
        robust_low = None
        robust_high = None
        binary_low = None
        binary_high = None
        global_model_space = False

    return {
        "robust_low": robust_low,
        "robust_high": robust_high,
        "binary_indices": binary_indices,
        "binary_low": binary_low,
        "binary_high": binary_high,
        "onehot_groups": detected_groups,
        "global_model_space": global_model_space,
    }


def trainer(
    model,
    train_loader,
    num_epochs: int = 1,
    lr: float = 1e-3,
    device=None,
    *,
    prox_mu: float = 0.0,
    global_params: list[torch.Tensor] | None = None,
    scaffold: bool = False,
    scaffold_c_global: "dict[str, torch.Tensor] | None" = None,
    scaffold_ci: "dict[str, torch.Tensor] | None" = None,
):
    """Train helper.

    Supported client-side objectives:
      - FedAvg (default): standard local training.
      - FedProx: if prox_mu>0 and global_params is provided, adds (mu/2)||w-w_global||^2.
      - SCAFFOLD: if scaffold=True, applies variance reduction using control variates.
        The server must provide `scaffold_c_global` (global control variate c), and the
        client should persist `scaffold_ci` (local control variate c_i) across rounds.

    Returns:
      (ci_new, steps)
        - ci_new: updated client control variate dict (CPU tensors) if scaffold=True, else None
        - steps:  number of optimizer steps performed (batches processed)
    """
    if device is None:
        device = torch.device("cpu")

    model.to(device)
    model.train()

    # Prepare FedProx reference parameters on the right device (no grad)
    use_fedprox = (not scaffold) and (prox_mu is not None) and (float(prox_mu) > 0.0) and (global_params is not None)
    if use_fedprox:
        gparams = [gp.detach().to(device) for gp in global_params]
        # If shapes don't align (shouldn't happen), disable FedProx rather than crash
        if len(gparams) != len(list(model.parameters())):
            use_fedprox = False
    else:
        gparams = []

    # SCAFFOLD state (device copies for fast per-step use)
    named_params = list(model.named_parameters())
    if scaffold:
        # default to zeros if missing
        c_dev = {}
        ci_dev = {}
        for n, p in named_params:
            z = torch.zeros_like(p.detach(), device=device)
            c_dev[n] = (scaffold_c_global.get(n).detach().to(device) if (scaffold_c_global and n in scaffold_c_global) else z)
            ci_dev[n] = (scaffold_ci.get(n).detach().to(device) if (scaffold_ci and n in scaffold_ci) else z)

        # snapshot of initial weights (on device)
        w0 = {n: p.detach().clone() for n, p in named_params}

        # SCAFFOLD is defined for SGD; use SGD for stability/faithfulness
        optimizer = optim.SGD(model.parameters(), lr=float(lr))
    else:
        optimizer = optim.Adam(model.parameters(), lr=float(lr))

    criterion = nn.CrossEntropyLoss()

    steps = 0

    for _ in range(int(num_epochs)):
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if use_fedprox:
                prox = 0.0
                for p, gp in zip(model.parameters(), gparams):
                    prox = prox + torch.sum((p - gp) ** 2)
                loss = loss + (float(prox_mu) / 2.0) * prox

            loss.backward()

            if scaffold:
                # Modify gradients: g' = g - c_i + c
                with torch.no_grad():
                    for n, p in named_params:
                        if p.grad is None:
                            continue
                        # grad <- grad + c - c_i
                        p.grad.add_(c_dev[n]).add_(-ci_dev[n])

            optimizer.step()
            steps += 1

    if not scaffold:
        return None, steps

    # Update c_i:
    #   c_i <- c_i - c + (w0 - wT) / (K * lr)
    # where K is the number of local steps (batches).
    if steps <= 0 or float(lr) <= 0.0:
        # Degenerate case: no updates; keep c_i unchanged
        ci_new_cpu = {n: (scaffold_ci.get(n).detach().cpu().clone() if scaffold_ci and n in scaffold_ci else torch.zeros_like(p.detach().cpu()))
                      for n, p in named_params}
        return ci_new_cpu, steps

    ci_new_cpu = {}
    with torch.no_grad():
        for n, p in named_params:
            wT = p.detach()
            ci_new = ci_dev[n] - c_dev[n] + (w0[n] - wT) / (float(steps) * float(lr))
            ci_new_cpu[n] = ci_new.detach().cpu().clone()

    return ci_new_cpu, steps


def evaluator(model, test_loader, device=None):
    if device is None:
        device = torch.device("cpu")

    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits = model(X_batch)
            total_loss += float(criterion(logits, y_batch).item())
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == y_batch).sum().item())
            total += int(y_batch.size(0))

    return total_loss / max(1, len(test_loader)), correct / max(1, total)

