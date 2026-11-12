"""Flower ClientApp: tabular training and adversarial-client simulations."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fltabular.task import (
    build_model,
    evaluator,
    load_data,
    trainer,
    malicious_ids_for_round,
    sampled_partition_ids_for_round,
    get_feature_poison_metadata,
    resolved_attack_config,
)


import torch
import random
import json
from torch.utils.data import DataLoader

# Flower ClientApp
app = ClientApp()


# ---------------------------------------------------------------------------
# SCAFFOLD client-side control variate state (persisted inside each ClientAppActor)
# Only used by ProbeGuard/ProbeGuard-EMA with probeguard_client_method="scaffold".
# ---------------------------------------------------------------------------
_SCAFFOLD_CI = None  # type: dict[str, 'torch.Tensor'] | None



# ---------------------------------------------------------------------------
# Query handler used by the ServerApp to map Flower node_id -> partition-id
# ---------------------------------------------------------------------------
@app.query("whoami")
def whoami(msg: Message, context: Context) -> Message:
    pid = int(context.node_config.get("partition-id", -1))
    metrics = MetricRecord({"client_id": pid, "partition-id": pid})
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)

# ===========================================================================

import numpy as np


import math
import os
import uuid
import time
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import TensorDataset

def _set_determinism(seed: int, *, deterministic_algos: bool = False) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    # reproducibility flags
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    if deterministic_algos:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # strict mode (may raise if a non-deterministic CUDA op is used)
        torch.use_deterministic_algorithms(True)

# ===========================================================================

# ============== Select Malicious Clients (global) ==============

# import random
# from math import ceil

# # configuration-based malicious ratio
# MAL_PCT = 0.2  # or read from context.run_config later
# NUM_CLIENTS = 5  # matches your simulation config

# client_ids = [str(i) for i in range(NUM_CLIENTS)]
# num_mal = ceil(NUM_CLIENTS * MAL_PCT)

# random.seed(42)
# malicious_ids = set(random.sample(client_ids, num_mal))

# print("Malicious clients:", malicious_ids)

# ===========================================================================

# ============ Attacks ============

def _rand_like(x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """rand_like that supports a dedicated Generator (compatible with your torch)."""
    if generator is None:
        return torch.rand_like(x)
    return torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)


def random_scaling_attack(model, poison_intensity=0.35, *, generator: torch.Generator | None = None):
    with torch.no_grad():
        eps = 0.5
        for _, param in model.named_parameters():
            param.mul_(1 + poison_intensity)
            # noise in [-eps, eps]
            noise = (_rand_like(param, generator=generator) * 2.0 - 1.0) * eps
            param.mul_(1.0 + noise)


def sign_flipping_attack(
    model,
    global_params: list[torch.Tensor],
    *,
    gamma: float = 5.0,
):
    """Sign-flipping (model-update) attack.

    Implements: Δw' = -gamma * Δw  where Δw = (w_local - w_global),
    so w_mal = w_global + Δw' = w_global - gamma * (w_local - w_global).
    """
    with torch.no_grad():
        for p, g in zip(model.parameters(), global_params):
            g = g.to(device=p.device, dtype=p.dtype)
            delta = p.detach() - g
            p.copy_(g - float(gamma) * delta)

def flatten_params(model):
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


def unflatten_params(model, flat_tensor):
    offset = 0
    for param in model.parameters():
        n = param.numel()
        param.data = flat_tensor[offset:offset+n].view_as(param).clone()
        offset += n


def compute_trim_stats(before_flat: torch.Tensor, global_flat: torch.Tensor):
    """Compute (mu, sigma, s_hat) for the Trim attack using flattened vectors."""
    if before_flat is None or global_flat is None:
        raise ValueError("compute_trim_stats: inputs must be tensors")

    before_flat = before_flat.view(-1)
    global_flat = global_flat.view(-1)

    if before_flat.numel() != global_flat.numel():
        raise ValueError(
            f"compute_trim_stats: size mismatch before={before_flat.numel()} global={global_flat.numel()}"
        )

    # Ensure same device/dtype
    if global_flat.device != before_flat.device:
        global_flat = global_flat.to(before_flat.device)
    if global_flat.dtype != before_flat.dtype:
        global_flat = global_flat.to(before_flat.dtype)

    mu = before_flat
    sigma = 0.1 * (before_flat.abs() + 1e-12)
    s_hat = torch.where(mu > global_flat, torch.ones_like(mu), -torch.ones_like(mu))
    return mu, sigma, s_hat


def craft_trim_vector(mu, sigma, s_hat, *, generator: torch.Generator | None = None):
    rand = _rand_like(mu, generator=generator)
    upper = (mu + 3*sigma) + rand * sigma
    lower = (mu - 4*sigma) + rand * sigma
    return torch.where(s_hat < 0, upper, lower)

# ===========================================================================

# ============ Label Flip Attack (data poisoning) ============

def _norm_attack_name(name: object) -> str:
    s = str(name) if name is not None else ""
    s = s.strip().lower()
    # normalize common separators
    for ch in ("-", "_", " "):
        s = s.replace(ch, "")
    return s


def _is_label_flip_attack(name: object) -> bool:
    s = _norm_attack_name(name)
    return s in {
        "labelflip",
        "labelflipping",
        "labelflipattack",
        "fliplabel",
        "fliplabels",
    }


def _parse_label_flip_mapping(raw: object) -> dict[int, int]:
    """Parse label-flip mapping from run_config.

    Accepts:
      - dict-like: {0:1, 1:0} or {"0": "1"}
      - JSON string: '{"0":1,"1":0}'
      - simple string: '0:1,1:0'  or  '0->1, 1->0'
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        out = {}
        for k, v in raw.items():
            try:
                out[int(k)] = int(v)
            except Exception:
                continue
        return out

    s = str(raw).strip()
    if not s:
        return {}

    # Try JSON first
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    try:
                        out[int(k)] = int(v)
                    except Exception:
                        continue
                return out
        except Exception:
            pass

    # Fallback: parse 'a:b' or 'a->b' pairs
    out = {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "->" in p:
            a, b = p.split("->", 1)
        elif ":" in p:
            a, b = p.split(":", 1)
        else:
            continue
        try:
            out[int(a.strip())] = int(b.strip())
        except Exception:
            continue
    return out


def _infer_num_classes_from_model(model: torch.nn.Module) -> int | None:
    """Best-effort: infer number of classes from the last Linear layer."""
    last_lin = None
    try:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_lin = m
    except Exception:
        last_lin = None

    if last_lin is not None:
        try:
            return int(last_lin.out_features)
        except Exception:
            return None
    return None


def _apply_label_flip_to_trainloader(
    train_loader: DataLoader,
    *,
    num_classes: int,
    flip_rate: float,
    mode: str = "random",
    target: int = 0,
    mapping: dict[int, int] | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """Flip labels in-place on the train_loader's TensorDataset.

    Returns number of flipped samples.
    """
    if train_loader is None or getattr(train_loader, "dataset", None) is None:
        return 0

    C = int(num_classes)
    if C <= 1:
        return 0

    try:
        ds = train_loader.dataset
        # Expect TensorDataset(X, y)
        y = ds.tensors[1]
    except Exception:
        return 0

    if y is None or int(y.numel()) == 0:
        return 0

    r = float(flip_rate)
    if not (r > 0.0):
        return 0
    if r > 1.0:
        r = 1.0

    # sample which indices to flip (deterministic if generator is provided)
    if r >= 1.0:
        mask = torch.ones_like(y, dtype=torch.bool)
    else:
        if generator is None:
            mask = (torch.rand(y.shape) < r)
        else:
            mask = (torch.rand(y.shape, generator=generator) < r)

    n_flip = int(mask.sum().item())
    if n_flip <= 0:
        return 0

    y_new = y.detach().clone()

    mode_l = str(mode).lower().strip()
    if mode_l in {"swap", "binary", "flip"} and C == 2:
        mode_l = "binary"
    if C == 2 and mode_l in {"random", "cyclic", "mapping"}:
        # For binary, all sensible modes reduce to 0<->1
        mode_l = "binary"

    if mode_l == "binary":
        y_new[mask] = 1 - y_new[mask]

    elif mode_l in {"cyclic", "shift"}:
        y_new[mask] = (y_new[mask] + 1) % C

    elif mode_l in {"targeted", "target"}:
        t = int(target) % C
        y_new[mask] = t

    elif mode_l in {"mapping", "map"}:
        mp = mapping or {}
        # build a vectorized mapping array
        map_arr = torch.arange(C, dtype=y_new.dtype)
        for a, b in mp.items():
            try:
                a_i = int(a)
                b_i = int(b) % C
                if 0 <= a_i < C:
                    map_arr[a_i] = b_i
            except Exception:
                continue
        y_new[mask] = map_arr[y_new[mask].clamp(min=0, max=C - 1)]

    else:
        # random: sample a different label for each selected index
        if C <= 1:
            return 0
        if generator is None:
            rint = torch.randint(low=0, high=C - 1, size=(n_flip,))
        else:
            rint = torch.randint(low=0, high=C - 1, size=(n_flip,), generator=generator)
        # (y + 1 + u) mod C => uniform over labels != y
        y_sel = y_new[mask].to(torch.long)
        y_new[mask] = ((y_sel + 1 + rint.to(torch.long)) % C).to(dtype=y_new.dtype)

    # write back in-place (keeps DataLoader intact)
    try:
        y.copy_(y_new)
    except Exception:
        # last-resort: replace tensor reference if possible
        try:
            ds.tensors = (ds.tensors[0], y_new)
        except Exception:
            pass

    return n_flip


def _is_clean_label_attack(name: object) -> bool:
    return _norm_attack_name(name) in {
        "cleanlabel",
        "cleanlabelpoison",
        "cleanlabelpoisoning",
        "cleanlabelfeatureperturbation",
        "cleanlabelfeatureperturbation",
        "cleanlabelattack",
    }


def _is_feature_poison_attack(name: object) -> bool:
    return _norm_attack_name(name) in {
        "featurepoison",
        "featurepoison1",
        "featurepoison2",
        "featurepoisoning",
        "samplepoison",
        "samplepoisoning",
        "featuresamplepoison",
    }


def _is_inverted_loss_attack(name: object) -> bool:
    return _norm_attack_name(name) in {
        "invertedloss",
        "invertedlosspoison",
        "invertedlosspoisoning",
        "invertedlossrelabel",
        "invertedlossrelabeling",
        "ilf",
    }


def _is_boundary_aware_label_flip_attack(name: object) -> bool:
    return _norm_attack_name(name) in {
        "boundaryawarelabelflipping",
        "boundaryawarelabelflip",
        "boundarylabelflipping",
        "boundarylabelflip",
        "BoundaryLabelFlip",
        "boundaryawarelabelpoisoning",
        "balf",
    }




def _get_dataset_tensors(train_loader: DataLoader) -> tuple[TensorDataset | None, torch.Tensor | None, torch.Tensor | None]:
    try:
        ds = train_loader.dataset
        X, y = ds.tensors
        return ds, X, y
    except Exception:
        return None, None, None


def _attack_active_this_round(
    run_config: dict,
    round_no: int,
    *,
    prefix: str,
    default_start_frac: float = 0.0,
) -> bool:
    total_rounds = int(run_config.get("num-server-rounds", 0) or 0)
    start_round = int(run_config.get(f"{prefix}_start_round", 0) or 0)
    start_frac = float(run_config.get(f"{prefix}_start_frac", default_start_frac) or default_start_frac)
    if start_round <= 0 and total_rounds > 0 and start_frac > 0.0:
        start_round = max(1, int(math.ceil(float(total_rounds) * float(start_frac))))
    if start_round <= 0:
        return True
    return int(round_no) >= int(start_round)


def _sample_poison_indices(
    y: torch.Tensor,
    *,
    poison_rate: float,
    generator: torch.Generator | None = None,
    allowed_mask: torch.Tensor | None = None,
    max_samples: int = 0,
) -> torch.Tensor:
    if y is None or int(y.numel()) == 0:
        return torch.empty((0,), dtype=torch.long)

    if allowed_mask is None:
        pool = torch.arange(int(y.numel()), dtype=torch.long)
    else:
        pool = torch.where(allowed_mask)[0].to(torch.long)

    m = int(pool.numel())
    if m <= 0:
        return torch.empty((0,), dtype=torch.long)

    r = float(max(0.0, min(1.0, poison_rate)))
    k = int(math.ceil(r * m))
    if max_samples and int(max_samples) > 0:
        k = min(k, int(max_samples))
    if k <= 0:
        return torch.empty((0,), dtype=torch.long)
    k = min(k, m)

    perm = torch.randperm(m, generator=generator)
    idx = pool[perm[:k]]
    idx, _ = torch.sort(idx)
    return idx.to(torch.long)


def _featurewise_bounds(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.min(X, dim=0).values
    high = torch.max(X, dim=0).values
    return low, high


def _project_linf_box(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    *,
    eps: float,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    eps_f = float(max(0.0, eps))
    delta = torch.clamp(x_adv - x_orig, min=-eps_f, max=eps_f)
    x_new = x_orig + delta
    x_new = torch.maximum(torch.minimum(x_new, high.unsqueeze(0)), low.unsqueeze(0))
    return x_new


def _flatten_grad_list(grads: tuple[torch.Tensor | None, ...] | list[torch.Tensor | None]) -> torch.Tensor:
    flat: list[torch.Tensor] = []
    for g in grads:
        if g is None:
            continue
        flat.append(g.reshape(-1))
    if not flat:
        return torch.zeros((1,), dtype=torch.float32)
    return torch.cat(flat, dim=0)


def _mode_label(y: torch.Tensor) -> int:
    if y is None or int(y.numel()) <= 0:
        return 0
    vals, counts = torch.unique(y.detach().cpu().to(torch.long), return_counts=True)
    if int(vals.numel()) <= 0:
        return 0
    return int(vals[int(torch.argmax(counts).item())].item())


def _choose_index_from_class(
    y: torch.Tensor,
    *,
    target_class: int,
    generator: torch.Generator | None = None,
    exclude_idx: torch.Tensor | None = None,
) -> int | None:
    mask = (y == int(target_class))
    if exclude_idx is not None and int(exclude_idx.numel()) > 0:
        mask[exclude_idx.to(torch.long)] = False
    pool = torch.where(mask)[0].to(torch.long)
    if int(pool.numel()) == 0:
        return None
    perm = torch.randperm(int(pool.numel()), generator=generator)
    return int(pool[perm[0]].item())


def _parse_int_list(raw: object) -> list[int]:
    """Parse a run-config int list from TOML/list/JSON/comma-separated text."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        out: list[int] = []
        for x in raw:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out

    s = str(raw).strip()
    if not s:
        return []

    # JSON list, e.g. "[0, 1, 2]"
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return _parse_int_list(obj)
        except Exception:
            pass

    out: list[int] = []
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    return out


def _rand_choice_int(values: list[int], *, generator: torch.Generator | None = None) -> int | None:
    if not values:
        return None
    if len(values) == 1:
        return int(values[0])
    if generator is None:
        j = int(torch.randint(low=0, high=len(values), size=(1,)).item())
    else:
        j = int(torch.randint(low=0, high=len(values), size=(1,), generator=generator).item())
    return int(values[j])


def _local_label_counts(y: torch.Tensor, *, num_classes: int) -> dict[int, int]:
    """Return local class counts for valid class IDs only."""
    out: dict[int, int] = {}
    if y is None or int(y.numel()) <= 0:
        return out
    vals, counts = torch.unique(y.detach().cpu().to(torch.long), return_counts=True)
    for v, c in zip(vals.tolist(), counts.tolist()):
        vi = int(v)
        if 0 <= vi < int(num_classes):
            out[vi] = int(c)
    return out


def _clean_label_target_is_eligible(
    counts: dict[int, int],
    *,
    target_class: int,
    requested_source_class: int,
    min_target_samples: int,
) -> bool:
    """A target is usable if it exists locally and leaves at least one non-target poison sample."""
    t = int(target_class)
    if int(counts.get(t, 0)) < int(max(1, min_target_samples)):
        return False
    if int(requested_source_class) >= 0:
        s = int(requested_source_class)
        return s != t and int(counts.get(s, 0)) > 0
    return sum(c for cls, c in counts.items() if int(cls) != t) > 0


def _select_clean_label_target_class(
    y: torch.Tensor,
    *,
    preferred_target_class: int,
    num_classes: int,
    requested_source_class: int,
    target_mode: str,
    target_classes: object = None,
    local_smallest_k: int = 3,
    min_target_samples: int = 1,
    generator: torch.Generator | None = None,
) -> int | None:
    """Choose a CleanLabel target class without silently failing on non-IID clients.

    Modes:
      - fixed: original behavior; use only preferred_target_class.
      - adaptive: use preferred target if possible, otherwise choose randomly from
        configured present targets or all locally eligible classes.
      - random_present: choose randomly from locally eligible classes.
      - smallest_local: choose randomly among the k smallest locally eligible classes.
      - adaptive_smallest: use preferred target if possible, otherwise use smallest_local.
      - multi: choose randomly from configured target_classes that are locally eligible.
    """
    C = int(num_classes)
    if C <= 1:
        return None

    preferred = int(preferred_target_class) % C
    mode = str(target_mode or "fixed").strip().lower().replace("-", "_").replace(" ", "_")
    counts = _local_label_counts(y, num_classes=C)
    if not counts:
        return None

    def eligible(c: int) -> bool:
        return _clean_label_target_is_eligible(
            counts,
            target_class=int(c) % C,
            requested_source_class=int(requested_source_class),
            min_target_samples=int(min_target_samples),
        )

    # Preserve the old exact behavior unless a new mode is explicitly enabled.
    if mode in {"", "fixed", "static", "global", "configured"}:
        return preferred if eligible(preferred) else None

    configured: list[int] = []
    for c in _parse_int_list(target_classes):
        ci = int(c) % C
        if ci not in configured:
            configured.append(ci)

    local_eligible = sorted([int(c) for c in counts.keys() if eligible(int(c))])

    if mode in {"adaptive", "fallback", "fallback_random"}:
        if eligible(preferred):
            return preferred
        pool = [c for c in configured if eligible(c)] if configured else local_eligible
        return _rand_choice_int(pool, generator=generator)

    if mode in {"multi", "configured_multi", "multi_target", "multitarget"}:
        pool = [c for c in configured if eligible(c)]
        # Safe fallback: if no configured target is locally usable, avoid poisoning
        # rather than changing the meaning of a multi-target experiment.
        return _rand_choice_int(pool, generator=generator)

    if mode in {"random", "random_present", "local_random"}:
        pool = [c for c in configured if eligible(c)] if configured else local_eligible
        return _rand_choice_int(pool, generator=generator)

    if mode in {"smallest", "smallest_local", "minority", "local_smallest"}:
        k = max(1, int(local_smallest_k))
        ranked = sorted(local_eligible, key=lambda c: (int(counts.get(c, 0)), int(c)))
        return _rand_choice_int(ranked[:k], generator=generator)

    if mode in {"adaptive_smallest", "fallback_smallest", "adaptive_minority"}:
        if eligible(preferred):
            return preferred
        k = max(1, int(local_smallest_k))
        ranked = sorted(local_eligible, key=lambda c: (int(counts.get(c, 0)), int(c)))
        return _rand_choice_int(ranked[:k], generator=generator)

    # Unknown mode: fail safely by preserving the old behavior.
    return preferred if eligible(preferred) else None


def _select_clean_label_source_class(
    y: torch.Tensor,
    *,
    target_class: int,
    requested_source_class: int,
    source_mode: str,
    min_source_samples: int = 1,
    generator: torch.Generator | None = None,
    X: torch.Tensor | None = None,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    num_classes: int | None = None,
    confusable_max_samples_per_class: int = 0,
    confusable_topk_sources: int = 1,
) -> int | list[int] | None:
    """Optionally choose CleanLabel source class(es) for tabular CleanLabel.

    Returning None means "do not restrict poison samples to one source class", which
    preserves the original source_class=-1 behavior.

    Single-source modes return one integer source class.

    For source_mode="model_confusable_topk"/"boundary_topk", the model is evaluated
    on all local training samples. We then focus on samples whose current top-1
    prediction is the selected target class. For those samples, we count which
    non-target classes appear as the top-2/second-logit class. The top-K such
    classes are returned as source classes. This selects classes that share the
    current decision region/boundary with the target while still poisoning several
    source classes instead of only one.
    """
    t = int(target_class)
    if int(requested_source_class) >= 0:
        s = int(requested_source_class)
        if s == t:
            return None
        mask = (y == s)
        return s if int(mask.sum().item()) >= int(max(1, min_source_samples)) else None

    mode = str(source_mode or "mode_poison").strip().lower().replace("-", "_").replace(" ", "_")
    if mode in {"", "none", "auto", "mode", "mode_poison", "selected_mode"}:
        return None

    if num_classes is None or int(num_classes) <= 0:
        try:
            num_classes = max(int(torch.max(y).item()) + 1 if int(y.numel()) > 0 else 0, t + 1)
        except Exception:
            num_classes = t + 1

    counts = _local_label_counts(y, num_classes=int(num_classes))
    candidates = [
        int(c) for c, n in counts.items()
        if int(c) != t and int(n) >= int(max(1, min_source_samples))
    ]
    if not candidates:
        return None

    # New multi-source, boundary-aware mode.
    if mode in {
        "model_confusable_topk",
        "confusable_topk",
        "boundary_topk",
        "boundary_aware_topk",
        "target_top2",
        "target_second_logit",
        "top2_confusable",
    }:
        k_src = int(max(1, confusable_topk_sources))
        if X is not None and model is not None:
            try:
                dev = device if device is not None else next(model.parameters()).device
                was_training = bool(model.training)
                model.eval()
                cand_set = set(int(c) for c in candidates)

                with torch.no_grad():
                    logits = model(X.to(dev))
                    if logits.ndim == 2 and int(logits.shape[1]) > 1 and int(logits.shape[1]) > t:
                        k_logits = min(2, int(logits.shape[1]))
                        top_vals, top_idx = torch.topk(logits, k=k_logits, dim=1, largest=True, sorted=True)
                        pred = top_idx[:, 0].detach().cpu().to(torch.long)
                        second = top_idx[:, 1].detach().cpu().to(torch.long)
                        probs = torch.softmax(logits, dim=1).detach().cpu()

                        # Focus only on points currently classified as the target class.
                        focus = (pred == int(t)) & (second != int(t))
                        stats: dict[int, dict[str, float]] = {}
                        for c in sorted(cand_set):
                            m = focus & (second == int(c))
                            n = int(m.sum().item())
                            if n <= 0:
                                continue
                            # Ranking is mainly frequency as the second-logit class.
                            # Average target probability is a tie-breaker for stronger target-boundary closeness.
                            target_prob = float(probs[m, int(t)].mean().item())
                            second_logit = float(top_vals.detach().cpu()[m, 1].mean().item())
                            if math.isfinite(target_prob) and math.isfinite(second_logit):
                                stats[int(c)] = {
                                    "n": float(n),
                                    "target_prob": float(target_prob),
                                    "second_logit": float(second_logit),
                                    "local_count": float(counts.get(int(c), 0)),
                                }

                        if was_training:
                            model.train()

                        if stats:
                            ranked = sorted(
                                stats.keys(),
                                key=lambda c: (
                                    stats[c]["n"],
                                    stats[c]["target_prob"],
                                    stats[c]["second_logit"],
                                    stats[c]["local_count"],
                                    -int(c),
                                ),
                                reverse=True,
                            )
                            return [int(c) for c in ranked[: min(k_src, len(ranked))]]
            except Exception:
                try:
                    if 'was_training' in locals() and was_training:
                        model.train()
                except Exception:
                    pass

        # Fallback: choose up to K largest local non-target classes rather than disabling poisoning.
        ranked = sorted(candidates, key=lambda c: (int(counts.get(c, 0)), -int(c)), reverse=True)
        return [int(c) for c in ranked[: min(k_src, len(ranked))]]

    if mode in {"model_confusable", "confusable", "boundary_aware", "target_prob", "target_probability"}:
        # Choose c != target with the largest average P(model predicts target | x from class c).
        # If model scoring fails for any reason, fall back to largest_non_target instead of
        # silently disabling the attack.
        if X is not None and model is not None:
            try:
                dev = device if device is not None else next(model.parameters()).device
                was_training = bool(model.training)
                model.eval()
                scores: dict[int, float] = {}
                max_per_class = int(max(0, confusable_max_samples_per_class))

                with torch.no_grad():
                    for c in sorted(candidates):
                        idx = torch.where(y == int(c))[0].to(torch.long)
                        if int(idx.numel()) <= 0:
                            continue
                        if max_per_class > 0 and int(idx.numel()) > max_per_class:
                            perm = torch.randperm(int(idx.numel()), generator=generator)
                            idx = idx[perm[:max_per_class]]

                        x_c = X[idx].to(dev)
                        logits = model(x_c)
                        if logits.ndim != 2 or int(logits.shape[1]) <= t:
                            continue
                        probs = torch.softmax(logits, dim=1)[:, t]
                        score = float(probs.mean().detach().cpu().item())
                        if math.isfinite(score):
                            scores[int(c)] = score

                if was_training:
                    model.train()

                if scores:
                    # Main criterion: highest average target probability.
                    # Tie-breakers: more local samples, then smaller class id for determinism.
                    return int(max(scores.keys(), key=lambda c: (scores[c], int(counts.get(c, 0)), -int(c))))
            except Exception:
                try:
                    if 'was_training' in locals() and was_training:
                        model.train()
                except Exception:
                    pass

        return int(max(candidates, key=lambda c: (int(counts.get(c, 0)), -int(c))))

    if mode in {"largest", "largest_non_target", "majority", "majority_non_target"}:
        return int(max(candidates, key=lambda c: (int(counts.get(c, 0)), -int(c))))

    if mode in {"smallest", "smallest_non_target", "minority", "minority_non_target"}:
        return int(min(candidates, key=lambda c: (int(counts.get(c, 0)), int(c))))

    if mode in {"random", "random_non_target"}:
        return _rand_choice_int(sorted(candidates), generator=generator)

    return None

def _least_likely_labels(logits: torch.Tensor, true_y: torch.Tensor | None = None) -> torch.Tensor:
    order = torch.argsort(logits, dim=1, descending=False)
    y_new = order[:, 0].clone()
    if true_y is not None and logits.shape[1] > 1:
        clash = (y_new == true_y)
        if bool(torch.any(clash)):
            y_new[clash] = order[clash, 1]
    return y_new.to(torch.long)


def _highest_non_true_logit_labels(logits: torch.Tensor, true_y: torch.Tensor) -> torch.Tensor:
    """Return the highest-logit class after excluding each sample's true class."""
    if logits.ndim != 2 or int(logits.shape[1]) <= 1:
        return true_y.to(torch.long).clone()

    masked_logits = logits.clone()
    rows = torch.arange(int(logits.shape[0]), device=logits.device)
    true_idx = true_y.to(device=logits.device, dtype=torch.long)
    valid = (true_idx >= 0) & (true_idx < int(logits.shape[1]))
    if bool(torch.any(valid)):
        masked_logits[rows[valid], true_idx[valid]] = float("-inf")
    return torch.argmax(masked_logits, dim=1).to(torch.long)


def _topk_mask_per_row(x: torch.Tensor, frac: float) -> torch.Tensor:
    if x.ndim != 2:
        return torch.ones_like(x)
    d = int(x.shape[1])
    k = int(math.ceil(float(max(0.0, min(1.0, frac))) * d))
    k = max(1, min(k, d))
    vals, idx = torch.topk(torch.abs(x), k=k, dim=1, largest=True, sorted=False)
    mask = torch.zeros_like(x)
    mask.scatter_(1, idx, 1.0)
    return mask


def _clean_label_order_by_boundary(
    logits_cpu: torch.Tensor,
    indices: torch.Tensor,
    *,
    target_class: int,
    source_class: int,
    role: str,
    metric: str,
) -> torch.Tensor:
    """Rank target/reference or source/poison samples by model confusability.

    For target/reference samples, good candidates already have relatively high
    source-class probability and a small target-vs-source margin.

    For source/poison samples, good candidates already have relatively high
    target-class probability and a small source-vs-target margin.
    """
    if indices is None or int(indices.numel()) <= 0:
        return torch.empty((0,), dtype=torch.long)

    idx = indices.detach().cpu().to(torch.long)
    logits = logits_cpu[idx]
    probs = torch.softmax(logits, dim=1)
    t = int(target_class)
    s = int(source_class)

    if str(role).strip().lower() == "target":
        cross_prob = probs[:, s]
    else:
        cross_prob = probs[:, t]

    margin = torch.abs(logits[:, t] - logits[:, s])
    mode = str(metric or "hybrid").strip().lower().replace("-", "_").replace(" ", "_")

    # Python sorting gives deterministic tie-breaking by original sample index.
    rows = []
    for pos, sample_idx in enumerate(idx.tolist()):
        p_cross = float(cross_prob[pos].item())
        m = float(margin[pos].item())
        if mode in {"margin", "small_margin", "logit_margin", "boundary"}:
            key = (m, -p_cross, int(sample_idx))
        elif mode in {"prob", "probability", "target_probability", "cross_probability"}:
            key = (-p_cross, m, int(sample_idx))
        else:  # hybrid: prioritize cross-class probability, then boundary closeness
            key = (-p_cross, m, int(sample_idx))
        rows.append((key, int(sample_idx)))

    rows.sort(key=lambda z: z[0])
    return torch.tensor([sample_idx for _, sample_idx in rows], dtype=torch.long)


def _clean_label_allocate_budget(
    source_classes: list[int],
    counts: dict[int, int],
    total_budget: int,
) -> dict[int, int]:
    """Allocate a total poison budget across source classes deterministically.

    Every selected source receives one sample when the budget allows it. Remaining
    samples are distributed approximately in proportion to local class size.
    """
    ordered = [int(c) for c in source_classes if int(counts.get(int(c), 0)) > 0]
    budget = int(max(0, total_budget))
    if not ordered or budget <= 0:
        return {}

    alloc = {c: 0 for c in ordered}
    if budget < len(ordered):
        for c in ordered[:budget]:
            alloc[c] = 1
        return alloc

    # Give each selected source one slot first so multi-source mode actually uses
    # every selected class whenever possible.
    for c in ordered:
        alloc[c] = 1
    remaining = budget - len(ordered)
    if remaining <= 0:
        return alloc

    capacities = {c: max(0, int(counts[c]) - alloc[c]) for c in ordered}
    total_capacity = sum(capacities.values())
    if total_capacity <= 0:
        return alloc

    raw = {c: remaining * capacities[c] / float(total_capacity) for c in ordered}
    for c in ordered:
        add = min(capacities[c], int(math.floor(raw[c])))
        alloc[c] += add
        remaining -= add

    # Largest-remainder distribution with stable source-class ordering.
    ranked = sorted(
        ordered,
        key=lambda c: (raw[c] - math.floor(raw[c]), capacities[c], -ordered.index(c)),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for c in ranked:
            if alloc[c] < int(counts[c]):
                alloc[c] += 1
                remaining -= 1
                progressed = True
                if remaining <= 0:
                    break
        if not progressed:
            break
    return alloc


def _clean_label_flat_gradient(
    model: torch.nn.Module,
    X: torch.Tensor,
    labels: torch.Tensor,
    params: tuple[torch.nn.Parameter, ...],
    *,
    create_graph: bool,
) -> torch.Tensor:
    loss = nn.CrossEntropyLoss()(model(X), labels)
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=bool(create_graph),
        create_graph=bool(create_graph),
        allow_unused=True,
    )
    return _flatten_grad_list(grads)


def _clean_label_gradient_similarity_matrix(
    model: torch.nn.Module,
    X_dev: torch.Tensor,
    candidate_idx: torch.Tensor,
    *,
    source_class: int,
    target_gradients: list[torch.Tensor],
    params: tuple[torch.nn.Parameter, ...],
) -> torch.Tensor:
    """Return cosine similarities for every source candidate/reference pair.

    Rows follow ``candidate_idx`` order and columns follow ``target_gradients`` order.
    The returned tensor is on CPU so deterministic Python-side assignment is cheap.
    """
    n_candidates = int(candidate_idx.numel()) if candidate_idx is not None else 0
    n_references = len(target_gradients)
    if n_candidates <= 0 or n_references <= 0:
        return torch.empty((n_candidates, n_references), dtype=torch.float32)

    rows: list[torch.Tensor] = []
    for sample_idx in candidate_idx.detach().cpu().tolist():
        x = X_dev[int(sample_idx) : int(sample_idx) + 1]
        y = torch.tensor([int(source_class)], dtype=torch.long, device=X_dev.device)
        source_gradient = _clean_label_flat_gradient(
            model,
            x,
            y,
            params,
            create_graph=False,
        ).detach()

        sims: list[torch.Tensor] = []
        for target_gradient in target_gradients:
            cos = torch.nn.functional.cosine_similarity(
                source_gradient,
                target_gradient,
                dim=0,
                eps=1e-12,
            )
            if not bool(torch.isfinite(cos)):
                cos = torch.tensor(-1.0, dtype=source_gradient.dtype, device=source_gradient.device)
            sims.append(cos.detach().to(dtype=torch.float32, device="cpu"))
        rows.append(torch.stack(sims, dim=0))

    return torch.stack(rows, dim=0)


def _clean_label_assign_candidates_to_references(
    candidate_idx: torch.Tensor,
    similarities: torch.Tensor,
    *,
    num_poison: int,
    mode: str = "balanced_similarity",
) -> dict[int, torch.Tensor]:
    """Select poison candidates and assign them to target references.

    ``balanced_similarity`` implements capacity-balanced matching:
      1) rank each candidate's target references by gradient similarity;
      2) give every reference an approximately equal capacity;
      3) process strong candidates first and use their best reference with capacity;
      4) when that reference is full, use the candidate's next-best reference.

    ``best_similarity`` keeps unrestricted best-reference matching and may therefore
    assign every selected candidate to the same target reference.
    """
    if candidate_idx is None:
        return {}
    idx = candidate_idx.detach().cpu().to(torch.long)
    sims = similarities.detach().cpu().to(torch.float32)
    if idx.ndim != 1 or sims.ndim != 2 or int(sims.shape[0]) != int(idx.numel()):
        return {}

    n_candidates = int(idx.numel())
    n_references = int(sims.shape[1])
    k = min(max(0, int(num_poison)), n_candidates)
    if k <= 0 or n_references <= 0:
        return {}

    mode_norm = str(mode or "balanced_similarity").strip().lower().replace("-", "_").replace(" ", "_")
    balanced = mode_norm not in {
        "best",
        "best_similarity",
        "unbalanced",
        "unrestricted",
        "unrestricted_similarity",
    }

    # Candidate order: strongest achievable match first. Earlier boundary rank and
    # lower sample index are deterministic tie-breakers.
    candidate_rows = list(range(n_candidates))
    candidate_rows.sort(
        key=lambda row: (
            -float(torch.max(sims[row]).item()),
            int(row),
            int(idx[row].item()),
        )
    )

    if balanced:
        base = k // n_references
        remainder = k % n_references

        # All capacities differ by at most one. Give the remainder slots to target
        # references that have the strongest available candidate match.
        reference_quality = []
        for ref in range(n_references):
            best = float(torch.max(sims[:, ref]).item()) if n_candidates > 0 else -float("inf")
            reference_quality.append((-best, int(ref)))
        reference_quality.sort()
        extra_refs = {ref for _, ref in reference_quality[:remainder]}
        capacity = [base + (1 if ref in extra_refs else 0) for ref in range(n_references)]
    else:
        capacity = [k] * n_references

    assigned: dict[int, list[int]] = {ref: [] for ref in range(n_references)}
    selected = 0
    for row in candidate_rows:
        preference = list(range(n_references))
        preference.sort(key=lambda ref: (-float(sims[row, ref].item()), int(ref)))
        for ref in preference:
            if capacity[ref] <= 0:
                continue
            assigned[ref].append(int(idx[row].item()))
            capacity[ref] -= 1
            selected += 1
            break
        if selected >= k:
            break

    return {
        ref: torch.tensor(sample_ids, dtype=torch.long)
        for ref, sample_ids in assigned.items()
        if sample_ids
    }


def _apply_clean_label_poison_to_trainloader(
    train_loader: DataLoader,
    model: torch.nn.Module,
    *,
    target_class: int,
    poison_rate: float,
    eps: float,
    steps: int,
    step_size: float,
    alpha: float,
    source_class: int,
    tabular_feature_frac: float,
    generator: torch.Generator | None,
    device: torch.device,
    max_poison_samples: int = 0,
    target_mode: str = "fixed",
    target_classes: object = None,
    local_smallest_k: int = 3,
    min_target_samples: int = 1,
    source_mode: str = "mode_poison",
    min_source_samples: int = 1,
    confusable_max_samples_per_class: int = 0,
    confusable_topk_sources: int = 1,
    sample_selection_mode: str = "boundary_gradient",
    target_selection_metric: str = "hybrid",
    source_selection_metric: str = "hybrid",
    target_reference_count: int = 3,
    reference_assignment_mode: str = "balanced_similarity",
    source_candidate_multiplier: float = 2.0,
    gradient_max_candidates_per_source: int = 24,
) -> int:
    """Apply pair-specific, boundary-aware CleanLabel poisoning.

    For each selected target/source class pair:
      1) choose target references already confusable with that source class;
      2) preselect source samples near the target boundary;
      3) match every candidate against every target-reference gradient;
      4) assign candidates to their most similar reference subject to balanced capacity;
      5) optimize each reference-specific poison group while retaining clean labels.

    Multi-source attacks are handled independently per source class, avoiding the
    previous majority-label objective for a mixed poison batch.
    """
    ds, X_cpu, y_cpu = _get_dataset_tensors(train_loader)
    if ds is None or X_cpu is None or y_cpu is None:
        return 0

    if int(steps) <= 0 or float(step_size) <= 0.0 or float(eps) <= 0.0:
        return 0

    num_classes = _infer_num_classes_from_model(model)
    if num_classes is None or int(num_classes) <= 1:
        return 0

    selected_target = _select_clean_label_target_class(
        y_cpu,
        preferred_target_class=int(target_class),
        num_classes=int(num_classes),
        requested_source_class=int(source_class),
        target_mode=str(target_mode),
        target_classes=target_classes,
        local_smallest_k=int(local_smallest_k),
        min_target_samples=int(min_target_samples),
        generator=generator,
    )
    if selected_target is None:
        return 0
    target_class = int(selected_target) % int(num_classes)

    selected_source = _select_clean_label_source_class(
        y_cpu,
        target_class=int(target_class),
        requested_source_class=int(source_class),
        source_mode=str(source_mode),
        min_source_samples=int(min_source_samples),
        generator=generator,
        X=X_cpu,
        model=model,
        device=device,
        num_classes=int(num_classes),
        confusable_max_samples_per_class=int(confusable_max_samples_per_class),
        confusable_topk_sources=int(confusable_topk_sources),
    )

    if isinstance(selected_source, (list, tuple, set)):
        source_classes = [int(c) for c in selected_source if int(c) != target_class]
    elif selected_source is not None:
        source_classes = [int(selected_source)] if int(selected_source) != target_class else []
    else:
        # In unrestricted mode, treat every locally present non-target class as a
        # separate source pair rather than collapsing them to a majority label.
        source_classes = sorted(
            int(c)
            for c in torch.unique(y_cpu.detach().cpu().to(torch.long)).tolist()
            if int(c) != target_class
        )

    # De-duplicate while preserving the source-selector ranking.
    source_classes = list(dict.fromkeys(source_classes))
    source_counts = {
        c: int((y_cpu == int(c)).sum().item())
        for c in source_classes
        if int((y_cpu == int(c)).sum().item()) >= int(max(1, min_source_samples))
    }
    source_classes = [c for c in source_classes if c in source_counts]
    if not source_classes:
        return 0

    eligible_total = int(sum(source_counts.values()))
    rate = float(max(0.0, min(1.0, poison_rate)))
    total_budget = int(math.ceil(rate * eligible_total))
    if int(max_poison_samples) > 0:
        total_budget = min(total_budget, int(max_poison_samples))
    total_budget = min(total_budget, eligible_total)
    if total_budget <= 0:
        return 0

    budget_by_source = _clean_label_allocate_budget(source_classes, source_counts, total_budget)
    if not budget_by_source:
        return 0

    was_training = bool(model.training)
    model.eval()

    X_dev = X_cpu.to(device)
    y_dev = y_cpu.to(device)
    low, high = _featurewise_bounds(X_dev)
    params = tuple(p for p in model.parameters() if p.requires_grad)
    if not params:
        if was_training:
            model.train()
        return 0

    selection_mode = str(sample_selection_mode or "boundary_gradient").strip().lower().replace("-", "_").replace(" ", "_")
    use_boundary = selection_mode not in {"random", "uniform", "legacy"}
    use_gradient = selection_mode in {
        "boundary_gradient",
        "boundary_and_gradient",
        "gradient",
        "gradient_similarity",
        "confusable_gradient",
    }

    alpha_f = float(max(0.0, min(1.0, alpha)))
    feat_frac = float(max(0.0, min(1.0, tabular_feature_frac)))
    reference_count = max(1, int(target_reference_count))
    assignment_mode = str(reference_assignment_mode or "balanced_similarity")
    multiplier = max(1.0, float(source_candidate_multiplier))
    grad_cap = int(max(0, gradient_max_candidates_per_source))

    total_poisoned = 0
    try:
        # Model is fixed during poison generation, so one forward pass is enough for
        # all boundary-based target/source ranking.
        with torch.no_grad():
            logits_cpu = model(X_dev).detach().cpu()

        for source_cls in source_classes:
            k_source = int(budget_by_source.get(int(source_cls), 0))
            if k_source <= 0:
                continue

            source_pool = torch.where(y_cpu == int(source_cls))[0].to(torch.long)
            target_pool = torch.where(y_cpu == int(target_class))[0].to(torch.long)
            if int(source_pool.numel()) <= 0 or int(target_pool.numel()) <= 0:
                continue

            if use_boundary:
                target_ranked = _clean_label_order_by_boundary(
                    logits_cpu,
                    target_pool,
                    target_class=int(target_class),
                    source_class=int(source_cls),
                    role="target",
                    metric=str(target_selection_metric),
                )
                source_ranked = _clean_label_order_by_boundary(
                    logits_cpu,
                    source_pool,
                    target_class=int(target_class),
                    source_class=int(source_cls),
                    role="source",
                    metric=str(source_selection_metric),
                )
            else:
                tperm = torch.randperm(int(target_pool.numel()), generator=generator)
                sperm = torch.randperm(int(source_pool.numel()), generator=generator)
                target_ranked = target_pool[tperm]
                source_ranked = source_pool[sperm]

            target_idx = target_ranked[: min(reference_count, int(target_ranked.numel()))]
            if int(target_idx.numel()) <= 0:
                continue

            # Preselect a manageable set of boundary-adjacent source candidates.
            candidate_n = max(k_source, int(math.ceil(k_source * multiplier)))
            if grad_cap > 0:
                candidate_n = max(k_source, min(candidate_n, grad_cap))
            candidate_n = min(candidate_n, int(source_ranked.numel()))
            candidate_idx = source_ranked[:candidate_n]
            if int(candidate_idx.numel()) <= 0:
                continue

            # Keep one separate target gradient per reference. Averaging them here
            # would erase the poison-to-reference matching that follows.
            target_gradients: list[torch.Tensor] = []
            for target_sample_idx in target_idx.detach().cpu().tolist():
                x_target_one = X_dev[int(target_sample_idx) : int(target_sample_idx) + 1].detach().clone()
                y_adv_one = torch.tensor(
                    [int(source_cls)],
                    dtype=torch.long,
                    device=device,
                )
                target_gradients.append(
                    _clean_label_flat_gradient(
                        model,
                        x_target_one,
                        y_adv_one,
                        params,
                        create_graph=False,
                    ).detach()
                )

            if use_gradient:
                similarities = _clean_label_gradient_similarity_matrix(
                    model,
                    X_dev,
                    candidate_idx,
                    source_class=int(source_cls),
                    target_gradients=target_gradients,
                    params=params,
                )
                poison_groups = _clean_label_assign_candidates_to_references(
                    candidate_idx,
                    similarities,
                    num_poison=int(k_source),
                    mode=assignment_mode,
                )
            else:
                # Without gradient matching, distribute boundary-ranked candidates
                # deterministically across references so the references are still used.
                k_keep = min(int(k_source), int(candidate_idx.numel()))
                selected_idx = candidate_idx[:k_keep]
                dummy = torch.zeros(
                    (int(selected_idx.numel()), len(target_gradients)),
                    dtype=torch.float32,
                )
                poison_groups = _clean_label_assign_candidates_to_references(
                    selected_idx,
                    dummy,
                    num_poison=int(k_keep),
                    mode="balanced_similarity",
                )

            for reference_pos, poison_idx in poison_groups.items():
                if int(poison_idx.numel()) <= 0:
                    continue
                target_gradient = target_gradients[int(reference_pos)]

                x_orig = X_dev[poison_idx].detach().clone()
                y_poison = y_dev[poison_idx].detach().clone()  # labels remain source_cls
                delta = torch.zeros_like(x_orig, requires_grad=True)

                for _ in range(int(steps)):
                    x_adv = _project_linf_box(
                        x_orig + delta,
                        x_orig,
                        eps=float(eps),
                        low=low,
                        high=high,
                    )

                    flat_poison = _clean_label_flat_gradient(
                        model,
                        x_adv,
                        y_poison,
                        params,
                        create_graph=True,
                    )

                    cos = torch.nn.functional.cosine_similarity(
                        flat_poison,
                        target_gradient,
                        dim=0,
                        eps=1e-12,
                    )
                    reg = torch.mean((x_adv - x_orig) ** 2)
                    obj = alpha_f * (1.0 - cos) + (1.0 - alpha_f) * reg

                    grad_delta = torch.autograd.grad(
                        obj,
                        delta,
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                    mask = _topk_mask_per_row(grad_delta, feat_frac)
                    grad_delta = grad_delta * mask

                    with torch.no_grad():
                        delta = (delta - float(step_size) * torch.sign(grad_delta)).detach()
                        x_new = _project_linf_box(
                            x_orig + delta,
                            x_orig,
                            eps=float(eps),
                            low=low,
                            high=high,
                        )
                        delta = (x_new - x_orig).detach().requires_grad_(True)

                X_new = _project_linf_box(
                    x_orig + delta.detach(),
                    x_orig,
                    eps=float(eps),
                    low=low,
                    high=high,
                ).detach().cpu()
                X_cpu[poison_idx] = X_new
                total_poisoned += int(poison_idx.numel())
    finally:
        if was_training:
            model.train()

    return int(total_poisoned)

def _normalize_feature_poison_mode(mode: object) -> str:
    """Normalize FeaturePoison mode names to the four supported implementations."""
    s = str(mode or "random_local").strip().lower().replace("-", "_").replace(" ", "_")
    if s in {"current", "legacy", "random", "random_local", "local_random"}:
        return "random_local"
    if s in {"opposite", "extreme", "opposite_extreme", "opposite_local_extreme"}:
        return "opposite_extreme"
    if s in {"gradient", "rf_gradient", "gradient_guided", "model_gradient"}:
        return "gradient"
    if s in {
        "global",
        "global_quantile",
        "robust_global",
        "random_global_quantile",
        "global_quantile_random",
    }:
        return "random_global_quantile"
    return "random_local"


def _feature_poison1_mode(run_config: dict) -> str:
    """FeaturePoison1 uses gradient by default; two randomized alternatives remain."""
    requested = _normalize_feature_poison_mode(run_config.get("feature_poison_mode", "gradient"))
    if requested not in {"gradient", "random_local", "random_global_quantile"}:
        raise ValueError(
            "FeaturePoison1 supports gradient, random_local, and "
            "random_global_quantile; use FeaturePoison2 for opposite_extreme"
        )
    return requested


def _apply_feature_poison_to_trainloader(
    train_loader: DataLoader,
    model: torch.nn.Module,
    *,
    poison_rate: float,
    topk_frac: float,
    feature_prob: float,
    rf_estimators: int,
    rf_max_samples: int,
    generator: torch.Generator | None,
    seed: int,
    mode: str = "random_local",
    feature_metadata: dict | None = None,
    gradient_strength: float = 1.0,
    device: torch.device | None = None,
) -> int:
    """Apply RF-guided, feature-type-aware poisoning to tabular training data.

    Supported modes:
      - random_local: legacy random replacement inside local min/max bounds.
      - opposite_extreme: move continuous features to the opposite local extreme.
      - gradient: use the neural model input gradient to choose a harmful direction.
      - random_global_quantile: random replacement inside robust global quantile bounds.

    Binary features are always kept at one of their two valid values. If globally
    detected/configured one-hot groups are available, changing any selected member
    switches the entire group to another valid one-hot category.
    """
    ds, X_cpu, y_cpu = _get_dataset_tensors(train_loader)
    if ds is None or X_cpu is None or y_cpu is None:
        return 0
    if int(X_cpu.shape[0]) <= 2 or int(X_cpu.shape[1]) <= 0:
        return 0

    poison_idx = _sample_poison_indices(y_cpu, poison_rate=float(poison_rate), generator=generator)
    if int(poison_idx.numel()) <= 0:
        return 0

    X_np = X_cpu.detach().cpu().numpy()
    y_np = y_cpu.detach().cpu().numpy()

    fit_idx = np.arange(X_np.shape[0], dtype=np.int64)
    max_fit = int(max(0, rf_max_samples))
    if max_fit > 0 and fit_idx.size > max_fit:
        rng = np.random.default_rng(int(seed))
        fit_idx = np.sort(rng.choice(fit_idx, size=max_fit, replace=False).astype(np.int64))

    # Preserve the reference implementation's requirement that the local RF sees
    # at least two labels. The variance fallback below is only for RF fit failures.
    if np.unique(y_np[fit_idx]).size <= 1:
        return 0

    try:
        rf = RandomForestClassifier(
            n_estimators=max(10, int(rf_estimators)),
            random_state=int(seed),
            n_jobs=1,
        )
        rf.fit(X_np[fit_idx], y_np[fit_idx])
        importances = np.asarray(rf.feature_importances_, dtype=np.float64)
    except Exception:
        importances = np.var(X_np[fit_idx], axis=0).astype(np.float64, copy=False)

    d = int(importances.shape[0])
    if d <= 0:
        return 0
    k = int(math.ceil(float(max(0.0, min(1.0, topk_frac))) * d))
    k = max(1, min(k, d))
    feat_idx = np.argsort(importances)[-k:].astype(np.int64, copy=False)
    selected_features = {int(f) for f in feat_idx.tolist()}

    mode_norm = _normalize_feature_poison_mode(mode)
    fp = float(max(0.0, min(1.0, feature_prob)))
    grad_strength = float(max(0.0, min(1.0, gradient_strength)))

    local_low = torch.min(X_cpu, dim=0).values.detach().cpu()
    local_high = torch.max(X_cpu, dim=0).values.detach().cpu()
    active_low = local_low.clone()
    active_high = local_high.clone()

    meta = feature_metadata if isinstance(feature_metadata, dict) else {}
    global_low_raw = meta.get("robust_low", None)
    global_high_raw = meta.get("robust_high", None)
    global_bounds_ok = bool(meta.get("global_model_space", False))
    if global_bounds_ok and global_low_raw is not None and global_high_raw is not None:
        try:
            global_low = torch.as_tensor(global_low_raw, dtype=X_cpu.dtype).reshape(-1)
            global_high = torch.as_tensor(global_high_raw, dtype=X_cpu.dtype).reshape(-1)
            if int(global_low.numel()) != d or int(global_high.numel()) != d:
                global_bounds_ok = False
        except Exception:
            global_bounds_ok = False
    if not global_bounds_ok:
        global_low = local_low
        global_high = local_high

    if mode_norm == "random_global_quantile":
        value_low = torch.minimum(global_low, global_high)
        value_high = torch.maximum(global_low, global_high)
    else:
        value_low = local_low
        value_high = local_high

    # Global binary levels are preferable because a small non-IID client can miss
    # one level locally. Otherwise infer valid levels from the local tensor.
    binary_indices = {
        int(j) for j in meta.get("binary_indices", [])
        if 0 <= int(j) < d
    }
    binary_low_raw = meta.get("binary_low", None)
    binary_high_raw = meta.get("binary_high", None)
    binary_levels_ok = bool(meta.get("global_model_space", False))
    if binary_levels_ok and binary_low_raw is not None and binary_high_raw is not None:
        try:
            b_lo = torch.as_tensor(binary_low_raw, dtype=X_cpu.dtype).reshape(-1)
            b_hi = torch.as_tensor(binary_high_raw, dtype=X_cpu.dtype).reshape(-1)
            if int(b_lo.numel()) != d or int(b_hi.numel()) != d:
                binary_levels_ok = False
        except Exception:
            binary_levels_ok = False
    if not binary_levels_ok:
        b_lo = local_low.clone()
        b_hi = local_high.clone()
        # Add locally observed two-level columns when global metadata is unavailable.
        for j in range(d):
            vals = torch.unique(X_cpu[:, j].detach().cpu())
            if int(vals.numel()) == 2:
                binary_indices.add(j)
                b_lo[j] = torch.min(vals)
                b_hi[j] = torch.max(vals)

    for j in binary_indices:
        if torch.isfinite(b_lo[j]) and torch.isfinite(b_hi[j]) and float(b_hi[j]) > float(b_lo[j]):
            active_low[j] = b_lo[j]
            active_high[j] = b_hi[j]

    # Validate one-hot groups and prevent a feature from belonging to two groups.
    onehot_groups: list[list[int]] = []
    used_group_features: set[int] = set()
    for raw_group in meta.get("onehot_groups", []) or []:
        group = [
            int(j) for j in raw_group
            if 0 <= int(j) < d and int(j) in binary_indices and int(j) not in used_group_features
        ]
        group = list(dict.fromkeys(group))
        if len(group) >= 2:
            onehot_groups.append(group)
            used_group_features.update(group)

    group_by_feature: dict[int, int] = {}
    for group_pos, group in enumerate(onehot_groups):
        for j in group:
            group_by_feature[j] = group_pos

    selected_group_positions = sorted({
        group_by_feature[j] for j in selected_features if j in group_by_feature
    })
    individual_features = sorted(j for j in selected_features if j not in group_by_feature)

    # Improvement 6: compute per-sample input gradients once. Cross-entropy is
    # maximized by moving in the positive input-gradient direction.
    grad_cpu = None
    if mode_norm == "gradient":
        if device is None:
            try:
                device = next(model.parameters()).device
            except Exception:
                device = torch.device("cpu")
        was_training = bool(model.training)
        try:
            model.eval()
            x_sel = X_cpu[poison_idx].to(device).detach().requires_grad_(True)
            y_sel = y_cpu[poison_idx].to(device)
            loss = nn.CrossEntropyLoss(reduction="sum")(model(x_sel), y_sel)
            grad_cpu = torch.autograd.grad(
                loss,
                x_sel,
                retain_graph=False,
                create_graph=False,
            )[0].detach().cpu()
        except Exception:
            grad_cpu = None
        finally:
            if was_training:
                model.train()
        # Safe fallback retains a working attack if gradient generation fails.
        if grad_cpu is None or tuple(grad_cpu.shape) != (int(poison_idx.numel()), d):
            mode_norm = "opposite_extreme"

    def _draw_probability() -> float:
        if generator is None:
            return float(torch.rand((1,)).item())
        return float(torch.rand((1,), generator=generator).item())

    def _draw_int(high_exclusive: int) -> int:
        if high_exclusive <= 1:
            return 0
        if generator is None:
            return int(torch.randint(0, high_exclusive, (1,)).item())
        return int(torch.randint(0, high_exclusive, (1,), generator=generator).item())

    changed_rows = 0
    with torch.no_grad():
        for poison_pos, row_idx in enumerate(poison_idx.tolist()):
            before = X_cpu[row_idx].detach().clone()
            grad_row = grad_cpu[poison_pos] if grad_cpu is not None else None

            # Handle each selected one-hot group as one atomic feature. This keeps
            # exactly one category active after the attack.
            for group_pos in selected_group_positions:
                if fp < 1.0 and _draw_probability() > fp:
                    continue
                group = onehot_groups[group_pos]
                cols = torch.tensor(group, dtype=torch.long)
                lows = active_low[cols]
                highs = active_high[cols]
                current = X_cpu[row_idx, cols].detach().cpu()
                active_pos = int(torch.argmin(torch.abs(current - highs)).item())
                alternatives = [i for i in range(len(group)) if i != active_pos]
                if not alternatives:
                    continue

                if mode_norm in {"random_local", "random_global_quantile"}:
                    new_active = alternatives[_draw_int(len(alternatives))]
                elif mode_norm == "opposite_extreme":
                    # For a categorical one-hot block there is no numeric opposite;
                    # choose the most RF-important different category.
                    new_active = max(alternatives, key=lambda pos: (float(importances[group[pos]]), -pos))
                else:  # gradient
                    best_pos = alternatives[0]
                    best_gain = -float("inf")
                    g = grad_row[cols]
                    for candidate_pos in alternatives:
                        candidate = lows.clone()
                        candidate[candidate_pos] = highs[candidate_pos]
                        gain = float(torch.sum(g * (candidate - current)).item())
                        if gain > best_gain:
                            best_gain = gain
                            best_pos = candidate_pos
                    new_active = best_pos

                replacement = lows.clone()
                replacement[new_active] = highs[new_active]
                X_cpu[row_idx, cols] = replacement.to(dtype=X_cpu.dtype)

            # Handle continuous and standalone binary features.
            for f in individual_features:
                if fp < 1.0 and _draw_probability() > fp:
                    continue
                current = float(X_cpu[row_idx, f].item())

                if f in binary_indices:
                    valid_lo = float(active_low[f].item())
                    valid_hi = float(active_high[f].item())
                    if not (math.isfinite(valid_lo) and math.isfinite(valid_hi)) or valid_hi <= valid_lo:
                        continue
                    # Binary variables only have one valid alternative, so every
                    # mode performs a valid 0/1-level flip instead of interpolation.
                    if abs(current - valid_lo) <= abs(current - valid_hi):
                        new_value = valid_hi
                    else:
                        new_value = valid_lo
                else:
                    lo = float(value_low[f].item())
                    hi = float(value_high[f].item())
                    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
                        continue
                    if mode_norm in {"random_local", "random_global_quantile"}:
                        new_value = lo + (hi - lo) * _draw_probability()
                    elif mode_norm == "opposite_extreme":
                        midpoint = 0.5 * (lo + hi)
                        new_value = lo if current >= midpoint else hi
                    else:  # gradient
                        g = float(grad_row[f].item())
                        target = hi if g >= 0.0 else lo
                        new_value = current + grad_strength * (target - current)
                        new_value = min(hi, max(lo, new_value))

                X_cpu[row_idx, f] = float(new_value)

            if bool(torch.any(torch.abs(X_cpu[row_idx] - before) > 1e-12)):
                changed_rows += 1

    return int(changed_rows)


def _prepare_inverted_loss_indices(
    train_loader: DataLoader,
    *,
    poison_rate: float,
    generator: torch.Generator | None,
    max_poison_samples: int = 0,
) -> torch.Tensor:
    _ds, _X, y = _get_dataset_tensors(train_loader)
    if y is None:
        return torch.empty((0,), dtype=torch.long)
    return _sample_poison_indices(
        y,
        poison_rate=float(poison_rate),
        generator=generator,
        max_samples=int(max_poison_samples),
    )


def _train_with_inverted_loss_relabeling(
    model,
    train_loader,
    poison_idx: torch.Tensor,
    *,
    label_mode: str = "least_likely",
    num_epochs: int = 1,
    lr: float = 1e-3,
    device=None,
    seed: int = 0,
    prox_mu: float = 0.0,
    global_params: list[torch.Tensor] | None = None,
    scaffold: bool = False,
    scaffold_c_global: "dict[str, torch.Tensor] | None" = None,
    scaffold_ci: "dict[str, torch.Tensor] | None" = None,
):
    if device is None:
        device = torch.device("cpu")

    ds, X_cpu, y_cpu = _get_dataset_tensors(train_loader)
    if ds is None or X_cpu is None or y_cpu is None or poison_idx is None or int(poison_idx.numel()) <= 0:
        from fltabular.task import trainer as _base_trainer
        return _base_trainer(
            model,
            train_loader,
            num_epochs=num_epochs,
            lr=lr,
            device=device,
            prox_mu=prox_mu,
            global_params=global_params,
            scaffold=scaffold,
            scaffold_c_global=scaffold_c_global,
            scaffold_ci=scaffold_ci,
        )

    model.to(device)
    model.train()

    use_fedprox = (not scaffold) and (prox_mu is not None) and (float(prox_mu) > 0.0) and (global_params is not None)
    if use_fedprox:
        gparams = [gp.detach().to(device) for gp in global_params]
        if len(gparams) != len(list(model.parameters())):
            use_fedprox = False
    else:
        gparams = []

    named_params = list(model.named_parameters())
    if scaffold:
        c_dev = {}
        ci_dev = {}
        for n, p in named_params:
            z = torch.zeros_like(p.detach(), device=device)
            c_dev[n] = (scaffold_c_global.get(n).detach().to(device) if (scaffold_c_global and n in scaffold_c_global) else z)
            ci_dev[n] = (scaffold_ci.get(n).detach().to(device) if (scaffold_ci and n in scaffold_ci) else z)
        w0 = {n: p.detach().clone() for n, p in named_params}
        optimizer = optim.SGD(model.parameters(), lr=float(lr))
    else:
        optimizer = optim.Adam(model.parameters(), lr=float(lr))

    criterion = nn.CrossEntropyLoss()
    poison_lookup = torch.zeros((int(y_cpu.numel()),), dtype=torch.bool)
    poison_idx_cpu = poison_idx.detach().cpu().to(torch.long)
    poison_lookup[poison_idx_cpu] = True

    batch_size = int(getattr(train_loader, "batch_size", 32) or 32)

    # Freeze each poisoned sample's wrong label once for this local training round.
    # These labels are computed from the model received at the start of the round and
    # then reused for every local epoch, so they cannot change as local training updates
    # the model.
    frozen_poison_labels = torch.full(
        (int(y_cpu.numel()),),
        fill_value=-1,
        dtype=torch.long,
    )
    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        for start in range(0, int(poison_idx_cpu.numel()), batch_size):
            idx_chunk = poison_idx_cpu[start : start + batch_size]
            X_chunk = X_cpu[idx_chunk].to(device, non_blocking=True)
            y_chunk = y_cpu[idx_chunk].to(device, non_blocking=True)
            logits_chunk = model(X_chunk)
            if str(label_mode).strip().lower() == "highest_non_true":
                frozen_chunk = _highest_non_true_logit_labels(logits_chunk, true_y=y_chunk)
            else:
                frozen_chunk = _least_likely_labels(logits_chunk, true_y=y_chunk)
            frozen_poison_labels[idx_chunk] = frozen_chunk.detach().cpu()
    if was_training:
        model.train()

    idx_tensor = torch.arange(int(y_cpu.numel()), dtype=torch.long)
    idx_ds = TensorDataset(X_cpu, y_cpu, idx_tensor)

    steps = 0
    for epoch in range(int(num_epochs)):
        g_epoch = torch.Generator(device="cpu")
        g_epoch.manual_seed(int(seed) + int(epoch))
        local_loader = DataLoader(idx_ds, batch_size=batch_size, shuffle=True, generator=g_epoch)

        for X_batch, y_batch, idx_batch in local_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            idx_cpu = idx_batch.to(torch.long)

            poison_mask_cpu = poison_lookup[idx_cpu]
            poison_mask = poison_mask_cpu.to(device)
            y_train = y_batch.clone()

            if bool(torch.any(poison_mask)):
                poisoned_batch_idx = idx_cpu[poison_mask_cpu]
                y_train[poison_mask] = frozen_poison_labels[poisoned_batch_idx].to(
                    device,
                    non_blocking=True,
                )

            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss = criterion(logits, y_train)

            if use_fedprox:
                prox = 0.0
                for p, gp in zip(model.parameters(), gparams):
                    prox = prox + torch.sum((p - gp) ** 2)
                loss = loss + (float(prox_mu) / 2.0) * prox

            loss.backward()

            if scaffold:
                with torch.no_grad():
                    for n, p in named_params:
                        if p.grad is None:
                            continue
                        p.grad.add_(c_dev[n]).add_(-ci_dev[n])

            optimizer.step()
            steps += 1

    if not scaffold:
        return None, steps

    if steps <= 0 or float(lr) <= 0.0:
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



# ===========================================================================
# ======================================================================================
# FedIMP: Parameter Importance-based Model Poisoning (Computers & Security 144 (2024) 103936)
# --------------------------------------------------------------------------------------
# High-level (paper):
#   1) Estimate parameter importance via (empirical) Fisher diag on an auxiliary set D_aux.
#   2) Create a binary mask M selecting top-k% important parameters.
#   3) Simulate benign updates to approximate μ and σ of benign updates.
#   4) Craft malicious update: Δw_mal = μ - δ * M * σ
#   5) Choose δ by (pre-poison in round 1) or binary-search to satisfy the paper inequality.
#
# In this Flower project, clients upload *weights*, so we forge:
#   w_mal = w_global + Δw_mal    (NOT: w_local + Δw_mal)
# ======================================================================================

def _fedimp_make_aux_loader(
    train_loader: DataLoader,
    *,
    max_batches: int,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Collect up to `max_batches` batches from `train_loader` and return a new small loader.

    This is used for both Fisher/importance and simulation, to keep FedIMP overhead bounded.
    """
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for b, (x, y) in enumerate(train_loader):
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
        if max_batches > 0 and (b + 1) >= int(max_batches):
            break

    if len(xs) == 0:
        raise ValueError("FedIMP: could not sample auxiliary batches from train_loader")

    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    ds = TensorDataset(X, y)

    bs = int(getattr(train_loader, "batch_size", 32) or 32)
    # Use a fixed generator for deterministic shuffling if provided
    if generator is None:
        return DataLoader(ds, batch_size=bs, shuffle=True)
    return DataLoader(ds, batch_size=bs, shuffle=True, generator=generator)


def _fedimp_empirical_fisher_diag(
    model: torch.nn.Module,
    aux_loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
) -> torch.Tensor:
    """Empirical Fisher diagonal approximation (up to a constant), returned as a flat CPU tensor."""
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_params = sum(int(p.numel()) for p in model.parameters())
    imp = torch.zeros(total_params, dtype=torch.float32, device="cpu")

    used = 0
    for b, (X, y) in enumerate(aux_loader):
        if max_batches > 0 and b >= int(max_batches):
            break

        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()

        grads = []
        for p in model.parameters():
            if p.grad is None:
                grads.append(torch.zeros(int(p.numel()), dtype=torch.float32, device="cpu"))
            else:
                grads.append(p.grad.detach().view(-1).float().cpu())
        gflat = torch.cat(grads, dim=0)
        imp += gflat * gflat
        used += 1

    if used > 0:
        imp /= float(used)
    return imp


def _fedimp_topk_mask(importance: torch.Tensor, topk_ratio: float) -> torch.Tensor:
    """Binary mask (float32 0/1) selecting top-k% important parameters."""
    if importance.ndim != 1:
        importance = importance.view(-1)

    r = float(topk_ratio)
    r = max(0.0, min(1.0, r))

    n = int(importance.numel())
    k = int(math.ceil(r * n))
    k = max(1, min(n, k))

    idx = torch.topk(importance, k=k, largest=True).indices
    mask = torch.zeros_like(importance, dtype=torch.float32)
    mask[idx] = 1.0
    return mask


def _fedimp_eval_loss(
    model: torch.nn.Module,
    aux_loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
) -> float:
    """Average CrossEntropy loss over up to max_batches batches."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total = 0.0
    used = 0
    with torch.no_grad():
        for b, (X, y) in enumerate(aux_loader):
            if max_batches > 0 and b >= int(max_batches):
                break
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            total += float(criterion(model(X), y).item())
            used += 1

    return total / max(1, used)


def _fedimp_simulate_benign_updates(
    *,
    global_state_dict: dict,
    num_clients: int,
    run_config: dict,
    sim_loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
    num_sims: int,
    base_seed: int,
    deterministic_algos: bool,
) -> torch.Tensor:
    """Simulate benign updates (Δw) using compromised data. Returns matrix [K, D] on CPU.

    The simulation model is constructed only once. Before every simulation, all
    parameters and buffers are restored from the same global state, while
    ``trainer`` creates a fresh optimizer. This avoids repeated model construction
    without changing the FedIMP simulation procedure.
    """
    # Build one reusable simulation model and flatten the global model once.
    sim_net = build_model(num_partitions=int(num_clients), run_config=run_config)
    sim_net.load_state_dict(global_state_dict)
    global_flat = flatten_params(sim_net).detach().float().cpu()
    sim_net.to(device)

    K = max(1, int(num_sims))
    sims: list[torch.Tensor] = []
    for s in range(K):
        sim_seed = int(base_seed) + 10_000 * int(s)
        _set_determinism(sim_seed, deterministic_algos=deterministic_algos)

        # Reset every parameter and buffer to the same global model. The trainer
        # constructs a new optimizer on every call, so optimizer state is not reused.
        sim_net.load_state_dict(global_state_dict)

        trainer(sim_net, sim_loader, num_epochs=int(epochs), lr=float(lr), device=device)

        sim_flat = flatten_params(sim_net).detach().float().cpu()
        sims.append(sim_flat - global_flat)

    return torch.stack(sims, dim=0)


def _fedimp_benign_knn_rhs(benign: torch.Tensor, *, k: int) -> tuple[int, float]:
    """Precompute the benign-only RHS of the FedIMP acceptance inequality.

    This quantity is constant throughout one delta binary search, so computing it
    once avoids repeatedly materializing the expensive ``[B, B, D]`` difference
    tensor at every search iteration.
    """
    if benign.ndim != 2:
        benign = benign.view(benign.size(0), -1)

    B = int(benign.size(0))
    if B <= 1:
        return 1, float("inf")

    k_nn = max(1, min(int(k), B - 1))

    diff = benign.unsqueeze(1) - benign.unsqueeze(0)     # [B, B, D]
    dist = (diff * diff).sum(dim=2)                      # [B, B]
    dist.fill_diagonal_(float("inf"))
    rhs = float(dist.topk(k=k_nn, largest=False).values.sum().item())

    # Release the largest temporary tensors before the binary search begins.
    del diff, dist
    return k_nn, rhs


def _fedimp_accept(
    mal_vec: torch.Tensor,
    benign: torch.Tensor,
    *,
    k: int,
    eps: float = 0.0,
    benign_rhs: float | None = None,
) -> bool:
    """FedIMP paper inequality test (Eq. 11 style, k-NN distances).

    LHS: sum of the k nearest squared distances from mal_vec to the benign simulated updates.
    RHS: sum over benign points of their own k-NN squared distances (self-distance excluded).

    ``benign_rhs`` can be supplied by the binary-search caller because the RHS is
    independent of the candidate malicious vector.
    """
    if benign is None:
        return True
    if benign.ndim != 2:
        benign = benign.view(benign.size(0), -1)

    B = int(benign.size(0))
    if B <= 1:
        return True

    # Clamp k to valid range
    k_nn = max(1, min(int(k), B - 1))
    mal_vec = mal_vec.view(-1)

    # LHS: k-NN squared distances from malicious vector to benign set
    dists_mal = ((benign - mal_vec) ** 2).sum(dim=1)
    lhs = float(dists_mal.topk(k=k_nn, largest=False).values.sum().item())

    # Direct callers may omit the cached value; preserve the standalone behavior.
    if benign_rhs is None:
        _, benign_rhs = _fedimp_benign_knn_rhs(benign, k=k_nn)

    return lhs <= (float(benign_rhs) + float(eps))


def _fedimp_binary_search_delta(
    *,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor,
    benign: torch.Tensor,
    k: int,
    delta_max: float,
    tau: float,
    eps: float = 0.0,
) -> float:
    """Binary search for the largest δ satisfying the paper inequality."""
    low = 0.0
    high = float(max(0.0, delta_max))
    tau = float(max(0.0, tau))

    if high <= 0.0:
        return 0.0

    # Ensure CPU float tensors for speed/stability.
    mu = mu.detach().float().cpu().view(-1)
    sigma = sigma.detach().float().cpu().view(-1)
    mask = mask.detach().float().cpu().view(-1)
    benign = benign.detach().float().cpu()
    if benign.ndim != 2:
        benign = benign.view(benign.size(0), -1)

    # The masked standard deviation and benign-only RHS are constant throughout
    # the search. Compute both once rather than at every candidate delta.
    masked_sigma = mask * sigma
    k_nn, benign_rhs = _fedimp_benign_knn_rhs(benign, k=k)

    # quick guard
    if tau == 0.0:
        tau = 1e-3

    # iterate ~log2(delta_max/tau)
    for _ in range(40):
        if (high - low) <= tau:
            break
        mid = (high + low) / 2.0
        mal = mu - mid * masked_sigma
        if _fedimp_accept(
            mal,
            benign,
            k=k_nn,
            eps=eps,
            benign_rhs=benign_rhs,
        ):
            low = mid
        else:
            high = mid

    return (high + low) / 2.0


def _fedimp_atomic_torch_save(path: Path, tensor: torch.Tensor) -> None:
    """Atomically save a tensor, safe under Ray multi-process concurrency.

    We use a *unique* temp file name to avoid races where another process renames
    or deletes a shared `.tmp` file between write and replace.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    torch.save(tensor, tmp)
    try:
        os.replace(tmp, path)
    except FileNotFoundError:
        # Extremely rare (e.g., concurrent cleanup on shared FS): retry once.
        torch.save(tensor, tmp)
        os.replace(tmp, path)


def _fedimp_atomic_text_save(path: Path, text: str) -> None:
    """Atomically save a small text file, safe under Ray multi-process concurrency."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text)
    try:
        os.replace(tmp, path)
    except FileNotFoundError:
        tmp.write_text(text)
        os.replace(tmp, path)


def _fedimp_wait_for(paths: list[Path], *, tries: int = 200, sleep_s: float = 0.05) -> bool:
    for _ in range(int(max(1, tries))):
        if all(p.exists() for p in paths):
            return True
        time.sleep(float(sleep_s))
    return False


# ===========================================================================

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    run_config = resolved_attack_config(context.run_config)
    # Load dataset
    partition_id = context.node_config["partition-id"]

    # Total number of clients:
    #  - primary: from [tool.flwr.app.config] -> run_config["num_clients"]
    #  - fallback: node_config["num-partitions"] (what Flower already sets)
    num_clients = run_config.get(
        "num_clients",
        context.node_config.get("num-partitions"),
    )


    cfg = msg.content["config"]
    round_no = int(cfg["server-round"])


    base_seed = int(run_config.get("seed", 42))
    local_seed = base_seed + int(partition_id) + 200_000 * int(round_no)
    _set_determinism(
        local_seed,
        deterministic_algos=bool(run_config.get("torch_deterministic", False)),
    )



    # ---------------------------------------------------------
    # Deterministic malicious-client selection PER ROUND
    # (must match server-side deterministic sampling)
    # ---------------------------------------------------------
    legacy_mal = float(run_config.get("Malicious-Client-Percentage", 0.0))
    mal_pct_min = float(run_config.get("Malicious-Client-Percentage-min", legacy_mal))
    mal_pct_max = float(run_config.get("Malicious-Client-Percentage-max", legacy_mal))

    fraction_fit = float(run_config.get("fraction-train", run_config.get("fraction-fit", 1.0)))
    sampled_ids = sampled_partition_ids_for_round(
        int(round_no),
        int(num_clients),
        float(fraction_fit),
        base_seed=int(base_seed),
        seed_offset=0,
    )

    malicious_ids = malicious_ids_for_round(
        int(round_no),
        int(num_clients),
        float(legacy_mal),
        seed_offset=0,
        selected_ids=sampled_ids,
        mal_pct_min=float(mal_pct_min),
        mal_pct_max=float(mal_pct_max),
        base_seed=int(base_seed),
    )

    partition_id_str = str(context.node_config["partition-id"])
    is_malicious = (partition_id_str in malicious_ids)
    context.node_config["is_malicious"] = bool(is_malicious)

    # Read attack type once (used for both data- and model-poisoning paths)
    attack_type = run_config.get("attack_type", "None")
    # ---------------------------------------------------------

    train_loader, _ = load_data(
        partition_id=partition_id,
        num_partitions=int(num_clients),
        run_config=run_config,
        server_round=round_no,
    )

    # Load model
    net = build_model(num_partitions=int(num_clients), run_config=run_config)
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Perform training
    lr = float(run_config.get("learning_rate"))
    local_epochs = int(run_config.get("local_epochs"))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # ---------------------------------------------------------------------------------
    # ProbeGuard local-client method selection:
    #   - probeguard_client_method="fedavg"   : plain local training
    #   - probeguard_client_method="fedprox"  : FedProx objective with (mu/2)||w-w_global||^2
    #   - probeguard_client_method="scaffold" : SCAFFOLD variance-reduction (server sends c, client keeps c_i)
    #
    # NOTE: This applies to both "probeguard" and "probeguard-ema"; their local training is identical.
    # ---------------------------------------------------------------------------------
    aggregation_method = str(run_config.get("aggregation_method", "fedavg")).lower().strip()
    probeguard_client_method = str(run_config.get("probeguard_client_method", "fedprox")).lower().strip()

    use_scaffold = False
    fedprox_mu = 0.0
    global_params_ref = None
    scaffold_c_global = None
    scaffold_ci = None
    scaffold_ci_before = None

    if aggregation_method in {"probeguard", "probeguard-ema"}:
        if probeguard_client_method in ("fedprox", "prox"):
            fedprox_mu = float(run_config.get("fedprox_mu", 0.01))
            if fedprox_mu > 0.0:
                # snapshot of the current (global) parameters before local training
                global_params_ref = [p.detach().clone() for p in net.parameters()]

        elif probeguard_client_method in ("scaffold", "scafold", "scaf"):
            use_scaffold = True

            # Server provides the global control variate `c` as an ArrayRecord
            ar_c = msg.content.get("scaffold_c", None)
            if ar_c is not None:
                try:
                    scaffold_c_global = ar_c.to_torch_state_dict()
                except Exception:
                    scaffold_c_global = None

            # Client keeps its local control variate `c_i` across rounds
            global _SCAFFOLD_CI
            if _SCAFFOLD_CI is None:
                _SCAFFOLD_CI = {n: torch.zeros_like(p.detach().cpu()) for n, p in net.named_parameters()}
            scaffold_ci_before = {k: v.detach().clone() for k, v in _SCAFFOLD_CI.items()}
            scaffold_ci = _SCAFFOLD_CI

        else:
            # "fedavg" (default): no prox term, no control variates
            pass

    # ---------------------- Data-poisoning attacks ----------------------
    # These attacks modify local TRAIN data (or labels) before local optimization.
    label_flip_applied = False
    label_flip_n = 0
    label_flip_rate = 0.0
    data_poison_applied = False
    data_poison_n = 0
    data_poison_rate = 0.0
    inverted_loss_idx = None
    boundary_label_flip_idx = None

    data_seed = int(base_seed) + int(partition_id) + 200_000 * int(round_no) + 1_234_567
    g_data = torch.Generator(device="cpu")
    g_data.manual_seed(int(data_seed))

    if bool(is_malicious):
        if _is_label_flip_attack(attack_type):
            try:
                label_flip_rate = float(run_config.get("label_flip_rate", 1.0))
            except Exception:
                label_flip_rate = 1.0

            label_flip_mode = str(run_config.get("label_flip_mode", "random")).lower().strip()
            try:
                label_flip_target = int(run_config.get("label_flip_target", 0))
            except Exception:
                label_flip_target = 0

            label_flip_mapping_raw = run_config.get("label_flip_mapping", "")
            label_flip_mapping = _parse_label_flip_mapping(label_flip_mapping_raw)

            num_classes = _infer_num_classes_from_model(net)
            if num_classes is None or int(num_classes) <= 1:
                try:
                    num_classes = int(run_config.get("num_classes", 0))
                except Exception:
                    num_classes = 0
            if num_classes is None or int(num_classes) <= 1:
                try:
                    y_tmp = train_loader.dataset.tensors[1]
                    num_classes = int(torch.max(y_tmp).item()) + 1 if int(y_tmp.numel()) > 0 else 0
                except Exception:
                    num_classes = 0

            label_flip_n = _apply_label_flip_to_trainloader(
                train_loader,
                num_classes=int(num_classes),
                flip_rate=float(label_flip_rate),
                mode=label_flip_mode,
                target=int(label_flip_target),
                mapping=label_flip_mapping,
                generator=g_data,
            )
            label_flip_applied = bool(label_flip_n > 0)
            data_poison_applied = label_flip_applied
            data_poison_n = int(label_flip_n)
            data_poison_rate = float(label_flip_rate)

        elif _is_clean_label_attack(attack_type):
            if _attack_active_this_round(run_config, int(round_no), prefix="clean_label", default_start_frac=0.8):
                data_poison_rate = float(run_config.get("clean_label_rate", 0.01))
                data_poison_n = _apply_clean_label_poison_to_trainloader(
                    train_loader,
                    net,
                    target_class=int(run_config.get("clean_label_target_class", 0)),
                    poison_rate=float(data_poison_rate),
                    eps=float(run_config.get("clean_label_eps", 0.15)),
                    steps=int(run_config.get("clean_label_steps", 10)),
                    step_size=float(run_config.get("clean_label_step_size", 0.02)),
                    alpha=float(run_config.get("clean_label_alpha", 0.8)),
                    source_class=int(run_config.get("clean_label_source_class", -1)),
                    tabular_feature_frac=float(run_config.get("clean_label_tabular_feature_frac", 0.10)),
                    generator=g_data,
                    device=device,
                    max_poison_samples=int(run_config.get("clean_label_max_poison_samples", 64)),
                    target_mode=str(run_config.get("clean_label_target_mode", "fixed")),
                    target_classes=run_config.get("clean_label_target_classes", ""),
                    local_smallest_k=int(run_config.get("clean_label_local_smallest_k", 3)),
                    min_target_samples=int(run_config.get("clean_label_min_target_samples", 1)),
                    source_mode=str(run_config.get("clean_label_source_mode", "mode_poison")),
                    min_source_samples=int(run_config.get("clean_label_min_source_samples", 1)),
                    confusable_max_samples_per_class=int(run_config.get("clean_label_confusable_max_samples_per_class", 128)),
                    confusable_topk_sources=int(run_config.get("clean_label_confusable_topk_sources", 3)),
                    sample_selection_mode=str(run_config.get("clean_label_sample_selection_mode", "boundary_gradient")),
                    target_selection_metric=str(run_config.get("clean_label_target_selection_metric", "hybrid")),
                    source_selection_metric=str(run_config.get("clean_label_source_selection_metric", "hybrid")),
                    target_reference_count=int(run_config.get("clean_label_target_reference_count", 3)),
                    reference_assignment_mode=str(run_config.get("clean_label_reference_assignment_mode", "balanced_similarity")),
                    source_candidate_multiplier=float(run_config.get("clean_label_source_candidate_multiplier", 2.0)),
                    gradient_max_candidates_per_source=int(run_config.get("clean_label_gradient_max_candidates_per_source", 24)),
                )
                data_poison_applied = bool(data_poison_n > 0)

        elif _is_feature_poison_attack(attack_type):
            if _attack_active_this_round(run_config, int(round_no), prefix="feature_poison", default_start_frac=0.0):
                data_poison_rate = float(run_config.get("feature_poison_rate", 0.10))
                feature_metadata = get_feature_poison_metadata(
                    num_partitions=int(num_clients),
                    run_config=run_config,
                    lower_quantile=float(run_config.get("feature_poison_global_lower_quantile", 0.05)),
                    upper_quantile=float(run_config.get("feature_poison_global_upper_quantile", 0.95)),
                    clip=float(run_config.get("feature_poison_global_clip", 8.0)),
                    onehot_groups=run_config.get("feature_poison_onehot_groups", "auto"),
                    onehot_min_group_size=int(run_config.get("feature_poison_onehot_min_group_size", 3)),
                )
                data_poison_n = _apply_feature_poison_to_trainloader(
                    train_loader,
                    net,
                    poison_rate=float(data_poison_rate),
                    topk_frac=float(run_config.get("feature_poison_topk_frac", 0.10)),
                    feature_prob=float(run_config.get("feature_poison_feature_prob", 1.0)),
                    rf_estimators=int(run_config.get("feature_poison_rf_estimators", 100)),
                    rf_max_samples=int(run_config.get("feature_poison_rf_max_samples", 512)),
                    generator=g_data,
                    seed=int(data_seed),
                    mode=(
                        "opposite_extreme" if _norm_attack_name(attack_type) == "featurepoison2"
                        else _feature_poison1_mode(run_config)
                    ),
                    feature_metadata=feature_metadata,
                    gradient_strength=float(run_config.get("feature_poison_gradient_strength", 1.0)),
                    device=device,
                )
                data_poison_applied = bool(data_poison_n > 0)

        elif _is_inverted_loss_attack(attack_type):
            if _attack_active_this_round(run_config, int(round_no), prefix="inverted_loss", default_start_frac=0.0):
                data_poison_rate = float(run_config.get("inverted_loss_rate", 0.10))
                inverted_loss_idx = _prepare_inverted_loss_indices(
                    train_loader,
                    poison_rate=float(data_poison_rate),
                    generator=g_data,
                    max_poison_samples=int(run_config.get("inverted_loss_max_poison_samples", 0)),
                )
                data_poison_n = int(inverted_loss_idx.numel()) if inverted_loss_idx is not None else 0
                data_poison_applied = bool(data_poison_n > 0)

        elif _is_boundary_aware_label_flip_attack(attack_type):
            if _attack_active_this_round(run_config, int(round_no), prefix="boundary_label_flip", default_start_frac=0.0):
                data_poison_rate = float(run_config.get("boundary_label_flip_rate", 0.10))
                boundary_label_flip_idx = _prepare_inverted_loss_indices(
                    train_loader,
                    poison_rate=float(data_poison_rate),
                    generator=g_data,
                    max_poison_samples=int(run_config.get("boundary_label_flip_max_poison_samples", 0)),
                )
                data_poison_n = int(boundary_label_flip_idx.numel()) if boundary_label_flip_idx is not None else 0
                data_poison_applied = bool(data_poison_n > 0)

    # -----------------------------------------------------------------------
    # FedIMP ultimately replaces the complete local model with
    # ``global_model + forged_update``. Therefore, ordinary local optimization
    # on a malicious FedIMP client is discarded and can be skipped safely.
    #
    # SCAFFOLD is excluded from this optimization because its local training also
    # updates the persistent client control variate c_i, which must still be sent
    # back to the server even when the model parameters are later forged.
    skip_local_training = (
        bool(is_malicious)
        and str(attack_type).strip().lower() == "fedimp"
        and not bool(use_scaffold)
    )

    # Train with the selected ProbeGuard client method
    if skip_local_training:
        ci_new = None
        _steps = 0
    elif bool(is_malicious) and _is_inverted_loss_attack(attack_type) and inverted_loss_idx is not None and int(inverted_loss_idx.numel()) > 0:
        ci_new, _steps = _train_with_inverted_loss_relabeling(
            net,
            train_loader,
            inverted_loss_idx,
            label_mode="least_likely",
            num_epochs=local_epochs,
            lr=lr,
            device=device,
            seed=int(local_seed) + 24680,
            prox_mu=fedprox_mu,
            global_params=global_params_ref,
            scaffold=use_scaffold,
            scaffold_c_global=scaffold_c_global,
            scaffold_ci=scaffold_ci,
        )
    elif (
        bool(is_malicious)
        and _is_boundary_aware_label_flip_attack(attack_type)
        and boundary_label_flip_idx is not None
        and int(boundary_label_flip_idx.numel()) > 0
    ):
        ci_new, _steps = _train_with_inverted_loss_relabeling(
            net,
            train_loader,
            boundary_label_flip_idx,
            label_mode="highest_non_true",
            num_epochs=local_epochs,
            lr=lr,
            device=device,
            seed=int(local_seed) + 24680,
            prox_mu=fedprox_mu,
            global_params=global_params_ref,
            scaffold=use_scaffold,
            scaffold_c_global=scaffold_c_global,
            scaffold_ci=scaffold_ci,
        )
    else:
        ci_new, _steps = trainer(
            net,
            train_loader,
            num_epochs=local_epochs,
            lr=lr,
            device=device,
            prox_mu=fedprox_mu,
            global_params=global_params_ref,
            scaffold=use_scaffold,
            scaffold_c_global=scaffold_c_global,
            scaffold_ci=scaffold_ci,
        )

    # If SCAFFOLD is enabled, persist updated c_i and attach delta c_i to the reply
    scaffold_delta_c = None
    if use_scaffold and (ci_new is not None) and (scaffold_ci_before is not None):
        try:
            _SCAFFOLD_CI = {k: v.detach().clone() for k, v in ci_new.items()}
            scaffold_delta_c = {
                k: (ci_new[k] - scaffold_ci_before.get(k, torch.zeros_like(ci_new[k])))
                for k in ci_new.keys()
            }
        except Exception:
            scaffold_delta_c = None



    # Store before-attack parameters (local model)
    before_attack_params = [p.detach().clone() for p in net.parameters()]
    global_net = build_model(num_partitions=int(num_clients), run_config=run_config)
    global_net.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    global_params = [p.detach().clone() for p in global_net.parameters()]

    # ---------------------- Apply Attacks ----------------------

    # `attack_type` / `is_malicious` were computed deterministically before training
    attack_type = str(attack_type)
    is_malicious = bool(is_malicious)


    attack_seed = base_seed + int(partition_id) + 200_000 * int(round_no) + 9_999_983
    device = next(net.parameters()).device
    gen_device = "cuda" if device.type == "cuda" else "cpu"
    g_attack = torch.Generator(device=gen_device)
    g_attack.manual_seed(int(attack_seed))


    if is_malicious:

        if attack_type == "RandScaling":
            random_scaling_attack(net, generator=g_attack)

        elif attack_type in {"SignFlip", "SignFlipping", "Sign-flipping"}:
            gamma = float(run_config.get("signflip_gamma", 5.0))
            sign_flipping_attack(net, global_params, gamma=gamma)

        elif attack_type == "Trim":
            # Craft an update that is hard for trimmed-mean style defenses to filter
            # (works on flattened parameter vectors to keep ordering consistent).
            before_flat = flatten_params(net).detach().clone()

            global_sd = msg.content["arrays"].to_torch_state_dict()
            global_net = build_model(num_partitions=int(num_clients), run_config=run_config)
            global_net.load_state_dict(global_sd)
            global_net.to(device)
            global_flat = flatten_params(global_net).detach().clone()

            mu, sigma, s_hat = compute_trim_stats(before_flat, global_flat)
            crafted = craft_trim_vector(mu, sigma, s_hat, generator=g_attack)

            if crafted.numel() != before_flat.numel():
                raise ValueError(
                    f"Trim crafted vector size mismatch crafted={crafted.numel()} expected={before_flat.numel()}"
                )

            unflatten_params(net, crafted)

        elif attack_type in {"ALIE", "IPM"}:
            # These are omniscient / same-round attacks implemented on the server side.
            pass

        elif _is_label_flip_attack(attack_type):
            # LabelFlip is a data-poisoning attack applied BEFORE local training.
            # No post-training model forging is required here.
            pass

        elif attack_type == "FedIMP":
            # FedIMP: parameter-importance model poisoning (paper-faithful shape):
            #   Δw_mal = μ - δ * M * σ ,   w_mal = w_global + Δw_mal
            global_sd = msg.content["arrays"].to_torch_state_dict()

            # Paper: all compromised clients upload *identical* malicious updates each round.
            # When enabled, clients share per-round artifacts via the filesystem.
            fedimp_share = bool(run_config.get("fedimp_share_across_malicious", True))
            fedimp_shared_root = Path(run_config.get("fedimp_shared_dir", "./.fedimp_shared")).resolve()
            run_tag = str(run_config.get("fedimp_run_tag", f"seed{int(base_seed)}"))
            fedimp_shared_dir = (fedimp_shared_root / run_tag).resolve()
            if fedimp_share:
                fedimp_shared_dir.mkdir(parents=True, exist_ok=True)

            fedimp_topk_ratio = float(run_config.get("fedimp_topk_ratio", 0.1))
            fedimp_fisher_batches = int(run_config.get("fedimp_fisher_batches", 2))
            fedimp_sim_batches = int(run_config.get("fedimp_sim_batches", 2))
            fedimp_sim_epochs = int(run_config.get("fedimp_sim_epochs", 1))
            fedimp_num_sims = int(run_config.get("fedimp_num_sims", 6))

            fedimp_delta_max = float(run_config.get("fedimp_delta_max", 20.0))
            fedimp_tau = float(run_config.get("fedimp_tau", 1e-3))
            fedimp_eps = float(run_config.get("fedimp_eps", 0.0))
            fedimp_prepoison_round = int(run_config.get("fedimp_prepoison_round", 1))
            fedimp_prepoison_batches = int(
                run_config.get("fedimp_prepoison_batches", max(1, fedimp_fisher_batches))
            )

            ## k = ceil((n_t - m_t)/2) per paper; approximate n_t using fraction-fit  !!!!!!
            # fraction_fit = float(run_config.get("fraction-fit", 1.0))
            # n_t = max(1, int(math.ceil(float(fraction_fit) * int(num_clients))))
            # m_t = int(math.floor(float(mal_pct) * float(n_t)))
            # b_t = max(1, n_t - m_t)
            # k_support = int(math.ceil(float(b_t) / 2.0))

            # malicious clients actually participating THIS round
            sampled_set = {str(x) for x in sampled_ids}
            mal_set_all = {str(x) for x in malicious_ids}
            mals_in_round = sorted(
                sampled_set & mal_set_all,
                key=lambda s: int(s) if s.isdigit() else s,
            )

            n_t = max(1, len(sampled_ids))
            m_t = min(n_t, len(mals_in_round))          # <-- IMPORTANT (paper uses m_t of this round)
            b_t = max(1, n_t - m_t)
            k_support = int(math.ceil(b_t / 2.0))       # k = ceil((n_t - m_t)/2)

            cid = str(context.node_config["partition-id"])

            # Choose a per-round leader among malicious clients participating THIS round.
            #
            # IMPORTANT: avoid relying on a shared-file "leader election" for correctness.
            # On some cluster filesystems, transient read errors can cause multiple clients
            # to believe they are the leader, which then races with per-round cleanup and
            # can delete each other's temp files. We therefore pick the leader *purely
            # deterministically* from `mals_in_round`, and only write the leader file as
            # a best-effort debug aid.
            leader_pref = str(run_config.get("fedimp_leader_cid", "min"))
            leader_pref_l = leader_pref.lower()

            # Deterministic leader among *malicious participants this round*
            if leader_pref_l in {"min", "auto", ""}:
                try:
                    leader_cid = str(min(int(x) for x in mals_in_round)) if mals_in_round else cid
                except Exception:
                    leader_cid = str(mals_in_round[0]) if mals_in_round else cid
            else:
                # explicit leader id (string)
                leader_cid = str(leader_pref)

            is_leader = (cid == leader_cid)

            # Best-effort: write leader file for debugging / late joiners (not used for election)
            if fedimp_share and leader_pref_l in {"min", "auto", ""}:
                leader_file = fedimp_shared_dir / f"fedimp_round{round_no}_leader.txt"
                try:
                    _fedimp_atomic_text_save(leader_file, str(leader_cid))
                except Exception:
                    pass

            mal_update_path = fedimp_shared_dir / f"fedimp_round{round_no}_mal_update.pt"
            l0_path = fedimp_shared_dir / "fedimp_l0.txt"
            use_large_flag = fedimp_shared_dir / "fedimp_use_large_delta.flag"

            # Use a CPU generator for deterministic aux sampling/shuffling
            g_cpu = torch.Generator(device="cpu")
            g_cpu.manual_seed(int(attack_seed) + 123_457)

            # --- FedIMP safety: clean stale shared artifacts for this round (avoids shape drift across reruns) ---
            if fedimp_share:
                cleanup_flag = fedimp_shared_dir / f"fedimp_round{round_no}_CLEANED.done"
                if is_leader:
                    tag = f"fedimp_round{round_no}_"
                    # Remove any stale per-round artifacts (e.g., from a previous crashed run)
                    for f in fedimp_shared_dir.glob(tag + "*"):
                        # Keep the leader election file; late-arriving clients may still need it.
                        if f.name.endswith("_leader.txt"):
                            continue
                        try:
                            f.unlink()
                        except Exception:
                            pass
                    # Also clear run-level flags at the beginning of a new run
                    if int(round_no) == 1:
                        for f in [l0_path, use_large_flag]:
                            try:
                                if f.exists():
                                    f.unlink()
                            except Exception:
                                pass
                    _fedimp_atomic_text_save(cleanup_flag, "ok")
                else:
                    ok = _fedimp_wait_for([cleanup_flag], tries=1200, sleep_s=0.05)
                    if not ok:
                        raise RuntimeError("FedIMP share-mode: timed out waiting for leader cleanup barrier")
            # -----------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------
            # Share-mode contributions (all malicious clients):
            #   - local empirical Fisher diag (importance proxy) and its weight |D_aux|
            #   - optional pre-poison loss on D_aux for l0/l1 bookkeeping
            # The leader aggregates these (weighted) per the paper.
            # --------------------------------------------------------------------
            if fedimp_share:
                local_fisher_path = fedimp_shared_dir / f"fedimp_round{round_no}_fisher_{cid}.pt"
                local_auxn_path = fedimp_shared_dir / f"fedimp_round{round_no}_auxn_{cid}.txt"
                local_loss_path = fedimp_shared_dir / f"fedimp_round{round_no}_loss_{cid}.txt"

                aux_loader_local = _fedimp_make_aux_loader(
                    train_loader, max_batches=max(1, fedimp_fisher_batches), generator=g_cpu
                )
                aux_n_local = int(len(getattr(aux_loader_local, "dataset", [])))
                _fedimp_atomic_text_save(local_auxn_path, str(aux_n_local))

                fisher_net_local = build_model(num_partitions=int(num_clients), run_config=run_config)
                fisher_net_local.load_state_dict(global_sd)
                fisher_net_local.to(device)

                importance_local = _fedimp_empirical_fisher_diag(
                    fisher_net_local,
                    aux_loader_local,
                    device=device,
                    max_batches=max(1, fedimp_fisher_batches),
                )
                _fedimp_atomic_torch_save(local_fisher_path, importance_local)

                if int(round_no) in {int(fedimp_prepoison_round), int(fedimp_prepoison_round) + 1}:
                    loss_local = _fedimp_eval_loss(
                        fisher_net_local,
                        aux_loader_local,
                        device=device,
                        max_batches=max(1, fedimp_prepoison_batches),
                    )
                    _fedimp_atomic_text_save(local_loss_path, str(loss_local))

            # --------------------------------------------------------------------
            # Non-leader in share-mode: wait for leader's crafted malicious update
            # --------------------------------------------------------------------
            mal_update = None
            if fedimp_share and (not is_leader):
                ok = _fedimp_wait_for([mal_update_path], tries=1200, sleep_s=0.05)  # ~60s
                if not ok:
                    raise RuntimeError(
                        "FedIMP share-mode: timed out waiting for leader mal_update"
                    )

                # NFS/Ray can occasionally show a TOCTOU here (exists->open). Retry a few times.
                mal_update = None
                for _ in range(40):
                    try:
                        mal_update = (
                            torch.load(mal_update_path, map_location="cpu").view(-1).float()
                        )
                        break
                    except FileNotFoundError:
                        time.sleep(0.05)

                if mal_update is None:
                    raise RuntimeError(
                        "FedIMP share-mode: leader mal_update file disappeared while loading"
                    )

            if mal_update is None:
                if fedimp_share and (not is_leader):
                    raise RuntimeError(
                        "FedIMP share-mode: non-leader must not compute mal_update"
                    )

                # ----------------------------------------------------------------
                # Compute FedIMP (leader in share-mode, or local computation when share disabled)
                # ----------------------------------------------------------------

                if fedimp_share:
                    # Leader aggregates importance across all malicious clients (weighted by |D_aux|)
                    def _stable_int_key(x: str):
                        try:
                            return (0, int(x))
                        except Exception:
                            return (1, str(x))

                    expected_mals = sorted(list(mals_in_round), key=_stable_int_key)
                    fisher_paths = [fedimp_shared_dir / f"fedimp_round{round_no}_fisher_{mid}.pt" for mid in expected_mals]
                    auxn_paths = [fedimp_shared_dir / f"fedimp_round{round_no}_auxn_{mid}.txt" for mid in expected_mals]

                    wait_paths = fisher_paths + auxn_paths
                    if int(round_no) in {int(fedimp_prepoison_round), int(fedimp_prepoison_round) + 1}:
                        loss_paths = [fedimp_shared_dir / f"fedimp_round{round_no}_loss_{mid}.txt" for mid in expected_mals]
                        wait_paths += loss_paths

                    _fedimp_wait_for(wait_paths, tries=240, sleep_s=0.05)

                    imps = []
                    ws = []
                    for mid in expected_mals:
                        try:
                            imp = torch.load(
                                fedimp_shared_dir / f"fedimp_round{round_no}_fisher_{mid}.pt",
                                map_location="cpu",
                            ).view(-1).float()
                        except Exception:
                            continue
                        try:
                            w = float((fedimp_shared_dir / f"fedimp_round{round_no}_auxn_{mid}.txt").read_text().strip())
                        except Exception:
                            w = 1.0
                        w = float(max(0.0, w))
                        imps.append(imp)
                        ws.append(w)

                    if len(imps) == 0:
                        raise ValueError("FedIMP: share-mode aggregation failed (no fisher vectors found)")

                    wsum = float(sum(ws)) if float(sum(ws)) > 0.0 else float(len(ws))
                    # Guard against stale/shape-mismatched vectors (can happen after reruns if shared_dir isn't cleaned)
                    if len(imps) > 1:
                        from collections import Counter
                        lens = [int(t.numel()) for t in imps]
                        mode_len = Counter(lens).most_common(1)[0][0]
                        if any(l != mode_len for l in lens):
                            keep = [(w, t) for w, t in zip(ws, imps) if int(t.numel()) == mode_len]
                            dropped = len(imps) - len(keep)
                            print(
                                f"[FedIMP] warning: dropping {dropped} fisher vectors due to length mismatch "
                                f"(mode_len={mode_len}, lens={lens}). If this persists, clear fedimp_shared_dir."
                            )
                            if not keep:
                                raise RuntimeError(
                                    "FedIMP share-mode: all fisher vectors were dropped due to length mismatch; "
                                    "clear fedimp_shared_dir and rerun."
                                )
                            ws, imps = zip(*keep)
                            ws, imps = list(ws), list(imps)

                    importance = sum((w / wsum) * imp for w, imp in zip(ws, imps))

                    mask = _fedimp_topk_mask(importance, fedimp_topk_ratio)

                    # Simulate benign updates (leader only)
                    sim_loader = _fedimp_make_aux_loader(
                        train_loader, max_batches=max(1, fedimp_sim_batches), generator=g_cpu
                    )
                    benign_updates = _fedimp_simulate_benign_updates(
                        global_state_dict=global_sd,
                        num_clients=int(num_clients),
                        run_config=run_config,
                        sim_loader=sim_loader,
                        device=device,
                        lr=float(lr),
                        epochs=int(fedimp_sim_epochs),
                        num_sims=int(fedimp_num_sims),
                        base_seed=int(attack_seed) + 777,
                        deterministic_algos=bool(run_config.get("torch_deterministic", False)),
                    )
                    mu = benign_updates.mean(dim=0)
                    sigma = benign_updates.std(dim=0, unbiased=False) + 1e-12

                    # --- Pre-poison bookkeeping (paper) ---
                    def _weighted_round_loss() -> float:
                        num = 0.0
                        den = 0.0
                        for mid in expected_mals:
                            try:
                                li = float(
                                    (fedimp_shared_dir / f"fedimp_round{round_no}_loss_{mid}.txt").read_text().strip()
                                )
                            except Exception:
                                continue
                            try:
                                wi = float(
                                    (fedimp_shared_dir / f"fedimp_round{round_no}_auxn_{mid}.txt").read_text().strip()
                                )
                            except Exception:
                                wi = 1.0
                            wi = float(max(0.0, wi))
                            num += wi * li
                            den += wi
                        if den <= 0.0:
                            return float(num)
                        return float(num / den)

                    if int(round_no) == int(fedimp_prepoison_round):
                        l0 = _weighted_round_loss()
                        _fedimp_atomic_text_save(l0_path, str(l0))
                        delta = fedimp_delta_max

                    elif int(round_no) == int(fedimp_prepoison_round) + 1:
                        l1 = _weighted_round_loss()
                        l0 = None
                        try:
                            if l0_path.exists():
                                l0 = float(l0_path.read_text().strip())
                        except Exception:
                            l0 = None

                        if (l0 is not None) and ((l1 - l0) > float(fedimp_eps)):
                            delta = fedimp_delta_max
                            _fedimp_atomic_text_save(use_large_flag, "1")
                        else:
                            delta = _fedimp_binary_search_delta(
                                mu=mu,
                                sigma=sigma,
                                mask=mask,
                                benign=benign_updates,
                                k=k_support,
                                delta_max=fedimp_delta_max,
                                tau=fedimp_tau,
                                eps=fedimp_eps,
                            )

                    else:
                        if use_large_flag.exists():
                            delta = fedimp_delta_max
                        else:
                            delta = _fedimp_binary_search_delta(
                                mu=mu,
                                sigma=sigma,
                                mask=mask,
                                benign=benign_updates,
                                k=k_support,
                                delta_max=fedimp_delta_max,
                                tau=fedimp_tau,
                                eps=fedimp_eps,
                            )

                    mal_update = (mu - float(delta) * (mask * sigma)).view(-1).float().cpu()
                    _fedimp_atomic_torch_save(mal_update_path, mal_update)

                else:
                    # Local (non-shared) FedIMP (legacy behavior; may differ across malicious clients)
                    aux_loader = _fedimp_make_aux_loader(
                        train_loader, max_batches=max(1, fedimp_fisher_batches), generator=g_cpu
                    )
                    sim_loader = _fedimp_make_aux_loader(
                        train_loader, max_batches=max(1, fedimp_sim_batches), generator=g_cpu
                    )

                    fisher_net = build_model(num_partitions=int(num_clients), run_config=run_config)
                    fisher_net.load_state_dict(global_sd)
                    fisher_net.to(device)

                    importance = _fedimp_empirical_fisher_diag(
                        fisher_net, aux_loader, device=device, max_batches=max(1, fedimp_fisher_batches)
                    )
                    mask = _fedimp_topk_mask(importance, fedimp_topk_ratio)

                    benign_updates = _fedimp_simulate_benign_updates(
                        global_state_dict=global_sd,
                        num_clients=int(num_clients),
                        run_config=run_config,
                        sim_loader=sim_loader,
                        device=device,
                        lr=float(lr),
                        epochs=int(fedimp_sim_epochs),
                        num_sims=int(fedimp_num_sims),
                        base_seed=int(attack_seed) + 777,
                        deterministic_algos=bool(run_config.get("torch_deterministic", False)),
                    )
                    mu = benign_updates.mean(dim=0)
                    sigma = benign_updates.std(dim=0, unbiased=False) + 1e-12

                    # --- Pre-poison bookkeeping (single-client) ---
                    if int(round_no) == int(fedimp_prepoison_round):
                        l0 = _fedimp_eval_loss(
                            fisher_net,
                            aux_loader,
                            device=device,
                            max_batches=max(1, fedimp_prepoison_batches),
                        )
                        _fedimp_atomic_text_save(l0_path, str(l0))
                        delta = fedimp_delta_max

                    elif int(round_no) == int(fedimp_prepoison_round) + 1:
                        l1 = _fedimp_eval_loss(
                            fisher_net,
                            aux_loader,
                            device=device,
                            max_batches=max(1, fedimp_prepoison_batches),
                        )
                        l0 = None
                        try:
                            if l0_path.exists():
                                l0 = float(l0_path.read_text().strip())
                        except Exception:
                            l0 = None

                        if (l0 is not None) and ((l1 - l0) > float(fedimp_eps)):
                            delta = fedimp_delta_max
                            _fedimp_atomic_text_save(use_large_flag, "1")
                        else:
                            delta = _fedimp_binary_search_delta(
                                mu=mu,
                                sigma=sigma,
                                mask=mask,
                                benign=benign_updates,
                                k=k_support,
                                delta_max=fedimp_delta_max,
                                tau=fedimp_tau,
                                eps=fedimp_eps,
                            )

                    else:
                        if use_large_flag.exists():
                            delta = fedimp_delta_max
                        else:
                            delta = _fedimp_binary_search_delta(
                                mu=mu,
                                sigma=sigma,
                                mask=mask,
                                benign=benign_updates,
                                k=k_support,
                                delta_max=fedimp_delta_max,
                                tau=fedimp_tau,
                                eps=fedimp_eps,
                            )

                    mal_update = (mu - float(delta) * (mask * sigma)).view(-1).float().cpu()

            # Forge weights: w_mal = w_global + Δw_mal  (NOT local + Δw_mal)
            global_net = build_model(num_partitions=int(num_clients), run_config=run_config)
            global_net.load_state_dict(global_sd)
            global_flat = flatten_params(global_net).detach().float().cpu()

            if mal_update.numel() != global_flat.numel():
                if fedimp_share:
                    # Likely stale artifact from a previous run. Remove and try once more.
                    try:
                        mal_update_path.unlink()
                    except Exception:
                        pass

                    ok = _fedimp_wait_for([mal_update_path], tries=600, sleep_s=0.05)
                    if ok:
                        try:
                            mal_update = (
                                torch.load(mal_update_path, map_location="cpu").view(-1).float()
                            )
                        except Exception:
                            pass

                if mal_update.numel() != global_flat.numel():
                    # Last-resort: don't crash the entire run. Disable the attack this round for this client.
                    print(
                        f"[FedIMP] WARN: mal_update/global length mismatch "
                        f"(mal_update={mal_update.numel()}, global_flat={global_flat.numel()}); using zeros."
                    )
                    mal_update = torch.zeros_like(global_flat)

            forged_flat = (global_flat + mal_update).to(device)
            unflatten_params(net, forged_flat)

        elif attack_type == "PoisonedFL":
            # PoisonedFL is forged on the server side (fake clients), to keep the attack consistent
            # under client sampling (malicious clients may not participate every round).
            pass


    # -----------------------------------------------------------

    # Construct and return reply Message
    sd_cpu = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    model_record = ArrayRecord(sd_cpu)

    scaffold_delta_record = None
    if 'scaffold_delta_c' in locals() and scaffold_delta_c is not None:
        try:
            scaffold_delta_record = ArrayRecord({k: v.detach().cpu() for k, v in scaffold_delta_c.items()})
        except Exception:
            scaffold_delta_record = None

    metrics = {
        # "num-examples": len(train_loader),
        "num-examples": len(train_loader.dataset),
        "client_id": int(context.node_config["partition-id"]),
        "is_malicious": int(bool(is_malicious)),
        "label_flip_n": int(label_flip_n) if bool(label_flip_applied) else 0,
        "label_flip_rate": float(label_flip_rate) if bool(label_flip_applied) else 0.0,
        "data_poison_n": int(data_poison_n) if bool(data_poison_applied) else 0,
        "data_poison_rate": float(data_poison_rate) if bool(data_poison_applied) else 0.0,
    }
    metric_record = MetricRecord(metrics)
    rd = {"arrays": model_record, "metrics": metric_record}
    if scaffold_delta_record is not None:
        rd["scaffold_delta_c"] = scaffold_delta_record
    content = RecordDict(rd)
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    # Load dataset
    partition_id = context.node_config["partition-id"]

    num_clients = context.run_config.get(
        "num_clients",
        context.node_config.get("num-partitions"),
    )

    _, test_loader = load_data(
        partition_id=partition_id,
        num_partitions=int(num_clients),
        run_config=context.run_config,
    )

    # Load model
    net = build_model(num_partitions=int(num_clients), run_config=context.run_config)
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Perform evaluation
    loss, accuracy = evaluator(net, test_loader, device=device)

    # Construct and return reply Message
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        # "num-examples": len(test_loader),
        "num-examples": len(test_loader.dataset),
        "client_id": int(context.node_config["partition-id"]),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
