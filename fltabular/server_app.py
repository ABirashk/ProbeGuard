"""Flower ServerApp: robust aggregation and ProbeGuard strategies."""

import atexit
import os
import json
import random
import math
import time
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from fltabular.task import build_model, load_data, load_root_data, trainer, malicious_ids_for_round, sampled_partition_ids_for_round, load_sdea_public_data, resolved_attack_config


from collections import OrderedDict, deque

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import norm
from sklearn.ensemble import IsolationForest

# Create ServerApp
app = ServerApp()

# ======================================================================

from contextlib import contextmanager

@contextmanager
def _temp_round_seed(seed: int):
    import random
    import numpy as np
    import torch

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

# ======================================================================

def _reply_to_state_dict(reply: Message) -> "OrderedDict[str, torch.Tensor]":
    arrays = reply.content["arrays"]
    return arrays.to_torch_state_dict()


def _reply_to_scaffold_delta_c(reply: Message) -> "OrderedDict[str, torch.Tensor] | None":
    """Return SCAFFOLD delta-c (c_i update) if present in client reply."""
    ar = reply.content.get("scaffold_delta_c", None)
    if ar is None:
        return None
    try:
        return ar.to_torch_state_dict()
    except Exception:
        return None

def _filter_replies_with_content(
    replies: list[Message],
    *,
    server_round: int | None = None,
    where: str = "",
) -> list[Message]:
    """Return replies only when every selected client produced valid content.

    For deterministic runs, silently continuing with missing/empty replies is unsafe,
    because the effective participant set can change across repeated runs.
    """
    kept: list[Message] = []
    dropped: list[str] = []
    for r in replies:
        try:
            if r is None:
                dropped.append("none")
                continue
            if hasattr(r, "has_content") and (not r.has_content()):
                meta = getattr(r, "metadata", None)
                src = getattr(meta, "src_node_id", None)
                dropped.append(str(src) if src is not None else "unknown")
                continue
        except Exception:
            dropped.append("unknown")
            continue
        kept.append(r)

    if dropped:
        tag = f"[Round {server_round}] " if server_round is not None else ""
        loc = f" ({where})" if where else ""
        raise RuntimeError(
            f"{tag}Deterministic run aborted: received {len(dropped)} empty/missing client reply/replies{loc}: {dropped}"
        )
    return kept


def _reply_num_examples(reply: Message) -> int:
    metrics = reply.content.get("metrics", None)
    if metrics is None:
        return 1
    # MetricRecord behaves like a mapping
    val = metrics.get("num-examples", metrics.get("num_examples", 1))
    return int(val)

def _reply_client_id(reply: Message) -> str:
    metrics = reply.content.get("metrics", None)
    if metrics is not None and "client_id" in metrics:
        return str(metrics["client_id"])
    # fallback if needed (Flower metadata often has src_node_id)
    meta = getattr(reply, "metadata", None)
    src = getattr(meta, "src_node_id", None)
    return str(src) if src is not None else "unknown"


def _reply_metrics(reply: Message) -> Dict[str, object]:
    """Return the reply MetricRecord as a plain dict (best-effort)."""
    m = reply.content.get("metrics", None)
    if m is None:
        return {}
    try:
        return dict(m)
    except Exception:
        out: Dict[str, object] = {}
        try:
            for k in m.keys():
                out[str(k)] = m[k]
        except Exception:
            pass
        return out


def _build_node_id_to_partition_id_map(grid: Grid, *, timeout: float = 30.0) -> dict[int, str]:
    """Build a mapping {node_id -> partition-id} by querying each node once.

    Requires the ClientApp to implement `@app.query("whoami")` which returns a MetricRecord
    containing `client_id` (the partition-id used across this project).
    """
    try:
        node_ids = list(grid.get_node_ids())
    except Exception:
        return {}

    if not node_ids:
        return {}

    group_id = "bootstrap.whoami"
    msgs: list[Message] = []
    for nid in node_ids:
        try:
            msg = grid.create_message(
                content=RecordDict({}),
                message_type="query.whoami",
                dst_node_id=int(nid),
                group_id=group_id,
                ttl=float(timeout),
            )
        except Exception:
            # Fallback: direct Message construction
            msg = Message(
                content=RecordDict({}),
                dst_node_id=int(nid),
                message_type="query.whoami",
                group_id=group_id,
                ttl=float(timeout),
            )
        msgs.append(msg)

    try:
        replies = list(grid.send_and_receive(msgs, timeout=float(timeout)))
    except Exception:
        return {}

    mapping: dict[int, str] = {}
    for rep in replies:
        try:
            src = int(rep.metadata.src_node_id)
            metrics = rep.content.get("metrics", None)
            if metrics is None:
                continue
            if "client_id" in metrics:
                pid = metrics["client_id"]
            elif "partition-id" in metrics:
                pid = metrics["partition-id"]
            elif "partition_id" in metrics:
                pid = metrics["partition_id"]
            else:
                continue
            mapping[src] = str(int(pid))
        except Exception:
            continue

    return mapping

def _extract_selected_partition_ids(train_instructions, node_id_to_partition_id: dict[int, str] | None = None) -> list[str]:
    """Extract *partition-id* strings for the nodes selected in this round.

    Flower ServerApp strategies typically return an iterable of Messages (or (dst_node_id, Message) pairs)
    where destination IDs are Flower *node_id*s (often large ints). In this project, we want to log the
    stable `partition-id` (0..num_clients-1) that each node is assigned.

    If `node_id_to_partition_id` is provided (built via a bootstrap `query.whoami`), we map node_id ->
    partition-id. If mapping is missing, we fall back to the raw destination id.
    """

    def _map_if_possible(val) -> str | None:
        if not node_id_to_partition_id:
            return None
        try:
            if isinstance(val, int):
                pid = node_id_to_partition_id.get(val)
                return str(pid) if pid is not None else None
            if isinstance(val, str) and val.isdigit():
                pid = node_id_to_partition_id.get(int(val))
                return str(pid) if pid is not None else None
        except Exception:
            return None
        return None

    def _normalize_id(x) -> str | None:
        if x is None:
            return None

        mapped = _map_if_possible(x)
        if mapped is not None:
            return mapped

        if isinstance(x, str):
            s = x
        else:
            for attr in ("cid", "partition_id", "node_id", "id"):
                if hasattr(x, attr):
                    try:
                        v = getattr(x, attr)
                        mapped2 = _map_if_possible(v)
                        if mapped2 is not None:
                            return mapped2
                        return str(v)
                    except Exception:
                        pass
            s = str(x)

        # Fallback: prefer extracting a bare integer token if present
        m = re.search(r"\d+", s)
        return m.group(0) if m else s

    def _maybe_add_from_message(msg: Message) -> list[str]:
        ids: list[str] = []
        meta = getattr(msg, "metadata", None)
        for attr in ("dst_node_id", "dst_node", "dst", "node_id"):
            dst = getattr(meta, attr, None) if meta is not None else None
            nid = _normalize_id(dst)
            if nid is not None:
                ids.append(nid)
                break
        return ids

    out: list[str] = []

    def _visit(obj):
        if obj is None:
            return
        # common: (dst_node_id, Message)
        if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], Message):
            nid = _normalize_id(obj[0])
            if nid is not None:
                out.append(nid)
            return
        if isinstance(obj, (list, tuple)):
            for it in obj:
                _visit(it)
            return
        if isinstance(obj, dict):
            # sometimes {dst: ..., message: ...}
            if "message" in obj and "dst" in obj:
                nid = _normalize_id(obj.get("dst"))
                if nid is not None:
                    out.append(nid)
            for v in obj.values():
                _visit(v)
            return
        if isinstance(obj, Message):
            out.extend(_maybe_add_from_message(obj))
            return

    _visit(train_instructions)

    # de-dup while preserving order
    uniq = list(dict.fromkeys(out))

    def _key(x: str):
        try:
            return (0, int(x))
        except Exception:
            return (1, x)

    return sorted(uniq, key=_key)



# ======================================================================  temp

# def mad(x: np.ndarray) -> float:
#     """Median Absolute Deviation (unscaled)."""
#     x = np.asarray(x, dtype=np.float32).ravel()
#     if x.size == 0:
#         return 0.0
#     med = np.median(x)
#     return float(np.median(np.abs(x - med)))

# ======================================================================
# ======================= Global / Helper State ========================

# Rolling accuracy buffer (per-round aggregated test accuracy)
recent_accuracies: List[float] = []
LAST_N_AVG_ROUNDS: int = 10

# We also keep track of the current round index from server side.
# This is incremented each time aggregate_evaluate is called
# (i.e., once per evaluation / round).
current_round: int = 0

# Global config for malicious-client schedule (set in main)
num_clients_global: int = 0
malicious_percentage_global: float = 0.0

# Snapshot of run_config for final result logging (set in main)
_RUN_CONFIG_SNAPSHOT: dict | None = None

# Track whether the run completed successfully (used by atexit logging).
_RUN_COMPLETED_OK: bool = False
_RUN_CRASHED: bool = False
_RUN_ERROR_MSG: str | None = None
_LAST_SERVER_ROUND: int = 0


# ======================================================================
# ========================= Aggregation Methods ========================

# --------------------- FedAvg --------------------- #

def agg_fedavg_delta(deltas, weights):
    total = float(sum(weights)) if sum(weights) > 0 else float(len(weights))
    out = OrderedDict()
    for k in deltas[0].keys():
        acc = None
        for d, w in zip(deltas, weights):
            t = d[k].detach().cpu()
            acc = t * (float(w) / total) if acc is None else acc + t * (float(w) / total)
        out[k] = acc
    return out

# ------------------------ FedAvg ------------------------ #

def agg_median(client_states):
    out = OrderedDict()
    for k in client_states[0].keys():
        stacked = torch.stack([sd[k].detach().cpu() for sd in client_states], dim=0)
        out[k] = torch.median(stacked, dim=0).values
    return out

# --------------------- Trimmed Mean --------------------- #

def agg_trimmed_mean(client_states, trim_ratio: float):
    n = len(client_states)
    trim_k = int(trim_ratio * n)

    out = OrderedDict()
    for k in client_states[0].keys():
        stacked = torch.stack([sd[k].detach().cpu() for sd in client_states], dim=0)
        flat = stacked.view(n, -1)
        flat_sorted, _ = torch.sort(flat, dim=0)

        lo = trim_k
        hi = n - trim_k
        if hi <= lo:  # too aggressive trimming => fallback to mean
            kept = flat_sorted
        else:
            kept = flat_sorted[lo:hi]

        mean = torch.mean(kept, dim=0).view_as(stacked[0])
        out[k] = mean
    return out

# --------------------- Krum --------------------- #

def _flatten_update(update_sd: "OrderedDict[str, torch.Tensor]") -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in update_sd.values()], dim=0)

def _flatten_update_with_keys(update_sd: "OrderedDict[str, torch.Tensor]", keys) -> torch.Tensor:
    # Enforce identical ordering across clients
    return torch.cat([update_sd[k].reshape(-1) for k in keys], dim=0)

def agg_krum(global_sd, client_states, f: int):
    n = len(client_states)
    if n == 0:
        return global_sd  # or raise

    # Clamp f to a valid range for this round
    f = int(max(0, f))
    f_max = (n - 3) // 2
    if f > f_max:
        f = f_max

    # If still not enough clients for Krum to be meaningful, fallback
    if n < 2 * f + 3:
        # Fallback: simple FedAvg on deltas (unweighted) to avoid undefined behavior
        weights = [1] * n
        deltas = [_sd_sub(sd, global_sd) for sd in client_states]
        delta = agg_fedavg_delta(deltas, weights)
        return _sd_add(global_sd, delta)

    keys = list(global_sd.keys())

    # compute flattened deltas
    updates = []
    for sd in client_states:
        upd = OrderedDict((k, (sd[k].detach().cpu() - global_sd[k].detach().cpu())) for k in keys)
        updates.append(_flatten_update_with_keys(upd, keys))

    # pairwise distances (squared L2)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = torch.sum((updates[i] - updates[j]) ** 2).item()
            d[i][j] = dist
            d[j][i] = dist

    nb_count = n - f - 2  # valid because n >= 2f+3
    scores = []
    for i in range(n):
        row = sorted(d[i])         # includes 0 for self
        scores.append(sum(row[1:1 + nb_count]))

    winner = int(min(range(n), key=lambda i: scores[i]))
    return client_states[winner]

# --------------------- Multi-Krum --------------------- #

def agg_multi_krum(global_sd, client_states, f: int, m: int):
    n = len(client_states)
    if n == 0:
        return global_sd

    # Clamp f to a valid range for this round
    f = int(max(0, f))
    f_max = (n - 3) // 2
    if f > f_max:
        f = f_max

    # Krum/Multi-Krum requires n >= 2f + 3
    if n < 2 * f + 3:
        # Fallback: FedAvg on deltas (unweighted)
        deltas = [_sd_sub(sd, global_sd) for sd in client_states]
        delta = agg_fedavg_delta(deltas, [1] * n)
        return _sd_add(global_sd, delta)

    # Robustness-recommended bound: m <= n - f
    m = int(max(1, min(m, n - f)))

    keys = list(global_sd.keys())

    # Build flattened deltas in a consistent order
    updates = []
    for sd in client_states:
        upd = OrderedDict((k, (sd[k].detach().cpu() - global_sd[k].detach().cpu())) for k in keys)
        updates.append(_flatten_update_with_keys(upd, keys))

    # Pairwise squared distances
    dmat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = torch.sum((updates[i] - updates[j]) ** 2).item()
            dmat[i][j] = dist
            dmat[j][i] = dist

    nb_count = n - f - 2  # valid since n >= 2f+3
    scores = []
    for i in range(n):
        row = sorted(dmat[i])  # includes 0 for self
        scores.append(sum(row[1:1 + nb_count]))

    selected = sorted(range(n), key=lambda i: scores[i])[:m]

    # Average selected deltas and apply to global
    out = OrderedDict()
    for k in keys:
        delta_mean = torch.mean(
            torch.stack([(client_states[i][k].detach().cpu() - global_sd[k].detach().cpu())
                         for i in selected], dim=0),
            dim=0,
        )
        out[k] = global_sd[k].detach().cpu() + delta_mean
    return out

# ----------------  Helper methods ------------------ #

def _to_deltas(global_sd, client_sds):
    return [_sd_sub(sd, global_sd) for sd in client_sds]

def _apply_delta(global_sd, delta_sd):
    return _sd_add(global_sd, delta_sd)

# --------------------- FLTrust --------------------- #

def _sd_sub(a, b):
    # a - b (state_dicts)
    out = OrderedDict()
    for k in a.keys():
        out[k] = a[k].detach().cpu() - b[k].detach().cpu()
    return out

def _sd_add(a, b):
    out = OrderedDict()
    for k in a.keys():
        out[k] = a[k].detach().cpu() + b[k].detach().cpu()
    return out

def _sd_scale(sd, s: float):
    out = OrderedDict()
    for k in sd.keys():
        out[k] = sd[k].detach().cpu() * float(s)
    return out

def _sd_zeros_like(sd):
    out = OrderedDict()
    for k in sd.keys():
        out[k] = torch.zeros_like(sd[k].detach().cpu())
    return out


def _flatten_sd(sd) -> np.ndarray:
    return torch.cat([v.detach().view(-1).cpu() for v in sd.values()]).numpy()


# Shared tensor/vector conversion helpers used by coordinated attacks and defenses.

def _flatten_sd_torch(sd: OrderedDict, keys: list[str]) -> torch.Tensor:
    """Flatten a (delta) state_dict into a 1-D float32 CPU tensor in a fixed key order."""
    return torch.cat([sd[k].detach().reshape(-1).to(torch.float32).cpu() for k in keys], dim=0)


def _unflatten_vec_to_delta_sd(
    vec: torch.Tensor,
    template_sd: OrderedDict,
    keys: list[str],
) -> OrderedDict:
    """Unflatten a vector into a delta state_dict matching template shapes/dtypes (CPU)."""
    out = OrderedDict()
    idx = 0
    vec = vec.detach().to(torch.float32).cpu()
    for k in keys:
        t = template_sd[k].detach().cpu()
        n = t.numel()
        out[k] = vec[idx : idx + n].view_as(t).to(dtype=t.dtype)
        idx += n
    if idx != int(vec.numel()):
        raise ValueError(f"Delta vector length mismatch: used {idx}, vec has {vec.numel()}")
    return out


# ----------------- ALIE / IPM (same-round, omniscient) ----------------- #
def _avg_l2_norm(G: torch.Tensor, eps: float = 1e-12) -> float:
    if G.numel() == 0:
        return 0.0
    nrm = torch.norm(G, dim=1, p=2)
    return float(nrm.mean().item()) if int(nrm.numel()) > 0 else 0.0


def _alie_zmax(n: int, f: int) -> float:
    """Compute z_max used in ALIE.

    Using the common closed-form from benchmark/survey descriptions:
        p = (n - floor(n/2 + 1)) / (n - f)
        z_max = Phi^{-1}(p)
    where Phi is the standard normal CDF.
    """
    n = int(max(1, n))
    f = int(max(0, f))
    denom = max(1, n - f)
    p = (n - math.floor(n / 2.0 + 1.0)) / float(denom)
    # keep inside (0,1) for numerical stability
    p = min(0.999999, max(1e-6, float(p)))
    return float(abs(norm.ppf(p)))


def forge_alie_ipm_same_round(
    *,
    attack_type: str,
    global_sd: OrderedDict,
    client_sds: list[OrderedDict],
    client_ids: list[str],
    malicious_ids: set[str],
    alie_z: float = -1.0,
    ipm_gamma: float = 20.0,
    ipm_norm_match: bool = True,
) -> list[OrderedDict]:
    """Overwrite malicious clients with a forged update for ALIE or IPM."""

    atk = str(attack_type)
    if atk not in {"ALIE", "IPM"}:
        return client_sds

    keys = list(global_sd.keys())

    benign_vecs: list[torch.Tensor] = []
    mal_indices: list[int] = []
    for i, (sd, cid) in enumerate(zip(client_sds, client_ids)):
        if str(cid) in malicious_ids:
            mal_indices.append(i)
        else:
            delta_sd = _sd_sub(sd, global_sd)
            benign_vecs.append(_flatten_sd_torch(delta_sd, keys))

    if not mal_indices or len(benign_vecs) < 1:
        return client_sds

    G = torch.stack(benign_vecs, dim=0)  # [B, d]
    mu = G.mean(dim=0)
    sigma = G.std(dim=0, unbiased=False)
    eps = 1e-12

    # A simple, commonly used direction choice that stays within coordinate-wise bounds:
    # move opposite the sign of the benign mean.
    sgn_mu = torch.sign(mu)
    sgn_mu = torch.where(sgn_mu == 0, torch.ones_like(sgn_mu), sgn_mu)

    if atk == "ALIE":
        if float(alie_z) > 0.0:
            z = float(alie_z)
        else:
            z = _alie_zmax(len(client_sds), len(mal_indices))
        forged_delta_vec = mu - z * sigma * sgn_mu
    elif atk == "IPM":
        if ipm_norm_match:
            avg_norm = _avg_l2_norm(G)
            forged_delta_vec = (-mu) / (mu.norm(p=2) + eps) * (float(ipm_gamma) * avg_norm)
        else:
            forged_delta_vec = -float(ipm_gamma) * mu
    else:
        raise ValueError(f"Unsupported coordinated attack: {atk}")

    forged_delta_sd = _unflatten_vec_to_delta_sd(forged_delta_vec, global_sd, keys)
    forged_sd = _sd_add(global_sd, forged_delta_sd)

    out = list(client_sds)
    for idx in mal_indices:
        out[idx] = forged_sd
    return out


def agg_fltrust(global_sd, client_states, root_sd, cos_clip: float = 0.0, eps: float = 1e-12):
    """
    FLTrust (Li et al.): compute a trusted 'server update' from root data,
    then trust-weight + norm-scale client updates based on cosine similarity
    to the trusted update.
    """
    # trusted/server delta
    trusted_delta = _sd_sub(root_sd, global_sd)
    t_vec = _flatten_sd(trusted_delta)
    t_norm = float(np.linalg.norm(t_vec)) + eps

    # accumulate scaled deltas
    acc_delta = _sd_zeros_like(global_sd)
    total_trust = 0.0
    trust_scores = []

    for sd in client_states:
        d = _sd_sub(sd, global_sd)
        d_vec = _flatten_sd(d)
        d_norm = float(np.linalg.norm(d_vec)) + eps

        cos = float(np.dot(d_vec, t_vec) / (d_norm * t_norm))
        trust = max(float(cos_clip), cos)  # cos_clip=0.0 => ReLU(cos)

        trust_scores.append(trust)

        if trust <= 0.0:
            continue

        # norm scaling: ||trusted_delta|| / ||client_delta||
        scale = trust * (t_norm / d_norm)

        for k in acc_delta.keys():
            acc_delta[k] += d[k] * scale

        total_trust += trust

    # If everyone is anti-aligned (all trust=0), FLTrust excludes all client updates.
    # Most faithful-to-main-paper behavior: skip the update for this round.
    if total_trust <= 0.0:
        return global_sd, trust_scores

    # average accumulated delta
    acc_delta = _sd_scale(acc_delta, 1.0 / total_trust)

    # apply to global
    new_sd = _sd_add(global_sd, acc_delta)
    return new_sd, trust_scores


# --------------------- Others --------------------- #

def agg_shieldfl(
    global_sd: dict[str, torch.Tensor],
    client_sds: list[dict[str, torch.Tensor]],
    weights: list[float],
    prev_agg_delta_vec: np.ndarray | None = None,
    eps: float = 1e-12,
):
    """ShieldFL (paper-faithful, non-crypto) cosine reweighting on *deltas*.

    Paper idea (Algorithm 3):
      1) Use last round's aggregated gradient/update direction as anchor.
      2) Pick baseline g* as the client update least aligned with that anchor.
      3) Weight clients by confidence eta_i = 1 - cos(g_i, g*), then aggregate g(t)=Σ η_i g_i (paper).

    Notes:
      - We operate on parameter deltas (client_sd - global_sd).
      - If prev_agg_delta_vec is missing (round 1) or near-zero, fall back to plain FedAvg.
    """
    if not client_sds:
        return global_sd

    # Convert to deltas and flatten
    deltas = [_sd_sub(sd, global_sd) for sd in client_sds]
    vecs = np.stack([_flatten_sd(d) for d in deltas], axis=0)

    # ShieldFL assumes *normalized* local gradients (SecJudge in the paper).
    # In our non-crypto setting, we enforce this by normalizing each client's delta to unit norm
    # before both cosine tests and aggregation (so magnitude-based poisoning is neutralized).

    raw_norms = np.linalg.norm(vecs, axis=1)

    valid = np.isfinite(raw_norms) & (raw_norms > eps)
    if not np.any(valid):
        return global_sd

    # Keep only valid clients (keep deltas/vecs aligned)
    vecs = vecs[valid]
    deltas = [d for d, ok in zip(deltas, valid) if ok]
    norms = raw_norms[valid]

    # Unit directions + unit-norm deltas
    u = vecs / (norms[:, None] + eps)
    deltas_unit = [_sd_scale(d, float(1.0 / (nrm + eps))) for d, nrm in zip(deltas, norms)]


    # Round-1 / no-anchor fallback: uniform average of normalized deltas
    if prev_agg_delta_vec is None:
        delta = agg_fedavg_delta(deltas_unit, [1.0] * len(deltas_unit))
        return _sd_add(global_sd, delta)

    anchor_norm = float(np.linalg.norm(prev_agg_delta_vec))
    if not np.isfinite(anchor_norm) or anchor_norm <= eps:
        delta = agg_fedavg_delta(deltas_unit, [1.0] * len(deltas_unit))
        return _sd_add(global_sd, delta)

    # Normalize anchor
    anchor = prev_agg_delta_vec / (anchor_norm + eps)

    # (1) baseline selection: least aligned with previous aggregate direction
    cos_to_anchor = u @ anchor
    baseline_idx = int(np.argmin(cos_to_anchor))
    baseline = u[baseline_idx]

    # (2) confidence: eta_i = 1 - cos(u_i, baseline)  in [0, 2]
    cos_to_base = u @ baseline
    conf = 1.0 - cos_to_base
    conf = np.clip(conf, 0.0, 2.0)

    # Paper: g(t) = sum_i eta_i * g_i  (no sample-size FedAvg weighting)
    new_w = conf.astype(np.float64).tolist()
    if not np.isfinite(sum(new_w)) or sum(new_w) <= eps:
        return global_sd  # paper-faithful: eta-sum=0 => zero update (skip round)

    delta = agg_fedavg_delta(deltas_unit, new_w)
    return _sd_add(global_sd, delta)

# --------------------- Others --------------------- #


# --------------------- Others --------------------- #

def agg_dnc(
    global_sd: dict[str, torch.Tensor],
    client_sds: list[dict[str, torch.Tensor]],
    weights: list[float],
    f: int,
    c: float = 1.5,
    niters: int = 5,
    b: int = 256,
    seed: int = 0,
    *,
    use_sample_weights: bool = False,
    eps: float = 1e-12,
):
    """DnC (Divide-and-Conquer) filtering (Algorithm 2), with robustness guards.

    Paper behavior (Algorithm 2):
      - Work on client deltas Δ_i
      - Repeat niters times:
          * sample random coordinate subset r (size b)
          * center projected deltas
          * compute top right singular vector v via SVD
          * score_i = <y_i, v>^2
          * keep the (n - c·m) smallest-score clients
      - Intersect keep-sets across iterations
      - Uniform-average remaining deltas (paper-faithful)

    Robustness additions:
      - Drop non-finite client delta vectors (NaN/Inf) before SVD.
      - Guard SVD failures (falls back to last non-empty keep-set).
    """
    n0 = len(client_sds)
    if n0 == 0:
        return global_sd

    f = int(f)
    if f < 0:
        f = 0

    # --- Compute deltas Δ_i with FIXED key-order (global_sd keys) ---
    keys = list(global_sd.keys())
    deltas: list[OrderedDict[str, torch.Tensor]] = []
    flat_list: list[np.ndarray] = []

    for sd in client_sds:
        out = OrderedDict()
        vec_parts = []
        for k in keys:
            dk = sd[k].detach().cpu() - global_sd[k].detach().cpu()
            out[k] = dk
            vec_parts.append(dk.view(-1))
        deltas.append(out)
        flat_list.append(torch.cat(vec_parts).numpy().astype(np.float64))

    X = np.stack(flat_list, axis=0)  # (n0, d)
    d = int(X.shape[1])
    if d <= 0:
        return global_sd

    # --- Robustness: remove non-finite rows (prevents SVD/score NaNs) ---
    finite_mask = np.isfinite(X).all(axis=1)
    if not bool(finite_mask.all()):
        valid_idx = np.where(finite_mask)[0].tolist()
        if len(valid_idx) == 0:
            return global_sd
        X = X[valid_idx, :]
        deltas = [deltas[i] for i in valid_idx]
        # keep weights aligned to filtered list
        if len(weights) == n0:
            weights = [weights[i] for i in valid_idx]
        n = len(valid_idx)
    else:
        n = n0

    # clamp f to participant count
    f = max(0, min(int(f), n))

    rng = np.random.default_rng(seed)

    # Keep (n - c·m) clients; remove ceil(c·m)
    target_keep = n - int(math.ceil(float(c) * float(f)))
    target_keep = max(1, min(target_keep, n))

    # If parameters imply "keep all", just aggregate directly
    if target_keep == n or int(niters) <= 0 or int(b) <= 0:
        kept_idx = list(range(n))
    else:
        keep = set(range(n))
        last_nonempty_keep = set(keep)

        for _ in range(int(niters)):
            subdim = min(int(b), d)

            # Paper: r is a sorted set of random dimensions
            idx = rng.choice(d, size=subdim, replace=False)
            idx.sort()

            Xs = X[:, idx]                       # (n, subdim)
            mu = Xs.mean(axis=0, keepdims=True)   # (1, subdim)
            Y = Xs - mu                           # centered

            # Top right singular vector of Y
            try:
                _, _, Vt = np.linalg.svd(Y, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            if Vt.size == 0:
                break

            v = Vt[0]
            scores = (Y @ v) ** 2
            order = np.argsort(scores, kind="stable")
            kept_now = set(order[:target_keep].tolist())

            new_keep = keep.intersection(kept_now)
            if len(new_keep) == 0:
                # Avoid empty-intersection failure; fall back to last non-empty keep-set
                break

            keep = new_keep
            last_nonempty_keep = set(keep)

        kept_idx = sorted(list(last_nonempty_keep))

    if len(kept_idx) == 0:
        return global_sd

    kept_deltas = [deltas[i] for i in kept_idx]

    # Paper uses uniform mean; optional FedAvg weighting after filtering
    if use_sample_weights:
        kept_w = [float(weights[i]) for i in kept_idx]
        w_arr = np.asarray(kept_w, dtype=np.float64)
        if (not np.isfinite(w_arr).all()) or float(w_arr.sum()) <= eps:
            kept_w = [1.0] * len(kept_idx)
    else:
        kept_w = [1.0] * len(kept_idx)

    delta = agg_fedavg_delta(kept_deltas, kept_w)
    return _sd_add(global_sd, delta)


# ------------------------------------------------------------------

class _BTBCNNode:
    """Binary-tree node for BTBCN hierarchical clustering."""
    __slots__ = ("idx", "centroid", "left", "right")

    def __init__(self, idx, centroid, left=None, right=None):
        self.idx = np.asarray(idx, dtype=np.int64)          # leaf indices under this node
        self.centroid = np.asarray(centroid, dtype=np.float64)
        self.left = left
        self.right = right

    @property
    def size(self) -> int:
        return int(self.idx.size)


def _btbcn_detect_labels(W: np.ndarray, min_cluster_size: int = 2) -> np.ndarray:
    """
    FedDMC BTBCN (Algorithm 2) — returns per-client labels S where:
      S[i] = 1 => benign, S[i] = 0 => malicious

    Steps:
      1) Build a hierarchical binary tree by repeatedly merging the two closest clusters
         (we use Euclidean distance between cluster centroids).
      2) Condensed binary tree pruning: starting from the root, keep splitting downward;
         if a child has size < min_cluster_size, mark that child as outlier and continue
         down the other child. Stop when both children are non-outliers or outliers >= floor(n/2).
      3) Client evaluation: label the larger of the final two clusters as benign; everything else
         (including outliers) as malicious.
    """
    n = int(W.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int8)
    if n == 1:
        return np.ones((1,), dtype=np.int8)

    # ---------- Build Binary Tree (agglomerative hierarchical clustering) ----------
    clusters: list[_BTBCNNode] = [_BTBCNNode([i], W[i]) for i in range(n)]

    while len(clusters) > 1:
        C = np.stack([c.centroid for c in clusters], axis=0)   # (m, k)
        diff = C[:, None, :] - C[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist2, np.inf)

        flat = int(np.argmin(dist2))
        i, j = divmod(flat, dist2.shape[1])
        if j < i:
            i, j = j, i

        a, b = clusters[i], clusters[j]
        new_idx = np.concatenate([a.idx, b.idx], axis=0)
        new_centroid = (a.centroid * a.size + b.centroid * b.size) / float(a.size + b.size)
        merged = _BTBCNNode(new_idx, new_centroid, left=a, right=b)

        # remove j then i (so indices remain valid)
        clusters.pop(j)
        clusters.pop(i)
        clusters.append(merged)

    root = clusters[0]

    # ---------- Condensed Binary Tree (prune noise/outliers) ----------
    outliers: set[int] = set()

    def _is_outlier(node: _BTBCNNode | None) -> bool:
        return node is None or node.size < min_cluster_size

    r = root
    L, R = r.left, r.right

    # Follow Algorithm 2 logic; handle the rare case where both children are outliers.
    while _is_outlier(L) or _is_outlier(R):
        if _is_outlier(L) and _is_outlier(R):
            # If both are too small, mark both as outliers and stop.
            if L is not None:
                outliers.update(L.idx.tolist())
            if R is not None:
                outliers.update(R.idx.tolist())
            break

        if _is_outlier(L) and L is not None:
            outliers.update(L.idx.tolist())
            r = R if R is not None else r

        elif _is_outlier(R) and R is not None:
            outliers.update(R.idx.tolist())
            r = L if L is not None else r

        if len(outliers) >= (n // 2):
            break
        if r.left is None or r.right is None:
            break

        L, R = r.left, r.right

    # ---------- ClientsEvaluation ----------
    labels = np.zeros((n,), dtype=np.int8)  # default malicious

    if L is None or R is None:
        benign = set(r.idx.tolist()) - outliers
    else:
        benign_node = L if L.size >= R.size else R
        benign = set(benign_node.idx.tolist()) - outliers

    for i in benign:
        if 0 <= i < n:
            labels[i] = 1

    return labels


def agg_feddmc(
    global_sd: dict[str, torch.Tensor],
    client_sds: list[dict[str, torch.Tensor]],
    client_ids: list[str],
    weights: list[float],
    trust: dict[str, float],
    *,
    server_round: int,
    pca_dim: int = 5,
    min_cluster_size: int = 2,
    ema_alpha: float = 0.8,
    trust_threshold: float = 0.5,
    use_sample_size_weights: bool = False,  # paper Alg.1 uses uniform averaging
    eps: float = 1e-12,
    pca_random_state: int = 0,
):
    """
    FedDMC-style aggregation aligned to:
      - Algorithm 1: PCA -> BTBCN -> SEDC -> aggregate benign
      - Algorithm 2: BTBCN (binary tree + condensed pruning)
      - Eq.(3): S_hat[t] = 0.5 if t=0 else alpha*S_hat[t-1] + (1-alpha)*S[t]

    Key behavior (paper-faithful):
      - Round t=0 (Flower server_round==1): initialize trust=0.5 for all participating clients.
        With the paper's threshold=0.5, this keeps all clients (i.e., FedAvg for round 1).
      - Round t>0: update trust via EMA and filter by trust_threshold.

    Note: We still compute BTBCN each round to produce S[t], but S[0] is not used to
    filter at t=0 because Eq.(3) defines S_hat[0]=0.5.
    """
    n = len(client_sds)
    if n == 0:
        return global_sd

    # Defensive parameter sanitization (paper: 0 <= alpha < 1, threshold in [0,1])
    pca_dim = max(1, int(pca_dim))
    min_cluster_size = max(1, int(min_cluster_size))
    ema_alpha = float(np.clip(float(ema_alpha), 0.0, 0.999999))
    trust_threshold = float(np.clip(float(trust_threshold), 0.0, 1.0))

    keys = list(global_sd.keys())

    def _delta_and_flat(
        sd: dict[str, torch.Tensor]
    ) -> tuple["OrderedDict[str, torch.Tensor]", np.ndarray, float]:
        d = OrderedDict()
        parts = []
        for k in keys:
            dv = sd[k].detach().cpu() - global_sd[k].detach().cpu()
            d[k] = dv
            parts.append(dv.reshape(-1))
        vec = torch.cat(parts, dim=0).numpy().astype(np.float64, copy=False)
        norm = float(np.linalg.norm(vec))
        return d, vec, norm

    # ---- Build (delta, flat) for each client; filter invalid vectors for PCA/BTBCN ----
    deltas_all: list[OrderedDict] = []
    X_all: list[np.ndarray] = []
    norms_all: list[float] = []

    valid_mask: list[bool] = []
    for sd in client_sds:
        d, v, nv = _delta_and_flat(sd)
        deltas_all.append(d)
        X_all.append(v)
        norms_all.append(nv)

        ok = np.isfinite(nv) and (nv > eps) and np.isfinite(v).all()
        valid_mask.append(ok)

    valid_idx = [i for i, ok in enumerate(valid_mask) if ok]
    if len(valid_idx) == 0:
        # No usable updates -> safest is skip update
        return global_sd

    X = np.stack([X_all[i] for i in valid_idx], axis=0)  # (n_valid, d)

    # ---- PCA dimensionality reduction ----
    k = int(min(pca_dim, X.shape[0], X.shape[1]))
    if k >= 1 and X.shape[1] > k:
        Xr = PCA(n_components=k, random_state=pca_random_state).fit_transform(X)
    else:
        Xr = X

    # ---- BTBCN detection for THIS round (on valid subset only): 1 benign / 0 malicious ----
    S_valid = _btbcn_detect_labels(Xr, min_cluster_size=min_cluster_size)

    # Expand labels back to full n with default "malicious" for invalid vectors
    S = np.zeros((n,), dtype=np.int8)
    for pos, i in enumerate(valid_idx):
        S[i] = int(S_valid[pos])

    # ---- SEDC trust update + selection ----
    is_t0 = (server_round <= 1)  # Flower round 1 corresponds to paper t=0

    if is_t0:
        # Eq.(3): S_hat[0] = 0.5
        for cid_raw in client_ids:
            trust[str(cid_raw)] = 0.5
        kept_idx = valid_idx.copy()  # threshold=0.5 => all benign at t=0
    else:
        kept_idx: list[int] = []
        for i, cid_raw in enumerate(client_ids):
            cid = str(cid_raw)

            prev = float(trust.get(cid, 0.5))
            s_t = float(S[i])  # 1.0 benign, 0.0 malicious
            trust[cid] = ema_alpha * prev + (1.0 - ema_alpha) * s_t

            if valid_mask[i] and (trust[cid] + eps >= trust_threshold):
                kept_idx.append(i)

    if len(kept_idx) == 0:
        # No one passes -> safest is skip update (zero update for this round)
        return global_sd

    # ---- Aggregate benign clients (paper Alg.1 uses uniform mean over C_B) ----
    kept_deltas = [deltas_all[i] for i in kept_idx]

    if use_sample_size_weights:
        kept_w = [float(weights[i]) for i in kept_idx]
    else:
        kept_w = [1.0] * len(kept_deltas)

    delta = agg_fedavg_delta(kept_deltas, kept_w)
    return _sd_add(global_sd, delta)



# ========================================================================

# ===================== ProbeGuard: Probe-based Peer Audit =====================

def _probeguard_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = logits - np.max(logits, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)


def _probeguard_pair_probs_from_logits(logits: np.ndarray, a: int, b: int) -> np.ndarray:
    """Return a 2D distribution on {a,b} by renormalizing softmax probs."""
    p = _probeguard_softmax(logits, axis=1)
    q = p[:, [a, b]]
    s = np.sum(q, axis=1, keepdims=True) + 1e-12
    return q / s


def _probeguard_logits_for_X(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32)
        logits = model(Xt).detach().cpu().numpy()
    return logits


def _probeguard_probe_pairs_for_X(model: torch.nn.Module, X: np.ndarray) -> list[tuple[int, int]]:
    """Assign one top-2 class pair to each probe point in X."""
    if X is None or X.size == 0:
        return []
    logits = _probeguard_logits_for_X(model, X)
    if logits is None or (not isinstance(logits, np.ndarray)) or logits.ndim != 2 or logits.shape[0] == 0:
        return []

    C = int(logits.shape[1]) if logits.ndim == 2 else 0
    if C <= 1:
        return [(0, 0)] * int(logits.shape[0])

    top2 = np.argsort(logits, axis=1)[:, -2:]
    out: list[tuple[int, int]] = []
    for r in range(top2.shape[0]):
        a, b = int(top2[r, 0]), int(top2[r, 1])
        if a > b:
            a, b = b, a
        out.append((a, b))
    return out


def _probeguard_compute_probe_anchors(
    model: torch.nn.Module,
    probe_X: np.ndarray,
    probe_pairs: list[tuple[int, int]],
    *,
    topk: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-probe class anchors using `model` (typically the previous global model).

    Args:
      topk: number of top logits/classes to keep as anchors (>=1). Output shape is (P, topk).

    Returns:
      - topk_anchors: (P, topk) int array; for probe j, classes are the top-`topk` predicted classes by `model`.
                      If num_classes < topk, values are repeated to keep shape.
      - comp_anchors: (P,) int array; for probe j with pair (a,b), comp is argmax over classes not in {a,b}.
                      If num_classes <= 2, comp is set to a (so dominance feature becomes ~0).
    """
    try:
        topk = int(topk)
    except Exception:
        topk = 3
    topk = max(1, topk)

    if probe_X is None or probe_X.size == 0:
        return np.zeros((0, topk), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    logits = _probeguard_logits_for_X(model, probe_X)  # (P,C)
    if logits is None or (not isinstance(logits, np.ndarray)) or logits.ndim != 2:
        return np.zeros((0, topk), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    P, C = int(logits.shape[0]), int(logits.shape[1])

    # Top-k anchors (pad by repeating when C < topk)
    k_eff = int(min(topk, C))
    if k_eff <= 0:
        topk_anchors = np.zeros((P, topk), dtype=np.int64)
    else:
        topk_idx = np.argsort(logits, axis=1)[:, -k_eff:][:, ::-1]  # (P,k_eff), descending
        topk_anchors = np.zeros((P, topk), dtype=np.int64)
        for j in range(P):
            idx = topk_idx[j]
            if idx.size == 0:
                topk_anchors[j, :] = 0
            else:
                # copy available
                n = int(min(idx.size, topk))
                topk_anchors[j, :n] = idx[:n].astype(np.int64)
                # pad remainder by repeating last available
                if n < topk:
                    topk_anchors[j, n:] = int(topk_anchors[j, n - 1])

    # Competitor anchors for each (a,b) probe
    comp = np.zeros((P,), dtype=np.int64)
    for j, (a, b) in enumerate(probe_pairs):
        a_i, b_i = int(a), int(b)
        if C <= 2:
            comp[j] = a_i
            continue
        row = logits[j].copy()
        if 0 <= a_i < C:
            row[a_i] = -np.inf
        if 0 <= b_i < C:
            row[b_i] = -np.inf
        c_i = int(np.argmax(row))
        # Safety: if argmax accidentally returns a/b due to infs, fall back to a
        if c_i == a_i or c_i == b_i:
            c_i = a_i
        comp[j] = c_i

    return topk_anchors, comp


def _probeguard_anchored_shift_invariant_features(
    logits: np.ndarray,
    topk_anchors: np.ndarray,
    *,
    feature_mode: str,
) -> np.ndarray:
    """Produce class-aligned, additive-logit-shift-invariant probe features.

    ``topk_anchors`` must be selected ONCE from the pre-aggregation global
    model and shared by all clients. Re-selecting the top-k classes separately
    for each client would discard class identity and hide label permutations.

    - global_top3_centered_logits: subtract the mean of the *distinct* anchored
      class logits for each probe. Thus z and z + c*1 have identical features.
      Repeated anchor IDs (when num_classes < topk) are not double-counted when
      computing the center, and retain the same padded output width.
    - global_top3_probs: stable softmax over ALL classes, then extract the
      same globally anchored coordinates, without re-normalizing top-k mass.

    In both cases, the output shape is (num_probes, num_anchors).
    """
    z = np.asarray(logits, dtype=np.float64)
    anchors = np.asarray(topk_anchors, dtype=np.int64)
    if z.ndim != 2 or anchors.ndim != 2 or z.shape[0] != anchors.shape[0]:
        raise ValueError("ProbeGuard anchored features: logits/anchors must have compatible 2D shapes")
    if z.shape[1] < 1 or anchors.shape[1] < 1:
        raise ValueError("ProbeGuard anchored features: empty class/anchor dimension")
    if not np.isfinite(z).all():
        raise ValueError("ProbeGuard anchored features: non-finite logits")
    if np.any(anchors < 0) or np.any(anchors >= z.shape[1]):
        raise ValueError("ProbeGuard anchored features: invalid global class anchor")

    if feature_mode == "global_top3_probs":
        # Softmax first: non-anchored classes contribute to normalization.
        probs = _probeguard_softmax(z, axis=1)
        return np.take_along_axis(probs, anchors, axis=1).astype(np.float32)
    if feature_mode != "global_top3_centered_logits":
        raise ValueError(f"Unsupported shift-invariant feature mode: {feature_mode!r}")

    selected = np.take_along_axis(z, anchors, axis=1)
    # Anchor padding repeats the last class when num_classes < topk.
    # Use only unique IDs for the mean so padding cannot bias the center.
    unique = np.ones_like(anchors, dtype=bool)
    for k in range(1, anchors.shape[1]):
        unique[:, k] = ~np.any(anchors[:, :k] == anchors[:, k, None], axis=1)
    center = (selected * unique).sum(axis=1, keepdims=True) / unique.sum(axis=1, keepdims=True)
    return (selected - center).astype(np.float32)


def _probeguard_compute_suspicion(
    *,
    global_sd: OrderedDict,
    client_sds: list[OrderedDict],
    client_ids: list[str],
    probe_X: np.ndarray,
    probe_pairs: list[tuple[int, int]],
    probe_top3_anchors: Optional[np.ndarray],
    probe_comp_anchors: Optional[np.ndarray],
    probeguard_probe_feature_mode: str,
    num_partitions: int,
    run_config: dict,
    mad_alpha: float,
    top_frac: float,
    use_kmeans: bool,
    probeguard_pca_var: float,
    probeguard_cluster_method: str,
    probeguard_iforest_contamination: float,
    rng: np.random.Generator,
    sklearn_random_state: int,
    use_mad: bool = True,
    probeguard_probe_mad_mode: str = "both",
    probeguard_probe_normalize_per_probe: bool = False,
    probeguard_probe_scale_floor_ratio: float = 0.05,
    probeguard_probe_threshold_normalized_ranking: bool = False,
) -> tuple[Dict[str, float], list[str]]:
    """Return Phase-1 probe scores/suspects; optional E1 and E2 are opt-in."""
    if probe_X is None or probe_X.shape[0] == 0:
        return {cid: 0.0 for cid in client_ids}, []

    probe_pairs = list(probe_pairs or [])

    # Build model shells
    model = build_model(num_partitions=num_partitions, run_config=run_config)

    P = int(probe_X.shape[0])

    # Number of global class-anchor logits. -1 selects all model output classes.
    try:
        probeguard_topk_logits = int(run_config.get("probeguard_topk_logits", 3))
    except (ValueError, TypeError):
        probeguard_topk_logits = 3
    if probeguard_topk_logits == -1:
        # Build_model derives its class count from the dataset when num_classes=0.
        probeguard_topk_logits = int(_probeguard_logits_for_X(model, probe_X[:1]).shape[1])
    elif probeguard_topk_logits < 1:
        raise ValueError("probeguard_topk_logits must be -1 or a positive integer")
    probeguard_topk_logits = max(2, int(probeguard_topk_logits))

    # Feature mode per probe; signature width depends on the selected representation.
    # Modes:
    #   1) "pair_probs" (default): current method, two-class probs for (a,b)
    #   2) "margin_conf": prob margin (p_a-p_b) and confidence proxy max(logit_a, logit_b)
    #   3) "top3_diffs": logit gaps (top1-top2, top2-top3) among top-3 logits
    mode = str(probeguard_probe_feature_mode).lower().strip() if probeguard_probe_feature_mode is not None else "pair_probs"
    if mode in {"1", "current", "pair", "pair_logits", "pair_probs"}:
        mode = "pair_probs"
    elif mode in {"2", "margin_conf", "margin+conf", "prob_margin_conf", "margin_conf_logitmax"}:
        mode = "margin_conf"
    elif mode in {"3", "top3diffs", "top3_diffs", "logit_top3_diffs", "rank_gaps"}:
        mode = "top3_diffs"
    elif mode in ("a", "pair_logit_margin_conf", "pair_logit_mc", "pair_logit"):
        mode = "pair_logit_margin_conf"
    elif mode in ("b", "global_top3_diffs", "anchored_top3_diffs", "global_top3"):
        mode = "global_top3_diffs"
    elif mode in ("b2", "global_top3_logits", "anchored_top3_logits", "global_top3_raw", "global_top3_values"):
        mode = "global_top3_logits"
    elif mode in ("global_top3_centered_logits", "anchored_top3_centered_logits", "global_topk_centered_logits"):
        mode = "global_top3_centered_logits"
    elif mode in ("global_top3_probs", "anchored_top3_probs", "global_topk_probs"):
        mode = "global_top3_probs"
    elif mode in ("c", "pair_competitor", "pair_comp", "pair_compet"):
        mode = "pair_competitor"
    else:
        mode = "pair_probs"

    # Distance-only random probes deliberately carry no top-two class-pair
    # metadata. Pair-dependent feature modes therefore fall back to the existing
    # pair-free top-k logit-gap representation.
    pair_dependent_modes = {
        "pair_probs",
        "margin_conf",
        "pair_logit_margin_conf",
        "pair_competitor",
    }
    if mode in pair_dependent_modes and len(probe_pairs) != P:
        mode = "top3_diffs"

    # Ensure anchors are available for anchored modes (B/C)
    top3_anchors = probe_top3_anchors
    comp_anchors = probe_comp_anchors
    try:
        need_anchors = mode in (
            "global_top3_diffs", "global_top3_logits",
            "global_top3_centered_logits", "global_top3_probs", "pair_competitor",
        )
        bad_top3 = (
            top3_anchors is None
            or (not isinstance(top3_anchors, np.ndarray))
            or (top3_anchors.shape[0] != P)
            or (top3_anchors.shape[1] != int(probeguard_topk_logits))
        )
        bad_comp = (
            comp_anchors is None
            or (not isinstance(comp_anchors, np.ndarray))
            or (comp_anchors.shape[0] != P)
        )
        if need_anchors and (bad_top3 or bad_comp):
            model.load_state_dict(global_sd)
            top3_anchors, comp_anchors = _probeguard_compute_probe_anchors(model, probe_X, probe_pairs, topk=int(probeguard_topk_logits))
    except Exception:
        # Anchor failure fallbacks keep features well-defined
        if mode in (
            "global_top3_diffs", "global_top3_logits",
            "global_top3_centered_logits", "global_top3_probs",
        ):
            mode = "top3_diffs"
        if mode == "pair_competitor":
            mode = "pair_logit_margin_conf"

    feats: list[list[float]] = []  # per-client feature vector (P*F), F depends on feature mode

    for _i, sd in enumerate(client_sds):
        model.load_state_dict(sd)
        logits = _probeguard_logits_for_X(model, probe_X)  # (P,C)
        parts: list[float] = []


        if mode == "top3_diffs":
            # logit gaps among top-k logits for THIS client (class IDs may vary across clients)
            for j in range(P):
                row = logits[j]
                vals = np.sort(row)[::-1]
                if vals.size <= 0:
                    top_vals = np.zeros((probeguard_topk_logits,), dtype=np.float32)
                else:
                    k_eff = min(int(probeguard_topk_logits), int(vals.size))
                    top_vals = np.zeros((probeguard_topk_logits,), dtype=np.float32)
                    top_vals[:k_eff] = vals[:k_eff].astype(np.float32)
                    if k_eff < int(probeguard_topk_logits):
                        top_vals[k_eff:] = float(top_vals[k_eff - 1])

                gaps = top_vals[:-1] - top_vals[1:]
                parts.extend([float(x) for x in gaps])

        elif mode == "global_top3_diffs":
            # (B) anchored diffs using the PREVIOUS GLOBAL model's per-probe top-k class IDs
            assert (
                isinstance(top3_anchors, np.ndarray)
                and top3_anchors.shape[0] == P
                and top3_anchors.shape[1] == int(probeguard_topk_logits)
            )
            for j in range(P):
                idxs = top3_anchors[j].tolist()
                z = []
                prev = 0.0
                for cc in idxs:
                    c_i = int(cc)
                    if 0 <= c_i < logits.shape[1]:
                        prev = float(logits[j, c_i])
                    z.append(prev)

                for t in range(max(0, len(z) - 1)):
                    parts.append(float(z[t] - z[t + 1]))

        elif mode in ("global_top3_centered_logits", "global_top3_probs"):
            # Same global class anchors for EVERY client, preserving class IDs.
            # Only the logit representation changes; Phase-1 MAD and clustering
            # receive the same shape as global_top3_logits.
            shift_invariant = _probeguard_anchored_shift_invariant_features(
                logits, top3_anchors, feature_mode=mode,
            )
            for j in range(P):
                parts.extend(shift_invariant[j].tolist())

        elif mode == "global_top3_logits":
            # (B2) anchored raw logits using the PREVIOUS GLOBAL model's per-probe top-k class IDs
            assert (
                isinstance(top3_anchors, np.ndarray)
                and top3_anchors.shape[0] == P
                and top3_anchors.shape[1] == int(probeguard_topk_logits)
            )
            for j in range(P):
                idxs = top3_anchors[j].tolist()
                z = [
                    float(logits[j, int(cc)]) if 0 <= int(cc) < logits.shape[1] else 0.0
                    for cc in idxs
                ]
                parts.extend([float(v) for v in z])

        else:
            # Pair-anchored modes: iterate probe_pairs iterate probe_pairs
            for j, (a, b) in enumerate(probe_pairs):
                a_i, b_i = int(a), int(b)
                za = float(logits[j, a_i]) if 0 <= a_i < logits.shape[1] else 0.0
                zb = float(logits[j, b_i]) if 0 <= b_i < logits.shape[1] else 0.0

                if mode == "pair_probs":
                    q = _probeguard_pair_probs_from_logits(logits[j : j + 1], a_i, b_i)[0]
                    parts.extend([float(q[0]), float(q[1])])

                elif mode == "margin_conf":
                    # margin from 2-class probs + confidence proxy from logits (max)
                    q = _probeguard_pair_probs_from_logits(logits[j : j + 1], a_i, b_i)[0]
                    margin = float(q[0] - q[1])
                    conf = float(max(za, zb))
                    parts.extend([margin, conf])

                elif mode == "pair_logit_margin_conf":
                    # (A) fully pair-anchored in logit space: (za - zb, max(za, zb))
                    parts.extend([float(za - zb), float(max(za, zb))])

                elif mode == "pair_competitor":
                    # (C) pair + global-anchored competitor: (za - zb, max(za,zb) - zc)
                    c_i = int(comp_anchors[j]) if isinstance(comp_anchors, np.ndarray) and comp_anchors.shape[0] == P else a_i
                    zc = float(logits[j, c_i]) if 0 <= c_i < logits.shape[1] else 0.0
                    parts.extend([float(za - zb), float(max(za, zb) - zc)])

                else:
                    # Safety fallback
                    q = _probeguard_pair_probs_from_logits(logits[j : j + 1], a_i, b_i)[0]
                    parts.extend([float(q[0]), float(q[1])])


        feats.append(parts)

    V = np.array(feats, dtype=np.float32)  # (n, P*F)
    n = int(V.shape[0])
    if n <= 1:
        return {cid: 0.0 for cid in client_ids}, []

    # Per-probe consensus (works for any per-probe feature dimension F)
    if V.shape[1] % P != 0:
        # Unexpected shape; fall back to no suspects
        return {cid: 0.0 for cid in client_ids}, []
    F = int(V.shape[1] // P)
    if F <= 0:
        return {cid: 0.0 for cid in client_ids}, []

    V_resh = V.reshape(n, P, F)
    consensus = np.median(V_resh, axis=0)  # (P,F)

    dists = np.linalg.norm(V_resh - consensus[None, :, :], axis=2)  # (n,P)

    # E1: standardize each probe's disagreement by its peer-to-peer variability.
    # This changes only the MAD scoring features, not the raw signatures used by
    # the optional KMeans/IForest branch. With E1 off, behavior is identical.
    if probeguard_probe_normalize_per_probe:
        probe_med = np.median(dists, axis=0)
        probe_scale = 1.4826 * np.median(np.abs(dists - probe_med[None, :]), axis=0)
        positive_scale = probe_scale[np.isfinite(probe_scale) & (probe_scale > 1e-12)]
        # A relative floor prevents nearly constant probes from overwhelming
        # scores. If all per-probe MADs vanish, use observed nonzero distances
        # to avoid dividing anomalous deviations by the machine-scale epsilon.
        ratio = max(0.0, float(probeguard_probe_scale_floor_ratio))
        positive_dist = dists[np.isfinite(dists) & (dists > 1e-12)]
        reference_scale = (
            float(np.median(positive_scale)) if positive_scale.size else
            (float(np.median(positive_dist)) if positive_dist.size else 0.0)
        )
        scale_floor = max(1e-12, ratio * reference_scale)
        dists = dists / np.maximum(probe_scale, scale_floor)[None, :]

    D_med = np.median(dists, axis=1)
    k_top = max(1, int(math.ceil(float(P) * float(top_frac))))
    D_top = np.mean(np.sort(dists, axis=1)[:, -k_top:], axis=1)

    # Whether to use MAD-based suspecting (default: True)
    try:
        use_mad = bool(run_config.get("probeguard_use_mad"))
    except Exception:
        use_mad = True

    # Robust thresholds (MAD). The scale is also used by E2 to make the
    # different disagreement statistics comparable for suspect ranking.
    def _center_scale_thr(x: np.ndarray, a: float) -> tuple[float, float, float]:
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        scale = 1.4826 * max(mad, 1e-12)
        return med, scale, med + float(a) * scale

    def _thr(x: np.ndarray, a: float) -> float:
        return _center_scale_thr(x, a)[2]


    thr_med = _thr(D_med, mad_alpha)
    thr_top = _thr(D_top, mad_alpha)

    # Probe MAD ablation: use a selected disagreement statistic or the original OR rule.
    # Only active when MAD is enabled; clustering behavior remains independent.
    D_all = None
    if use_mad:
        if probeguard_probe_mad_mode == "med_only":
            suspect_mask = D_med > thr_med
        elif probeguard_probe_mad_mode == "top_only":
            suspect_mask = D_top > thr_top
        elif probeguard_probe_mad_mode == "l2_all":
            # Euclidean distance of the full probe-output signature from its
            # coordinate-wise median (equivalent to flattening all probe residuals).
            D_all = np.linalg.norm(dists, axis=1)
            suspect_mask = D_all > _thr(D_all, mad_alpha)
        else:  # "both": original implementation
            suspect_mask = (D_med > thr_med) | (D_top > thr_top)
    else:
        suspect_mask = np.zeros((n,), dtype=bool)

    iso_scores = None  # used only when MAD disabled + IF is enabled

    # Optional clustering / outlier signal on logit-signatures (PCA space)
    # Controlled by probeguard_cluster_method: "kmeans" | "iforest" | "none".
    # Backward compatibility: if probeguard_cluster_method is empty, falls back to ("kmeans" if use_kmeans else "none").
    method = str(probeguard_cluster_method).lower().strip() if probeguard_cluster_method is not None else ("kmeans" if use_kmeans else "none")
    if method in ("kmeans", "iforest") and n >= 4:
        # PCA to reduce dimension for stability

        pca_var = float(getattr(self, "probeguard_pca_var", 0.95)) if False else None
        # (If you don't have access to self here, pass probeguard_pca_var into the function as an argument instead.)

        # --- guard: if V has ~zero variance, PCA/KMeans is meaningless ---
        if (not np.isfinite(V).all()) or (float(np.nanvar(V)) < 1e-12):
            labels = None  # skip kmeans contribution
        else:
            if 0.0 < probeguard_pca_var < 1.0:
                pca = PCA(n_components=float(probeguard_pca_var), svd_solver="full", random_state=int(sklearn_random_state))
                Vr = pca.fit_transform(V)
            else:
                k = int(min(10, V.shape[1], n))
                Vr = PCA(n_components=k, random_state=int(sklearn_random_state)).fit_transform(V)

        if method == "kmeans":
            km = KMeans(n_clusters=2, n_init=10, random_state=int(sklearn_random_state))
            labels = km.fit_predict(Vr)
            c0 = int(np.sum(labels == 0))
            c1 = int(np.sum(labels == 1))
            # NOTE: "small cluster = suspicious" only makes sense if attackers are a minority.
            small = 0 if c0 < c1 else 1
            if min(c0, c1) >= 1:
                suspect_mask = suspect_mask | (labels == small)

        elif method == "iforest":
            # IsolationForest: outliers are predicted as -1
            # contamination: if <=0 -> "auto", else float in (0, 0.5]
            cont = "auto"
            try:
                if float(probeguard_iforest_contamination) > 0.0:
                    cont = float(probeguard_iforest_contamination)
            except Exception:
                cont = "auto"

            iso = IsolationForest(
                n_estimators=200,
                contamination=cont,
                random_state=int(sklearn_random_state),
            )
            y = iso.fit_predict(Vr)  # -1 outlier, +1 inlier
            if not use_mad:
                try:
                    iso_scores = -iso.score_samples(Vr)  # larger => more anomalous
                except Exception:
                    iso_scores = None
            suspect_mask = suspect_mask | (y == -1)

    # Rank flagged clients by the same statistic used in the selected MAD rule.
    # When MAD is disabled, keep the original max-score / IForest behavior.
    if use_mad and probeguard_probe_mad_mode == "med_only":
        score = D_med
    elif use_mad and probeguard_probe_mad_mode == "top_only":
        score = D_top
    elif use_mad and probeguard_probe_mad_mode == "l2_all":
        score = D_all
    else:  # "both" or MAD disabled: original implementation
        score = np.maximum(D_med, D_top)
    if (not use_mad) and (method == "iforest") and (iso_scores is not None):
        score = iso_scores

    # E2: sort suspects by standardized MAD threshold *exceedance*, rather than
    # comparing raw D_med and D_top magnitudes. Preserve original suspect masks.
    if probeguard_probe_threshold_normalized_ranking and use_mad:
        if probeguard_probe_mad_mode == "med_only":
            score = (D_med - thr_med) / _center_scale_thr(D_med, mad_alpha)[1]
        elif probeguard_probe_mad_mode == "top_only":
            score = (D_top - thr_top) / _center_scale_thr(D_top, mad_alpha)[1]
        elif probeguard_probe_mad_mode == "l2_all":
            if D_all is None:
                D_all = np.linalg.norm(dists, axis=1)
            _, all_scale, all_thr = _center_scale_thr(D_all, mad_alpha)
            score = (D_all - all_thr) / all_scale
        else:  # "both" uses the larger standardized exceedance
            med_excess = (D_med - thr_med) / _center_scale_thr(D_med, mad_alpha)[1]
            top_excess = (D_top - thr_top) / _center_scale_thr(D_top, mad_alpha)[1]
            score = np.maximum(med_excess, top_excess)

    score_by = {client_ids[i]: float(score[i]) for i in range(n)}

    suspects = [client_ids[i] for i in range(n) if bool(suspect_mask[i])]
    # Order by score descending
    suspects = sorted(suspects, key=lambda cid: score_by.get(cid, 0.0), reverse=True)
    return score_by, suspects


def _probeguard_compute_update_based_suspicion(
    *,
    global_sd: OrderedDict,
    client_sds: list[OrderedDict],
    client_ids: list[str],
    mad_alpha: float,
    use_mad: bool = True,
) -> tuple[Dict[str, float], list[str]]:
    """Return (score_by_client, suspect_ids) using MAD on client update vectors.

    Each client update is flattened into a single vector. We then compute a robust
    center (coordinate-wise median), convert each update to a scalar outlier score
    using its L2 distance to that center, and apply MAD on those distances.
    """
    if not client_sds:
        return {}, []

    keys = list(global_sd.keys())
    delta_vecs: list[torch.Tensor] = []
    for sd in client_sds:
        dsd = _sd_sub(sd, global_sd)
        delta_vecs.append(_flatten_sd_torch(dsd, keys).detach().to(torch.float32).cpu())

    n = len(delta_vecs)
    if n <= 0:
        return {}, []

    G = torch.stack(delta_vecs, dim=0)
    try:
        center = torch.median(G, dim=0).values
    except Exception:
        center = torch.mean(G, dim=0)

    dist = torch.norm(G - center[None, :], dim=1, p=2).detach().cpu().numpy().astype(np.float64, copy=False)
    score_by = {client_ids[i]: float(dist[i]) for i in range(n)}

    if (not use_mad) or n <= 1:
        return score_by, []

    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)))
    mad = max(mad, 1e-12)
    thr = med + float(mad_alpha) * 1.4826 * mad

    suspects = [client_ids[i] for i in range(n) if float(dist[i]) > thr]
    suspects = sorted(suspects, key=lambda cid: score_by.get(cid, 0.0), reverse=True)
    return score_by, suspects


def _probeguard_fuse_phase1_scores(
    *,
    probe_scores: Dict[str, float],
    update_scores: Dict[str, float],
    client_ids: list[str],
    probe_weight: float,
    mad_alpha: float,
    use_mad: bool,
) -> tuple[Dict[str, float], list[str]]:
    """E3: independently robust-calibrate probe and update scores, then fuse.

    The fused MAD decision replaces (not OR-combines) the two individual
    suspect lists. Thus the user can compare an actual hybrid detector with
    the original probe-only and update-only modes. It never uses attack labels.
    """
    n = len(client_ids)
    if n == 0:
        return {}, []
    w = float(probe_weight)
    if not (math.isfinite(w) and 0.0 <= w <= 1.0):
        raise ValueError("probeguard_phase1_hybrid_probe_weight must be in [0, 1]")

    def robust_z(scores: Dict[str, float]) -> np.ndarray:
        v = np.array([float(scores.get(cid, 0.0)) for cid in client_ids], dtype=np.float64)
        if not np.isfinite(v).all():
            raise ValueError("Non-finite Phase-1 scores cannot be fused")
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        # Standard deviation fallback keeps a zero-MAD majority from causing
        # extreme weights. This is a scale floor, not a new detector signal.
        scale = max(1.4826 * mad, 0.05 * float(np.std(v)), 1e-12)
        return (v - med) / scale

    zp = robust_z(probe_scores)
    zu = robust_z(update_scores)
    fused = w * zp + (1.0 - w) * zu
    score_by = {cid: float(fused[i]) for i, cid in enumerate(client_ids)}
    if (not use_mad) or n <= 1:
        return score_by, []

    med = float(np.median(fused))
    mad = float(np.median(np.abs(fused - med)))
    scale = max(1.4826 * mad, 0.05 * float(np.std(fused)), 1e-12)
    thr = med + float(mad_alpha) * scale
    suspects = [cid for i, cid in enumerate(client_ids) if float(fused[i]) > thr]
    suspects.sort(key=lambda cid: score_by[cid], reverse=True)
    return score_by, suspects


# (ProbeGuardStrategy class moved below _BaseMonitoredStrategy)


# ======================================================================


class _BaseMonitoredStrategy(FedAvg):
    def __init__(
        self,
        *,
        num_clients_global: int,
        malicious_percentage_global: float,
        malicious_percentage_min: float | None = None,
        malicious_percentage_max: float | None = None,
        attack_type: str = "None",
        # ALIE/IPM (same-round omniscient attacks)
        alie_z: float = -1.0,
        ipm_gamma: float = 20.0,
        ipm_norm_match: bool = True,
        # PoisonedFL (USENIX Security'22) attack params
        poisonedfl_c0: float = 8.0,
        poisonedfl_beta: float = 0.7,
        poisonedfl_e: int = 50,
        poisonedfl_min_c: float = 0.5,
        poisonedfl_pval: float = 0.01,
        poisonedfl_sign_seed: int = 12345,
        # PoisonedFL-Adapt (optional): evade GMM-Sign (alpha) / GMM-Magnitude (gamma)
        poisonedfl_flip_rate: float = 0.0,
        poisonedfl_flip_eps: float = 1e-6,
        poisonedfl_noise_gamma: float = 0.0,
        oracle_drop: bool = False,
        root_trainloader=None,
        root_lr: float = 1e-3,
        fltrust_root_epochs: int = 1,
        fltrust_cos_clip: float = 0.0,
        model_fn=None,
        base_seed: int, 
        deterministic_sampling: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_clients_global = num_clients_global
        self.model_fn = model_fn
        self.malicious_percentage_global = malicious_percentage_global
        # Optional deterministic per-round malicious-percentage range
        self.malicious_percentage_min = (
            float(malicious_percentage_global)
            if malicious_percentage_min is None
            else float(malicious_percentage_min)
        )
        self.malicious_percentage_max = (
            float(malicious_percentage_global)
            if malicious_percentage_max is None
            else float(malicious_percentage_max)
        )
        self.oracle_drop = oracle_drop

        # Attack configuration for coordinated ALIE/IPM and other attack paths.
        self.attack_type = str(attack_type)

        # Same-round omniscient attack hyperparams
        self.alie_z = float(alie_z)
        self.ipm_gamma = float(ipm_gamma)
        self.ipm_norm_match = bool(ipm_norm_match)

        # PoisonedFL attack configuration/state
        self.poisonedfl_c0 = float(poisonedfl_c0)
        self.poisonedfl_beta = float(poisonedfl_beta)
        self.poisonedfl_e = int(poisonedfl_e)
        self.poisonedfl_min_c = float(poisonedfl_min_c)
        self.poisonedfl_pval = float(poisonedfl_pval)
        self.poisonedfl_sign_seed = int(poisonedfl_sign_seed)
        self.poisonedfl_flip_rate = float(poisonedfl_flip_rate)
        self.poisonedfl_flip_eps = float(poisonedfl_flip_eps)
        self.poisonedfl_noise_gamma = float(poisonedfl_noise_gamma)

        # Persistent PoisonedFL state across rounds (server-side forging)
        self._poisonedfl_s: torch.Tensor | None = None
        self._poisonedfl_k_prev: torch.Tensor | None = None
        self._poisonedfl_c: float = float(poisonedfl_c0)
        self._poisonedfl_w_prev: torch.Tensor | None = None
        self._poisonedfl_w_prev2: torch.Tensor | None = None
        self._poisonedfl_w_hist = deque(maxlen=max(2, int(poisonedfl_e) + 1))
        self._poisonedfl_keys: list[str] | None = None
        # FLTrust root training support (only used by FLTrust strategy)
        self.root_trainloader = root_trainloader
        self.root_lr = root_lr

        # cached global params as state_dict (updated each round)
        self._global_sd: dict[str, torch.Tensor] | None = None

        self.fltrust_root_epochs = fltrust_root_epochs
        self.fltrust_cos_clip = fltrust_cos_clip

        self.base_seed = int(base_seed)
        self.deterministic_sampling = bool(deterministic_sampling)
        # node_id -> partition-id mapping (populated once in main via query.whoami)
        self._node_id_to_partition_id: dict[int, str] = {}
        self._dump_dir = None

    def _ensure_full_whoami_map(self, grid: Grid, *, attempts: int = 20, sleep_s: float = 0.1) -> None:
        # Ensure self._node_id_to_partition_id is complete. This is critical for determinism
        # because we order runtime node IDs by stable partition-id before sampling.
        target = int(getattr(self, 'num_clients_global', 0) or 0)
        if target <= 0:
            return
        cur = len(getattr(self, '_node_id_to_partition_id', {}) or {})
        if cur >= target:
            return
        for _ in range(int(max(1, attempts))):
            try:
                m = _build_node_id_to_partition_id_map(grid, timeout=float(getattr(self, 'whoami_timeout', 30.0)))
                if m:
                    self._node_id_to_partition_id.update(m)
            except Exception:
                pass
            cur = len(getattr(self, '_node_id_to_partition_id', {}) or {})
            if cur >= target:
                return
            time.sleep(float(sleep_s))
    def _extra_train_records(self, server_round: int, *, arrays: ArrayRecord, config, grid) -> dict:
        """Hook to attach extra records to each train message.

        Subclasses can override to add extra ArrayRecord/MetricRecord entries
        (e.g., SCAFFOLD global control variate).
        """
        return {}

    @staticmethod
    def _sorted_partition_ids(ids: list[str]) -> list[str]:
        def _key(x: str):
            try:
                return (0, int(x))
            except Exception:
                return (1, str(x))
        return sorted([str(x) for x in ids], key=_key)

    @staticmethod
    def _sort_replies_by_client_id(replies: list[Message]) -> list[Message]:
        def _key(r: Message):
            cid = _reply_client_id(r)
            try:
                return (0, int(cid))
            except Exception:
                return (1, str(cid))
        return sorted(list(replies), key=_key)

    def _require_complete_whoami_map(self, grid: Grid, *, where: str, attempts: int = 20, sleep_s: float = 0.1) -> None:
        self._ensure_full_whoami_map(grid, attempts=attempts, sleep_s=sleep_s)
        target = int(getattr(self, 'num_clients_global', 0) or 0)
        have = len(getattr(self, '_node_id_to_partition_id', {}) or {})
        if target > 0 and have < target:
            raise RuntimeError(
                f"Deterministic run aborted: incomplete whoami mapping in {where} (have {have}, need {target})."
            )

    def _validate_round_replies(self, server_round: int, replies, *, where: str, expected_attr: str) -> list[Message]:
        replies = _filter_replies_with_content(list(replies), server_round=server_round, where=where)
        expected_map = getattr(self, expected_attr, {}) if hasattr(self, expected_attr) else {}
        expected = expected_map.get(int(server_round)) if isinstance(expected_map, dict) else None
        if expected is not None:
            expected_ids = self._sorted_partition_ids(list(expected))
            actual_ids = self._sorted_partition_ids([_reply_client_id(r) for r in replies])
            if actual_ids != expected_ids:
                raise RuntimeError(
                    f"Deterministic run aborted: reply set mismatch in {where} for round {server_round}. "
                    f"Expected partition-ids {expected_ids}, got {actual_ids}."
                )
        return self._sort_replies_by_client_id(replies)

    def _validate_train_replies(self, server_round: int, replies) -> list[Message]:
        return self._validate_round_replies(
            server_round,
            replies,
            where="train",
            expected_attr="_selected_partition_ids_by_round",
        )

    def _validate_evaluate_replies(self, server_round: int, replies) -> list[Message]:
        return self._validate_round_replies(
            server_round,
            replies,
            where="evaluate",
            expected_attr="_selected_eval_partition_ids_by_round",
        )

    def configure_train(self, server_round: int, arrays: ArrayRecord, config, grid):
        """Deterministic per-round client sampling (by partition-id) and message creation."""
        # Track progress for crash-aware final logging.
        global _LAST_SERVER_ROUND
        _LAST_SERVER_ROUND = int(server_round)

        self._global_sd = arrays.to_torch_state_dict()

        # Update PoisonedFL global-model history (needed for g^{t-1} and hypothesis test)
        if self.attack_type == 'PoisonedFL':
            if self._poisonedfl_keys is None:
                self._poisonedfl_keys = list(self._global_sd.keys())
            w_flat = _flatten_sd_torch(self._global_sd, self._poisonedfl_keys)
            self._poisonedfl_w_prev2 = self._poisonedfl_w_prev
            self._poisonedfl_w_prev = w_flat
            self._poisonedfl_w_hist.append(w_flat)

        # Ensure node_id -> partition-id mapping is complete (required to map deterministic partitions to runtime nodes).
        self._require_complete_whoami_map(grid, where="configure_train")

        # Deterministically reproduce sampled partition-ids for this round (must match ClientApp).
        fraction_fit = float(getattr(self, '_fraction_fit_for_sampling', getattr(self, 'fraction_train', 1.0)))
        selected_pids = sampled_partition_ids_for_round(
            int(server_round),
            int(self.num_clients_global),
            float(fraction_fit),
            base_seed=int(self.base_seed),
            seed_offset=0,
        )

        # Build deterministic partition-id -> node_id mapping (pick smallest node_id if duplicates).
        pid_to_node: dict[str, int] = {}
        for nid, pid in (getattr(self, '_node_id_to_partition_id', {}) or {}).items():
            try:
                pid_s = str(int(pid))
            except Exception:
                pid_s = str(pid)
            nid_i = int(nid)
            if pid_s not in pid_to_node or nid_i < pid_to_node[pid_s]:
                pid_to_node[pid_s] = nid_i

        # If mapping is still incomplete, retry and fail fast instead of falling back to
        # runtime node-id sampling, which would break determinism.
        if any(pid not in pid_to_node for pid in selected_pids):
            self._require_complete_whoami_map(grid, where="configure_train.retry", attempts=50, sleep_s=0.05)
            pid_to_node.clear()
            for nid, pid in (getattr(self, '_node_id_to_partition_id', {}) or {}).items():
                try:
                    pid_s = str(int(pid))
                except Exception:
                    pid_s = str(pid)
                nid_i = int(nid)
                if pid_s not in pid_to_node or nid_i < pid_to_node[pid_s]:
                    pid_to_node[pid_s] = nid_i

        missing = [pid for pid in selected_pids if pid not in pid_to_node]
        if missing:
            raise RuntimeError(
                f"Deterministic run aborted: missing partition-id to node-id mapping for round {server_round}: {missing}"
            )

        # Create train messages for the deterministically selected runtime nodes.
        cfg = MetricRecord({'server-round': int(server_round)})

        extra = {}
        try:
            extra = self._extra_train_records(int(server_round), arrays=arrays, config=config, grid=grid) or {}
        except Exception:
            extra = {}

        rd = {'arrays': arrays, 'config': cfg}
        try:
            rd.update(dict(extra))
        except Exception:
            pass
        content = RecordDict(rd)
        group_id = f'train.{int(server_round)}'
        msgs: list[Message] = []
        for pid in selected_pids:
            nid = int(pid_to_node[pid])
            msg = grid.create_message(
                content=content,
                message_type='train',
                dst_node_id=nid,
                group_id=group_id,
            )
            msgs.append(msg)

        # Cache selection (partition-ids) for other components (oracle/attacks/logging).
        if not hasattr(self, '_selected_partition_ids_by_round'):
            self._selected_partition_ids_by_round = {}
        self._selected_partition_ids_by_round[int(server_round)] = list(selected_pids)

        print(f'[Round {server_round}] selected_clients_sorted={selected_pids}', flush=True)
        return msgs


    def _filter_oracle(self, server_round: int, results: list[Message]) -> list[Message]:
        """If oracle_drop=True, remove known-malicious clients for this round."""
        if not self.oracle_drop:
            return results

        # task.py returns IDs as strings: {"0","5",...}
        # Oracle-drop should remove the (deterministically chosen) malicious subset *of this round's selected clients*.
        selected_ids = None
        if hasattr(self, "_selected_partition_ids_by_round"):
            selected_ids = self._selected_partition_ids_by_round.get(int(server_round))
        if selected_ids is None:
            selected_ids = [str(_reply_client_id(r)) for r in results]

        mal_ids = malicious_ids_for_round(
            int(server_round),
            int(self.num_clients_global),
            float(self.malicious_percentage_global),
            seed_offset=0,
            selected_ids=list(selected_ids),
            mal_pct_min=float(getattr(self, "malicious_percentage_min", self.malicious_percentage_global)),
            mal_pct_max=float(getattr(self, "malicious_percentage_max", self.malicious_percentage_global)),
            base_seed=int(self.base_seed),
        )

        kept: list[Message] = []
        for reply in results:
            cid = _reply_client_id(reply)   # should be "0","1",...
            if str(cid) in mal_ids:
                continue
            kept.append(reply)

        return kept

    def _maybe_forge_alie_ipm(
        self,
        server_round: int,
        replies: list[Message],
        global_sd: OrderedDict,
        client_sds: list[OrderedDict],
    ) -> list[OrderedDict]:
        """Apply coordinated ALIE/IPM forging; leave other attacks unchanged."""
        if getattr(self, "attack_type", None) not in {"ALIE", "IPM"}:
            return client_sds

        client_ids = [_reply_client_id(r) for r in replies]

        selected_ids = None
        if hasattr(self, "_selected_partition_ids_by_round"):
            selected_ids = self._selected_partition_ids_by_round.get(int(server_round))
        if selected_ids is None:
            selected_ids = [str(cid) for cid in client_ids]

        mal_ids = set(
            malicious_ids_for_round(
                int(server_round),
                int(self.num_clients_global),
                float(self.malicious_percentage_global),
                seed_offset=0,
                selected_ids=list(selected_ids),
                mal_pct_min=float(getattr(self, "malicious_percentage_min", self.malicious_percentage_global)),
                mal_pct_max=float(getattr(self, "malicious_percentage_max", self.malicious_percentage_global)),
                base_seed=int(self.base_seed),
            )
        )

        return forge_alie_ipm_same_round(
            attack_type=self.attack_type,
            global_sd=global_sd,
            client_sds=client_sds,
            client_ids=client_ids,
            malicious_ids=mal_ids,
            alie_z=getattr(self, "alie_z", -1.0),
            ipm_gamma=getattr(self, "ipm_gamma", 20.0),
            ipm_norm_match=bool(getattr(self, "ipm_norm_match", True)),
        )



    def _maybe_forge_poisonedfl(
        self,
        server_round: int,
        replies: List[Message],
        global_sd: OrderedDict,
        client_sds: list[OrderedDict],
    ) -> list[OrderedDict]:
        """Server-side PoisonedFL forging (USENIX Security '22).

        Simulates fake clients with coordinated malicious updates:
            g_i^t = k^t ⊙ s
        where s is a fixed random sign vector and k^t adapts across rounds.
        """
        if getattr(self, "attack_type", None) != "PoisonedFL":
            return client_sds

        client_ids = [_reply_client_id(r) for r in replies]

        selected_ids = None
        if hasattr(self, "_selected_partition_ids_by_round"):
            selected_ids = self._selected_partition_ids_by_round.get(int(server_round))
        if selected_ids is None:
            selected_ids = [str(cid) for cid in client_ids]

        mal_ids = set(
            malicious_ids_for_round(
                int(server_round),
                int(self.num_clients_global),
                float(self.malicious_percentage_global),
                seed_offset=0,
                selected_ids=list(selected_ids),
                mal_pct_min=float(getattr(self, "malicious_percentage_min", self.malicious_percentage_global)),
                mal_pct_max=float(getattr(self, "malicious_percentage_max", self.malicious_percentage_global)),
                base_seed=int(self.base_seed),
            )
        )

        # Only forge if at least one malicious client is present in replies
        if not any(cid in mal_ids for cid in client_ids):
            return client_sds

        # Need g^{t-1} = w_{t-1} - w_{t-2}
        if self._poisonedfl_w_prev is None or self._poisonedfl_w_prev2 is None:
            return client_sds

        keys = self._poisonedfl_keys or list(global_sd.keys())
        self._poisonedfl_keys = keys

        w_prev = self._poisonedfl_w_prev.detach().to(torch.float32).cpu()
        w_prev2 = self._poisonedfl_w_prev2.detach().to(torch.float32).cpu()
        g_prev = (w_prev - w_prev2)

        eps = 1e-12
        g_norm = float(g_prev.norm(p=2).item())
        if g_norm <= eps:
            return client_sds

        # Init fixed random sign vector s once (shared across fake clients)
        if self._poisonedfl_s is None or int(self._poisonedfl_s.numel()) != int(g_prev.numel()):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(self.poisonedfl_sign_seed))
            r = torch.randn(int(g_prev.numel()), generator=gen)
            self._poisonedfl_s = torch.where(r >= 0, torch.ones_like(r), -torch.ones_like(r))
            self._poisonedfl_k_prev = None
            self._poisonedfl_c = float(self.poisonedfl_c0)

        s = self._poisonedfl_s

        # Update c^t via hypothesis test (paper): if Pr(x >= X) <= p keep c, else shrink c <- beta*c.
        c = float(self._poisonedfl_c)
        e = int(self.poisonedfl_e)
        if e > 0 and len(self._poisonedfl_w_hist) >= e:
            # paper uses w_{t-1} - w_{t-e} as the total aggregated update in past e rounds
            w_te = self._poisonedfl_w_hist[-e].detach().to(torch.float32).cpu()
            total_delta = (w_prev - w_te)
            X = int((torch.sign(total_delta) == s).sum().item())
            d = int(s.numel())
            p_tail = float(binom.sf(X - 1, d, 0.5))  # Pr(x >= X)
            if p_tail > float(self.poisonedfl_pval):
                c = max(float(self.poisonedfl_min_c), float(self.poisonedfl_beta) * c)

        self._poisonedfl_c = c

        # Compute v^t. If k^{t-1} missing: v^t = |g^{t-1}| / || |g^{t-1}| ||.
        if self._poisonedfl_k_prev is None:
            v_raw = g_prev.abs()
            v = v_raw / (v_raw.norm(p=2) + eps)
        else:
            k_prev = self._poisonedfl_k_prev.detach().to(torch.float32).cpu()
            signed_k_prev = s * k_prev
            if float(signed_k_prev.norm(p=2).item()) <= eps:
                v_raw = g_prev.abs()
                v = v_raw / (v_raw.norm(p=2) + eps)
            else:
                # v^t ∝ | g^{t-1} - (k^{t-1}⊙s)/||k^{t-1}⊙s|| * ||g^{t-1}|| |
                est = g_prev - (signed_k_prev / (signed_k_prev.norm(p=2) + eps)) * g_prev.norm(p=2)
                v_raw = est.abs()
                v = v_raw / (v_raw.norm(p=2) + eps)

        # λ^t = c^t * ||g^{t-1}||, k^t = λ^t v^t, delta = k^t ⊙ s
        lam = float(c) * g_prev.norm(p=2)
        k = lam * v
        self._poisonedfl_k_prev = k.detach()
        delta_base = k * s

        alpha = float(self.poisonedfl_flip_rate)
        gamma = float(self.poisonedfl_noise_gamma)

        # Base PoisonedFL: same malicious update for all malicious clients
        if alpha <= 0.0 and gamma <= 0.0:
            delta_sd = _unflatten_vec_to_delta_sd(delta_base, global_sd, keys)
            forged_sd = _sd_add(global_sd, delta_sd)
            for i, cid in enumerate(client_ids):
                if cid in mal_ids:
                    client_sds[i] = forged_sd
            return client_sds

        # PoisonedFL-Adapt: per-client sign flips (alpha) and/or magnitude noise (gamma)
        for i, cid in enumerate(client_ids):
            if cid not in mal_ids:
                continue

            try:
                cid_int = int(cid)
            except Exception:
                cid_int = abs(hash(cid)) % 1_000_000_007

            seed = int(self.base_seed) + 99991 * int(server_round) + cid_int
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)

            s_i = s
            k_i = k

            if alpha > 0.0:
                mask = torch.rand(int(s.numel()), generator=gen) < alpha
                s_i = torch.where(mask, -s, s)
                k_i = k.clone()
                k_i[mask] = float(self.poisonedfl_flip_eps)

            if gamma > 0.0:
                eps_vec = torch.randn(int(s.numel()), generator=gen)
                eps_norm = eps_vec.norm(p=2) + eps
                k_norm = k_i.norm(p=2) + eps
                k_i = k_i + float(gamma) * k_norm * (eps_vec / eps_norm)

            delta_i = k_i * s_i
            delta_sd = _unflatten_vec_to_delta_sd(delta_i, global_sd, keys)
            client_sds[i] = _sd_add(global_sd, delta_sd)

        return client_sds


    def configure_evaluate(self, server_round: int, arrays, config, grid):
        # Keep evaluation sampling deterministic as well (if enabled).
        if self.deterministic_sampling:
            self._require_complete_whoami_map(grid, where="configure_evaluate")
            round_seed = int(self.base_seed) + 100_000 * int(server_round) + 33_333

            orig_get_node_ids = getattr(grid, "get_node_ids", None)
            if callable(orig_get_node_ids):
                def _sorted_get_node_ids():
                    try:
                        node_ids = list(orig_get_node_ids())
                        if getattr(self, "_node_id_to_partition_id", None):
                            m = self._node_id_to_partition_id

                            def _k(nid):
                                pid = m.get(int(nid))
                                if pid is None:
                                    return (1, int(nid))
                                try:
                                    return (0, int(pid))
                                except Exception:
                                    return (0, str(pid))

                            return sorted(node_ids, key=_k)
                        return sorted(node_ids)
                    except Exception:
                        return orig_get_node_ids()
                try:
                    setattr(grid, "get_node_ids", _sorted_get_node_ids)
                except Exception:
                    orig_get_node_ids = None

            try:
                with _temp_round_seed(round_seed):
                    eval_ins = super().configure_evaluate(server_round, arrays, config, grid)
            finally:
                if callable(orig_get_node_ids):
                    try:
                        setattr(grid, "get_node_ids", orig_get_node_ids)
                    except Exception:
                        pass

            selected = _extract_selected_partition_ids(eval_ins, self._node_id_to_partition_id)
            if not hasattr(self, '_selected_eval_partition_ids_by_round'):
                self._selected_eval_partition_ids_by_round = {}
            self._selected_eval_partition_ids_by_round[int(server_round)] = list(selected)
            return eval_ins

        eval_ins = super().configure_evaluate(server_round, arrays, config, grid)
        selected = _extract_selected_partition_ids(eval_ins, self._node_id_to_partition_id)
        if not hasattr(self, '_selected_eval_partition_ids_by_round'):
            self._selected_eval_partition_ids_by_round = {}
        self._selected_eval_partition_ids_by_round[int(server_round)] = list(selected)
        return eval_ins


    def aggregate_evaluate(self, server_round: int, replies):
        replies = self._validate_evaluate_replies(server_round, replies)
        if not replies:
            return None

        metrics_avg = super().aggregate_evaluate(server_round, replies)
        if metrics_avg is None:
            return None

        # Remove per-client identifier fields from the aggregated metrics
        try:
            metrics_avg.pop("client_id", None)
            metrics_avg.pop("partition-id", None)
            metrics_avg.pop("partition_id", None)
        except Exception:
            pass

        acc = metrics_avg.get("accuracy", None)
        loss = metrics_avg.get("loss", None)

        if acc is not None:
            mal_ids = malicious_ids_for_round(
                server_round,
                self.num_clients_global,
                self.malicious_percentage_global,
                seed_offset=0,
            )
            # if loss is None:
            #     print(f"[Round {server_round}] Eval acc={float(acc):.4f} | malicious_ids={mal_ids}")
            # else:
            #     print(f"[Round {server_round}] Eval loss={float(loss):.4f} acc={float(acc):.4f} | malicious_ids={mal_ids}")

            recent_accuracies.append(float(acc))
            if len(recent_accuracies) > 200:
                del recent_accuracies[:-200]

        return metrics_avg


    # subclasses override this
    def aggregate_train(self, server_round: int, replies, failures):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# GAS (ICML 2023): GrAdient Splitting, applied to local model DELTAS.
# Paper: Liu et al., Byzantine-Robust Learning on Heterogeneous Data via
# Gradient Splitting.  Original reference: YuchenLiu-a/byzantine-gas.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _gas_dnc_reference(
    group: torch.Tensor,
    *,
    f: int,
    c: float,
    niters: int,
    b: int,
    seed: int,
) -> torch.Tensor:
    """DnC base AGR on one GAS subvector group; return the robust mean.

    Matches the selection rule in the existing standalone ``agg_dnc``:
    random coordinates -> centered SVD -> squared top-direction projection
    scores -> keep n-ceil(c*f) -> intersect across iterations -> mean.
    The standalone DnC function and its configuration are not changed.
    """
    if group.ndim != 2 or group.shape[0] == 0 or group.shape[1] == 0:
        raise ValueError("GAS(DnC) requires a nonempty [clients, dimensions] group.")
    if not bool(torch.isfinite(group).all()):
        raise ValueError("GAS(DnC) received non-finite split updates.")
    if not math.isfinite(float(c)) or float(c) <= 0.0:
        raise ValueError("GAS(DnC) requires dnc_c > 0 and finite.")
    if int(niters) < 1 or int(b) < 1:
        raise ValueError("GAS(DnC) requires dnc_niters >= 1 and dnc_b >= 1.")

    n, dim = (int(v) for v in group.shape)
    if not 0 <= int(f) < n:
        raise ValueError(f"GAS(DnC) requires 0 <= f < n, got f={f}, n={n}.")
    # Identical (n - ceil(c * f)) selection rule and clipping as agg_dnc.
    keep_count = max(1, min(n, n - int(math.ceil(float(c) * int(f)))))
    if keep_count == n:
        return group.mean(dim=0)

    X = group.detach().cpu().numpy().astype(np.float64, copy=False)
    rng = np.random.default_rng(int(seed))
    keep = set(range(n))
    last_nonempty_keep = keep.copy()
    # When b >= dim every iteration sees ALL dimensions; repeating SVD would
    # produce exactly the same keep-set and wastes up to 3x the CPU time.
    n_effective_iters = 1 if int(b) >= dim else int(niters)
    for _ in range(n_effective_iters):
        subdim = min(int(b), dim)
        columns = np.sort(rng.choice(dim, size=subdim, replace=False))
        Y = X[:, columns]
        Y = Y - Y.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(Y, full_matrices=False)
        except np.linalg.LinAlgError:
            break  # same guard as standalone DnC
        if vt.size == 0:
            break
        scores = (Y @ vt[0]) ** 2
        if not np.isfinite(scores).all():
            raise FloatingPointError("GAS(DnC) projection produced non-finite scores.")
        chosen = set(np.argsort(scores, kind="stable")[:keep_count].tolist())
        next_keep = keep.intersection(chosen)
        if not next_keep:
            break  # retain last nonempty keep-set, as in standalone DnC
        keep = next_keep
        last_nonempty_keep = keep.copy()

    kept = sorted(last_nonempty_keep)
    # DnC's internal reference is an UNWEIGHTED mean of its kept sub-vectors;
    # GAS still performs its own n-f selection and full-update mean afterwards.
    result = X[kept].mean(axis=0)
    return torch.as_tensor(result, dtype=group.dtype, device=group.device)


@torch.no_grad()
def agg_gas(
    global_sd: "OrderedDict[str, torch.Tensor]",
    client_sds: list["OrderedDict[str, torch.Tensor]"],
    *,
    f: int,
    p: int = 1000,
    base_agr: str = "median",
    seed: int = 42,
    dnc_c: float = 1.0,
    dnc_niters: int = 1,
    dnc_b: int = 2000,
) -> tuple["OrderedDict[str, torch.Tensor]", list[int], list[float]]:
    """Randomly split flattened DELTAS, score via group AGR, mean n-f full deltas.

    Scores are the SUM of L2 distances to p groupwise robust references, per
    Eq.(7)-(11) of GAS. For base_agr='dnc', group references use the existing
    standalone DnC selection rule with configurable c/niters/b. Aggregation
    of selected full updates is UNWEIGHTED, as in the paper. This is a
    multi-local-epoch FL adaptation of gradient GAS.
    """
    n = len(client_sds)
    if n <= 0:
        raise ValueError("GAS requires at least one client update.")
    keys = list(global_sd)
    float_keys = [key for key in keys if torch.is_floating_point(global_sd[key])]
    if not float_keys:
        raise ValueError("GAS needs at least one floating-point model parameter.")
    for sd in client_sds:
        if set(sd) != set(keys):
            raise ValueError("GAS client/global parameter keys differ.")

    # Ignore integral tracking counters (e.g. BatchNorm num_batches_tracked):
    # they are not gradients or trainable parameter deltas.
    updates = torch.stack([_flatten_sd_torch(_sd_sub(sd, global_sd), float_keys)
                           for sd in client_sds], dim=0)
    if not bool(torch.isfinite(updates).all()):
        raise ValueError("GAS received non-finite client updates.")
    dim = int(updates.shape[1])
    if dim == 0:
        raise ValueError("GAS requires a nonempty model update.")
    if not 0 <= int(f) < n:
        raise ValueError(f"GAS requires 0 <= f < n; got f={f} n={n}.")
    if int(p) < 1:
        raise ValueError("gas_p must be >= 1.")
    p = min(int(p), dim)
    agr = str(base_agr).strip().lower()
    if agr not in {"median", "trimmed_mean", "multi_krum", "dnc"}:
        raise ValueError("gas_base_agr must be median, trimmed_mean, multi_krum or dnc.")

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(dim, generator=gen)
    scores = torch.zeros(n, dtype=torch.float64)
    for group_idx, index in enumerate(torch.tensor_split(perm, p)):
        group = updates[:, index]
        if agr == "median":
            reference = group.median(dim=0).values
        elif agr == "trimmed_mean":
            cut = min(int(f), (n - 1) // 2)
            sorted_group = group.sort(dim=0).values
            reference = sorted_group[cut:n-cut].mean(dim=0)
        elif agr == "dnc":
            reference = _gas_dnc_reference(
                group, f=int(f), c=float(dnc_c), niters=int(dnc_niters),
                b=int(dnc_b), seed=int(seed) + 101_003 * (group_idx + 1),
            )
        else:  # GAS with Multi-Krum as the subvector reference.
            f_krum = min(int(f), max(0, (n - 3) // 2))
            distances = torch.cdist(group, group, p=2).square()
            neighbors = max(1, n - f_krum - 2)
            krum_scores = distances.sort(dim=1).values[:, 1:1+neighbors].sum(dim=1)
            selected_ref = torch.argsort(krum_scores, stable=True)[: n - f_krum]
            reference = group[selected_ref].mean(dim=0)
        scores += torch.linalg.vector_norm(group - reference, ord=2, dim=1).double()

    # Stable client order resolves ties and preserves repeatability.
    selected = torch.argsort(scores, stable=True)[:n-int(f)].tolist()
    mean_update = updates[selected].mean(dim=0)
    out = OrderedDict()
    offset = 0
    for key in keys:
        value = global_sd[key].detach().cpu()
        if key in float_keys:
            count = value.numel()
            out[key] = value + mean_update[offset:offset+count].reshape(value.shape).to(value.dtype)
            offset += count
        else:
            out[key] = value.clone()
    return out, selected, scores.tolist()


# ---------------------------------------------------------------------------
# SDEA (ICML 2024): Self-Driven Entropy Aggregation.
# Paper: Huang et al. Official reference: WenkeHuang/SDEA.
# ---------------------------------------------------------------------------
def _sdea_finch_first_partition(weights: np.ndarray) -> np.ndarray:
    """FINCH first-partition connectivity on 1-D weights (no external package).

    Uses first-neighbor and shared-first-neighbor links. This is the first
    FINCH partition, NOT K-Means and not an oracle Byzantine partition.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = len(w)
    if n < 2:
        return np.zeros(n, dtype=np.int64)
    dist = np.abs(w[:, None] - w[None, :])
    np.fill_diagonal(dist, np.inf)
    nearest = np.argmin(dist, axis=1)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    first_owner = {}
    for i, j in enumerate(nearest):
        union(i, int(j))
        j = int(j)
        if j in first_owner:
            union(i, first_owner[j])
        else:
            first_owner[j] = i
    roots = [find(i) for i in range(n)]
    lookup = {root: idx for idx, root in enumerate(sorted(set(roots)))}
    return np.asarray([lookup[r] for r in roots], dtype=np.int64)


def _sdea_cooperative_cluster(weights: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """Paper Eq.(7): equalize the mass of all groups other than minimum-mean group."""
    w = weights.detach().cpu().double().numpy().copy()
    partitions = _sdea_finch_first_partition(w)
    labels = np.unique(partitions)
    if len(labels) <= 1:
        return weights.detach().clone(), []
    min_cluster = min(labels.tolist(), key=lambda label: (float(w[partitions == label].mean()), int(label)))
    bad_indices = np.flatnonzero(partitions == min_cluster).tolist()
    good = partitions != min_cluster
    if not np.any(good):
        return weights.detach().clone(), []
    w[good] = float(w[good].sum()) / int(good.sum())
    # Preserve malicious-group weights, precisely as SDEA Eq.(7).
    result = torch.tensor(w, dtype=weights.dtype, device=weights.device)
    result = result / result.sum()
    return result, bad_indices


def agg_sdea(
    global_sd: "OrderedDict[str, torch.Tensor]",
    client_sds: list["OrderedDict[str, torch.Tensor]"],
    *,
    public_X: torch.Tensor,
    model_fn,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 0.005,
    use_cc: bool = True,
    seed: int = 42,
    device: str = "cpu",
) -> tuple["OrderedDict[str, torch.Tensor]", list[float], list[int]]:
    """Optimize softmax mixture weights using IS+CD on unlabeled public inputs.

    Unlike client-logit ensembling, parameter mixing happens BEFORE the forward
    pass, matching SDEA Eq.(3)-(6). Optimizes only aggregation logits M, never
    model parameters; CC applies a FINCH first partition at round end.
    """
    from torch.func import functional_call

    n = len(client_sds)
    if n <= 0:
        raise ValueError("SDEA requires at least one client model.")
    if epochs < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("SDEA epochs/batch_size/lr must all be positive.")
    if public_X is None or public_X.ndim != 2 or public_X.shape[0] == 0:
        raise ValueError("SDEA requires a nonempty 2-D unlabeled public-input tensor.")
    if not torch.isfinite(public_X).all():
        raise ValueError("SDEA public data contain non-finite values.")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("sdea_device='cuda' requested but CUDA is unavailable.")

    # Do not mutate the shared model or the client state dictionaries.
    with _temp_round_seed(int(seed)):
        model = model_fn().to(dev).eval()
    keys = list(model.state_dict())
    if set(keys) != set(global_sd):
        raise ValueError("SDEA server model keys do not match global state.")
    for sd in client_sds:
        if set(sd) != set(global_sd):
            raise ValueError("SDEA client/global state keys differ.")
    buffer_keys = set(dict(model.named_buffers()))
    # Stack each floating parameter and buffer ONCE; make no graph for clients.
    stacks = {}
    non_float = {}
    for key in keys:
        ref = global_sd[key]
        if torch.is_floating_point(ref):
            stacks[key] = torch.stack([sd[key].detach().to(device=dev, dtype=ref.dtype)
                                       for sd in client_sds], dim=0)
            if not bool(torch.isfinite(stacks[key]).all()):
                raise ValueError(f"SDEA non-finite client state tensor: {key}")
        else:
            non_float[key] = ref.detach().to(device=dev).clone()

    X = public_X.detach().to(device=dev, dtype=torch.float32)
    M = torch.nn.Parameter(torch.full((n,), 1.0 / n, device=dev, dtype=torch.float32))
    optimizer = torch.optim.Adam([M], lr=float(lr), weight_decay=1e-5)
    perm_gen = torch.Generator(device="cpu").manual_seed(int(seed) + 31)
    # SDEA's original implementation starts new learnable weights each round.
    for _ in range(int(epochs)):
        permutation = torch.randperm(int(X.shape[0]), generator=perm_gen)
        for indices in permutation.split(int(batch_size)):
            batch = X[indices.to(device=dev)]
            coeff = torch.softmax(M, dim=0)
            combined = {}
            for key in keys:
                if key in stacks:
                    value = stacks[key]
                    # BatchNorm running statistics cannot accept gradients with
                    # respect to the statistics themselves in eval mode. Mix
                    # them in value, but differentiate only model parameters.
                    c = coeff.detach() if key in buffer_keys else coeff
                    combined[key] = torch.tensordot(c.to(value.dtype), value, dims=([0], [0]))
                else:
                    # e.g. BatchNorm's integral num_batches_tracked buffer.
                    combined[key] = non_float[key]
            output = functional_call(model, combined, (batch,))
            if isinstance(output, dict):
                output = output["logits"]
            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                raise ValueError("SDEA model must produce a [batch, classes] logit tensor")
            logp = torch.log_softmax(output, dim=-1)
            probs = logp.exp()
            # Eq.(4) per-example entropy, Eq.(5) minus entropy of batch mean.
            # Both terms use the same entropy normalization; the common factor
            # does not change the optimum and preserves original repo scaling.
            instance_entropy = -(probs * logp).sum(dim=-1).mean()
            batch_mean = probs.mean(dim=0).clamp_min(1e-12)
            class_diversity = (batch_mean * batch_mean.log()).sum()
            loss = instance_entropy + class_diversity
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("SDEA entropy objective became non-finite.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        weights = torch.softmax(M, dim=0)
        cc_excluded = []
        if use_cc:
            weights, cc_excluded = _sdea_cooperative_cluster(weights)
        if not bool(torch.isfinite(weights).all()):
            raise FloatingPointError("SDEA produced non-finite aggregation weights.")
        out = OrderedDict()
        for key in keys:
            if key in stacks:
                merged = torch.tensordot(weights.to(stacks[key].dtype), stacks[key], dims=([0], [0]))
                out[key] = merged.detach().to(device="cpu", dtype=global_sd[key].dtype)
            else:
                out[key] = non_float[key].detach().to(device="cpu", dtype=global_sd[key].dtype)
    return out, weights.detach().cpu().tolist(), cc_excluded



class FedAvgStrategy(_BaseMonitoredStrategy):
    def aggregate_train(self, server_round: int, replies):
        replies = self._validate_train_replies(server_round, replies)
        if not replies:
            return (None, None)

        global_sd = self._global_sd
        assert global_sd is not None

        client_sds = [_reply_to_state_dict(r) for r in replies]
        client_sds = self._maybe_forge_alie_ipm(server_round, replies, global_sd, client_sds)
        client_sds = self._maybe_forge_poisonedfl(server_round, replies, global_sd, client_sds)
        weights = [_reply_num_examples(r) for r in replies]

        deltas = _to_deltas(global_sd, client_sds)
        delta_new = agg_fedavg_delta(deltas, weights)
        new_sd = _apply_delta(global_sd, delta_new)

        new_params = ArrayRecord(new_sd)
        return new_params, MetricRecord({"round": server_round})


class MedianStrategy(_BaseMonitoredStrategy):
    def aggregate_train(self, server_round: int, replies):
        replies = self._validate_train_replies(server_round, replies)
        if not replies:
            return (None, None)

        global_sd = self._global_sd
        assert global_sd is not None

        client_sds = [_reply_to_state_dict(r) for r in replies]
        client_sds = self._maybe_forge_alie_ipm(server_round, replies, global_sd, client_sds)
        client_sds = self._maybe_forge_poisonedfl(server_round, replies, global_sd, client_sds)

        deltas = _to_deltas(global_sd, client_sds)
        delta_new = agg_median(deltas)

        new_sd = _apply_delta(global_sd, delta_new)

        new_params = ArrayRecord(new_sd)
        return new_params, MetricRecord({"round": server_round})


class TrimmedMeanStrategy(_BaseMonitoredStrategy):
    def __init__(self, *, trim_ratio: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.trim_ratio = trim_ratio

    def aggregate_train(self, server_round: int, replies):
        replies = self._validate_train_replies(server_round, replies)
        if not replies:
            return (None, None)

        global_sd = self._global_sd
        assert global_sd is not None

        client_sds = [_reply_to_state_dict(r) for r in replies]
        client_sds = self._maybe_forge_alie_ipm(server_round, replies, global_sd, client_sds)
        client_sds = self._maybe_forge_poisonedfl(server_round, replies, global_sd, client_sds)

        deltas = _to_deltas(global_sd, client_sds)
        delta_new = agg_trimmed_mean(deltas, self.trim_ratio)
        new_sd = _apply_delta(global_sd, delta_new)

        new_params = ArrayRecord(new_sd)
        return new_params, MetricRecord({"round": server_round})


class KrumStrategy(_BaseMonitoredStrategy):
    def __init__(self, *, krum_f: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.krum_f = krum_f

    def aggregate_train(self, server_round: int, replies):
        replies = self._validate_train_replies(server_round, replies)
        if not replies:
            return (None, None)

        global_sd = self._global_sd
        assert global_sd is not None

        client_sds = [_reply_to_state_dict(r) for r in replies]
        client_sds = self._maybe_forge_alie_ipm(server_round, replies, global_sd, client_sds)
        client_sds = self._maybe_forge_poisonedfl(server_round, replies, global_sd, client_sds)
        new_sd = agg_krum(global_sd, client_sds, f=self.krum_f)

        new_params = ArrayRecord(new_sd)
        return new_params, MetricRecord({"round": server_round})


class MultiKrumStrategy(_BaseMonitoredStrategy):
    def __init__(self, *, krum_f: int = 1, multi_krum_m: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.krum_f = krum_f
        self.multi_krum_m = multi_krum_m

    def _effective_multi_krum_m(self, replies) -> int:
        """
        If multi_krum_m >= 0:
            use the configured value.
        If multi_krum_m < 0:
            automatically set m using only the pessimistic malicious rate:
                m = n - ceil(n * Malicious-Client-Percentage)

        This does NOT use exact malicious IDs.
        """
        if int(self.multi_krum_m) >= 0:
            return int(self.multi_krum_m)

        n = int(len(replies))
        if n <= 0:
            return 1

        mal_pct = float(self.malicious_percentage_global)
        mal_pct = max(0.0, min(1.0, mal_pct))

        estimated_malicious = int(math.ceil(n * mal_pct))
        m_auto = max(1, n - estimated_malicious)

        return int(m_auto)
    

    def aggregate_train(self, server_round: int, replies):
        replies = self._validate_train_replies(server_round, replies)
        if not replies:
            return (None, None)

        global_sd = self._global_sd
        assert global_sd is not None

        client_sds = [_reply_to_state_dict(r) for r in replies]
        client_sds = self._maybe_forge_alie_ipm(server_round, replies, global_sd, client_sds)
        client_sds = self._maybe_forge_poisonedfl(server_round, replies, global_sd, client_sds)

        m_eff = self._effective_multi_krum_m(replies)

        print(
            f"[Round {server_round}] Multi-Krum m: "
            f"configured={self.multi_krum_m}, "
            f"n={len(replies)}, "
            f"mal_pct={float(self.malicious_percentage_global):.4f}, "
            f"effective_m={m_eff}",
            flush=True,
        )

        new_sd = agg_multi_krum(global_sd, client_sds, f=self.krum_f, m=m_eff)

        new_params = ArrayRecord(new_sd)
        return new_params, MetricRecord({"round": server_round, "multi_krum_m": int(m_eff)})


class FLTrustStrategy(_BaseMonitoredStrategy):
    def __init__(self, *, root_trainloader=None, root_run_config=None, **kwargs):
        super().__init__(root_trainloader=root_trainloader, **kwargs)
        self.root_run_config = dict(root_run_config) if root_run_config is not None else None

    def aggregate_train(self, server_round: int, replies):
        replies = self._validate_train_replies(server_round, replies)
        if not replies:
            return (None, None)

        client_ids = [_reply_client_id(r) for r in replies]

        global_sd = self._global_sd
        assert global_sd is not None

        root_trainloader = self.root_trainloader
        if self.root_run_config is not None:
            root_trainloader = load_root_data(
                num_partitions=int(self.num_clients_global),
                run_config=self.root_run_config,
                server_round=int(server_round),
            )
        assert root_trainloader is not None, "FLTrust requires root_trainloader"

        # Train root model on trusted data starting from global
        assert self.model_fn is not None, "FLTrust requires model_fn"
        root_model = self.model_fn()
        root_model.load_state_dict({k: v.detach().cpu() for k, v in global_sd.items()})

        with _temp_round_seed(self.base_seed + 100_000 * server_round + 12345):
            trainer(
                root_model,
                root_trainloader,
                num_epochs=int(self.fltrust_root_epochs),
                lr=float(self.root_lr),
            )

        root_sd = root_model.state_dict()

        # ✅ Change (3): read client params from Message content (not r.parameters)
        client_sds = [_reply_to_state_dict(r) for r in replies]
        client_sds = self._maybe_forge_alie_ipm(server_round, replies, global_sd, client_sds)
        client_sds = self._maybe_forge_poisonedfl(server_round, replies, global_sd, client_sds)
        weights = [_reply_num_examples(r) for r in replies]

        # ✅ agg_fltrust returns (new_sd, trust_scores)
        new_sd, trust_scores = agg_fltrust(
            global_sd,
            client_sds,
            root_sd,
            cos_clip=self.fltrust_cos_clip,
        )

        # -------------------- FLTrust logging --------------------
        # Ground-truth malicious IDs among SELECTED participants (simulation only)
        try:
            selected_set = {str(cid) for cid in client_ids}
            mal_selected_set = malicious_ids_for_round(
                int(server_round),
                int(self.num_clients_global),
                float(self.malicious_percentage_global),
                seed_offset=0,
                selected_ids=sorted(selected_set, key=lambda x: int(x) if str(x).isdigit() else str(x)),
                mal_pct_min=float(self.malicious_percentage_min),
                mal_pct_max=float(self.malicious_percentage_max),
                base_seed=int(self.base_seed),
            )
            mal_selected = sorted(
                list(mal_selected_set),
                key=lambda x: int(x) if str(x).isdigit() else str(x),
            )
        except Exception:
            mal_selected = []

        # Detected/excluded by FLTrust (trust <= 0 => excluded from aggregation)
        excluded = [str(cid) for cid, t in zip(client_ids, trust_scores) if float(t) <= 0.0]
        try:
            excluded_sorted = sorted(excluded, key=lambda x: int(x))
        except Exception:
            excluded_sorted = sorted(excluded)

        print(f"[Round {server_round}] malicious_ids_selected={mal_selected}")
        print(f"[Round {server_round}] FLTrust excluded IDs  ={excluded_sorted}")
        not_excluded = sorted(set(map(str, mal_selected)) - set(map(str, excluded_sorted)), key=int)
        print(f"[Round {server_round}] malicious_not_excluded={not_excluded}")
        wrong_included = sorted(set(map(str, excluded_sorted)) - set(map(str,mal_selected )), key=int)
        print(f"[Round {server_round}] Wrong_included={wrong_included}")
        not_included_in_suspicious = sorted(set(map(str, mal_selected)) - set(map(str, excluded_sorted)), key=int)
        print(f"[Round {server_round}] Malicious_not_included_in_suspicious={not_included_in_suspicious}", flush=True)
        # ---------------------------------------------------------

        # ✅ Return ArrayRecord (not ParametersRecords)
        metrics = {"round": server_round}
        if trust_scores:
            metrics["fltrust_mean_trust"] = float(np.mean(trust_scores))

        return ArrayRecord(new_sd), MetricRecord(metrics)


class ShieldFLStrategy(_BaseMonitoredStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store last round aggregated update direction (flattened delta vector)
        self._prev_agg_delta_vec: np.ndarray | None = None

    def aggregate_train(self, server_round: int, replies):
        replies = self._validate_train_replies(server_round, replies)
        if not replies:
            return (None, None)

        global_sd = self._global_sd
        assert global_sd is not None

        client_sds = [_reply_to_state_dict(r) for r in replies]
        client_sds = self._maybe_forge_alie_ipm(server_round, replies, global_sd, client_sds)
        client_sds = self._maybe_forge_poisonedfl(server_round, replies, global_sd, client_sds)

        # ShieldFL paper uses confidence weights η_i (not sample-size FedAvg weights)
        weights = [1.0] * len(replies)

        new_sd = agg_shieldfl(
            global_sd,
            client_sds,
            weights,
            prev_agg_delta_vec=self._prev_agg_delta_vec,
        )

        # Update anchor for next round: aggregated delta of this round
        agg_delta_sd = _sd_sub(new_sd, global_sd)
        agg_delta_vec = _flatten_sd(agg_delta_sd)
        
        delta_norm = float(np.linalg.norm(agg_delta_vec))
        if np.isfinite(delta_norm) and delta_norm > 1e-12:
            self._prev_agg_delta_vec = agg_delta_vec

        return ArrayRecord(new_sd), MetricRecord({"round": server_round})





class DnCStrategy(_BaseMonitoredStrategy):
    def __init__(
        self,
        *,
        dnc_f: int = -1,
        dnc_c: float = 1.5,
        dnc_niters: int = 5,
        dnc_b: int = 256,
        dnc_use_sample_weights: bool = False,  # keep False for paper-faithful behavior
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dnc_f = dnc_f
        self.dnc_c = dnc_c
        self.dnc_niters = dnc_niters
        self.dnc_b = dnc_b
        self.dnc_use_sample_weights = dnc_use_sample_weights

    def aggregate_train(self, server_round: int, replies):
        replies = self._validate_train_replies(server_round, replies)
        if not replies:
            return (None, None)

        global_sd = self._global_sd
        assert global_sd is not None

        client_sds = [_reply_to_state_dict(r) for r in replies]
        client_sds = self._maybe_forge_alie_ipm(server_round, replies, global_sd, client_sds)
        client_sds = self._maybe_forge_poisonedfl(server_round, replies, global_sd, client_sds)
        weights = [_reply_num_examples(r) for r in replies]

        # If dnc_f < 0, estimate f for the sampled participants this round
        n = len(client_sds)
        if self.dnc_f < 0:
            f = int(math.ceil(float(self.malicious_percentage_global) * float(n)))
        else:
            f = int(self.dnc_f)

        new_sd = agg_dnc(
            global_sd,
            client_sds,
            weights,
            f=f,
            c=self.dnc_c,
            niters=self.dnc_niters,
            b=self.dnc_b,
            seed=server_round,
            use_sample_weights=self.dnc_use_sample_weights,
        )

        return ArrayRecord(new_sd), MetricRecord({"round": server_round})




class FedDMCStrategy(_BaseMonitoredStrategy):
    def __init__(
        self,
        *,
        pca_dim: int = 5,
        min_cluster_size: int = 2,
        ema_alpha: float = 0.8,
        trust_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pca_dim = pca_dim
        self.min_cluster_size = min_cluster_size
        self.ema_alpha = ema_alpha
        self.trust_threshold = trust_threshold
        self._trust: dict[str, float] = {}

    def aggregate_train(self, server_round: int, replies):
        replies = self._validate_train_replies(server_round, replies)
        if not replies:
            return (None, None)

        global_sd = self._global_sd
        assert global_sd is not None

        client_sds = [_reply_to_state_dict(r) for r in replies]
        client_sds = self._maybe_forge_alie_ipm(server_round, replies, global_sd, client_sds)
        client_sds = self._maybe_forge_poisonedfl(server_round, replies, global_sd, client_sds)
        client_ids = [_reply_client_id(r) for r in replies]
        weights = [_reply_num_examples(r) for r in replies]

        new_sd = agg_feddmc(
            global_sd=global_sd,
            client_sds=client_sds,
            client_ids=client_ids,
            weights=weights,
            trust=self._trust,
            server_round=server_round,  # IMPORTANT: needed to match paper Eq.(3) base case
            pca_dim=self.pca_dim,
            min_cluster_size=self.min_cluster_size,
            ema_alpha=self.ema_alpha,
            trust_threshold=self.trust_threshold,
        )

        return ArrayRecord(new_sd), MetricRecord({"round": server_round})


class GASStrategy(_BaseMonitoredStrategy):
    """GAS selection/aggregation, with the shared unchanged attack hooks."""
    def __init__(self, *, gas_p: int = 1000, gas_f: int = -1,
                 gas_base_agr: str = "median", gas_dnc_c: float = 1.0,
                 gas_dnc_niters: int = 1, gas_dnc_b: int = 2000, **kwargs):
        super().__init__(**kwargs)
        self.gas_p = int(gas_p)
        self.gas_f = int(gas_f)
        self.gas_base_agr = str(gas_base_agr)
        self.gas_dnc_c = float(gas_dnc_c)
        self.gas_dnc_niters = int(gas_dnc_niters)
        self.gas_dnc_b = int(gas_dnc_b)

    def aggregate_train(self, server_round: int, replies):
        replies = self._validate_train_replies(server_round, replies)
        if not replies:
            return None, None
        global_sd = self._global_sd
        assert global_sd is not None
        client_sds = [_reply_to_state_dict(r) for r in replies]
        client_sds = self._maybe_forge_alie_ipm(server_round, replies, global_sd, client_sds)
        client_sds = self._maybe_forge_poisonedfl(server_round, replies, global_sd, client_sds)
        n = len(client_sds)
        # Do not use oracle per-round malicious IDs. Auto uses the configured
        # maximum attack rate, as a fixed conservative Byzantine upper bound.
        f_requested = (int(math.floor(float(self.malicious_percentage_max) * n))
                       if self.gas_f < 0 else int(self.gas_f))
        if self.gas_f < -1 or f_requested < 0:
            raise ValueError("gas_f must be -1 (auto) or nonnegative")
        f = min(f_requested, (n - 1) // 2)  # GAS theory assumes f < n/2.
        if f_requested != f:
            print(f"[GAS] round={server_round} requested f={f_requested}, capped at "
                  f"{f} because GAS assumes f<n/2", flush=True)
        new_sd, selected, scores = agg_gas(
            global_sd, client_sds, f=f, p=self.gas_p,
            base_agr=self.gas_base_agr,
            seed=int(self.base_seed) + 100_000 * int(server_round) + 123,
            dnc_c=self.gas_dnc_c,
            dnc_niters=self.gas_dnc_niters,
            dnc_b=self.gas_dnc_b,
        )
        ids = [_reply_client_id(r) for r in replies]
        if self.gas_base_agr.strip().lower() == "dnc":
            print(f"[GAS(DnC)][Round {server_round}] "
                  f"c={self.gas_dnc_c} niters={self.gas_dnc_niters} "
                  f"b={self.gas_dnc_b}", flush=True)
        print(f"[GAS][Round {server_round}] base={self.gas_base_agr} "
              f"groups={self.gas_p} f={f} selected="
              f"{[ids[j] for j in selected]}", flush=True)
        return ArrayRecord(new_sd), MetricRecord({"round": server_round})


class SDEAStrategy(_BaseMonitoredStrategy):
    """SDEA entropy-trained weighted model aggregation with unlabeled public inputs."""
    def __init__(self, *, sdea_public_X: torch.Tensor,
                 sdea_public_epochs: int = 20, sdea_public_batch_size: int = 64,
                 sdea_public_lr: float = 0.005, sdea_use_cc: bool = True,
                 sdea_device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self.sdea_public_X = sdea_public_X
        self.sdea_public_epochs = int(sdea_public_epochs)
        self.sdea_public_batch_size = int(sdea_public_batch_size)
        self.sdea_public_lr = float(sdea_public_lr)
        self.sdea_use_cc = bool(sdea_use_cc)
        self.sdea_device = str(sdea_device)

    def aggregate_train(self, server_round: int, replies):
        replies = self._validate_train_replies(server_round, replies)
        if not replies:
            return None, None
        global_sd = self._global_sd
        assert global_sd is not None
        client_sds = [_reply_to_state_dict(r) for r in replies]
        client_sds = self._maybe_forge_alie_ipm(server_round, replies, global_sd, client_sds)
        client_sds = self._maybe_forge_poisonedfl(server_round, replies, global_sd, client_sds)
        new_sd, weights, excluded = agg_sdea(
            global_sd, client_sds, public_X=self.sdea_public_X,
            model_fn=self.model_fn, epochs=self.sdea_public_epochs,
            batch_size=self.sdea_public_batch_size, lr=self.sdea_public_lr,
            use_cc=self.sdea_use_cc,
            seed=int(self.base_seed) + 100_000 * int(server_round) + 917,
            device=self.sdea_device,
        )
        ids = [_reply_client_id(r) for r in replies]
        print(f"[SDEA][Round {server_round}] CC_low_cluster="
              f"{[ids[i] for i in excluded]} "
              f"weights={[(ids[i], round(float(w), 6)) for i,w in enumerate(weights)]}", flush=True)
        return ArrayRecord(new_sd), MetricRecord({"round": server_round})


class OracleDropStrategy(_BaseMonitoredStrategy):
    """
    Oracle defense: server "knows" malicious clients and drops them before aggregation.
    Uses deterministic malicious IDs computed on server side (no client flag needed).
    """
    def aggregate_train(self, server_round: int, replies):
        replies = self._validate_train_replies(server_round, replies)
        if not replies:
            return (None, None)

        global_sd = self._global_sd
        assert global_sd is not None
        # Ground-truth malicious IDs among SELECTED participants (simulation only)
        selected_ids = None
        if hasattr(self, '_selected_partition_ids_by_round'):
            selected_ids = self._selected_partition_ids_by_round.get(int(server_round))
        if selected_ids is None:
            selected_ids = sorted({str(_reply_client_id(r)) for r in replies}, key=lambda x: int(x) if str(x).isdigit() else str(x))

        mal_set = set(
            malicious_ids_for_round(
                int(server_round),
                int(self.num_clients_global),
                float(self.malicious_percentage_global),
                seed_offset=0,
                selected_ids=list(selected_ids),
                mal_pct_min=float(getattr(self, 'malicious_percentage_min', self.malicious_percentage_global)),
                mal_pct_max=float(getattr(self, 'malicious_percentage_max', self.malicious_percentage_global)),
                base_seed=int(self.base_seed),
            )
        )

        kept = []
        dropped = 0
        for r in replies:
            cid = _reply_client_id(r)  # should be "0","1",... as strings
            if str(cid) in mal_set:
                dropped += 1
                continue
            kept.append(r)

        if not kept:
            kept = list(replies)
            dropped = 0

        # ✅ Change (3): read from Message content
        client_sds = [_reply_to_state_dict(r) for r in kept]
        weights = [_reply_num_examples(r) for r in kept]

        # FedAvg on deltas
        deltas = _to_deltas(global_sd, client_sds)
        delta_new = agg_fedavg_delta(deltas, weights)
        new_sd = _apply_delta(global_sd, delta_new)

        # ✅ Return ArrayRecord
        return ArrayRecord(new_sd), MetricRecord(
            {"round": server_round, "oracle_dropped": dropped, "clients_used": len(kept)}
        )


def _probeguard_phase1_detection_metrics(
    client_ids: list[str],
    suspect_ids: list[str],
    malicious_ids: set[str],
) -> tuple[float, float, float]:
    """Return (F1, MCC, Precision) for Phase-1 malicious-positive classification.

    Count only clients whose updates were actually received this round. Both
    metrics use 0.0 when their denominator is zero (undefined classification).
    No ground-truth labels enter any detector; they are only used for evaluation.
    """
    participated = {str(cid) for cid in client_ids}
    predicted = {str(cid) for cid in suspect_ids} & participated
    actual = {str(cid) for cid in malicious_ids} & participated
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    tn = len(participated - (predicted | actual))

    f1_denom = 2 * tp + fp + fn
    f1 = (2.0 * tp / f1_denom) if f1_denom else 0.0

    # Multiply as floats to avoid fixed-width integer overflow in larger runs.
    mcc_denom = math.sqrt(
        float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    )
    mcc = (float(tp * tn - fp * fn) / mcc_denom) if mcc_denom > 0.0 else 0.0
    precision = (float(tp) / float(tp + fp)) if (tp + fp) else 0.0
    return float(f1), float(mcc), float(precision)


class ProbeGuardStrategy(_BaseMonitoredStrategy):
    """ProbeGuard: random behavioral probing, trust-weighting, and sanitization.

    The Random Phase-1 mode keeps a bounded-uniform probe bank and refreshes it
    on a deterministic schedule. Oracle/update/none remain for evaluation and
    ablation. Phase 2 scores, sanitizes, and aggregates client model updates.
    """

    def __init__(
        self,
        *,
        probeguard_warmup_rounds: int = 0,
        probeguard_probe_bank_size: int = 60,
        probeguard_probe_refresh: int = 20,
        probeguard_refresh_every: int = 10,
        probeguard_bounds_low: float = -3.0,
        probeguard_bounds_high: float = 3.0,
        probeguard_use_data_bounds: bool = True,
        probeguard_data_bounds_clip: float = 8.0,
        probeguard_suspect_max: int = 20,
        probeguard_mad_alpha: float = 1.0,
        probeguard_use_mad: bool = True,
        probeguard_probe_mad_mode: str = "both",
        probeguard_probe_normalize_per_probe: bool = False,
        probeguard_probe_scale_floor_ratio: float = 0.3,
        probeguard_probe_threshold_normalized_ranking: bool = False,
        probeguard_phase1_hybrid_update: bool = True,
        probeguard_phase1_hybrid_probe_weight: float = 0.5,
        probeguard_top_frac: float = 0.06,
        probeguard_probe_feature_mode: str = "global_top3_logits",
        probeguard_use_kmeans: bool = True,
        probeguard_cluster_method: str = "none",
        probeguard_iforest_contamination: float = -1.0,
        probeguard_phase1_mode: str = "random",
        use_ema_reference: bool = False,
        probeguard_reference_ema_beta: float = 0.9,
        probeguard_reference_use_benign_median: bool = False,
        probeguard_trust_only_suspicious: bool = True,
        probeguard_trust_use_cos: bool = True,
        probeguard_trust_cos_tau: float = 0.0,
        probeguard_trust_cos_gamma: float = 5.0,
        probeguard_trust_use_norm: bool = True,
        probeguard_trust_norm_cap: float = 0.0,
        probeguard_trust_norm_cap_mult: float = 1.5,
        probeguard_trust_norm_lambda: float = 1.0,
        probeguard_trust_use_ortho: bool = True,
        probeguard_trust_ortho_lambda: float = 2.0,
        probeguard_trust_use_suspect_weight: bool = False,
        probeguard_suspect_weight: float = 0.1,
        probeguard_sanitize_only_suspicious: bool = False,
        probeguard_sanitize_use_cone: bool = True,
        probeguard_sanitize_cone_deg: float = 60.0,
        probeguard_sanitize_use_norm_clip: bool = True,
        probeguard_sanitize_norm_cap: float = 0.0,
        probeguard_sanitize_norm_cap_mult: float = 1.0,
        probeguard_seed: int | None = None,
        probeguard_pca_var: float = 0.98,
        probeguard_client_method: str = "fedprox",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.probeguard_warmup_rounds = max(0, int(probeguard_warmup_rounds))
        self.probeguard_probe_bank_size = max(1, int(probeguard_probe_bank_size))
        self.probeguard_probe_refresh = max(0, int(probeguard_probe_refresh))
        self.probeguard_refresh_every = max(1, int(probeguard_refresh_every))
        self.probeguard_bounds_low = float(probeguard_bounds_low)
        self.probeguard_bounds_high = float(probeguard_bounds_high)
        self.probeguard_use_data_bounds = bool(probeguard_use_data_bounds)
        self.probeguard_data_bounds_clip = float(probeguard_data_bounds_clip)
        self.probeguard_suspect_max = int(probeguard_suspect_max)
        self.probeguard_mad_alpha = float(probeguard_mad_alpha)
        self.probeguard_use_mad = bool(probeguard_use_mad)
        self.probeguard_probe_mad_mode = str(probeguard_probe_mad_mode).lower().strip()
        self.probeguard_probe_normalize_per_probe = bool(probeguard_probe_normalize_per_probe)
        self.probeguard_probe_scale_floor_ratio = float(probeguard_probe_scale_floor_ratio)
        if not math.isfinite(self.probeguard_probe_scale_floor_ratio) or self.probeguard_probe_scale_floor_ratio < 0.0:
            raise ValueError("probeguard_probe_scale_floor_ratio must be finite and >= 0")
        self.probeguard_probe_threshold_normalized_ranking = bool(probeguard_probe_threshold_normalized_ranking)
        self.probeguard_phase1_hybrid_update = bool(probeguard_phase1_hybrid_update)
        self.probeguard_phase1_hybrid_probe_weight = float(probeguard_phase1_hybrid_probe_weight)
        if not 0.0 <= self.probeguard_phase1_hybrid_probe_weight <= 1.0:
            raise ValueError("probeguard_phase1_hybrid_probe_weight must be in [0, 1]")
        self.probeguard_top_frac = float(probeguard_top_frac)
        self.probeguard_probe_feature_mode = str(probeguard_probe_feature_mode)
        self.probeguard_use_kmeans = bool(probeguard_use_kmeans)
        self.probeguard_cluster_method = str(probeguard_cluster_method)
        self.probeguard_iforest_contamination = float(probeguard_iforest_contamination)
        self.probeguard_phase1_mode = str(probeguard_phase1_mode).lower().strip()
        if self.probeguard_phase1_mode not in {"random", "oracle", "update", "none"}:
            raise ValueError(
                f"Unsupported ProbeGuard Phase-1 mode: {probeguard_phase1_mode!r}; "
                "expected random, oracle, update, or none"
            )

        # The aggregation method fixes the reference rule; it is not a run-config option.
        # ProbeGuard: current-round coordinate-wise median.
        # ProbeGuard-EMA: EMA of current-round coordinate-wise medians.
        # The existing benign-only median flag still controls the client subset.
        self._probeguard_use_ema_reference = bool(use_ema_reference)
        self.probeguard_reference_use_benign_median = bool(probeguard_reference_use_benign_median)
        self.probeguard_reference_ema_beta = float(probeguard_reference_ema_beta)

        self.probeguard_trust_only_suspicious = bool(probeguard_trust_only_suspicious)
        self.probeguard_trust_use_cos = bool(probeguard_trust_use_cos)
        self.probeguard_trust_cos_tau = float(probeguard_trust_cos_tau)
        self.probeguard_trust_cos_gamma = float(probeguard_trust_cos_gamma)
        self.probeguard_trust_use_norm = bool(probeguard_trust_use_norm)
        self.probeguard_trust_norm_cap = float(probeguard_trust_norm_cap)
        self.probeguard_trust_norm_cap_mult = float(probeguard_trust_norm_cap_mult)
        self.probeguard_trust_norm_lambda = float(probeguard_trust_norm_lambda)
        self.probeguard_trust_use_ortho = bool(probeguard_trust_use_ortho)
        self.probeguard_trust_ortho_lambda = float(probeguard_trust_ortho_lambda)
        self.probeguard_trust_use_suspect_weight = bool(probeguard_trust_use_suspect_weight)

        self.probeguard_suspect_weight = float(probeguard_suspect_weight)

        self.probeguard_sanitize_only_suspicious = bool(probeguard_sanitize_only_suspicious)
        self.probeguard_sanitize_use_cone = bool(probeguard_sanitize_use_cone)
        self.probeguard_sanitize_cone_deg = float(probeguard_sanitize_cone_deg)
        self.probeguard_sanitize_use_norm_clip = bool(probeguard_sanitize_use_norm_clip)
        self.probeguard_sanitize_norm_cap = float(probeguard_sanitize_norm_cap)
        self.probeguard_sanitize_norm_cap_mult = float(probeguard_sanitize_norm_cap_mult)

        # Persistent reference direction state (flattened deltas, CPU float32)
        self._probeguard_prev_agg_delta_vec: Optional[torch.Tensor] = None
        self._probeguard_ref_ema_vec: Optional[torch.Tensor] = None

        self.probeguard_pca_var = float(probeguard_pca_var)

        # Probe bank
        self._probe_X: Optional[np.ndarray] = None
        self._probe_pairs: list[tuple[int, int]] = []

        # Probe anchors (computed from previous global model on the current probe bank)
        self._probe_top3_anchors: Optional[np.ndarray] = None
        self._probe_comp_anchors: Optional[np.ndarray] = None



        # Latest suspicion scores (for diagnostics)
        self._last_score: Dict[str, float] = {}
        # Per-round Phase-1 detection evaluation, excluding warmup rounds.
        self._probeguard_detection_f1: list[float] = []
        self._probeguard_detection_mcc: list[float] = []
        self._probeguard_detection_precision: list[float] = []
        self._probeguard_warned_invalid_no_phase1_ref = False

        self.base_seed = int(kwargs.get("base_seed"))
        self.probeguard_seed = int(self.base_seed if probeguard_seed is None else probeguard_seed)

        self.probeguard_client_method = str(probeguard_client_method).lower().strip()
        # SCAFFOLD global control variate (initialized lazily on first round)
        self._scaffold_c_sd: Optional[OrderedDict[str, torch.Tensor]] = None

    def _probeguard_true_malicious_ids(self, server_round: int, client_ids: list[str]) -> set[str]:
        """Get the same deterministic ground truth as client attack assignment."""
        selected_by_round = getattr(self, "_selected_partition_ids_by_round", {})
        selected_ids = selected_by_round.get(int(server_round), client_ids)
        return set(map(str, malicious_ids_for_round(
            int(server_round),
            int(self.num_clients_global),
            float(self.malicious_percentage_global),
            seed_offset=0,
            selected_ids=list(selected_ids),
            mal_pct_min=float(getattr(self, "malicious_percentage_min", self.malicious_percentage_global)),
            mal_pct_max=float(getattr(self, "malicious_percentage_max", self.malicious_percentage_global)),
            base_seed=int(self.base_seed),
        )))

    def write_phase1_detection_report(self, run_config: dict) -> tuple[str, str, str]:
        """Append one mean Phase-1 F1, MCC, and Precision value per completed run.

        Each metric has its own numerical-only file. Files are shared by runs
        with the same dataset/attack/mode/malicious-percentage configuration;
        the seed is deliberately not part of the filename. Per-round values
        are used only to compute each run's mean, not written individually.
        """
        f1_values = self._probeguard_detection_f1
        mcc_values = self._probeguard_detection_mcc
        precision_values = self._probeguard_detection_precision
        if not (len(f1_values) == len(mcc_values) == len(precision_values)):
            raise RuntimeError("ProbeGuard Phase-1 F1/MCC/Precision series lengths differ")
        if not f1_values:
            print("[ProbeGuard] No active Phase-1 rounds; no detection report generated.", flush=True)
            return "", "", ""

        prefix = _results_filename_from_config(run_config).removesuffix(".txt")
        mal_min = _sanitize_filename_part(getattr(self, "malicious_percentage_min", "na"))
        mal_max = _sanitize_filename_part(getattr(self, "malicious_percentage_max", "na"))
        mode = _sanitize_filename_part(self.probeguard_phase1_mode)
        common_name = f"{prefix}-phase1_{mode}-malrange_{mal_min}_{mal_max}"
        f1_filename = f"{common_name}-F1.txt"
        mcc_filename = f"{common_name}-MCC.txt"
        precision_filename = f"{common_name}-Precision.txt"

        # Append exactly one numeric value per completed run in each file.
        with open(f1_filename, "a", encoding="utf-8") as file:
            file.write(f"{float(np.mean(f1_values)):.8f}\n")
        with open(mcc_filename, "a", encoding="utf-8") as file:
            file.write(f"{float(np.mean(mcc_values)):.8f}\n")
        with open(precision_filename, "a", encoding="utf-8") as file:
            file.write(f"{float(np.mean(precision_values)):.8f}\n")
        print(
            f"[ProbeGuard] Phase-1 run means appended: {f1_filename}; {mcc_filename}; {precision_filename}",
            flush=True,
        )
        return f1_filename, mcc_filename, precision_filename

    def _extra_train_records(self, server_round: int, *, arrays: ArrayRecord, config, grid) -> dict:
        # Only attach SCAFFOLD records when ProbeGuard is configured to use SCAFFOLD locally
        if str(getattr(self, "probeguard_client_method", "")).lower().strip() != "scaffold":
            return {}

        # Lazy init: zeros like the global model params
        if self._scaffold_c_sd is None:
            try:
                sd = arrays.to_torch_state_dict()
            except Exception:
                sd = getattr(self, "_global_sd", None) or {}
            self._scaffold_c_sd = OrderedDict((k, torch.zeros_like(v).cpu()) for k, v in sd.items())

        return {"scaffold_c": ArrayRecord(self._scaffold_c_sd)}

    def configure_train(self, server_round: int, arrays, config, grid):
        # Let base class cache global state_dict (self._global_sd)
        return super().configure_train(server_round, arrays, config, grid)

    @staticmethod
    def _quick_acc_state_dict(
        *,
        sd: "OrderedDict[str, torch.Tensor]",
        train_loader,
        num_partitions: int,
        run_config: dict,
        max_batches: int,
        device: torch.device,
    ) -> float:
        # Build a fresh model and evaluate accuracy (no gradients).
        model = build_model(num_partitions=num_partitions, run_config=run_config)
        model.load_state_dict(sd)
        model.to(device)
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for b, batch in enumerate(train_loader):
                if b >= max_batches:
                    break
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()
        return float(correct / total) if total > 0 else 0.0

    def _update_scaffold_control_variates(self, replies, global_sd) -> None:
        """Update SCAFFOLD server controls (also used by warmup fallback)."""
        if self.probeguard_client_method != "scaffold":
            return
        if self._scaffold_c_sd is None:
            self._scaffold_c_sd = OrderedDict(
                (k, torch.zeros_like(v).cpu()) for k, v in global_sd.items()
            )
        dc_sum = OrderedDict(
            (k, torch.zeros_like(v).cpu()) for k, v in self._scaffold_c_sd.items()
        )
        dc_count = 0
        for reply in replies:
            dc = _reply_to_scaffold_delta_c(reply)
            if dc is None:
                continue
            dc_count += 1
            for key, value in dc.items():
                if key in dc_sum and dc_sum[key].shape == value.shape:
                    dc_sum[key] += value.detach().cpu()
        if dc_count:
            for key in self._scaffold_c_sd:
                self._scaffold_c_sd[key] = (
                    self._scaffold_c_sd[key] + dc_sum[key] / float(dc_count)
                ).detach().cpu()

    def aggregate_train(self, server_round: int, replies):
        replies = self._validate_train_replies(server_round, replies)

        replies = sorted(
            replies,
            key=lambda r: int(_reply_client_id(r)) if str(_reply_client_id(r)).isdigit() else str(_reply_client_id(r)),
        )

        if not replies:
            return (None, None)

        global_sd = self._global_sd
        assert global_sd is not None

        # Oracle drop (optional)
        replies = self._filter_oracle(server_round, replies)

        # Parse client updates
        client_ids = [_reply_client_id(r) for r in replies]
        client_sds = [_reply_to_state_dict(r) for r in replies]
        client_sds = self._maybe_forge_alie_ipm(server_round, replies, global_sd, client_sds)
        client_sds = self._maybe_forge_poisonedfl(server_round, replies, global_sd, client_sds)
        weights = [_reply_num_examples(r) for r in replies]

        # --- Warmup: plain FedAvg on deltas ---
        if int(server_round) < int(self.probeguard_warmup_rounds):
            delta = agg_fedavg_delta(_to_deltas(global_sd, client_sds), weights)
            new_sd = _apply_delta(global_sd, delta)
            self._probeguard_prev_global_sd = global_sd
            self._update_scaffold_control_variates(replies, global_sd)
            return ArrayRecord(new_sd), MetricRecord({"round": server_round, "probeguard_warmup": 1})

        rng = np.random.default_rng(self.probeguard_seed + int(server_round))
        phase1_mode = self.probeguard_phase1_mode
        score_by: Dict[str, float] = {str(cid): 0.0 for cid in client_ids}
        suspects: list[str] = []

        if phase1_mode == "random":
            base_model = build_model(num_partitions=self.num_clients_global,
                                     run_config=_RUN_CONFIG_SNAPSHOT or {})
            base_model.load_state_dict(global_sd)
            base_model.eval()
            try:
                in_dim = int(base_model.net[0].in_features)
            except (AttributeError, IndexError, TypeError):
                in_dim = int(getattr(base_model, "input_dim", 0) or 0)
            if in_dim <= 0:
                # Preserve the original FedAvg fallback when the input dimension is unavailable.
                delta = agg_fedavg_delta(_to_deltas(global_sd, client_sds), weights)
                new_sd = _apply_delta(global_sd, delta)
                self._probeguard_prev_global_sd = global_sd
                self._update_scaffold_control_variates(replies, global_sd)
                return ArrayRecord(new_sd), MetricRecord({"round": server_round, "probeguard_no_input_dim": 1})

            if self.probeguard_use_data_bounds:
                try:
                    from fltabular.task import get_standardized_feature_bounds
                    low, high = get_standardized_feature_bounds(
                        num_partitions=self.num_clients_global,
                        run_config=_RUN_CONFIG_SNAPSHOT or {},
                        clip=self.probeguard_data_bounds_clip,
                    )
                except Exception:
                    low = np.full((in_dim,), self.probeguard_bounds_low, dtype=np.float32)
                    high = np.full((in_dim,), self.probeguard_bounds_high, dtype=np.float32)
            else:
                low = np.full((in_dim,), self.probeguard_bounds_low, dtype=np.float32)
                high = np.full((in_dim,), self.probeguard_bounds_high, dtype=np.float32)

            bank_missing = self._probe_X is None or self._probe_X.size == 0 or not self._probe_pairs
            if bank_missing:
                candidate_X = rng.uniform(
                    low=low, high=high,
                    size=(max(1, self.probeguard_probe_bank_size), in_dim),
                ).astype(np.float32)
                self._probe_X = candidate_X[:self.probeguard_probe_bank_size]
                self._probe_pairs = _probeguard_probe_pairs_for_X(base_model, self._probe_X)
            else:
                refresh = self.probeguard_probe_refresh
                refresh_due = (
                    (int(server_round) - self.probeguard_warmup_rounds) % self.probeguard_refresh_every == 0
                )
                if refresh > 0 and refresh_due:
                    new_X = rng.uniform(low=low, high=high,
                                        size=(refresh, in_dim)).astype(np.float32)
                    new_pairs = _probeguard_probe_pairs_for_X(base_model, new_X)
                    keep = max(0, self.probeguard_probe_bank_size - len(new_X))
                    if keep > 0 and self._probe_X is not None and self._probe_X.size > 0:
                        self._probe_X = np.vstack([self._probe_X[-keep:], new_X]).astype(np.float32)
                        self._probe_pairs = list(self._probe_pairs[-keep:]) + new_pairs
                    else:
                        self._probe_X = new_X[-self.probeguard_probe_bank_size:]
                        self._probe_pairs = new_pairs[-self.probeguard_probe_bank_size:]

            # In the original Random mode the class anchors are re-derived from
            # the global model at scoring time; the same cached-anchor fallback
            # is retained in _probeguard_compute_suspicion.
            sklearn_rs = int((self.base_seed + int(server_round)) % (2**32 - 1))
            score_by, suspects = _probeguard_compute_suspicion(
                global_sd=global_sd,
                client_sds=client_sds,
                client_ids=client_ids,
                probe_X=self._probe_X,
                probe_pairs=self._probe_pairs,
                num_partitions=self.num_clients_global,
                run_config=_RUN_CONFIG_SNAPSHOT or {},
                mad_alpha=self.probeguard_mad_alpha,
                top_frac=self.probeguard_top_frac,
                use_kmeans=self.probeguard_use_kmeans,
                probeguard_pca_var=self.probeguard_pca_var,
                probeguard_cluster_method=self.probeguard_cluster_method,
                probeguard_iforest_contamination=self.probeguard_iforest_contamination,
                probeguard_probe_feature_mode=self.probeguard_probe_feature_mode,
                rng=rng,
                probe_top3_anchors=self._probe_top3_anchors,
                probe_comp_anchors=self._probe_comp_anchors,
                sklearn_random_state=sklearn_rs,
                use_mad=self.probeguard_use_mad,
                probeguard_probe_mad_mode=self.probeguard_probe_mad_mode,
                probeguard_probe_normalize_per_probe=self.probeguard_probe_normalize_per_probe,
                probeguard_probe_scale_floor_ratio=self.probeguard_probe_scale_floor_ratio,
                probeguard_probe_threshold_normalized_ranking=self.probeguard_probe_threshold_normalized_ranking,
            )
            if self.probeguard_phase1_hybrid_update:
                update_scores, _ = _probeguard_compute_update_based_suspicion(
                    global_sd=global_sd, client_sds=client_sds, client_ids=client_ids,
                    mad_alpha=self.probeguard_mad_alpha, use_mad=self.probeguard_use_mad,
                )
                score_by, suspects = _probeguard_fuse_phase1_scores(
                    probe_scores=score_by, update_scores=update_scores,
                    client_ids=client_ids,
                    probe_weight=self.probeguard_phase1_hybrid_probe_weight,
                    mad_alpha=self.probeguard_mad_alpha,
                    use_mad=self.probeguard_use_mad,
                )
        elif phase1_mode == "oracle":
            true_malicious_ids = self._probeguard_true_malicious_ids(server_round, client_ids)
            suspects = [str(cid) for cid in client_ids if str(cid) in true_malicious_ids]
            score_by = {str(cid): float(str(cid) in true_malicious_ids) for cid in client_ids}
        elif phase1_mode == "update":
            score_by, suspects = _probeguard_compute_update_based_suspicion(
                global_sd=global_sd, client_sds=client_sds, client_ids=client_ids,
                mad_alpha=self.probeguard_mad_alpha, use_mad=self.probeguard_use_mad,
            )
        # mode == "none" leaves all clients unflagged.

        self._last_score = score_by

        # All non-oracle detectors retain the existing suspect cap. The oracle
        # must identify *all* malicious clients, including when count > cap.
        suspects = [str(s) for s in suspects]
        if phase1_mode != "oracle":
            suspects = suspects[: max(0, int(self.probeguard_suspect_max))]

        # Evaluate the *actual post-cap Phase-1 decision* before downstream
        # trust weighting/sanitization (no labels are used by the detector).
        true_malicious_ids = self._probeguard_true_malicious_ids(server_round, client_ids)
        f1, mcc, precision = _probeguard_phase1_detection_metrics(client_ids, suspects, true_malicious_ids)
        self._probeguard_detection_f1.append(f1)
        self._probeguard_detection_mcc.append(mcc)
        self._probeguard_detection_precision.append(precision)

        # ------------------------------------------------------------------
        # Soft trust-weighting + sanitization (replaces audit + hard removal)
        # ------------------------------------------------------------------
        suspect_set = set(map(str, suspects or []))

        # Build deltas and flatten (fixed key order)
        keys = list(global_sd.keys())
        deltas_sd: list[OrderedDict[str, torch.Tensor]] = []
        delta_vecs: list[torch.Tensor] = []
        for sd in client_sds:
            dsd = _sd_sub(sd, global_sd)
            deltas_sd.append(dsd)
            delta_vecs.append(_flatten_sd_torch(dsd, keys).detach().to(torch.float32).cpu())

        n = len(delta_vecs)
        if n == 0:
            return (None, None)

        G = torch.stack(delta_vecs, dim=0)  # [n, d]
        eps = 1e-12
        norms = torch.norm(G, dim=1, p=2) + eps

        # Benign-looking mask from phase 1 (clients not flagged as suspicious this round).
        # When phase 1 is disabled, there is intentionally no benign/suspicious list.
        phase1_has_labels = (phase1_mode != "none")
        if phase1_has_labels:
            benign_mask = torch.tensor([str(cid) not in suspect_set for cid in client_ids], dtype=torch.bool)
            has_benign = bool(torch.any(benign_mask))
        else:
            benign_mask = torch.zeros((n,), dtype=torch.bool)
            has_benign = False

        # Sample-size weights (FedAvg-style)
        w_samples = torch.tensor([float(w) for w in (weights or [1.0] * n)], dtype=torch.float32)
        if int(w_samples.numel()) != n:
            w_samples = torch.ones((n,), dtype=torch.float32)

        # Benign-only statistics from phase-1 benign clients.
        # If no client is marked benign in a round, use an all-client fallback only to avoid a degenerate zero-reference round.
        w_b = w_samples * benign_mask.to(torch.float32)
        w_b_sum = float(w_b.sum().item())
        if has_benign and w_b_sum > eps:
            mean_benign = (G * w_b[:, None]).sum(dim=0) / max(eps, w_b_sum)
        else:
            mean_benign = (G * w_samples[:, None]).sum(dim=0) / max(eps, float(w_samples.sum().item()))


        # Coordinate-wise median of client deltas for the reference direction.
        # Keep the existing optional benign-only median behavior unchanged.
        use_benign_median = self.probeguard_reference_use_benign_median
        try:
            if use_benign_median and has_benign:
                median_vec = torch.median(G[benign_mask], dim=0).values
            else:
                median_vec = torch.median(G, dim=0).values
        except Exception:
            median_vec = mean_benign

        # Both variants use the median; only ProbeGuard-EMA carries it across rounds.
        ref_source_vec = median_vec
        if self._probeguard_use_ema_reference:
            if phase1_mode == "none" and not self._probeguard_warned_invalid_no_phase1_ref:
                print(
                    "[ProbeGuard-EMA] WARNING: Phase 1 is disabled; using all-client "
                    "statistics for the EMA reference.",
                    flush=True,
                )
                self._probeguard_warned_invalid_no_phase1_ref = True
            ref_vec = self._probeguard_ref_ema_vec
            if ref_vec is not None:
                ref_vec = ref_vec.detach().to(torch.float32).cpu()
            else:
                ref_vec = ref_source_vec
        else:
            ref_vec = median_vec

        ref_norm = float(torch.norm(ref_vec, p=2).item())
        if (not math.isfinite(ref_norm)) or ref_norm <= eps:
            ref_vec = ref_source_vec
            ref_norm = float(torch.norm(ref_vec, p=2).item())
        ref_norm = max(ref_norm, eps)
        u_ref = ref_vec / ref_norm

        # ---- Determine norm cap (for trust + sanitization) ----
        norm_cap = float(getattr(self, "probeguard_trust_norm_cap", 0.0))
        if not (norm_cap > 0.0):
            mult = float(getattr(self, "probeguard_trust_norm_cap_mult", 1.5))
            try:
                norm_source = norms[benign_mask] if has_benign else norms
                med_norm = float(torch.median(norm_source).item())
            except Exception:
                med_norm = float(torch.median(norms).item())
            if not math.isfinite(med_norm) or med_norm <= eps:
                med_norm = float(torch.mean(norms).item())
            norm_cap = max(eps, med_norm * mult)

        # ---- Trust scoring (continuous weights) ----
        dots = torch.sum(G * ref_vec[None, :], dim=1)
        cos = dots / (norms * ref_norm)
        cos = torch.clamp(cos, -1.0, 1.0)

        cos_tau = float(getattr(self, "probeguard_trust_cos_tau", 0.0))
        cos_gamma = float(getattr(self, "probeguard_trust_cos_gamma", 5.0))
        if bool(getattr(self, "probeguard_trust_use_cos", True)):
            # sigmoid((cos - tau) * gamma)
            w_cos = torch.sigmoid((cos - cos_tau) * cos_gamma)
        else:
            w_cos = torch.ones_like(norms)

        norm_lambda = float(getattr(self, "probeguard_trust_norm_lambda", 1.0))
        if bool(getattr(self, "probeguard_trust_use_norm", True)):
            w_norm = torch.minimum(torch.ones_like(norms), torch.tensor(norm_cap, dtype=torch.float32) / norms)
            if norm_lambda != 1.0:
                w_norm = torch.pow(w_norm, float(norm_lambda))
        else:
            w_norm = torch.ones_like(norms)

        ortho_lambda = float(getattr(self, "probeguard_trust_ortho_lambda", 2.0))
        ortho = torch.sqrt(torch.clamp(1.0 - cos * cos, min=0.0))
        if bool(getattr(self, "probeguard_trust_use_ortho", True)):
            w_ortho = torch.exp(-float(ortho_lambda) * ortho)
        else:
            w_ortho = torch.ones_like(norms)

        trust = w_cos * w_norm * w_ortho

        trust_only_suspicious = bool(getattr(self, "probeguard_trust_only_suspicious", False)) if phase1_has_labels else False
        trust_use_suspect_weight = bool(getattr(self, "probeguard_trust_use_suspect_weight", True)) if phase1_has_labels else False

        # Optional mode: only suspicious clients receive trust-based weighting; benign clients keep weight 1.
        if trust_only_suspicious:
            trust = torch.where(benign_mask, torch.ones_like(trust), trust)

        # Downweight suspects (no hard removal)
        if trust_use_suspect_weight:
            suspect_w = float(getattr(self, "probeguard_suspect_weight", 0.1))
            trust = trust * torch.where(
                benign_mask,
                torch.ones_like(trust),
                torch.full_like(trust, float(max(0.0, min(1.0, suspect_w)))),
            )

        # ---- Sanitization: cone projection toward reference ----
        G_san = G

        sanitize_only_suspicious = bool(getattr(self, "probeguard_sanitize_only_suspicious", False)) if phase1_has_labels else False
        sanitize_mask = (~benign_mask) if sanitize_only_suspicious else torch.ones((n,), dtype=torch.bool)

        cone_deg = float(getattr(self, "probeguard_sanitize_cone_deg", 60.0))
        if bool(getattr(self, "probeguard_sanitize_use_cone", True)) and cone_deg > 0.0 and math.isfinite(cone_deg) and cone_deg < 180.0:
            cos_th = float(math.cos(math.radians(cone_deg)))
            sin_th = float(math.sin(math.radians(cone_deg)))

            proj_mag = torch.sum(G_san * u_ref[None, :], dim=1)  # [n]
            a = proj_mag[:, None] * u_ref[None, :]
            b = G_san - a
            b_norm = torch.norm(b, dim=1, p=2) + eps
            b_dir = b / b_norm[:, None]

            need = (cos < cos_th) & sanitize_mask
            if bool(torch.any(need)):
                new_dir = cos_th * u_ref[None, :] + sin_th * b_dir
                G_san = torch.where(need[:, None], norms[:, None] * new_dir, G_san)

        # ---- Sanitization: norm clipping ----
        if bool(getattr(self, "probeguard_sanitize_use_norm_clip", True)):
            sanitize_cap = float(getattr(self, "probeguard_sanitize_norm_cap", 0.0))
            if not (sanitize_cap > 0.0):
                mult2 = float(getattr(self, "probeguard_sanitize_norm_cap_mult", 1.0))
                sanitize_cap = max(eps, float(norm_cap) * float(mult2))
            sanitize_cap = max(eps, float(sanitize_cap))

            norms2 = torch.norm(G_san, dim=1, p=2) + eps
            clip_scale = torch.minimum(torch.ones_like(norms2), torch.tensor(sanitize_cap, dtype=torch.float32) / norms2)
            clip_scale = torch.where(sanitize_mask, clip_scale, torch.ones_like(clip_scale))
            G_san = G_san * clip_scale[:, None]

        # ---- Aggregate with trust weights (and sample-size weights) ----
        final_w = trust * w_samples
        sum_w = float(final_w.sum().item())
        if (not math.isfinite(sum_w)) or sum_w <= eps:
            final_w = w_samples
            sum_w = float(final_w.sum().item())

        agg_vec = (G_san * final_w[:, None]).sum(dim=0) / max(eps, sum_w)

        # Convert to delta state_dict and apply
        delta_sd = _unflatten_vec_to_delta_sd(agg_vec, global_sd, keys)
        new_sd = _sd_add(global_sd, delta_sd)

        # ---- Update reference state for next rounds ----
        self._probeguard_prev_agg_delta_vec = agg_vec.detach().to(torch.float32).cpu()

        if self._probeguard_use_ema_reference:
            beta = float(getattr(self, "probeguard_reference_ema_beta", 0.9))
            beta = float(np.clip(beta, 0.0, 0.999999))
            prev = getattr(self, "_probeguard_ref_ema_vec", None)
            target_ref = ref_source_vec.detach().to(torch.float32).cpu()
            if prev is None or (not isinstance(prev, torch.Tensor)) or int(prev.numel()) != int(agg_vec.numel()):
                self._probeguard_ref_ema_vec = target_ref
            else:
                self._probeguard_ref_ema_vec = (beta * prev.detach().to(torch.float32).cpu() + (1.0 - beta) * target_ref)

        # ---------------- SCAFFOLD: update global control variate c ----------------
        scaffold_c_updated = 0
        if str(getattr(self, 'probeguard_client_method', '')).lower().strip() == 'scaffold':
            if self._scaffold_c_sd is None:
                self._scaffold_c_sd = OrderedDict((k, torch.zeros_like(v).cpu()) for k, v in global_sd.items())

            # Weighted average delta-c over replies (aligned with final_w)
            dc_sum = OrderedDict((k, torch.zeros_like(v).cpu()) for k, v in self._scaffold_c_sd.items())
            dc_wsum = 0.0

            for i in range(n):
                dc = _reply_to_scaffold_delta_c(replies[i])
                if dc is None:
                    continue
                wi = float(final_w[i].item())
                if wi <= 0.0 or (not math.isfinite(wi)):
                    continue
                dc_wsum += wi
                for k, v in dc.items():
                    if k in dc_sum and dc_sum[k].shape == v.shape:
                        dc_sum[k] += v.detach().cpu() * wi

            if dc_wsum > eps:
                for k in self._scaffold_c_sd.keys():
                    self._scaffold_c_sd[k] = (self._scaffold_c_sd[k] + (dc_sum[k] / float(dc_wsum))).detach().cpu()
                scaffold_c_updated = 1
        # --------------------------------------------------------------------------

        # Metrics (avoid strings; lists must be numeric)
        phase1_map = {"update": 2, "none": 3, "oracle": 11, "random": 12}
        out_metrics: Dict[str, object] = {
            "round": int(server_round),
            "probeguard_phase1_mode": int(phase1_map.get(phase1_mode, 0)),
            "probeguard_suspects": int(len(suspects)),
            "probeguard_trust_mean": float(torch.mean(trust).item()) if trust.numel() > 0 else 0.0,
            "probeguard_trust_min": float(torch.min(trust).item()) if trust.numel() > 0 else 0.0,
            "probeguard_trust_max": float(torch.max(trust).item()) if trust.numel() > 0 else 0.0,
            "probeguard_scaffold_c_updated": int(scaffold_c_updated),
        }

        # Store suspect ids (as list[int]) if possible
        if suspects:
            try:
                sus_sorted = sorted(set(map(str, suspects)), key=lambda s: int(s))
                out_metrics["probeguard_suspect_ids"] = [int(s) for s in sus_sorted]
            except Exception:
                pass

        self._probeguard_prev_global_sd = global_sd
        return ArrayRecord(new_sd), MetricRecord(out_metrics)


# ======================================================================
# ======================= Final Summary at Exit ========================

def _sanitize_filename_part(x: object) -> str:
    """Sanitize arbitrary config values for safe filenames."""
    s = str(x)
    s = s.replace("/", "_").replace("\\", "_").strip()
    out = []
    for ch in s:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out)
    return cleaned if cleaned else "na"


def _results_filename_from_config(run_config: dict) -> str:
    dataset_name = _sanitize_filename_part(run_config.get("dataset_name", "unknown"))
    alpha = _sanitize_filename_part(run_config.get("dirichlet_alpha", "na"))
    attack_type = _sanitize_filename_part(run_config.get("attack_type", "None"))
    aggregation_method = _sanitize_filename_part(run_config.get("aggregation_method", "None"))
    Mal_Percentage = _sanitize_filename_part(run_config.get("Malicious-Client-Percentage", "None"))
    return f"{dataset_name}({alpha})-{attack_type}({Mal_Percentage})-{aggregation_method}.txt"


def print_final_results() -> None:
    """Print and save the final metric: avg accuracy over the last 10 eval rounds."""
    global _RUN_CONFIG_SNAPSHOT
    global _RUN_COMPLETED_OK, _RUN_CRASHED, _RUN_ERROR_MSG, _LAST_SERVER_ROUND

    # If the run crashed (or did not finish cleanly), write an explicit error marker.
    # This avoids saving/printing a misleading rolling average based on partial progress.
    if _RUN_CRASHED or not _RUN_COMPLETED_OK:
        err_msg = _RUN_ERROR_MSG or "Run did not complete"
        print("")
        print("[FINAL] Summary of evaluation:")
        print(f"[FINAL] ERROR: run crashed around round {_LAST_SERVER_ROUND}: {err_msg}")
        print("")

        filename = "results.txt"
        if isinstance(_RUN_CONFIG_SNAPSHOT, dict) and _RUN_CONFIG_SNAPSHOT:
            filename = _results_filename_from_config(_RUN_CONFIG_SNAPSHOT)

        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(filename, "a", encoding="utf-8") as f:
            # Keep the first field numeric-friendly for downstream parsing.
            f.write(f"nan\tERROR\tround={_LAST_SERVER_ROUND}\t{err_msg}\n")
        return

    if not recent_accuracies:
        print("[FINAL] No evaluation rounds were recorded.")
        return

    last_k = recent_accuracies[-LAST_N_AVG_ROUNDS:]  # if <30, use what's available
    avg_last_10 = sum(last_k) / len(last_k)

    print("")
    print("[FINAL] Summary of evaluation:")
    print(f"[FINAL] Rolling Avg Accuracy over last {len(last_k)} rounds: {avg_last_10:.4f}")
    print("")

    # File name: dataset_name-dirichlet_alpha-attack_type-aggregation_method.txt
    filename = "results.txt"
    if isinstance(_RUN_CONFIG_SNAPSHOT, dict) and _RUN_CONFIG_SNAPSHOT:
        filename = _results_filename_from_config(_RUN_CONFIG_SNAPSHOT)

    # Append one line per run (same config => same file)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"{avg_last_10:.6f}\n")


atexit.register(print_final_results)


# ======================================================================
# ============================= main() =================================

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    run_config = resolved_attack_config(context.run_config)

    print(f"[SERVER] run_config = {run_config}")

    global _RUN_CONFIG_SNAPSHOT
    _RUN_CONFIG_SNAPSHOT = dict(run_config)

    global num_clients_global, malicious_percentage_global

    # Read run config
    num_rounds: int = int(run_config["num-server-rounds"])

    aggregation_method = str(run_config.get("aggregation_method", "fedavg"))
    trim_ratio = float(run_config.get("trim_ratio", 0.2))
    krum_f = int(run_config.get("krum_f", -1))
    multi_krum_m = int(run_config.get("multi_krum_m", 3))

    fltrust_root_epochs = int(run_config.get("fltrust_root_epochs", 1))
    fltrust_root_lr = float(run_config.get("fltrust_root_lr", 0.001))
    fltrust_cos_clip = float(run_config.get("fltrust_cos_clip", 0.0))

    # Read federated setup from run_config
    num_clients_global = int(run_config.get("num_clients"))
    # Malicious percentage configuration:
    #   - legacy: Malicious-Client-Percentage
    #   - new: Malicious-Client-Percentage-min / -max (per-round deterministic in [min,max])
    legacy_mal = float(run_config.get("Malicious-Client-Percentage", 0.0))
    malicious_percentage_min = float(
        run_config.get("Malicious-Client-Percentage-min", legacy_mal)
    )
    malicious_percentage_max = float(
        run_config.get("Malicious-Client-Percentage-max", legacy_mal)
    )
    # Keep a single "global" value for backward-compatible parts of the code (e.g., Krum f derivation)
    malicious_percentage_global = float(malicious_percentage_max)

    # Build FLTrust root loader from a random subset of the full dataset
    # (the subset is sampled once in task.py and re-used across rounds).
    root_train_loader = load_root_data(
        num_partitions=num_clients_global,
        run_config=run_config,
        server_round=0,
    )

    def model_fn():
        return build_model(num_partitions=num_clients_global, run_config=run_config)

    print(
        f"[SERVER] num_clients_global={num_clients_global}, "
        f"malicious_percentage_global={malicious_percentage_global}, "
        f"malicious_percentage_min={malicious_percentage_min}, "
        f"malicious_percentage_max={malicious_percentage_max}"
    )

    # ---------------------------------------------------------------------

    fraction_train = float(run_config.get("fraction-train", run_config.get("fraction-fit", 1.0)))
    fraction_evaluate = float(run_config.get("fraction-evaluate", 1.0))

    min_available_nodes = num_clients_global
    min_train_nodes = max(1, int(round(fraction_train * num_clients_global)))
    min_evaluate_nodes = max(1, int(round(fraction_evaluate * num_clients_global)))

    def derive_f_if_needed(f_val: int) -> int:
        if f_val >= 0:
            return f_val
        # expected number of malicious among the sampled fit clients
        return int(math.ceil(min_train_nodes * malicious_percentage_global))

    def weighted_average(metrics):
        # metrics is typically list of tuples: (num_examples, {"accuracy":..., ...})
        total = sum(n for n, _ in metrics) or 1
        out = {}
        for n, m in metrics:
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    out[k] = out.get(k, 0.0) + (n / total) * float(v)
        return out

    base_seed = int(run_config.get("seed"))

    # ---------------------------------------------------------------------
    # FedIMP: clear the shared artifact directory at the *start of the run*
    # to avoid stale leader/update artifacts from previous runs (especially
    # when reusing the same seed/run_tag).
    attack_type_cfg = str(run_config.get("attack_type", "None"))
    fedimp_share = bool(run_config.get("fedimp_share_across_malicious", False))
    if attack_type_cfg.lower() == "fedimp" and fedimp_share and aggregation_method.strip().lower() != "no-attack":
        try:
            shared_root = Path(str(run_config.get("fedimp_shared_dir", "./.fedimp_shared"))).expanduser().resolve()
            run_tag = str(run_config.get("fedimp_run_tag", f"seed{base_seed}"))
            shared_dir = (shared_root / run_tag).resolve()
            # Safety: only delete the run_tag subdir under shared_root.
            shared_dir.relative_to(shared_root)
            if shared_dir.exists():
                print(f"[SERVER] Clearing FedIMP shared dir at startup: {shared_dir}", flush=True)
                shutil.rmtree(shared_dir, ignore_errors=True)
            shared_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[SERVER] WARNING: could not clear FedIMP shared dir: {e}", flush=True)

    common = dict(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=min_train_nodes,
        min_evaluate_nodes=min_evaluate_nodes,
        min_available_nodes=min_available_nodes,
        # NOTE: ServerApp FedAvg uses evaluate_metrics_aggr_fn (different signature).
        # If you don't need custom aggregation, just omit it and let FedAvg do the default.
        num_clients_global=num_clients_global,
        malicious_percentage_global=malicious_percentage_global,
        malicious_percentage_min=malicious_percentage_min,
        malicious_percentage_max=malicious_percentage_max,
        model_fn=model_fn,
        # Attack configuration for ALIE/IPM and other attack paths.
        attack_type=str(run_config.get("attack_type", "None")),
        alie_z=float(run_config.get("alie_z", -1.0)),
        ipm_gamma=float(run_config.get("ipm_gamma", 20.0)),
        ipm_norm_match=bool(run_config.get("ipm_norm_match", True)),
        base_seed=base_seed,
        deterministic_sampling=True,
    )

    # ---------------------------------------------------------------------------------------

    # Init global model (deterministic init)
    with _temp_round_seed(base_seed):
        net = model_fn()
    arrays = ArrayRecord(net.state_dict())

    # Strategy which calls our aggregate_evaluate() each round
    aggregation_method = str(run_config.get("aggregation_method", "fedavg")).lower()


    if aggregation_method in {"fedavg", "no-attack"}:
        strategy = FedAvgStrategy(**common)
    elif aggregation_method == "median":
        strategy = MedianStrategy(**common)
    elif aggregation_method == "trimmed_mean":
        trim_ratio = float(run_config.get("trim_ratio", 0.2))
        strategy = TrimmedMeanStrategy(trim_ratio=trim_ratio, **common)
    elif aggregation_method == "krum":
        krum_f = int(run_config.get("krum_f", 1))
        krum_f = derive_f_if_needed(krum_f)
        strategy = KrumStrategy(krum_f=krum_f, **common)
    elif aggregation_method == "multi_krum":
        krum_f = int(run_config.get("krum_f", 1))
        krum_f = derive_f_if_needed(krum_f)
        multi_krum_m = int(run_config.get("multi_krum_m", 5))
        strategy = MultiKrumStrategy(krum_f=krum_f, multi_krum_m=multi_krum_m, **common)
    elif aggregation_method == "fltrust":
        strategy = FLTrustStrategy(
            root_trainloader=root_train_loader,
            root_run_config=dict(run_config),
            root_lr=fltrust_root_lr,
            fltrust_root_epochs=fltrust_root_epochs,
            fltrust_cos_clip=fltrust_cos_clip,
            **common,
        )
    elif aggregation_method == "shieldfl":
        strategy = ShieldFLStrategy(**common)
    elif aggregation_method == "dnc":
        dnc_f = int(run_config.get("dnc_f", -1))  # <-- changed (was 1)
        dnc_c = float(run_config.get("dnc_c", 1.5))
        dnc_niters = int(run_config.get("dnc_niters", 5))
        dnc_b = int(run_config.get("dnc_b", 256))
        strategy = DnCStrategy(dnc_f=dnc_f, dnc_c=dnc_c, dnc_niters=dnc_niters, dnc_b=dnc_b, **common)
    elif aggregation_method == "feddmc":
        pca_dim = int(run_config.get("feddmc_pca_dim", 5))
        min_cluster_size = int(run_config.get("feddmc_min_cluster_size", 2))
        ema_alpha = float(run_config.get("feddmc_ema_alpha", 0.8))
        trust_threshold = float(run_config.get("feddmc_trust_threshold", 0.5))
        strategy = FedDMCStrategy(
            pca_dim=pca_dim,
            min_cluster_size=min_cluster_size,
            ema_alpha=ema_alpha,
            trust_threshold=trust_threshold,
            **common,
        )
    elif aggregation_method == "gas":
        strategy = GASStrategy(
            gas_p=int(run_config.get("gas_p", 1000)),
            gas_f=int(run_config.get("gas_f", -1)),
            gas_base_agr=str(run_config.get("gas_base_agr", "median")),
            # GAS(DnC) parameters are entirely separate from standalone DnC.
            # With gas_p=1000 on the Covertype MLP, b=2000 exceeds each split
            # group dimension, so one full-group spectral pass is sufficient.
            gas_dnc_c=float(run_config.get("gas_dnc_c", 1.0)),
            gas_dnc_niters=int(run_config.get("gas_dnc_niters", 1)),
            gas_dnc_b=int(run_config.get("gas_dnc_b", 2000)),
            **common,
        )
    elif aggregation_method == "sdea":
        # Only SDEA loads unlabeled public data; never uses test examples.
        sdea_X = load_sdea_public_data(
            num_partitions=num_clients_global, run_config=run_config,
        )
        strategy = SDEAStrategy(
            sdea_public_X=sdea_X,
            sdea_public_epochs=int(run_config.get("sdea_public_epochs", 20)),
            sdea_public_batch_size=int(run_config.get("sdea_public_batch_size", 64)),
            sdea_public_lr=float(run_config.get("sdea_public_lr", 0.005)),
            sdea_use_cc=bool(run_config.get("sdea_use_cc", True)),
            sdea_device=str(run_config.get("sdea_device", "cpu")),
            **common,
        )
    elif aggregation_method in {"probeguard", "probeguard-ema"}:
        strategy = ProbeGuardStrategy(
            probeguard_warmup_rounds=int(run_config.get("probeguard_warmup_rounds", 5)),
            probeguard_probe_bank_size=int(run_config.get("probeguard_probe_bank_size", 64)),
            probeguard_probe_refresh=int(run_config.get("probeguard_probe_refresh", 16)),
            probeguard_refresh_every=int(run_config.get("probeguard_refresh_every", 1)),
            probeguard_bounds_low=float(run_config.get("probeguard_bounds_low", -3.0)),
            probeguard_bounds_high=float(run_config.get("probeguard_bounds_high", 3.0)),
            probeguard_use_data_bounds=bool(run_config.get("probeguard_use_data_bounds", False)),
            probeguard_data_bounds_clip=float(run_config.get("probeguard_data_bounds_clip", 8.0)),
            probeguard_suspect_max=int(run_config.get("probeguard_suspect_max", 16)),
            probeguard_mad_alpha=float(run_config.get("probeguard_mad_alpha", 3.0)),
            probeguard_probe_mad_mode=str(run_config.get("probeguard_probe_mad_mode", "both")),
            probeguard_probe_normalize_per_probe=bool(run_config.get("probeguard_probe_normalize_per_probe", False)),
            probeguard_probe_scale_floor_ratio=float(run_config.get("probeguard_probe_scale_floor_ratio", 0.05)),
            probeguard_probe_threshold_normalized_ranking=bool(run_config.get("probeguard_probe_threshold_normalized_ranking", False)),
            probeguard_phase1_hybrid_update=bool(run_config.get("probeguard_phase1_hybrid_update", False)),
            probeguard_phase1_hybrid_probe_weight=float(run_config.get("probeguard_phase1_hybrid_probe_weight", 0.5)),
            probeguard_top_frac=float(run_config.get("probeguard_top_frac", 0.1)),
            probeguard_use_kmeans=bool(run_config.get("probeguard_use_kmeans", True)),
            probeguard_cluster_method=str(run_config.get("probeguard_cluster_method", "kmeans")),
            probeguard_iforest_contamination=float(run_config.get("probeguard_iforest_contamination", -1.0)),
            probeguard_probe_feature_mode=str(run_config.get("probeguard_probe_feature_mode", "pair_probs")),
            probeguard_phase1_mode=str(run_config.get("probeguard_phase1_mode", "random")),
            use_ema_reference=(aggregation_method == "probeguard-ema"),
            probeguard_reference_ema_beta=float(run_config.get("probeguard_reference_ema_beta", 0.9)),
            probeguard_reference_use_benign_median=bool( run_config.get("probeguard_reference_use_benign_median", False )),
            probeguard_trust_only_suspicious=bool(run_config.get("probeguard_trust_only_suspicious", False)),
            probeguard_trust_use_cos=bool(run_config.get("probeguard_trust_use_cos", True)),
            probeguard_trust_cos_tau=float(run_config.get("probeguard_trust_cos_tau", 0.0)),
            probeguard_trust_cos_gamma=float(run_config.get("probeguard_trust_cos_gamma", 5.0)),
            probeguard_trust_use_norm=bool(run_config.get("probeguard_trust_use_norm", True)),
            probeguard_trust_norm_cap=float(run_config.get("probeguard_trust_norm_cap", 0.0)),
            probeguard_trust_norm_cap_mult=float(run_config.get("probeguard_trust_norm_cap_mult", 1.5)),
            probeguard_trust_norm_lambda=float(run_config.get("probeguard_trust_norm_lambda", 1.0)),
            probeguard_trust_use_ortho=bool(run_config.get("probeguard_trust_use_ortho", True)),
            probeguard_trust_ortho_lambda=float(run_config.get("probeguard_trust_ortho_lambda", 2.0)),
            probeguard_trust_use_suspect_weight=bool(run_config.get("probeguard_trust_use_suspect_weight", True)),
            probeguard_suspect_weight=float(run_config.get("probeguard_suspect_weight", 0.1)),
            probeguard_sanitize_only_suspicious=bool(run_config.get("probeguard_sanitize_only_suspicious", False)),
            probeguard_sanitize_use_cone=bool(run_config.get("probeguard_sanitize_use_cone", True)),
            probeguard_sanitize_cone_deg=float(run_config.get("probeguard_sanitize_cone_deg", 60.0)),
            probeguard_sanitize_use_norm_clip=bool(run_config.get("probeguard_sanitize_use_norm_clip", True)),
            probeguard_sanitize_norm_cap=float(run_config.get("probeguard_sanitize_norm_cap", 0.0)),
            probeguard_sanitize_norm_cap_mult=float(run_config.get("probeguard_sanitize_norm_cap_mult", 1.0)),
            probeguard_seed=run_config.get("probeguard_seed", None),
            probeguard_pca_var=float(run_config.get("probeguard_pca_var", 0.95)),
            probeguard_use_mad=bool(run_config.get("probeguard_use_mad", True)),
            probeguard_client_method=str(run_config.get("probeguard_client_method", "fedprox")),
            **common,
        )
    elif aggregation_method == "oracle_drop":
        strategy = OracleDropStrategy(**common)
    else:
        raise ValueError(f"Unknown aggregation_method={aggregation_method}")


    # ---------------------------------------------------------------------
    # Bootstrap node_id -> partition-id mapping so round logs can print stable
    # partition numbers (0..num_clients-1) instead of Flower runtime node IDs.
    # This requires the ClientApp to implement `@app.query("whoami")`.
    try:
        whoami_timeout = float(run_config.get("whoami_timeout", 30.0))
        # Make timeout accessible to the strategy (for internal retries)
        try:
            setattr(strategy, "whoami_timeout", whoami_timeout)
        except Exception:
            pass

        # Retry a few times to reliably build a COMPLETE node_id->partition-id map.
        # In Ray simulation, some actors may not be ready immediately.
        node_map = {}
        for _ in range(20):
            node_map = _build_node_id_to_partition_id_map(grid, timeout=whoami_timeout)
            if node_map and len(node_map) >= int(num_clients_global):
                break
            time.sleep(0.1)

        if not node_map or len(node_map) < int(num_clients_global):
            raise RuntimeError(
                f"Deterministic run aborted: whoami node_id->partition-id map is incomplete "
                f"(have {len(node_map)}, need {int(num_clients_global)})."
            )

        # All our strategies inherit from _BaseMonitoredStrategy and provide this attr
        getattr(strategy, "_node_id_to_partition_id", {}).update(node_map)
        print(f"[SERVER] whoami node_id->partition-id map size={len(getattr(strategy, '_node_id_to_partition_id', {}))}")
    except Exception as e:
        raise RuntimeError(f"Deterministic run aborted during whoami bootstrap: {e}") from e


    # Start strategy, run for `num_rounds`
    global _RUN_COMPLETED_OK, _RUN_CRASHED, _RUN_ERROR_MSG
    try:
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=num_rounds,
        )
        _RUN_COMPLETED_OK = True

        # Save final model
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")
        # ProbeGuard-only, completed-run detection report; other strategies unchanged.
        if isinstance(strategy, ProbeGuardStrategy):
            strategy.write_phase1_detection_report(run_config)
    except Exception as e:
        _RUN_CRASHED = True
        _RUN_ERROR_MSG = f"{type(e).__name__}: {e}"
        raise

