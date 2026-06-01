#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cGAN for GridOPT per-bus P,Q forecasting (UPDATED to match MLP pipeline)
----------------------------------------------------------------------
Goals:
- Same evaluation protocol + figure style as your MLP baseline, so you can compare apples-to-apples.

Fixes / updates applied:
1) ✅ Correct time alignment:
   past = arr[:Tin], future = arr[Tin:Tin+Tout]  (contiguous)
2) ✅ Test normalization policy is now well-defined:
   - Compute GLOBAL stats from TRAIN folders only
   - Use those same stats for ALL test folders (strict unseen eval)
3) ✅ Faster test evaluation:
   - Batch inference over ALL buses in one forward pass (no per-bus loop)
4) ✅ Same figure rules as MLP:
   - PDF (not PNG), no title, no legend
   - Short axis labels with LaTeX mathtext (e.g. epoch, MAE, RMSE, KS, Mean |Δω|)
   - Save NPZ for curves and PQ plots
5) ✅ Evaluate TEST (5 folders) after every global epoch:
   - Save overall test metrics vs epoch (CSV + NPZ)
   - Plot overall test metric curves vs epoch (PDF)

Notes:
- Generator is stochastic; to make metric curves reproducible, we use a deterministic torch.Generator seeded by global_epoch.
"""

import os
import json
import time
import warnings
import csv
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = r"C:\Users\omar shadafny\Desktop\Omar_Elinor\data\extracted_datasets\gridopt-dataset-tmp\dataset_release_1\pglib_opf_case118_ieee"
OUT_ROOT  = r"C:\Users\omar shadafny\Desktop\Omar_Elinor\realdatacgan"

TRAIN_FOLDERS = 1
TEST_FOLDERS  = 1
MAX_JSONS_PER_FOLDER = 1500
SPLIT_RATIO = 0.75

EPOCHS_PER_FOLDER = 30
BATCH_SIZE = 64

LR_G = 2e-4
LR_D = 2e-4
Z_DIM = 32
LAMBDA_L1 = 10.0

SEED = 7

# Evaluate test every N global epochs
EVAL_TEST_EVERY = 1

# Save PQ plots once for first test folder
SAVE_PQ_PLOTS_FOR_FIRST_TEST_FOLDER = True
PQ_BUSES = (0, 1)

# ======================== PJM REAL-DATA EXTERNAL TEST ========================
# This is a real operational load dataset converted to pseudo-bus P/Q sequences.
# Each PJM load area becomes one pseudo-bus. P is real measured MW load as negative injection.
# Q is estimated using an assumed power factor.
RUN_PJM_REAL_TEST = True
PJM_REAL_CSV_PATH = r"C:\Users\omar shadafny\Desktop\Omar_Elinor\pjm_hrl_load_metered_1_1_3_23_2024.csv"
PJM_POWER_FACTOR = 0.95
PJM_SAVE_PLOTS = True
PJM_PLOT_BUSES = (0, 1)

# Plot scaling only: keep training/evaluation metrics in MW/MVar, but display plots in GW/GVar.
# 1000 MW = 1 GW, 1000 MVar = 1 GVar.
PLOT_POWER_SCALE = 1000.0
PLOT_P_LABEL = r"$P~[\mathrm{GW}]$"
PLOT_Q_LABEL = r"$Q~[\mathrm{GVar}]$"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

# Output dirs
FIGS_DIR = os.path.join(OUT_ROOT, "figs")
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# =========================================================
# FIGURE STYLE GUIDELINES (match your MLP)
# =========================================================
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "Times New Roman",
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "axes.unicode_minus": True,
})

COLORS_QUAL = [
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854",
    "#ffd92f", "#e5c494", "#b3b3b3", "#1b9e77", "#d95f02"
]

XLAB_EPOCH = r"$\mathrm{epoch}$"


# =========================================================
# UTILS: KS, metrics, saving, smoothing
# =========================================================
def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a.reshape(-1))
    b = np.sort(b.reshape(-1))
    na, nb = a.size, b.size
    i = j = 0
    cdf_a = cdf_b = 0.0
    d = 0.0
    while i < na and j < nb:
        if a[i] < b[j]:
            i += 1
            cdf_a = i / na
        elif b[j] < a[i]:
            j += 1
            cdf_b = j / nb
        else:
            i += 1
            j += 1
            cdf_a = i / na
            cdf_b = j / nb
        d = max(d, abs(cdf_a - cdf_b))
    while i < na:
        i += 1
        cdf_a = i / na
        d = max(d, abs(cdf_a - cdf_b))
    while j < nb:
        j += 1
        cdf_b = j / nb
        d = max(d, abs(cdf_a - cdf_b))
    return float(d)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    eps = 1e-6
    mape = float(np.mean(np.abs(diff) / (np.abs(y_true) + eps)))
    ks_p = ks_statistic(y_true[..., 0], y_pred[..., 0])
    ks_q = ks_statistic(y_true[..., 1], y_pred[..., 1])
    ks_val = float(max(ks_p, ks_q))
    return {"mae": mae, "rmse": rmse, "mape": mape, "ks": ks_val}


def save_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1:
        return x
    w = min(int(w), x.size)
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(x, kernel, mode="same")


# =========================================================
# PLOTTING HELPERS (PDF, no legend, no title)
# =========================================================
def plot_single_curve_pdf(x, y, xlabel, ylabel, pdf_path, color="#66c2a5", ylog=False):
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if ylog:
        ax.set_yscale("log")
        pos = y[y > 0]
        if pos.size > 0:
            ax.set_ylim(float(np.min(pos)) * 0.7, None)
    plt.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def save_power_flow_figs_and_data(
    fut_all: np.ndarray,   # (Tout,B,2), values in MW/MVar; plotted after /PLOT_POWER_SCALE
    preds: np.ndarray,     # (Tout,B,2), values in MW/MVar; plotted after /PLOT_POWER_SCALE
    tout: int,
    gname: str,
    out_dir: str,
    buses=(0, 1),
):
    t_axis = np.arange(tout)

    for b in buses:
        if b >= fut_all.shape[1]:
            continue

        # P
        fig, ax = plt.subplots()
        ax.plot(t_axis, fut_all[:, b, 0] / PLOT_POWER_SCALE, linewidth=2, color=COLORS_QUAL[0])
        ax.plot(t_axis, preds[:, b, 0] / PLOT_POWER_SCALE, "--", linewidth=2, color=COLORS_QUAL[1])
        ax.set_xlabel(r"$\mathrm{t}$")
        ax.set_ylabel(PLOT_P_LABEL)
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"cgan_{gname}_bus{b}_P.pdf"), bbox_inches="tight")
        plt.close(fig)

        np.savez_compressed(
            os.path.join(out_dir, f"cgan_{gname}_bus{b}_P_data.npz"),
            time=t_axis, real=fut_all[:, b, 0], pred=preds[:, b, 0]
        )

        # Q
        fig, ax = plt.subplots()
        ax.plot(t_axis, fut_all[:, b, 1] / PLOT_POWER_SCALE, linewidth=2, color=COLORS_QUAL[0])
        ax.plot(t_axis, preds[:, b, 1] / PLOT_POWER_SCALE, "--", linewidth=2, color=COLORS_QUAL[1])
        ax.set_xlabel(r"$\mathrm{t}$")
        ax.set_ylabel(PLOT_Q_LABEL)
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"cgan_{gname}_bus{b}_Q.pdf"), bbox_inches="tight")
        plt.close(fig)

        np.savez_compressed(
            os.path.join(out_dir, f"cgan_{gname}_bus{b}_Q_data.npz"),
            time=t_axis, real=fut_all[:, b, 1], pred=preds[:, b, 1]
        )


# =========================================================
# DATA LOADING
# =========================================================
def list_jsons(folder: str, limit: Optional[int]) -> List[str]:
    def extract_number(fn: str) -> int:
        try:
            return int(os.path.splitext(fn)[0].split("_")[-1])
        except Exception:
            return 10 ** 9
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    files.sort(key=extract_number)
    if limit and limit > 0:
        files = files[:limit]
    return files


def _parse_load(ld) -> Tuple[float, float]:
    p, q = 0.0, 0.0
    if isinstance(ld, dict):
        p = ld.get("p_mw", ld.get("p", 0.0))
        q = ld.get("q_mvar", ld.get("q", 0.0))
    elif isinstance(ld, (list, tuple)) and len(ld) >= 2:
        p, q = ld[0], ld[1]
    return float(p or 0.0), float(q or 0.0)


def _parse_gen(gn) -> float:
    if isinstance(gn, (list, tuple)) and len(gn) >= 2:
        return float(gn[1] or 0.0)
    if isinstance(gn, dict):
        return float(gn.get("p_mw", gn.get("p", 0.0)) or 0.0)
    return 0.0


def load_pq_sequence(group_path: str, max_jsons: int) -> Optional[np.ndarray]:
    files = list_jsons(group_path, max_jsons)
    if len(files) < 4:
        return None

    seq_list = []
    bus_count = None

    for fn in files:
        fp = os.path.join(group_path, fn)
        try:
            with open(fp, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        grid = data.get("grid", {})
        nodes = grid.get("nodes", {})
        edges = grid.get("edges", {})

        buses = nodes.get("bus", None)
        if buses is None:
            continue
        cur_B = len(buses)
        if bus_count is None:
            bus_count = cur_B
        if cur_B != bus_count:
            continue

        P = np.zeros(bus_count, dtype=np.float32)
        Q = np.zeros(bus_count, dtype=np.float32)

        # loads
        loads = nodes.get("load", [])
        llink = edges.get("load_link", {})
        lsend = llink.get("senders", [])
        lrecv = llink.get("receivers", [])
        if loads and lrecv:
            for li, bi in zip(lsend, lrecv):
                if 0 <= li < len(loads) and 0 <= bi < bus_count:
                    lp, lq = _parse_load(loads[li])
                    P[bi] -= lp
                    Q[bi] -= lq
        elif loads:
            for i, ld in enumerate(loads):
                if i >= bus_count:
                    break
                lp, lq = _parse_load(ld)
                P[i] -= lp
                Q[i] -= lq

        # generators
        gens = nodes.get("generator", [])
        glink = edges.get("generator_link", {})
        gsend = glink.get("senders", [])
        grecv = glink.get("receivers", [])
        if gens and grecv:
            for gi, bi in zip(gsend, grecv):
                if 0 <= gi < len(gens) and 0 <= bi < bus_count:
                    gp = _parse_gen(gens[gi])
                    P[bi] += gp
        elif gens:
            for i, gn in enumerate(gens):
                if i >= bus_count:
                    break
                gp = _parse_gen(gn)
                P[i] += gp

        seq_list.append(np.stack([P, Q], axis=1))

    if len(seq_list) < 4:
        return None
    return np.stack(seq_list, axis=0)  # (T,B,2)


# =========================================================
# MODELS
# =========================================================
class CGAN_G(nn.Module):
    def __init__(self, tin: int, tout: int, z_dim: int = 32, hid: int = 128):
        super().__init__()
        self.tout = tout
        self.enc = nn.GRU(2, hid, batch_first=True)
        self.fc = nn.Linear(hid + z_dim, tout * 2)

    def forward(self, past, z):
        _, h = self.enc(past)  # (1,B,H)
        h = h.squeeze(0)
        hz = torch.cat([h, z], dim=1)
        out = self.fc(hz)
        return out.view(-1, self.tout, 2)


class CGAN_D(nn.Module):
    def __init__(self, tin: int, tout: int, hid: int = 128):
        super().__init__()
        self.gru = nn.GRU(2, hid, batch_first=True)
        self.fc = nn.Linear(hid, 1)

    def forward(self, past, future):
        x = torch.cat([past, future], dim=1)  # (B, Tin+Tout, 2)
        _, h = self.gru(x)
        h = h.squeeze(0)
        return self.fc(h)


# =========================================================
# GLOBAL TRAIN STATS (for strict test normalization)
# =========================================================
def compute_global_train_stats(
    train_groups: List[str],
    cache: Dict[str, Optional[np.ndarray]],
    common_Tin: int,
    common_Tout: int
) -> Dict[str, np.ndarray]:
    X_all, Y_all = [], []
    for g in train_groups:
        arr = cache.get(g, None)
        if arr is None:
            continue
        T, B, _ = arr.shape
        if T < (common_Tin + common_Tout):
            continue

        past_all = arr[:common_Tin]
        fut_all  = arr[common_Tin:common_Tin + common_Tout]

        X = np.transpose(past_all, (1, 0, 2))  # (B,Tin,2)
        Y = np.transpose(fut_all,  (1, 0, 2))  # (B,Tout,2)
        X_all.append(X)
        Y_all.append(Y)

    if not X_all:
        raise RuntimeError("No valid training data for global stats.")
    Xc = np.concatenate(X_all, axis=0)
    Yc = np.concatenate(Y_all, axis=0)

    x_mean = Xc.mean(axis=(0, 1), keepdims=True)  # (1,1,2)
    x_std  = Xc.std(axis=(0, 1), keepdims=True) + 1e-6
    y_mean = Yc.mean(axis=(0, 1), keepdims=True)  # (1,1,2)
    y_std  = Yc.std(axis=(0, 1), keepdims=True) + 1e-6

    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


# =========================================================
# FAST BATCH PREDICTION (ALL buses at once)
# =========================================================
@torch.no_grad()
def predict_folder_batch(
    G: nn.Module,
    arr: np.ndarray,                 # (T,B,2)
    common_Tin: int,
    common_Tout: int,
    stats: Dict[str, np.ndarray],    # means/stds (1,1,2)
    z_gen: torch.Generator,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    T, B, _ = arr.shape
    if T < (common_Tin + common_Tout):
        return None

    past_all = arr[:common_Tin]                                # (Tin,B,2)
    fut_all  = arr[common_Tin:common_Tin + common_Tout]        # (Tout,B,2)

    x_mean, x_std = stats["x_mean"], stats["x_std"]
    y_mean, y_std = stats["y_mean"], stats["y_std"]

    X = np.transpose(past_all, (1, 0, 2))                      # (B,Tin,2)
    Xn = (X - x_mean.squeeze(0)) / x_std.squeeze(0)

    past_t = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)   # (B,Tin,2)
    z = torch.randn(B, Z_DIM, device=DEVICE, generator=z_gen)
    outn = G(past_t, z).detach().cpu().numpy()                      # (B,Tout,2)

    out = outn * y_std.squeeze(0) + y_mean.squeeze(0)               # (B,Tout,2)
    preds = np.transpose(out, (1, 0, 2))                             # (Tout,B,2)
    return fut_all, preds


# =========================================================
# TEST EVAL EACH EPOCH (overall + per-folder)
# =========================================================
@torch.no_grad()
def eval_on_test_folders(
    G: nn.Module,
    test_groups: List[str],
    common_Tin: int,
    common_Tout: int,
    global_stats: Dict[str, np.ndarray],
    seed: int,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], Optional[Tuple[str, np.ndarray, np.ndarray]]]:
    G.eval()

    all_true_list, all_pred_list = [], []
    per_folder_rows: List[Dict[str, Any]] = []
    first_plot_payload = None

    z_gen = torch.Generator(device=DEVICE)
    z_gen.manual_seed(seed)

    for g in test_groups:
        arr = load_pq_sequence(os.path.join(DATA_ROOT, g), MAX_JSONS_PER_FOLDER)
        if arr is None:
            continue

        pred_pack = predict_folder_batch(G, arr, common_Tin, common_Tout, global_stats, z_gen=z_gen)
        if pred_pack is None:
            continue

        fut_all, preds = pred_pack
        m = compute_all_metrics(fut_all, preds)

        per_folder_rows.append({"folder": g, **m})
        all_true_list.append(fut_all)
        all_pred_list.append(preds)

        if first_plot_payload is None:
            first_plot_payload = (g, fut_all, preds)

    if not all_true_list:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "ks": np.nan}, per_folder_rows, first_plot_payload

    all_true = np.concatenate(all_true_list, axis=0)
    all_pred = np.concatenate(all_pred_list, axis=0)
    overall = compute_all_metrics(all_true, all_pred)
    return overall, per_folder_rows, first_plot_payload




# =========================================================
# PJM REAL-DATA LOADER + EXTERNAL TEST
# =========================================================
def load_pjm_real_load_sequence(csv_path: str, pf: float = 0.95) -> Tuple[np.ndarray, List[str]]:
    """
    Load real PJM hourly metered load data and convert it to the model format:
        arr.shape = (T, B, 2)
    where:
        T = number of timestamps
        B = number of PJM load areas treated as pseudo-buses
        arr[..., 0] = P [MW] as negative load injection
        arr[..., 1] = Q [MVar] estimated from assumed power factor

    Expected columns in the PJM file:
        datetime, load_area, mw
    Some files may use slightly different capitalization; this function handles that.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"PJM CSV path does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Flexible column detection
    datetime_col = None
    for c in ["datetime", "datetime_beginning_ept", "datetime_beginning_utc", "timestamp", "time"]:
        if c in df.columns:
            datetime_col = c
            break

    area_col = None
    for c in ["load_area", "zone", "area", "name"]:
        if c in df.columns:
            area_col = c
            break

    mw_col = None
    for c in ["mw", "load", "load_mw", "metered_load_mw"]:
        if c in df.columns:
            mw_col = c
            break

    if datetime_col is None or area_col is None or mw_col is None:
        raise ValueError(
            "Could not find required PJM columns. Need datetime, load_area/zone, and mw/load column. "
            f"Found columns: {list(df.columns)}"
        )

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df[mw_col] = pd.to_numeric(df[mw_col], errors="coerce")
    df = df.dropna(subset=[datetime_col, area_col, mw_col])

    # Average duplicates if same timestamp/area appears more than once
    pivot = (
        df.groupby([datetime_col, area_col], as_index=False)[mw_col]
          .mean()
          .pivot(index=datetime_col, columns=area_col, values=mw_col)
          .sort_index()
    )

    # Keep areas with enough data, then interpolate missing values over time
    pivot = pivot.dropna(axis=1, how="all")
    pivot = pivot.interpolate(method="time", limit_direction="both")
    pivot = pivot.dropna(axis=1, how="any")

    if pivot.shape[0] < 4 or pivot.shape[1] < 1:
        raise RuntimeError(f"Not enough valid PJM data after preprocessing. Shape={pivot.shape}")

    area_names = [str(c) for c in pivot.columns]

    # Demand is treated as negative injection, matching GridOPT load convention in load_pq_sequence().
    P = -pivot.to_numpy(dtype=np.float32)  # (T,B)

    # Estimate reactive power from assumed power factor: |Q| = |P| tan(arccos(pf)).
    # Since P is negative for load, Q is also negative, matching inductive load convention.
    pf = float(pf)
    if not (0.0 < pf <= 1.0):
        raise ValueError("pf must be in (0, 1].")
    Q = P * np.tan(np.arccos(pf)).astype(np.float32) if hasattr(np.tan(np.arccos(pf)), 'astype') else P * np.float32(np.tan(np.arccos(pf)))

    arr = np.stack([P, Q], axis=2).astype(np.float32)  # (T,B,2)
    print(f"Loaded PJM real data: T={arr.shape[0]}, pseudo-buses={arr.shape[1]}, features={arr.shape[2]}")
    print("First PJM pseudo-buses:", area_names[:min(10, len(area_names))])
    return arr, area_names


@torch.no_grad()
def eval_on_pjm_real_data(
    G: nn.Module,
    csv_path: str,
    common_Tin: int,
    common_Tout: int,
    global_stats: Dict[str, np.ndarray],
    pf: float = 0.95,
    seed: int = 987654,
) -> Tuple[Optional[Dict[str, float]], Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Evaluate the trained cGAN generator on real PJM load data converted to pseudo-bus P/Q.
    Uses the same train-only global normalization as the GridOPT test.
    """
    print("\n=== PJM REAL-DATA TEST (cGAN) ===")
    arr, area_names = load_pjm_real_load_sequence(csv_path, pf=pf)

    if arr.shape[0] < (common_Tin + common_Tout):
        print(
            f"PJM data too short: T={arr.shape[0]}, "
            f"need at least Tin+Tout={common_Tin + common_Tout}. Skipping."
        )
        return None, None, None, area_names

    z_gen = torch.Generator(device=DEVICE)
    z_gen.manual_seed(seed)

    G.eval()
    pred_pack = predict_folder_batch(
        G=G,
        arr=arr,
        common_Tin=common_Tin,
        common_Tout=common_Tout,
        stats=global_stats,
        z_gen=z_gen,
    )

    if pred_pack is None:
        print("PJM prediction failed / data too short. Skipping.")
        return None, None, None, area_names

    fut_all, preds = pred_pack
    metrics = compute_all_metrics(fut_all, preds)

    print("\n=== PJM REAL-DATA TEST METRICS ===")
    print(
        f"MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, "
        f"MAPE={metrics['mape']:.6f}, KS={metrics['ks']:.6f}"
    )

    return metrics, fut_all, preds, area_names

# =========================================================
# MAIN
# =========================================================
def main():
    all_groups = sorted([d for d in os.listdir(DATA_ROOT) if d.startswith("group_")])
    train_groups = all_groups[:TRAIN_FOLDERS]
    test_groups  = all_groups[TRAIN_FOLDERS:TRAIN_FOLDERS + TEST_FOLDERS]
    print("train groups:", train_groups)
    print("test  groups:", test_groups)

    # ---- first pass: find common Tin/Tout + cache train arrays ----
    tins, touts = [], []
    cache: Dict[str, Optional[np.ndarray]] = {}

    for g in train_groups:
        arr = load_pq_sequence(os.path.join(DATA_ROOT, g), MAX_JSONS_PER_FOLDER)
        cache[g] = arr
        if arr is None:
            continue
        T, _, _ = arr.shape
        Tin = int(T * SPLIT_RATIO)
        Tout = T - Tin
        tins.append(Tin)
        touts.append(Tout)

    if not tins:
        raise RuntimeError("No train data in any folder")

    common_Tin  = min(tins)
    common_Tout = min(touts)
    print(f"common Tin={common_Tin}, common Tout={common_Tout}")

    # ---- init cGAN ----
    G = CGAN_G(tin=common_Tin, tout=common_Tout, z_dim=Z_DIM, hid=128).to(DEVICE)
    D = CGAN_D(tin=common_Tin, tout=common_Tout, hid=128).to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    l1  = nn.L1Loss()

    # ---- GLOBAL train stats for TEST normalization ----
    global_stats = compute_global_train_stats(train_groups, cache, common_Tin, common_Tout)
    print("Computed GLOBAL train stats for test normalization.")

    # logs
    training_logs: List[Dict[str, Any]] = []
    test_logs: List[Dict[str, Any]] = []

    test_epoch_overall_logs: List[Dict[str, Any]] = []
    test_epoch_per_folder_logs: List[Dict[str, Any]] = []

    # global epoch counter (for cross-model comparability)
    global_epoch = 0

    # ---- train folder by folder ----
    for g in train_groups:
        print(f"\n=== Training on folder: {g} ===")
        arr = cache[g]
        if arr is None:
            print(f"  skip {g} (no data)")
            continue

        T, B, _ = arr.shape
        if T < (common_Tin + common_Tout):
            print("  skip, too short for common Tin/Tout")
            continue

        # ✅ FIXED alignment
        past_all = arr[:common_Tin]
        fut_all  = arr[common_Tin:common_Tin + common_Tout]

        # per-bus samples: (B,Tin,2) and (B,Tout,2)
        X = np.transpose(past_all, (1, 0, 2))
        Y = np.transpose(fut_all,  (1, 0, 2))
        N = X.shape[0]

        # per-folder normalization (training only, same as your original behavior)
        x_mean = X.mean(axis=(0, 1), keepdims=True)
        x_std  = X.std(axis=(0, 1), keepdims=True) + 1e-6
        y_mean = Y.mean(axis=(0, 1), keepdims=True)
        y_std  = Y.std(axis=(0, 1), keepdims=True) + 1e-6

        Xn = (X - x_mean) / x_std
        Yn = (Y - y_mean) / y_std

        idxs = np.arange(N)

        for ep in range(1, EPOCHS_PER_FOLDER + 1):
            global_epoch += 1
            np.random.shuffle(idxs)

            G.train()
            D.train()
            d_sum = 0.0
            g_sum = 0.0
            t0 = time.time()

            if DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats(DEVICE)

            for start in range(0, N, BATCH_SIZE):
                end = start + BATCH_SIZE
                bidx = idxs[start:end]

                past = torch.tensor(Xn[bidx], dtype=torch.float32, device=DEVICE)      # (B,Tin,2)
                real_future = torch.tensor(Yn[bidx], dtype=torch.float32, device=DEVICE)  # (B,Tout,2)
                Bcur = past.size(0)

                real_label = torch.ones(Bcur, 1, device=DEVICE)
                fake_label = torch.zeros(Bcur, 1, device=DEVICE)

                # ---------------- D step ----------------
                z = torch.randn(Bcur, Z_DIM, device=DEVICE)
                fake_future = G(past, z).detach()

                d_real = D(past, real_future)
                d_fake = D(past, fake_future)

                loss_D = 0.5 * (bce(d_real, real_label) + bce(d_fake, fake_label))
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

                # ---------------- G step ----------------
                z = torch.randn(Bcur, Z_DIM, device=DEVICE)
                fake_future = G(past, z)
                d_fake_for_g = D(past, fake_future)

                adv_loss = bce(d_fake_for_g, real_label)
                recon_loss = l1(fake_future, real_future)
                loss_G = adv_loss + LAMBDA_L1 * recon_loss

                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

                d_sum += loss_D.item() * Bcur
                g_sum += loss_G.item() * Bcur

            dt = time.time() - t0
            mem_mb = float("nan")
            if DEVICE.type == "cuda":
                mem_mb = float(torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2))

            # ---- training metrics on full folder (deterministic z for reproducibility) ----
            G.eval()
            with torch.no_grad():
                z_gen = torch.Generator(device=DEVICE)
                z_gen.manual_seed(global_epoch)  # deterministic per global epoch

                all_past_t = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)  # (N,Tin,2)
                preds_list = []
                for s in range(0, N, BATCH_SIZE):
                    pb = all_past_t[s:s + BATCH_SIZE]
                    Bc = pb.size(0)
                    zz = torch.randn(Bc, Z_DIM, device=DEVICE, generator=z_gen)
                    outn = G(pb, zz)
                    preds_list.append(outn.cpu().numpy())
                Yn_pred_n = np.concatenate(preds_list, axis=0)  # (N,Tout,2)

            Yn_pred = Yn_pred_n * y_std + y_mean
            train_metrics = compute_all_metrics(Y, Yn_pred)

            print(
                f"  [folder {g} ep {ep:02d} (global {global_epoch})] "
                f"D={d_sum/N:.4f} | G={g_sum/N:.4f} | {dt:.1f}s | "
                f"train_MAE={train_metrics['mae']:.6f} train_RMSE={train_metrics['rmse']:.6f}"
            )

            training_logs.append({
                "folder": g,
                "epoch": ep,
                "global_epoch": global_epoch,
                "loss_D": d_sum / N,
                "loss_G": g_sum / N,
                "time_sec": dt,
                "gpu_mem_mb": mem_mb,
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_mape": train_metrics["mape"],
                "train_ks": train_metrics["ks"],
            })

            # ====== TEST EVAL EACH EPOCH (GLOBAL TRAIN STATS) ======
            if (global_epoch % EVAL_TEST_EVERY) == 0 and len(test_groups) > 0:
                overall_test, per_folder_rows, _ = eval_on_test_folders(
                    G=G,
                    test_groups=test_groups,
                    common_Tin=common_Tin,
                    common_Tout=common_Tout,
                    global_stats=global_stats,
                    seed=global_epoch,  # deterministic
                )

                test_epoch_overall_logs.append({
                    "global_epoch": global_epoch,
                    "test_mae_all": overall_test["mae"],
                    "test_rmse_all": overall_test["rmse"],
                    "test_mape_all": overall_test["mape"],
                    "test_ks_all": overall_test["ks"],
                })

                for row in per_folder_rows:
                    test_epoch_per_folder_logs.append({
                        "global_epoch": global_epoch,
                        "folder": row["folder"],
                        "mae": row["mae"],
                        "rmse": row["rmse"],
                        "mape": row["mape"],
                        "ks": row["ks"],
                    })

        # save checkpoints after each folder
        torch.save(G.state_dict(), os.path.join(OUT_ROOT, f"G_after_{g}.pt"))
        torch.save(D.state_dict(), os.path.join(OUT_ROOT, f"D_after_{g}.pt"))

    # =====================================================
    # FINAL TEST (folder by folder) + PQ PDF plots once
    # =====================================================
    print("\n=== FINAL TEST (cGAN) ===")
    G.eval()

    all_true_list, all_pred_list = [], []
    first_figs_done = False

    z_gen = torch.Generator(device=DEVICE)
    z_gen.manual_seed(123456)  # fixed for final test reproducibility

    for g in test_groups:
        arr = load_pq_sequence(os.path.join(DATA_ROOT, g), MAX_JSONS_PER_FOLDER)
        if arr is None:
            print(f"  skip test {g}")
            continue

        pred_pack = predict_folder_batch(G, arr, common_Tin, common_Tout, global_stats, z_gen=z_gen)
        if pred_pack is None:
            print(f"  test {g} too short, skipping")
            continue

        fut_all, preds = pred_pack
        test_metrics = compute_all_metrics(fut_all, preds)

        print(
            f"  [test {g}] MAE={test_metrics['mae']:.6f} RMSE={test_metrics['rmse']:.6f} "
            f"MAPE={test_metrics['mape']:.6f} KS={test_metrics['ks']:.6f}"
        )

        test_logs.append({"folder": g, **test_metrics})
        all_true_list.append(fut_all)
        all_pred_list.append(preds)

        if SAVE_PQ_PLOTS_FOR_FIRST_TEST_FOLDER and not first_figs_done:
            save_power_flow_figs_and_data(
                fut_all=fut_all, preds=preds, tout=common_Tout, gname=g, out_dir=FIGS_DIR, buses=PQ_BUSES
            )
            first_figs_done = True
            print(f"     saved 4 power-flow PDFs + NPZ in {FIGS_DIR}")

    # overall final test metrics
    if all_true_list:
        all_true = np.concatenate(all_true_list, axis=0)
        all_pred = np.concatenate(all_pred_list, axis=0)
        overall = compute_all_metrics(all_true, all_pred)
        print("\n=== OVERALL FINAL TEST METRICS (all folders combined) ===")
        print(
            f"MAE={overall['mae']:.6f}, RMSE={overall['rmse']:.6f}, "
            f"MAPE={overall['mape']:.6f}, KS={overall['ks']:.6f}"
        )
        test_logs.append({"folder": "ALL", **overall})

    # =====================================================
    # PJM REAL-DATA EXTERNAL TEST
    # =====================================================
    pjm_metrics = None
    pjm_true = None
    pjm_pred = None
    pjm_area_names: List[str] = []

    if RUN_PJM_REAL_TEST:
        try:
            pjm_metrics, pjm_true, pjm_pred, pjm_area_names = eval_on_pjm_real_data(
                G=G,
                csv_path=PJM_REAL_CSV_PATH,
                common_Tin=common_Tin,
                common_Tout=common_Tout,
                global_stats=global_stats,
                pf=PJM_POWER_FACTOR,
                seed=987654,
            )

            if pjm_metrics is not None and pjm_true is not None and pjm_pred is not None:
                # Save PJM prediction arrays for reproducibility
                np.savez_compressed(
                    os.path.join(OUT_ROOT, "pjm_real_test_predictions_cgan.npz"),
                    real=pjm_true,
                    pred=pjm_pred,
                    area_names=np.array(pjm_area_names, dtype=object),
                    power_factor=np.array([PJM_POWER_FACTOR], dtype=np.float32),
                    note=np.array(["PJM real load converted to pseudo-bus P/Q. P=-load MW, Q=P*tan(arccos(pf))."], dtype=object),
                )

                if PJM_SAVE_PLOTS:
                    save_power_flow_figs_and_data(
                        fut_all=pjm_true,
                        preds=pjm_pred,
                        tout=common_Tout,
                        gname="PJM_REAL",
                        out_dir=FIGS_DIR,
                        buses=PJM_PLOT_BUSES,
                    )
                    print(f"Saved PJM real-data P/Q PDFs + NPZ in {FIGS_DIR}")

        except Exception as e:
            print("PJM real-data test failed:", repr(e))

    # =====================================================
    # SAVE LOGS (CSV)
    # =====================================================
    train_csv_path = os.path.join(OUT_ROOT, "training_log.csv")
    save_csv(
        train_csv_path,
        training_logs,
        ["folder", "epoch", "global_epoch", "loss_D", "loss_G", "time_sec", "gpu_mem_mb",
         "train_mae", "train_rmse", "train_mape", "train_ks"]
    )

    test_csv_path = os.path.join(OUT_ROOT, "test_log.csv")
    save_csv(test_csv_path, test_logs, ["folder", "mae", "rmse", "mape", "ks"])

    # Save PJM real-data metrics separately
    pjm_csv_path = os.path.join(OUT_ROOT, "pjm_real_load_test_log_cgan.csv")
    if RUN_PJM_REAL_TEST and pjm_metrics is not None:
        save_csv(
            pjm_csv_path,
            [{"dataset": "PJM_REAL_LOAD", "pseudo_buses": len(pjm_area_names), "power_factor": PJM_POWER_FACTOR, **pjm_metrics}],
            ["dataset", "pseudo_buses", "power_factor", "mae", "rmse", "mape", "ks"],
        )

    # NEW: test-vs-epoch CSVs
    test_epoch_overall_csv = os.path.join(OUT_ROOT, "test_overall_vs_epoch.csv")
    if test_epoch_overall_logs:
        save_csv(
            test_epoch_overall_csv,
            test_epoch_overall_logs,
            ["global_epoch", "test_mae_all", "test_rmse_all", "test_mape_all", "test_ks_all"]
        )

    test_epoch_per_folder_csv = os.path.join(OUT_ROOT, "test_per_folder_vs_epoch.csv")
    if test_epoch_per_folder_logs:
        save_csv(
            test_epoch_per_folder_csv,
            test_epoch_per_folder_logs,
            ["global_epoch", "folder", "mae", "rmse", "mape", "ks"]
        )

    # =====================================================
    # PLOTS: TRAIN METRICS vs GLOBAL EPOCH (PDF + NPZ)
    # =====================================================
    if training_logs:
        epochs = np.array([r["global_epoch"] for r in training_logs], dtype=np.int32)
        lossD  = np.array([r["loss_D"] for r in training_logs], dtype=np.float32)
        lossG  = np.array([r["loss_G"] for r in training_logs], dtype=np.float32)
        maes   = np.array([r["train_mae"] for r in training_logs], dtype=np.float32)
        rmses  = np.array([r["train_rmse"] for r in training_logs], dtype=np.float32)
        kss    = np.array([r["train_ks"] for r in training_logs], dtype=np.float32)

        np.savez_compressed(
            os.path.join(FIGS_DIR, "cgan_train_metrics_vs_epoch.npz"),
            epoch=epochs, lossD=lossD, lossG=lossG, mae=maes, rmse=rmses, ks=kss
        )

        plot_single_curve_pdf(epochs, lossD, XLAB_EPOCH, r"$\mathrm{D\ loss}$",
                              os.path.join(FIGS_DIR, "cgan_train_lossD_vs_epoch.pdf"),
                              color=COLORS_QUAL[0], ylog=False)
        plot_single_curve_pdf(epochs, lossG, XLAB_EPOCH, r"$\mathrm{G\ loss}$",
                              os.path.join(FIGS_DIR, "cgan_train_lossG_vs_epoch.pdf"),
                              color=COLORS_QUAL[1], ylog=False)
        plot_single_curve_pdf(epochs, maes, XLAB_EPOCH, r"$\mathrm{MAE}$",
                              os.path.join(FIGS_DIR, "cgan_train_mae_vs_epoch.pdf"),
                              color=COLORS_QUAL[2], ylog=False)
        plot_single_curve_pdf(epochs, rmses, XLAB_EPOCH, r"$\mathrm{RMSE}$",
                              os.path.join(FIGS_DIR, "cgan_train_rmse_vs_epoch.pdf"),
                              color=COLORS_QUAL[3], ylog=False)
        plot_single_curve_pdf(epochs, kss, XLAB_EPOCH, r"$\mathrm{KS}$",
                              os.path.join(FIGS_DIR, "cgan_train_ks_vs_epoch.pdf"),
                              color=COLORS_QUAL[4], ylog=False)

    # =====================================================
    # PLOTS: OVERALL TEST METRICS vs GLOBAL EPOCH (PDF + NPZ)
    # =====================================================
    if test_epoch_overall_logs:
        te    = np.array([r["global_epoch"] for r in test_epoch_overall_logs], dtype=np.int32)
        tmae  = np.array([r["test_mae_all"] for r in test_epoch_overall_logs], dtype=np.float32)
        trmse = np.array([r["test_rmse_all"] for r in test_epoch_overall_logs], dtype=np.float32)
        tmape = np.array([r["test_mape_all"] for r in test_epoch_overall_logs], dtype=np.float32)
        tks   = np.array([r["test_ks_all"] for r in test_epoch_overall_logs], dtype=np.float32)

        np.savez_compressed(
            os.path.join(FIGS_DIR, "cgan_test_overall_metrics_vs_epoch.npz"),
            epoch=te, mae=tmae, rmse=trmse, mape=tmape, ks=tks
        )

        plot_single_curve_pdf(te, tmae, XLAB_EPOCH, r"$\mathrm{MAE}$",
                              os.path.join(FIGS_DIR, "cgan_test_overall_mae_vs_epoch.pdf"),
                              color=COLORS_QUAL[1], ylog=False)
        plot_single_curve_pdf(te, trmse, XLAB_EPOCH, r"$\mathrm{RMSE}$",
                              os.path.join(FIGS_DIR, "cgan_test_overall_rmse_vs_epoch.pdf"),
                              color=COLORS_QUAL[2], ylog=False)
        plot_single_curve_pdf(te, tks, XLAB_EPOCH, r"$\mathrm{KS}$",
                              os.path.join(FIGS_DIR, "cgan_test_overall_ks_vs_epoch.pdf"),
                              color=COLORS_QUAL[3], ylog=False)
        plot_single_curve_pdf(te, tmape, XLAB_EPOCH, r"$\mathrm{MAPE}$",
                              os.path.join(FIGS_DIR, "cgan_test_overall_mape_vs_epoch.pdf"),
                              color=COLORS_QUAL[4], ylog=True)

    print("\nSaved training log to:", train_csv_path)
    print("Saved final test log to:", test_csv_path)
    if RUN_PJM_REAL_TEST and pjm_metrics is not None:
        print("Saved PJM real-data test log to:", pjm_csv_path)
    if test_epoch_overall_logs:
        print("Saved test-overall-vs-epoch CSV to:", test_epoch_overall_csv)
        print("Saved test-overall-vs-epoch NPZ/PDFs in:", FIGS_DIR)
    if test_epoch_per_folder_logs:
        print("Saved test-per-folder-vs-epoch CSV to:", test_epoch_per_folder_csv)
    print("All figures/NPZ saved in:", FIGS_DIR)


if __name__ == "__main__":
    main()
