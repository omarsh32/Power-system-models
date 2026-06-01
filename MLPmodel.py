#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline MLP for GridOPT per-bus P,Q forecasting (UPDATED)
----------------------------------------------------------
Fixes applied:
1) ✅ Correct time alignment:
   past = arr[:Tin], future = arr[Tin:Tin+Tout]  (contiguous)
2) ✅ Test normalization is now well-defined:
   - Compute GLOBAL stats from TRAIN folders only
   - Use those same stats for ALL test folders (strict “unseen” eval)
3) ✅ Much faster test evaluation:
   - Batch inference over ALL buses in a single forward pass
4) ✅ Short axis labels + LaTeX-style mathtext everywhere:
   - x-axis: r"$\\mathrm{epoch}$"
   - short y labels like r"$\\mathrm{MAE}$", r"$\\mathrm{Mean}\\ |\\Delta\\omega|$"
"""

import os
import json
import warnings
import time
import csv
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = r"DATA_FOLDER"
OUT_ROOT  = r"realdata_mlb_final"

# External REAL measured load data test: PJM hourly metered load
# Download/copy your CSV here, or change this path to where the file is on your PC.
PJM_REAL_CSV_PATH = r"pjm_hrl_load_metered_1_1_3_23_2024.csv"
RUN_PJM_REAL_TEST = True
PJM_POWER_FACTOR = 0.95
PJM_USE_TOP_N_LOAD_AREAS = None  # None = use all load_area values; or set e.g. 30

# ======================== DISPLAY / METRIC UNIT SCALING ========================
# Raw data remains in MW/MVar. Only reported MAE/RMSE and plotted curves are scaled.
# 1000 MW = 1 GW, and 1000 MVar = 1 GVar.
POWER_SCALE = 1000.0
POWER_SCALE_NAME = "GW/GVar"
P_AXIS_LABEL = r"$P~[\mathrm{GW}]$"
Q_AXIS_LABEL = r"$Q~[\mathrm{GVar}]$"

TRAIN_FOLDERS = 10
TEST_FOLDERS  = 1

MAX_JSONS_PER_FOLDER = 1500
SPLIT_RATIO = 0.75

EPOCHS_PER_FOLDER = 30
BATCH_SIZE = 64
LR = 1e-3
HID1 = 512
HID2 = 256

SEED = 7
TOPK_PARAMS = 10
SMOOTH_W = 9  # moving average window

# Evaluate test every N global epochs
EVAL_TEST_EVERY = 1

# Power-flow plots for which test folder + which buses
SAVE_PQ_PLOTS_FOR_FIRST_TEST_FOLDER = True
PQ_BUSES = (0, 1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

# Output dirs
FIGS_DIR   = os.path.join(OUT_ROOT, "figs")
PARAMS_DIR = os.path.join(OUT_ROOT, "params")
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

# =========================================================
# FIGURE STYLE GUIDELINES
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

# ColorBrewer-ish palette (you already used this)
COLORS_QUAL = [
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854",
    "#ffd92f", "#e5c494", "#b3b3b3", "#1b9e77", "#d95f02"
]

# Short labels (LaTeX-like mathtext)
XLAB_EPOCH = r"$\mathrm{epoch}$"


# =========================================================
# UTILS
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
    """
    Metrics are computed from raw MW/MVar values.
    MAE and RMSE are reported after dividing by POWER_SCALE, so with POWER_SCALE=1000:
        MW  -> GW
        MVar -> GVar
    MAPE and KS are dimensionless and unchanged.
    """
    diff = y_pred - y_true

    mae_raw = float(np.mean(np.abs(diff)))
    rmse_raw = float(np.sqrt(np.mean(diff ** 2)))

    mae = mae_raw / POWER_SCALE
    rmse = rmse_raw / POWER_SCALE

    eps = 1e-6
    mape = float(np.mean(np.abs(diff) / (np.abs(y_true) + eps)))
    ks_p = ks_statistic(y_true[..., 0], y_pred[..., 0])
    ks_q = ks_statistic(y_true[..., 1], y_pred[..., 1])
    ks_val = float(max(ks_p, ks_q))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "ks": ks_val,
        "mae_raw_mw_mvar": mae_raw,
        "rmse_raw_mw_mvar": rmse_raw,
    }


def save_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1:
        return x
    w = min(int(w), x.size)
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(x, kernel, mode="same")


def compute_global_train_stats(
    train_groups: List[str],
    cache: Dict[str, Optional[np.ndarray]],
    common_Tin: int,
    common_Tout: int
) -> Dict[str, np.ndarray]:
    """
    Compute GLOBAL normalization stats from TRAIN folders only.
    Uses contiguous future: [Tin : Tin+Tout].
    Returns stats shaped: (1,1,2) for mean/std.
    """
    X_all, Y_all = [], []

    for g in train_groups:
        arr = cache.get(g, None)
        if arr is None:
            continue
        T, B, _ = arr.shape
        if T < (common_Tin + common_Tout):
            continue

        past_all = arr[:common_Tin]                                # (Tin,B,2)
        fut_all  = arr[common_Tin:common_Tin + common_Tout]        # (Tout,B,2)

        # per-bus samples -> (B,Tin,2) and (B,Tout,2)
        X = np.transpose(past_all, (1, 0, 2))
        Y = np.transpose(fut_all,  (1, 0, 2))

        X_all.append(X)
        Y_all.append(Y)

    if not X_all:
        raise RuntimeError("Failed to compute global train stats: no valid training data.")

    Xc = np.concatenate(X_all, axis=0)  # (sumB, Tin, 2)
    Yc = np.concatenate(Y_all, axis=0)  # (sumB, Tout,2)

    x_mean = Xc.mean(axis=(0, 1), keepdims=True)           # (1,1,2)
    x_std  = Xc.std(axis=(0, 1), keepdims=True) + 1e-6
    y_mean = Yc.mean(axis=(0, 1), keepdims=True)           # (1,1,2)
    y_std  = Yc.std(axis=(0, 1), keepdims=True) + 1e-6

    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


# =========================================================
# FIGS + NPZ (NO TITLE, NO LEGEND)
# =========================================================
def save_power_flow_figs_and_data(
    fut_all: np.ndarray,   # (Tout,B,2)
    preds: np.ndarray,     # (Tout,B,2)
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
        ax.plot(t_axis, fut_all[:, b, 0] / POWER_SCALE, linewidth=2, color=COLORS_QUAL[0])
        ax.plot(t_axis, preds[:, b, 0] / POWER_SCALE, "--", linewidth=2, color=COLORS_QUAL[1])
        ax.set_xlabel(r"$\mathrm{t}$")
        ax.set_ylabel(P_AXIS_LABEL)
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"mlp_{gname}_bus{b}_P.pdf"), bbox_inches="tight")
        plt.close(fig)

        np.savez_compressed(
            os.path.join(out_dir, f"mlp_{gname}_bus{b}_P_data.npz"),
            time=t_axis, real=fut_all[:, b, 0], pred=preds[:, b, 0]
        )

        # Q
        fig, ax = plt.subplots()
        ax.plot(t_axis, fut_all[:, b, 1] / POWER_SCALE, linewidth=2, color=COLORS_QUAL[0])
        ax.plot(t_axis, preds[:, b, 1] / POWER_SCALE, "--", linewidth=2, color=COLORS_QUAL[1])
        ax.set_xlabel(r"$\mathrm{t}$")
        ax.set_ylabel(Q_AXIS_LABEL)
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"mlp_{gname}_bus{b}_Q.pdf"), bbox_inches="tight")
        plt.close(fig)

        np.savez_compressed(
            os.path.join(out_dir, f"mlp_{gname}_bus{b}_Q_data.npz"),
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
            return 10**9

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
# MLP MODEL
# =========================================================
class MLPForecast(nn.Module):
    """ past: (B,Tin,2) -> flatten -> MLP -> (B,Tout,2) """
    def __init__(self, tin: int, tout: int, hid1: int = 512, hid2: int = 256):
        super().__init__()
        self.tout = tout
        in_dim = tin * 2
        out_dim = tout * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid1),
            nn.ReLU(inplace=True),
            nn.Linear(hid1, hid2),
            nn.ReLU(inplace=True),
            nn.Linear(hid2, out_dim),
        )

    def forward(self, past):
        b, t, c = past.shape
        x = past.reshape(b, t * c)
        out = self.net(x)
        return out.reshape(b, self.tout, 2)


# =========================================================
# PLOTTING HELPERS (SHORT LABELS + LaTeX)
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


def plot_multi_curves_pdf(x, Y, xlabel, ylabel, pdf_path, colors, ylog=False):
    fig, ax = plt.subplots()
    for i in range(Y.shape[1]):
        ax.plot(x, Y[:, i], linewidth=2, color=colors[i % len(colors)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if ylog:
        ax.set_yscale("log")
        pos = Y[Y > 0]
        if pos.size > 0:
            ax.set_ylim(float(np.min(pos)) * 0.7, None)
    plt.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# FAST BATCH PREDICTION OVER ALL BUSES
# =========================================================
@torch.no_grad()
def predict_folder_batch(
    model: nn.Module,
    arr: np.ndarray,                 # (T,B,2)
    common_Tin: int,
    common_Tout: int,
    stats: Dict[str, np.ndarray],    # (1,1,2) means/stds
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns:
      fut_all: (Tout,B,2)
      preds  : (Tout,B,2)
    """
    T, B, _ = arr.shape
    if T < (common_Tin + common_Tout):
        return None

    past_all = arr[:common_Tin]                                  # (Tin,B,2)
    fut_all  = arr[common_Tin:common_Tin + common_Tout]          # (Tout,B,2)

    x_mean, x_std = stats["x_mean"], stats["x_std"]
    y_mean, y_std = stats["y_mean"], stats["y_std"]

    # Build batch: (B,Tin,2)
    X = np.transpose(past_all, (1, 0, 2))
    Xn = (X - x_mean.squeeze(0)) / x_std.squeeze(0)

    past_t = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)  # (B,Tin,2)
    outn = model(past_t).detach().cpu().numpy()                    # (B,Tout,2)

    out = outn * y_std.squeeze(0) + y_mean.squeeze(0)              # (B,Tout,2)
    preds = np.transpose(out, (1, 0, 2))                            # (Tout,B,2)
    return fut_all, preds


# =========================================================
# TEST EVAL (overall + per-folder), using GLOBAL TRAIN stats
# =========================================================
@torch.no_grad()
def eval_on_test_folders(
    model: nn.Module,
    test_groups: List[str],
    common_Tin: int,
    common_Tout: int,
    global_stats: Dict[str, np.ndarray],
) -> Tuple[Dict[str, float], List[Dict[str, Any]], Optional[Tuple[str, np.ndarray, np.ndarray]]]:
    model.eval()

    all_true_list, all_pred_list = [], []
    per_folder_rows: List[Dict[str, Any]] = []
    first_plot_payload = None

    for g in test_groups:
        arr = load_pq_sequence(os.path.join(DATA_ROOT, g), MAX_JSONS_PER_FOLDER)
        if arr is None:
            continue

        pred_pack = predict_folder_batch(model, arr, common_Tin, common_Tout, global_stats)
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
# REAL PJM LOAD DATA CONVERSION + EXTERNAL TEST
# =========================================================
def load_pjm_real_load_sequence(
    csv_path: str,
    pf: float = 0.95,
    top_n_load_areas: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Convert real PJM hourly metered load data to the model format:
        arr.shape = (T, B, 2)

    Where:
        T = number of timestamps
        B = number of PJM load areas, treated as pseudo-buses
        arr[...,0] = P [MW], demand represented as negative injection
        arr[...,1] = Q [MVar], approximated from P using assumed power factor

    Important scientific wording:
        This is a real measured load-driven external robustness test,
        not a full physical bus-level AC power-flow validation.
    """
    if not os.path.isfile(csv_path):
        print("PJM CSV path does not exist:", csv_path)
        return None, []

    df = pd.read_csv(csv_path)

    required = {"datetime_beginning_utc", "load_area", "mw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"PJM CSV is missing required columns: {missing}")

    df = df[["datetime_beginning_utc", "load_area", "mw"]].copy()
    df["datetime_beginning_utc"] = pd.to_datetime(df["datetime_beginning_utc"], errors="coerce")
    df["mw"] = pd.to_numeric(df["mw"], errors="coerce")
    df = df.dropna(subset=["datetime_beginning_utc", "load_area", "mw"])

    # Pivot to time x load_area. Each load_area becomes one pseudo-bus.
    pivot = df.pivot_table(
        index="datetime_beginning_utc",
        columns="load_area",
        values="mw",
        aggfunc="mean",
    ).sort_index()

    # Keep the most complete / largest load areas if requested.
    if top_n_load_areas is not None and top_n_load_areas > 0 and top_n_load_areas < pivot.shape[1]:
        energy_rank = pivot.abs().sum(axis=0).sort_values(ascending=False)
        keep_cols = list(energy_rank.index[:top_n_load_areas])
        pivot = pivot[keep_cols]

    # Fill small missing gaps if any.
    pivot = pivot.interpolate(method="time", limit_direction="both")
    pivot = pivot.dropna(axis=1, how="any")

    if pivot.shape[0] < 4 or pivot.shape[1] < 1:
        print("PJM data is not sufficient after cleaning.")
        return None, []

    load_area_names = [str(c) for c in pivot.columns]

    # PJM mw is load demand. To match your GridOPT parser convention,
    # load is negative net injection.
    P = -pivot.to_numpy(dtype=np.float32)  # (T,B)

    # Estimate Q using fixed power factor: Q = P * tan(arccos(pf)).
    # Since P is negative for load, Q is also negative by this convention.
    pf = float(pf)
    if not (0.0 < pf <= 1.0):
        raise ValueError("Power factor must be in (0, 1].")
    Q = P * np.tan(np.arccos(pf)).astype(np.float32)

    arr = np.stack([P, Q], axis=2).astype(np.float32)  # (T,B,2)
    return arr, load_area_names


@torch.no_grad()
def eval_on_pjm_real_data(
    model: nn.Module,
    csv_path: str,
    common_Tin: int,
    common_Tout: int,
    global_stats: Dict[str, np.ndarray],
    pf: float = 0.95,
    top_n_load_areas: Optional[int] = None,
    save_plots: bool = True,
) -> List[Dict[str, Any]]:
    """Run final external real-data test on PJM load-area measurements."""
    print("\n=== EXTERNAL REAL-DATA TEST: PJM hourly metered load ===")

    arr, load_area_names = load_pjm_real_load_sequence(
        csv_path=csv_path,
        pf=pf,
        top_n_load_areas=top_n_load_areas,
    )

    if arr is None:
        return []

    print(f"PJM converted shape: {arr.shape} = (T, pseudo_buses, [P,Q])")
    print(f"PJM pseudo-buses/load areas ({len(load_area_names)}): {load_area_names}")

    pred_pack = predict_folder_batch(
        model=model,
        arr=arr,
        common_Tin=common_Tin,
        common_Tout=common_Tout,
        stats=global_stats,
    )

    if pred_pack is None:
        print("PJM data is too short for common_Tin + common_Tout, skipping.")
        return []

    fut_all, preds = pred_pack
    metrics = compute_all_metrics(fut_all, preds)

    print("\n=== PJM REAL-DATA TEST METRICS ===")
    print(
        f"MAE={metrics['mae']:.6f} {POWER_SCALE_NAME}, RMSE={metrics['rmse']:.6f} {POWER_SCALE_NAME}, "
        f"MAPE={metrics['mape']:.6f}, KS={metrics['ks']:.6f}"
    )

    # Save the converted real data and predictions for reproducibility.
    np.savez_compressed(
        os.path.join(OUT_ROOT, "pjm_real_test_predictions.npz"),
        true=fut_all,
        pred=preds,
        load_area_names=np.array(load_area_names, dtype=object),
        pf=np.array([pf], dtype=np.float32),
    )

    if save_plots:
        save_power_flow_figs_and_data(
            fut_all=fut_all,
            preds=preds,
            tout=common_Tout,
            gname="PJM_REAL",
            out_dir=FIGS_DIR,
            buses=PQ_BUSES,
        )
        print(f"Saved PJM real-data P/Q plots + NPZ in {FIGS_DIR}")

    return [{"folder": "PJM_REAL_LOAD", **metrics, "num_pseudo_buses": len(load_area_names), "pf": pf}]


# =========================================================
# MAIN
# =========================================================
def main():
    # 1) discover folders
    all_groups = sorted([d for d in os.listdir(DATA_ROOT) if d.startswith("group_")])
    train_groups = all_groups[:TRAIN_FOLDERS]
    test_groups  = all_groups[TRAIN_FOLDERS:TRAIN_FOLDERS + TEST_FOLDERS]
    print("train groups:", train_groups)
    print("test  groups:", test_groups)

    # 2) first pass: find common Tin/Tout, cache train arrays
    tins, touts, cache = [], [], {}
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
        raise RuntimeError("No train data found")

    common_Tin  = min(tins)
    common_Tout = min(touts)
    print(f"common Tin={common_Tin}, common Tout={common_Tout}")

    # 3) model
    model = MLPForecast(common_Tin, common_Tout, hid1=HID1, hid2=HID2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # 4) compute GLOBAL train stats for testing
    global_stats = compute_global_train_stats(train_groups, cache, common_Tin, common_Tout)
    print("Computed GLOBAL train stats for test normalization.")

    # logs
    training_logs: List[Dict[str, Any]] = []
    test_logs: List[Dict[str, Any]] = []  # FINAL test after training

    # test-vs-epoch logs
    test_epoch_overall_logs: List[Dict[str, Any]] = []
    test_epoch_per_folder_logs: List[Dict[str, Any]] = []

    # param change tracking
    global_epoch = 0
    prev_params: Dict[str, torch.Tensor] = {}
    change_history: List[Dict[str, Any]] = []
    cumulative_change: Dict[str, float] = {}

    # ============ TRAIN FOLDER BY FOLDER ============
    for g in train_groups:
        print(f"\n=== Training on folder: {g} ===")
        arr = cache[g]
        if arr is None:
            print("  skip, no data")
            continue

        T, B, _ = arr.shape
        if T < (common_Tin + common_Tout):
            print("  skip, too short for common Tin/Tout")
            continue

        # ✅ FIXED alignment: contiguous future after past
        past_all = arr[:common_Tin]
        fut_all  = arr[common_Tin:common_Tin + common_Tout]

        # per-bus samples
        X = np.transpose(past_all, (1, 0, 2))  # (B,Tin,2)
        Y = np.transpose(fut_all,  (1, 0, 2))  # (B,Tout,2)
        N = X.shape[0]

        # per-folder normalization (kept for training)
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

            model.train()
            total_loss = 0.0
            t0 = time.time()

            if DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats(DEVICE)

            for start in range(0, N, BATCH_SIZE):
                end = start + BATCH_SIZE
                bidx = idxs[start:end]
                past = torch.tensor(Xn[bidx], dtype=torch.float32, device=DEVICE)
                fut  = torch.tensor(Yn[bidx], dtype=torch.float32, device=DEVICE)

                opt.zero_grad()
                pred = model(past)
                loss = loss_fn(pred, fut)
                loss.backward()
                opt.step()

                total_loss += loss.item() * past.size(0)

            dt = time.time() - t0
            mem_mb = float("nan")
            if DEVICE.type == "cuda":
                mem_mb = float(torch.cuda.max_memory_allocated(DEVICE) / (1024**2))

            # ---- evaluate on full TRAIN folder ----
            model.eval()
            with torch.no_grad():
                all_past_t = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)
                preds_list = []
                for s in range(0, N, BATCH_SIZE):
                    pb = all_past_t[s:s + BATCH_SIZE]
                    outn = model(pb)
                    preds_list.append(outn.cpu().numpy())
                Yn_pred_n = np.concatenate(preds_list, axis=0)

            Yn_pred = Yn_pred_n * y_std + y_mean
            train_metrics = compute_all_metrics(Y, Yn_pred)

            print(
                f"  [folder {g} ep {ep:02d} (global {global_epoch})] "
                f"loss={total_loss/N:.6f} | {dt:.1f}s | "
                f"train_MAE={train_metrics['mae']:.6f} {POWER_SCALE_NAME} train_RMSE={train_metrics['rmse']:.6f} {POWER_SCALE_NAME}"
            )

            training_logs.append({
                "folder": g,
                "epoch": ep,
                "global_epoch": global_epoch,
                "loss": total_loss / N,
                "time_sec": dt,
                "gpu_mem_mb": mem_mb,
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_mape": train_metrics["mape"],
                "train_ks": train_metrics["ks"],
            })

            # ====== PARAMETER SNAPSHOT + CHANGE TRACKING ======
            snap_path = os.path.join(PARAMS_DIR, f"params_epoch_{global_epoch}.npz")
            with torch.no_grad():
                current_params = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
            np.savez_compressed(snap_path, **{n: p.numpy() for n, p in current_params.items()})

            epoch_changes: Dict[str, float] = {}
            if prev_params:
                for n, p in current_params.items():
                    if n in prev_params:
                        delta = torch.abs(p - prev_params[n]).mean().item()
                        epoch_changes[n] = float(delta)
                        cumulative_change[n] = cumulative_change.get(n, 0.0) + float(delta)
                    else:
                        epoch_changes[n] = 0.0
                        cumulative_change.setdefault(n, 0.0)
            else:
                for n in current_params.keys():
                    epoch_changes[n] = 0.0
                    cumulative_change.setdefault(n, 0.0)

            change_history.append({"global_epoch": global_epoch, "changes": epoch_changes})
            prev_params = current_params

            # ====== TEST EVAL EACH EPOCH (GLOBAL TRAIN STATS) ======
            if (global_epoch % EVAL_TEST_EVERY) == 0 and len(test_groups) > 0:
                overall_test, per_folder_rows, _ = eval_on_test_folders(
                    model=model,
                    test_groups=test_groups,
                    common_Tin=common_Tin,
                    common_Tout=common_Tout,
                    global_stats=global_stats,
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

        # checkpoint per folder
        torch.save(model.state_dict(), os.path.join(OUT_ROOT, f"mlp_after_{g}.pt"))

    # ============ FINAL TEST (after training) ============
    print("\n=== FINAL TEST (MLP baseline) ===")
    model.eval()

    all_true_list, all_pred_list = [], []
    first_figs_done = False

    for g in test_groups:
        arr = load_pq_sequence(os.path.join(DATA_ROOT, g), MAX_JSONS_PER_FOLDER)
        if arr is None:
            print(f"  skip test {g}")
            continue

        pred_pack = predict_folder_batch(model, arr, common_Tin, common_Tout, global_stats)
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

    # overall metrics (final)
    if all_true_list:
        all_true = np.concatenate(all_true_list, axis=0)
        all_pred = np.concatenate(all_pred_list, axis=0)
        overall = compute_all_metrics(all_true, all_pred)
        print("\n=== OVERALL FINAL TEST METRICS (all folders combined) ===")
        print(
            f"MAE={overall['mae']:.6f} {POWER_SCALE_NAME}, RMSE={overall['rmse']:.6f} {POWER_SCALE_NAME}, "
            f"MAPE={overall['mape']:.6f}, KS={overall['ks']:.6f}"
        )
        test_logs.append({"folder": "ALL", **overall})

    # ============ EXTERNAL REAL-DATA TEST: PJM ============
    pjm_real_logs: List[Dict[str, Any]] = []
    if RUN_PJM_REAL_TEST:
        pjm_real_logs = eval_on_pjm_real_data(
            model=model,
            csv_path=PJM_REAL_CSV_PATH,
            common_Tin=common_Tin,
            common_Tout=common_Tout,
            global_stats=global_stats,
            pf=PJM_POWER_FACTOR,
            top_n_load_areas=PJM_USE_TOP_N_LOAD_AREAS,
            save_plots=True,
        )

    # ============ SAVE LOGS ============
    train_csv_path = os.path.join(OUT_ROOT, "training_log.csv")
    save_csv(
        train_csv_path,
        training_logs,
        ["folder", "epoch", "global_epoch", "loss", "time_sec", "gpu_mem_mb",
         "train_mae", "train_rmse", "train_mape", "train_ks"]
    )

    test_csv_path = os.path.join(OUT_ROOT, "test_log.csv")
    save_csv(test_csv_path, test_logs, ["folder", "mae", "rmse", "mape", "ks"])

    pjm_real_csv_path = os.path.join(OUT_ROOT, "pjm_real_load_test_log.csv")
    if pjm_real_logs:
        save_csv(
            pjm_real_csv_path,
            pjm_real_logs,
            ["folder", "mae", "rmse", "mape", "ks", "mae_raw_mw_mvar", "rmse_raw_mw_mvar", "num_pseudo_buses", "pf"]
        )

    # save test-vs-epoch CSVs
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
        losses = np.array([r["loss"] for r in training_logs], dtype=np.float32)
        maes   = np.array([r["train_mae"] for r in training_logs], dtype=np.float32)
        rmses  = np.array([r["train_rmse"] for r in training_logs], dtype=np.float32)
        kss    = np.array([r["train_ks"] for r in training_logs], dtype=np.float32)

        np.savez_compressed(
            os.path.join(FIGS_DIR, "mlp_train_metrics_vs_epoch.npz"),
            epoch=epochs, loss=losses, mae=maes, rmse=rmses, ks=kss
        )

        plot_single_curve_pdf(epochs, losses, XLAB_EPOCH, r"$\mathrm{loss}$",
                              os.path.join(FIGS_DIR, "mlp_train_loss_vs_epoch.pdf"),
                              color=COLORS_QUAL[0], ylog=False)
        plot_single_curve_pdf(epochs, maes, XLAB_EPOCH, r"$\mathrm{MAE}$",
                              os.path.join(FIGS_DIR, "mlp_train_mae_vs_epoch.pdf"),
                              color=COLORS_QUAL[1], ylog=False)
        plot_single_curve_pdf(epochs, rmses, XLAB_EPOCH, r"$\mathrm{RMSE}$",
                              os.path.join(FIGS_DIR, "mlp_train_rmse_vs_epoch.pdf"),
                              color=COLORS_QUAL[2], ylog=False)
        plot_single_curve_pdf(epochs, kss, XLAB_EPOCH, r"$\mathrm{KS}$",
                              os.path.join(FIGS_DIR, "mlp_train_ks_vs_epoch.pdf"),
                              color=COLORS_QUAL[3], ylog=False)

    # =====================================================
    # PLOTS: OVERALL TEST METRICS vs GLOBAL EPOCH (PDF + NPZ)
    # =====================================================
    if test_epoch_overall_logs:
        te = np.array([r["global_epoch"] for r in test_epoch_overall_logs], dtype=np.int32)
        tmae = np.array([r["test_mae_all"] for r in test_epoch_overall_logs], dtype=np.float32)
        trmse = np.array([r["test_rmse_all"] for r in test_epoch_overall_logs], dtype=np.float32)
        tmape = np.array([r["test_mape_all"] for r in test_epoch_overall_logs], dtype=np.float32)
        tks = np.array([r["test_ks_all"] for r in test_epoch_overall_logs], dtype=np.float32)

        np.savez_compressed(
            os.path.join(FIGS_DIR, "mlp_test_overall_metrics_vs_epoch.npz"),
            epoch=te, mae=tmae, rmse=trmse, mape=tmape, ks=tks
        )

        plot_single_curve_pdf(te, tmae, XLAB_EPOCH, r"$\mathrm{MAE}$",
                              os.path.join(FIGS_DIR, "mlp_test_overall_mae_vs_epoch.pdf"),
                              color=COLORS_QUAL[1], ylog=False)
        plot_single_curve_pdf(te, trmse, XLAB_EPOCH, r"$\mathrm{RMSE}$",
                              os.path.join(FIGS_DIR, "mlp_test_overall_rmse_vs_epoch.pdf"),
                              color=COLORS_QUAL[2], ylog=False)
        plot_single_curve_pdf(te, tks, XLAB_EPOCH, r"$\mathrm{KS}$",
                              os.path.join(FIGS_DIR, "mlp_test_overall_ks_vs_epoch.pdf"),
                              color=COLORS_QUAL[3], ylog=False)

        # MAPE usually huge -> log scale helps
        plot_single_curve_pdf(te, tmape, XLAB_EPOCH, r"$\mathrm{MAPE}$",
                              os.path.join(FIGS_DIR, "mlp_test_overall_mape_vs_epoch.pdf"),
                              color=COLORS_QUAL[4], ylog=True)

    # =====================================================
    # TOP-10 MOST-CHANGING PARAMS: ABS + REL (PDF + NPZ)
    # =====================================================
    if cumulative_change and change_history:
        ranked = sorted(cumulative_change.items(), key=lambda kv: kv[1], reverse=True)
        top_params = ranked[:TOPK_PARAMS]
        top_names = [n for n, _ in top_params]

        top_json = os.path.join(OUT_ROOT, "top10_changed_params.json")
        with open(top_json, "w") as f:
            json.dump([{"name": n, "total_mean_abs_change": float(v)} for n, v in top_params], f, indent=2)

        top_txt = os.path.join(OUT_ROOT, "top10_changed_params.txt")
        with open(top_txt, "w") as f:
            for n, v in top_params:
                f.write(f"{n}\t{v:.10f}\n")

        epochs_vec = np.array([entry["global_epoch"] for entry in change_history], dtype=np.int32)
        E = len(change_history)
        abs_change_mat = np.zeros((E, TOPK_PARAMS), dtype=np.float32)

        for e_idx, entry in enumerate(change_history):
            ch = entry["changes"]
            for j, name in enumerate(top_names):
                abs_change_mat[e_idx, j] = float(ch.get(name, 0.0))

        mean_abs_w_mat = np.zeros((E, TOPK_PARAMS), dtype=np.float32)
        eps = 1e-12

        for e_idx, ge in enumerate(epochs_vec):
            snap_path = os.path.join(PARAMS_DIR, f"params_epoch_{int(ge)}.npz")
            if not os.path.exists(snap_path):
                continue
            snap = np.load(snap_path, allow_pickle=True)
            for j, name in enumerate(top_names):
                if name in snap:
                    w = snap[name]
                    mean_abs_w_mat[e_idx, j] = float(np.mean(np.abs(w)))

        rel_change_mat = abs_change_mat / (mean_abs_w_mat + eps)

        abs_sm = np.stack([moving_average(abs_change_mat[:, j], SMOOTH_W) for j in range(TOPK_PARAMS)], axis=1)
        rel_sm = np.stack([moving_average(rel_change_mat[:, j], SMOOTH_W) for j in range(TOPK_PARAMS)], axis=1)

        np.savez_compressed(
            os.path.join(FIGS_DIR, "mlp_top10_param_change_abs_rel_vs_epoch.npz"),
            epoch=epochs_vec,
            names=np.array(top_names, dtype=object),
            abs_change=abs_change_mat,
            rel_change=rel_change_mat,
            abs_change_smooth=abs_sm,
            rel_change_smooth=rel_sm,
            mean_abs_w=mean_abs_w_mat,
        )

        # ✅ Short labels + LaTeX-like mathtext (like your Fig. 5a)
        plot_multi_curves_pdf(
            epochs_vec, abs_sm, XLAB_EPOCH, r"$\mathrm{Mean}\ |\Delta\omega|$",
            os.path.join(FIGS_DIR, "mlp_top10_param_abs_change_vs_epoch.pdf"),
            colors=COLORS_QUAL, ylog=True
        )
        plot_multi_curves_pdf(
            epochs_vec, rel_sm, XLAB_EPOCH, r"$\mathrm{Mean}\ |\Delta\omega|/(|\omega|+\epsilon)$",
            os.path.join(FIGS_DIR, "mlp_top10_param_rel_change_vs_epoch.pdf"),
            colors=COLORS_QUAL, ylog=True
        )

    print("\nSaved training log to:", train_csv_path)
    print("Saved final test log to:", test_csv_path)
    if test_epoch_overall_logs:
        print("Saved test-overall-vs-epoch CSV to:", test_epoch_overall_csv)
        print("Saved test-overall-vs-epoch NPZ/PDFs in:", FIGS_DIR)
    if test_epoch_per_folder_logs:
        print("Saved test-per-folder-vs-epoch CSV to:", test_epoch_per_folder_csv)
    print("All figures/NPZ saved in:", FIGS_DIR)
    print("All epoch parameter snapshots saved in:", PARAMS_DIR)


if __name__ == "__main__":
    main()
