#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure CONDITIONAL VAE for GridOPT per-bus P,Q forecasting (UPDATED to match MLP/cGAN pipeline)
------------------------------------------------------------------------------------------
What was changed to make results comparable across ALL models:

A) ✅ Correct time alignment (contiguous split)
   past = arr[:Tin]
   future = arr[Tin:Tin+Tout]

B) ✅ Same figure rules as MLP
   - PDF only, no title, no legend
   - Short axis labels (epoch, MAE, RMSE, KS, loss, recon, KL, t)
   - LaTeX mathtext (no external latex dependency)
   - Save NPZ for every curve + PQ plots

C) ✅ Evaluate TEST (5 folders) after every global epoch
   - Save overall test MAE/RMSE/MAPE/KS vs epoch (CSV + NPZ)
   - Plot overall test metric curves vs epoch (PDF)
   - Optionally save per-folder test metrics per epoch (CSV)

D) ✅ Strict test normalization (same as the updated cGAN version)
   - Compute GLOBAL stats from TRAIN folders only
   - Use those stats for ALL test folders

E) ✅ Fast inference (batch over all buses)
   - Avoid per-bus loops in testing

IMPORTANT NOTE about cVAE test inference:
- A real cVAE at test time should NOT use the future in the encoder (future is unknown).
- For fair comparison with MLP/cGAN, we do:
    z ~ N(0, I),  decode(past, z)
- For reproducible curves, we seed z by global_epoch.

If you also want the "oracle" mode (encode with future for upper-bound), I can add it as an extra column.
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

# ======================== CONFIG ========================
DATA_ROOT = r"DATA_FOLDER"
OUT_ROOT  = r"realdata_cvae_1"

# External REAL measured load data test: PJM hourly metered load
# Each PJM load area is treated as a pseudo-bus.
PJM_REAL_CSV_PATH = r"C:\Users\omar shadafny\Desktop\Omar_Elinor\pjm_hrl_load_metered_1_1_3_23_2024.csv"
RUN_PJM_REAL_TEST = True
PJM_POWER_FACTOR = 0.95
PJM_USE_TOP_N_LOAD_AREAS = None  # None = use all load_area values; or set e.g. 30


TRAIN_FOLDERS = 1
TEST_FOLDERS  = 1
MAX_JSONS_PER_FOLDER = 1500
SPLIT_RATIO = 0.75

EPOCHS_PER_FOLDER = 30
BATCH_SIZE = 64
LR = 2e-4
HID_DIM = 128
LATENT_DIM = 32
BETA_KL = 1e-3   # KL weight
RECON_MODE = "l1"  # "l1" or "mse"

SEED = 7
EVAL_TEST_EVERY = 1

SAVE_PQ_PLOTS_FOR_FIRST_TEST_FOLDER = True
PQ_BUSES = (0, 1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

FIGS_DIR = os.path.join(OUT_ROOT, "figs")
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# ======================== FIGURE STYLE (match MLP) ========================
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


# ======================== UTILS (metrics, csv) ========================
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


# ======================== PLOTTING HELPERS (PDF, no legend, no title) ========================
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
        ax.plot(t_axis, fut_all[:, b, 0], linewidth=2, color=COLORS_QUAL[0])
        ax.plot(t_axis, preds[:, b, 0], "--", linewidth=2, color=COLORS_QUAL[1])
        ax.set_xlabel(r"$\mathrm{t}$")
        ax.set_ylabel(r"$P~[\mathrm{MW}]$")
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"cvae_{gname}_bus{b}_P.pdf"), bbox_inches="tight")
        plt.close(fig)

        np.savez_compressed(
            os.path.join(out_dir, f"cvae_{gname}_bus{b}_P_data.npz"),
            time=t_axis, real=fut_all[:, b, 0], pred=preds[:, b, 0]
        )

        # Q
        fig, ax = plt.subplots()
        ax.plot(t_axis, fut_all[:, b, 1], linewidth=2, color=COLORS_QUAL[0])
        ax.plot(t_axis, preds[:, b, 1], "--", linewidth=2, color=COLORS_QUAL[1])
        ax.set_xlabel(r"$\mathrm{t}$")
        ax.set_ylabel(r"$Q~[\mathrm{MVar}]$")
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"cvae_{gname}_bus{b}_Q.pdf"), bbox_inches="tight")
        plt.close(fig)

        np.savez_compressed(
            os.path.join(out_dir, f"cvae_{gname}_bus{b}_Q_data.npz"),
            time=t_axis, real=fut_all[:, b, 1], pred=preds[:, b, 1]
        )


# ======================== DATA LOADING ========================
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


# ======================== CVAE MODEL ========================
class CVAE(nn.Module):
    """
    Conditional VAE:
    - Encoder sees (past, future) -> (mu, logvar)
    - Decoder gets (past, z)      -> predicted future
    """
    def __init__(self, tin: int, tout: int, hid: int = 128, z_dim: int = 32):
        super().__init__()
        self.tin = tin
        self.tout = tout
        self.z_dim = z_dim

        self.enc_gru = nn.GRU(2, hid, batch_first=True)
        self.fc_mu = nn.Linear(hid, z_dim)
        self.fc_logvar = nn.Linear(hid, z_dim)

        self.dec_gru = nn.GRU(2, hid, batch_first=True)
        self.dec_fc = nn.Linear(hid + z_dim, tout * 2)

    def encode(self, past, future):
        x = torch.cat([past, future], dim=1)  # (B, Tin+Tout, 2)
        _, h = self.enc_gru(x)
        h = h.squeeze(0)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, past, z):
        _, h = self.dec_gru(past)
        h = h.squeeze(0)
        hz = torch.cat([h, z], dim=1)
        out = self.dec_fc(hz)
        return out.view(-1, self.tout, 2)

    def forward(self, past, future):
        mu, logvar = self.encode(past, future)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(past, z)
        return recon, mu, logvar


def cvae_loss(recon, target, mu, logvar, beta=1e-3, mode="l1"):
    if mode == "l1":
        recon_loss = torch.abs(recon - target).mean()
    else:
        recon_loss = nn.functional.mse_loss(recon, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


# ======================== GLOBAL TRAIN STATS ========================
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
        fut_all  = arr[common_Tin:common_Tin + common_Tout]  # ✅ contiguous

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


# ======================== FAST TEST PREDICTION (batch buses) ========================
@torch.no_grad()
def predict_folder_batch(
    model: nn.Module,
    arr: np.ndarray,               # (T,B,2)
    common_Tin: int,
    common_Tout: int,
    stats: Dict[str, np.ndarray],  # (1,1,2)
    z_gen: torch.Generator,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    T, B, _ = arr.shape
    if T < (common_Tin + common_Tout):
        return None

    past_all = arr[:common_Tin]
    fut_all  = arr[common_Tin:common_Tin + common_Tout]  # ✅ contiguous

    x_mean, x_std = stats["x_mean"], stats["x_std"]
    y_mean, y_std = stats["y_mean"], stats["y_std"]

    X = np.transpose(past_all, (1, 0, 2))  # (B,Tin,2)
    Xn = (X - x_mean.squeeze(0)) / x_std.squeeze(0)

    past_t = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)  # (B,Tin,2)

    # cVAE proper test: sample z ~ N(0,I) without future
    z = torch.randn(B, LATENT_DIM, device=DEVICE, generator=z_gen)
    outn = model.decode(past_t, z).detach().cpu().numpy()          # (B,Tout,2)

    out = outn * y_std.squeeze(0) + y_mean.squeeze(0)              # (B,Tout,2)
    preds = np.transpose(out, (1, 0, 2))                             # (Tout,B,2)
    return fut_all, preds


# ======================== TEST EVAL EACH EPOCH ========================
@torch.no_grad()
def eval_on_test_folders(
    model: nn.Module,
    test_groups: List[str],
    common_Tin: int,
    common_Tout: int,
    global_stats: Dict[str, np.ndarray],
    seed: int,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], Optional[Tuple[str, np.ndarray, np.ndarray]]]:
    model.eval()

    all_true_list, all_pred_list = [], []
    per_folder_rows: List[Dict[str, Any]] = []
    first_plot_payload = None

    z_gen = torch.Generator(device=DEVICE)
    z_gen.manual_seed(seed)

    for g in test_groups:
        arr = load_pq_sequence(os.path.join(DATA_ROOT, g), MAX_JSONS_PER_FOLDER)
        if arr is None:
            continue

        pack = predict_folder_batch(model, arr, common_Tin, common_Tout, global_stats, z_gen=z_gen)
        if pack is None:
            continue

        fut_all, preds = pack
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
    Q = P * np.float32(np.tan(np.arccos(pf)))

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
    seed: int = 987654,
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

    z_gen = torch.Generator(device=DEVICE)
    z_gen.manual_seed(seed)

    pred_pack = predict_folder_batch(
        model=model,
        arr=arr,
        common_Tin=common_Tin,
        common_Tout=common_Tout,
        stats=global_stats,
        z_gen=z_gen,
    )

    if pred_pack is None:
        print("PJM data is too short for common_Tin + common_Tout, skipping.")
        return []

    fut_all, preds = pred_pack
    metrics = compute_all_metrics(fut_all, preds)

    print("\n=== PJM REAL-DATA TEST METRICS ===")
    print(
        f"MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, "
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




# ======================== MAIN ========================
def main():
    # logs
    training_logs: List[Dict[str, Any]] = []
    test_logs: List[Dict[str, Any]] = []

    test_epoch_overall_logs: List[Dict[str, Any]] = []
    test_epoch_per_folder_logs: List[Dict[str, Any]] = []

    # 1) discover folders
    all_groups = sorted([d for d in os.listdir(DATA_ROOT) if d.startswith("group_")])
    train_groups = all_groups[:TRAIN_FOLDERS]
    test_groups  = all_groups[TRAIN_FOLDERS:TRAIN_FOLDERS + TEST_FOLDERS]
    print("train groups:", train_groups)
    print("test  groups:", test_groups)

    # 2) cache train folders, find common Tin/Tout
    tins, touts = [], []
    cache: Dict[str, Optional[np.ndarray]] = {}
    for g in train_groups:
        arr = load_pq_sequence(os.path.join(DATA_ROOT, g), MAX_JSONS_PER_FOLDER)
        cache[g] = arr
        if arr is None:
            continue
        T, _, _ = arr.shape
        Tin  = int(T * SPLIT_RATIO)
        Tout = T - Tin
        tins.append(Tin)
        touts.append(Tout)

    if not tins:
        raise RuntimeError("no train data")

    common_Tin  = min(tins)
    common_Tout = min(touts)
    print(f"common Tin={common_Tin}, common Tout={common_Tout}")

    # 3) GLOBAL stats (train-only) used for ALL test folders
    global_stats = compute_global_train_stats(train_groups, cache, common_Tin, common_Tout)
    print("Computed GLOBAL train stats for test normalization.")

    # 4) init model
    model = CVAE(tin=common_Tin, tout=common_Tout, hid=HID_DIM, z_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    global_epoch = 0

    # 5) train folder-by-folder (folder normalization for TRAIN metrics/loss is optional;
    #    here we keep GLOBAL normalization to be consistent with strict test)
    for g in train_groups:
        print(f"\n=== Training on folder: {g} ===")
        arr = cache[g]
        if arr is None:
            print("  skip (no data)")
            continue

        T, B, _ = arr.shape
        if T < (common_Tin + common_Tout):
            print("  skip, too short for common Tin/Tout")
            continue

        # ✅ contiguous split
        past_all = arr[:common_Tin]
        fut_all  = arr[common_Tin:common_Tin + common_Tout]

        # samples as (B,Tin,2) and (B,Tout,2)
        X = np.transpose(past_all, (1, 0, 2))
        Y = np.transpose(fut_all,  (1, 0, 2))
        N = X.shape[0]

        x_mean, x_std = global_stats["x_mean"], global_stats["x_std"]
        y_mean, y_std = global_stats["y_mean"], global_stats["y_std"]

        Xn = (X - x_mean.squeeze(0)) / x_std.squeeze(0)
        Yn = (Y - y_mean.squeeze(0)) / y_std.squeeze(0)

        idxs = np.arange(N)

        for ep in range(1, EPOCHS_PER_FOLDER + 1):
            global_epoch += 1
            np.random.shuffle(idxs)

            model.train()
            loss_sum = 0.0
            recon_sum = 0.0
            kl_sum = 0.0
            t0 = time.time()

            if DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats(DEVICE)

            for start in range(0, N, BATCH_SIZE):
                end = start + BATCH_SIZE
                bidx = idxs[start:end]

                past   = torch.tensor(Xn[bidx], dtype=torch.float32, device=DEVICE)
                future = torch.tensor(Yn[bidx], dtype=torch.float32, device=DEVICE)

                optimizer.zero_grad()
                recon, mu, logvar = model(past, future)
                loss, rec_l, kl_l = cvae_loss(recon, future, mu, logvar, beta=BETA_KL, mode=RECON_MODE)
                loss.backward()
                optimizer.step()

                Bcur = past.size(0)
                loss_sum  += loss.item() * Bcur
                recon_sum += rec_l.item() * Bcur
                kl_sum    += kl_l.item() * Bcur

            dt = time.time() - t0
            mem_mb = float("nan")
            if DEVICE.type == "cuda":
                mem_mb = float(torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2))

            # ---- train metrics (use deterministic z sample per global epoch) ----
            model.eval()
            with torch.no_grad():
                z_gen = torch.Generator(device=DEVICE)
                z_gen.manual_seed(global_epoch)

                past_t = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)  # (N,Tin,2)
                z = torch.randn(N, LATENT_DIM, device=DEVICE, generator=z_gen)
                predn = model.decode(past_t, z).cpu().numpy()  # (N,Tout,2)

            pred = predn * y_std.squeeze(0) + y_mean.squeeze(0)
            train_metrics = compute_all_metrics(Y, pred)

            print(
                f"  [folder {g} ep {ep:02d} (global {global_epoch})] "
                f"loss={loss_sum/N:.6f} | recon={recon_sum/N:.6f} | kl={kl_sum/N:.6f} | "
                f"{dt:.1f}s | train_MAE={train_metrics['mae']:.6f} train_RMSE={train_metrics['rmse']:.6f}"
            )

            training_logs.append({
                "folder": g,
                "epoch": ep,
                "global_epoch": global_epoch,
                "loss": loss_sum / N,
                "recon": recon_sum / N,
                "kl": kl_sum / N,
                "time_sec": dt,
                "gpu_mem_mb": mem_mb,
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_mape": train_metrics["mape"],
                "train_ks": train_metrics["ks"],
            })

            # ===== NEW: TEST EVAL EACH EPOCH =====
            if (global_epoch % EVAL_TEST_EVERY) == 0 and len(test_groups) > 0:
                overall_test, per_folder_rows, _ = eval_on_test_folders(
                    model=model,
                    test_groups=test_groups,
                    common_Tin=common_Tin,
                    common_Tout=common_Tout,
                    global_stats=global_stats,
                    seed=global_epoch,
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

        torch.save(model.state_dict(), os.path.join(OUT_ROOT, f"cvae_after_{g}.pt"))

    # ======================== FINAL TEST ========================
    print("\n=== FINAL TEST (cVAE) ===")
    model.eval()

    all_true_list, all_pred_list = [], []
    first_figs_done = False

    z_gen = torch.Generator(device=DEVICE)
    z_gen.manual_seed(123456)  # fixed for final test reproducibility

    for g in test_groups:
        arr = load_pq_sequence(os.path.join(DATA_ROOT, g), MAX_JSONS_PER_FOLDER)
        if arr is None:
            print(f"  skip test {g}")
            continue

        pack = predict_folder_batch(model, arr, common_Tin, common_Tout, global_stats, z_gen=z_gen)
        if pack is None:
            print(f"  test {g} too short, skipping")
            continue

        fut_all, preds = pack
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
            seed=987654,
        )

    # ======================== SAVE LOGS ========================
    train_csv_path = os.path.join(OUT_ROOT, "training_log.csv")
    save_csv(
        train_csv_path,
        training_logs,
        ["folder", "epoch", "global_epoch", "loss", "recon", "kl", "time_sec", "gpu_mem_mb",
         "train_mae", "train_rmse", "train_mape", "train_ks"]
    )

    test_csv_path = os.path.join(OUT_ROOT, "test_log.csv")
    save_csv(test_csv_path, test_logs, ["folder", "mae", "rmse", "mape", "ks"])

    pjm_real_csv_path = os.path.join(OUT_ROOT, "pjm_real_load_test_log.csv")
    if pjm_real_logs:
        save_csv(
            pjm_real_csv_path,
            pjm_real_logs,
            ["folder", "mae", "rmse", "mape", "ks", "num_pseudo_buses", "pf"]
        )

    # NEW: save test-vs-epoch CSVs
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

    # ======================== PLOTS: TRAIN METRICS vs GLOBAL EPOCH ========================
    if training_logs:
        epochs = np.array([r["global_epoch"] for r in training_logs], dtype=np.int32)
        losses = np.array([r["loss"] for r in training_logs], dtype=np.float32)
        recons = np.array([r["recon"] for r in training_logs], dtype=np.float32)
        kls    = np.array([r["kl"] for r in training_logs], dtype=np.float32)
        maes   = np.array([r["train_mae"] for r in training_logs], dtype=np.float32)
        rmses  = np.array([r["train_rmse"] for r in training_logs], dtype=np.float32)
        kss    = np.array([r["train_ks"] for r in training_logs], dtype=np.float32)

        np.savez_compressed(
            os.path.join(FIGS_DIR, "cvae_train_metrics_vs_epoch.npz"),
            epoch=epochs, loss=losses, recon=recons, kl=kls, mae=maes, rmse=rmses, ks=kss
        )

        plot_single_curve_pdf(epochs, losses, XLAB_EPOCH, r"$\mathrm{loss}$",
                              os.path.join(FIGS_DIR, "cvae_train_loss_vs_epoch.pdf"),
                              color=COLORS_QUAL[0], ylog=False)
        plot_single_curve_pdf(epochs, recons, XLAB_EPOCH, r"$\mathrm{recon}$",
                              os.path.join(FIGS_DIR, "cvae_train_recon_vs_epoch.pdf"),
                              color=COLORS_QUAL[1], ylog=False)
        plot_single_curve_pdf(epochs, kls, XLAB_EPOCH, r"$\mathrm{KL}$",
                              os.path.join(FIGS_DIR, "cvae_train_kl_vs_epoch.pdf"),
                              color=COLORS_QUAL[2], ylog=True)
        plot_single_curve_pdf(epochs, maes, XLAB_EPOCH, r"$\mathrm{MAE}$",
                              os.path.join(FIGS_DIR, "cvae_train_mae_vs_epoch.pdf"),
                              color=COLORS_QUAL[3], ylog=False)
        plot_single_curve_pdf(epochs, rmses, XLAB_EPOCH, r"$\mathrm{RMSE}$",
                              os.path.join(FIGS_DIR, "cvae_train_rmse_vs_epoch.pdf"),
                              color=COLORS_QUAL[4], ylog=False)
        plot_single_curve_pdf(epochs, kss, XLAB_EPOCH, r"$\mathrm{KS}$",
                              os.path.join(FIGS_DIR, "cvae_train_ks_vs_epoch.pdf"),
                              color=COLORS_QUAL[5], ylog=False)

    # ======================== PLOTS: OVERALL TEST METRICS vs GLOBAL EPOCH ========================
    if test_epoch_overall_logs:
        te    = np.array([r["global_epoch"] for r in test_epoch_overall_logs], dtype=np.int32)
        tmae  = np.array([r["test_mae_all"] for r in test_epoch_overall_logs], dtype=np.float32)
        trmse = np.array([r["test_rmse_all"] for r in test_epoch_overall_logs], dtype=np.float32)
        tmape = np.array([r["test_mape_all"] for r in test_epoch_overall_logs], dtype=np.float32)
        tks   = np.array([r["test_ks_all"] for r in test_epoch_overall_logs], dtype=np.float32)

        np.savez_compressed(
            os.path.join(FIGS_DIR, "cvae_test_overall_metrics_vs_epoch.npz"),
            epoch=te, mae=tmae, rmse=trmse, mape=tmape, ks=tks
        )

        plot_single_curve_pdf(te, tmae, XLAB_EPOCH, r"$\mathrm{MAE}$",
                              os.path.join(FIGS_DIR, "cvae_test_overall_mae_vs_epoch.pdf"),
                              color=COLORS_QUAL[1], ylog=False)
        plot_single_curve_pdf(te, trmse, XLAB_EPOCH, r"$\mathrm{RMSE}$",
                              os.path.join(FIGS_DIR, "cvae_test_overall_rmse_vs_epoch.pdf"),
                              color=COLORS_QUAL[2], ylog=False)
        plot_single_curve_pdf(te, tks, XLAB_EPOCH, r"$\mathrm{KS}$",
                              os.path.join(FIGS_DIR, "cvae_test_overall_ks_vs_epoch.pdf"),
                              color=COLORS_QUAL[3], ylog=False)
        plot_single_curve_pdf(te, tmape, XLAB_EPOCH, r"$\mathrm{MAPE}$",
                              os.path.join(FIGS_DIR, "cvae_test_overall_mape_vs_epoch.pdf"),
                              color=COLORS_QUAL[4], ylog=True)

    print("\nSaved training log to:", train_csv_path)
    print("Saved final test log to:", test_csv_path)
    if pjm_real_logs:
        print("Saved PJM real-data test log to:", pjm_real_csv_path)
    if test_epoch_overall_logs:
        print("Saved test-overall-vs-epoch CSV to:", test_epoch_overall_csv)
        print("Saved test-overall-vs-epoch NPZ/PDFs in:", FIGS_DIR)
    if test_epoch_per_folder_logs:
        print("Saved test-per-folder-vs-epoch CSV to:", test_epoch_per_folder_csv)
    print("All figures/NPZ saved in:", FIGS_DIR)


if __name__ == "__main__":
    main()
