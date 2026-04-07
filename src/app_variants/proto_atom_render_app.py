#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FIX IMPORT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============================================================
# IMPORT CORE FUNCTIONS
# ============================================================

from src.proto_atoms.proto_atom_final_render_best import (
    laplacian,
    compute_flux,
    compute_divergence,
    compute_node_field,
    normalize,
    gaussian_center_mask,
    edge_mask,
    make_local_phase_field,
    compute_occupancy,
    weighted_centroid,
)

PHI_CLIP = 6.0

OUT = ROOT / "results" / "app_variants" / "proto_atom_render"
OUT.mkdir(parents=True, exist_ok=True)

FIG_OUT = ROOT / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# METRICS
# ============================================================

def centroid_offset(occ):
    cx, cy = weighted_centroid(occ)
    cx0 = (occ.shape[1] - 1) / 2
    cy0 = (occ.shape[0] - 1) / 2
    return float(np.sqrt((cx - cx0) ** 2 + (cy - cy0) ** 2))


def weighted_covariance(w, cx, cy):
    total = float(np.sum(w))
    if total <= 1e-12:
        return np.eye(2)

    y, x = np.indices(w.shape)
    dx = x - cx
    dy = y - cy

    cxx = float(np.sum(w * dx * dx) / total)
    cyy = float(np.sum(w * dy * dy) / total)
    cxy = float(np.sum(w * dx * dy) / total)

    return np.array([[cxx, cxy], [cxy, cyy]])


def anisotropy_ratio(occ):
    cx, cy = weighted_centroid(occ)
    cov = weighted_covariance(occ, cx, cy)
    eigvals = np.linalg.eigvalsh(cov)
    return float(eigvals.max() / (eigvals.min() + 1e-8))


def core_mass_fraction(occ, r_core=10.0):
    total = float(np.sum(occ))
    if total <= 1e-12:
        return 0.0

    h, w = occ.shape
    y, x = np.indices(occ.shape)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    core = occ[r < r_core]
    return float(core.sum() / total)


def split_score(occ):
    total = float(np.sum(occ))
    if total <= 1e-12:
        return 0.0

    h, w = occ.shape
    cy, cx = (h - 1) / 2, (w - 1) / 2
    y, x = np.indices(occ.shape)

    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    ring = (r > 3.0) & (r < 18.0)

    top = float(occ[ring & (y < cy)].sum())
    bottom = float(occ[ring & (y > cy)].sum())
    left = float(occ[ring & (x < cx)].sum())
    right = float(occ[ring & (x > cx)].sum())

    tb = abs(top - bottom) / (top + bottom + 1e-8)
    lr = abs(left - right) / (left + right + 1e-8)

    return float(max(tb, lr))


def stability_index(final_std, offset, anisotropy, core, split):
    score = 0.0
    score += np.exp(-final_std)
    score += np.exp(-0.5 * offset)
    score += 1.0 / (1.0 + (anisotropy - 1.0))
    score += core
    score -= split
    return float(score)

# ============================================================
# SIMULATION
# ============================================================

def run_simulation(**params):
    rng = np.random.default_rng(params["seed"])

    phi0 = rng.uniform(-3, 3, (params["size"], params["size"]))
    phi = phi0.copy()
    phase0 = make_local_phase_field(phi.shape, rng)

    std_hist = []
    flux_hist = []
    node_hist = []

    for step in range(params["n_steps"]):
        lap = laplacian(phi)
        gx, gy, flux_mag = compute_flux(phi)
        div_flux = compute_divergence(gx, gy)
        node_field = compute_node_field(phi, flux_mag)

        background = params["background_gain"] * np.sin(params["omega_bg"] * step) * phi0
        local_beating = params["local_beat_gain"] * np.sin(params["omega_local"] * step + phase0) * phi

        matter_feedback = params["matter_gain"] * normalize(-div_flux)
        node_feedback = params["node_gain"] * normalize(node_field)
        flux_feedback = params["flux_gain"] * normalize(flux_mag)

        center = gaussian_center_mask(phi.shape)
        edges = edge_mask(phi.shape)

        center_capture = params["center_gain"] * center * normalize(node_field + np.maximum(-div_flux, 0.0))
        anti_edge = -params["edge_penalty"] * edges * normalize(node_field)

        stabilizer = -0.015 * phi * np.abs(phi)

        phi = (
            phi
            + 0.085 * lap
            - 0.008 * params["beta"] * phi
            + background
            + local_beating
            + matter_feedback
            + node_feedback
            + flux_feedback
            + center_capture
            + anti_edge
            + stabilizer
        )

        phi = np.clip(phi, -PHI_CLIP, PHI_CLIP)

        std_hist.append(float(np.std(phi)))
        flux_hist.append(float(np.max(flux_mag)))
        node_hist.append(float(np.max(node_field)))

    gx, gy, flux_mag = compute_flux(phi)
    node_field = compute_node_field(phi, flux_mag)
    occ = compute_occupancy(node_field)

    cx, cy = weighted_centroid(occ)

    # ===============================
    # METRICS
    # ===============================

    final_std = std_hist[-1]
    offset = centroid_offset(occ)
    anis = anisotropy_ratio(occ)
    core = core_mass_fraction(occ)
    split = split_score(occ)

    score = stability_index(final_std, offset, anis, core, split)

    if score > 2.5:
        stability = "stable"
    elif score > 1.5:
        stability = "meta-stable"
    else:
        stability = "unstable"

    # ===============================
    # FIGURES
    # ===============================

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(phi)
    axes[1].imshow(node_field)
    axes[2].imshow(occ)
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIG_OUT / "app_proto_atom_render.png", dpi=300)
    plt.close()

    # ===============================
    # PRINT REPORT
    # ===============================

    print("\n=== STABILITY REPORT ===")
    print(f"final_std={final_std}")
    print(f"centroid_offset={offset}")
    print(f"anisotropy={anis}")
    print(f"core_mass_fraction={core}")
    print(f"split_score={split}")
    print(f"stability_index={score}")
    print(f"stability_class={stability}")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--beta", type=float, default=8.75)
    parser.add_argument("--center_gain", type=float, default=0.012)
    parser.add_argument("--node_gain", type=float, default=0.1)
    parser.add_argument("--matter_gain", type=float, default=0.098)
    parser.add_argument("--omega_bg", type=float, default=0.22)
    parser.add_argument("--background_gain", type=float, default=0.035)
    parser.add_argument("--omega_local", type=float, default=0.47)
    parser.add_argument("--local_beat_gain", type=float, default=0.085)
    parser.add_argument("--flux_gain", type=float, default=0.045)
    parser.add_argument("--edge_penalty", type=float, default=0.1)

    args = parser.parse_args()

    run_simulation(**vars(args))


if __name__ == "__main__":
    main()
