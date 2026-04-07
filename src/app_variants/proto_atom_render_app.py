#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def run_simulation(
    size: int,
    n_steps: int,
    seed: int,
    beta: float,
    center_gain: float,
    node_gain: float,
    matter_gain: float,
    omega_bg: float,
    background_gain: float,
    omega_local: float,
    local_beat_gain: float,
    flux_gain: float,
    edge_penalty: float,
):
    rng = np.random.default_rng(seed)

    phi0 = rng.uniform(-3, 3, (size, size))
    phi = phi0.copy()
    phase0 = make_local_phase_field(phi.shape, rng)

    std_hist = []
    flux_hist = []
    node_hist = []

    for step in range(n_steps):
        lap = laplacian(phi)
        gx, gy, flux_mag = compute_flux(phi)
        div_flux = compute_divergence(gx, gy)
        node_field = compute_node_field(phi, flux_mag)

        background = background_gain * np.sin(omega_bg * step) * phi0
        local_beating = local_beat_gain * np.sin(omega_local * step + phase0) * phi

        matter_feedback = matter_gain * normalize(-div_flux)
        node_feedback = node_gain * normalize(node_field)
        flux_feedback = flux_gain * normalize(flux_mag)

        center = gaussian_center_mask(phi.shape)
        edges = edge_mask(phi.shape)

        center_capture = center_gain * center * normalize(node_field + np.maximum(-div_flux, 0.0))
        anti_edge = -edge_penalty * edges * normalize(node_field)

        stabilizer = -0.015 * phi * np.abs(phi)
        stabilizer = np.nan_to_num(stabilizer, nan=0.0, posinf=0.0, neginf=0.0)

        phi = (
            phi
            + 0.085 * lap
            - 0.008 * beta * phi
            + background
            + local_beating
            + matter_feedback
            + node_feedback
            + flux_feedback
            + center_capture
            + anti_edge
            + stabilizer
        )

        phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        phi = np.clip(phi, -PHI_CLIP, PHI_CLIP)

        std_hist.append(float(np.std(phi)))
        flux_hist.append(float(np.max(flux_mag)))
        node_hist.append(float(np.max(node_field)))

    gx, gy, flux_mag = compute_flux(phi)
    node_field = compute_node_field(phi, flux_mag)
    occ = compute_occupancy(node_field)
    cx, cy = weighted_centroid(occ)

    np.save(OUT / "phi_final.npy", phi)
    np.save(OUT / "node_field_final.npy", node_field)
    np.save(OUT / "occupancy_final.npy", occ)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    axes[0].imshow(phi, cmap="viridis")
    axes[0].set_title("Field φ")

    axes[1].imshow(node_field, cmap="inferno")
    axes[1].set_title("Node field")

    axes[2].imshow(occ, cmap="hot")
    axes[2].scatter(cx, cy, c="cyan", s=50)
    axes[2].set_title("Occupancy + centroid")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    main_fig = FIG_OUT / "app_proto_atom_render.png"
    plt.savefig(main_fig, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(std_hist)
    axes[0].set_title("phi std")

    axes[1].plot(flux_hist)
    axes[1].set_title("flux max")

    axes[2].plot(node_hist)
    axes[2].set_title("node max")

    plt.tight_layout()
    diag_fig = FIG_OUT / "app_proto_atom_render_diagnostics.png"
    plt.savefig(diag_fig, dpi=220)
    plt.close(fig)

    final_std = std_hist[-1]

    if final_std < 0.5:
        stability = "stable"
    elif final_std < 1.5:
        stability = "meta-stable"
    else:
        stability = "unstable"

    print(f"[OK] wrote {main_fig}")
    print(f"[OK] wrote {diag_fig}")
    print(f"[INFO] final_std={final_std:.6f}")
    print(f"[INFO] stability={stability}")


def main():
    parser = argparse.ArgumentParser(description="Interactive proto-atom render based on the real OMM code.")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--beta", type=float, default=8.75)
    parser.add_argument("--center_gain", type=float, default=0.012)
    parser.add_argument("--node_gain", type=float, default=0.100)
    parser.add_argument("--matter_gain", type=float, default=0.098)
    parser.add_argument("--omega_bg", type=float, default=0.22)
    parser.add_argument("--background_gain", type=float, default=0.035)
    parser.add_argument("--omega_local", type=float, default=0.47)
    parser.add_argument("--local_beat_gain", type=float, default=0.085)
    parser.add_argument("--flux_gain", type=float, default=0.045)
    parser.add_argument("--edge_penalty", type=float, default=0.10)

    args = parser.parse_args()

    run_simulation(
        size=args.size,
        n_steps=args.n_steps,
        seed=args.seed,
        beta=args.beta,
        center_gain=args.center_gain,
        node_gain=args.node_gain,
        matter_gain=args.matter_gain,
        omega_bg=args.omega_bg,
        background_gain=args.background_gain,
        omega_local=args.omega_local,
        local_beat_gain=args.local_beat_gain,
        flux_gain=args.flux_gain,
        edge_penalty=args.edge_penalty,
    )


if __name__ == "__main__":
    main()
