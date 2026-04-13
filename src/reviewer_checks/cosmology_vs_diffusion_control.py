#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT = Path("results/reviewer_checks/cosmology_vs_diffusion")
OUT.mkdir(parents=True, exist_ok=True)


def laplacian(field: np.ndarray) -> np.ndarray:
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    )


def gaussian_blob(size: int = 160, sigma: float = 7.0, amp: float = 1.0) -> np.ndarray:
    y, x = np.indices((size, size))
    cx = cy = size / 2.0
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return amp * np.exp(-r2 / (2.0 * sigma**2))


def scale_factor_proxy(field: np.ndarray) -> float:
    y, x = np.indices(field.shape)
    cy = field.shape[0] / 2.0
    cx = field.shape[1] / 2.0
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    w = np.abs(field)
    denom = np.sum(w) + 1e-12
    return float(np.sqrt(np.sum(r2 * w) / denom))


def run_diffusion(
    psi0: np.ndarray,
    *,
    dt: float = 0.05,
    n_steps: int = 220,
    diff_coeff: float = 0.18,
):
    psi = psi0.copy()
    history = []

    for _ in range(n_steps):
        psi = psi + dt * diff_coeff * laplacian(psi)
        history.append(scale_factor_proxy(psi))

    return psi, np.array(history, dtype=float)


def run_omm_like(
    psi0: np.ndarray,
    *,
    dt: float = 0.05,
    n_steps: int = 220,
    wave_gain: float = 0.18,
    damp: float = 0.998,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    psi = psi0.copy()
    vel = np.zeros_like(psi)
    history = []

    for _ in range(n_steps):
        nonlinear = alpha * psi - beta * psi**3
        vel = damp * vel + dt * (wave_gain * laplacian(psi) + nonlinear)
        psi = psi + dt * vel
        history.append(scale_factor_proxy(psi))

    return psi, np.array(history, dtype=float)


def hubble_proxy(a: np.ndarray, dt: float) -> np.ndarray:
    da = np.gradient(a, dt)
    return da / np.maximum(a, 1e-12)


def main():
    print("=== Cosmology vs Diffusion Control ===")

    dt = 0.05
    n_steps = 240
    psi0 = gaussian_blob()

    diff_field, a_diff = run_diffusion(psi0, dt=dt, n_steps=n_steps)
    omm_field, a_omm = run_omm_like(psi0, dt=dt, n_steps=n_steps)

    H_diff = hubble_proxy(a_diff, dt)
    H_omm = hubble_proxy(a_omm, dt)

    summary = pd.DataFrame(
        [
            {
                "model": "diffusion",
                "a_initial": float(a_diff[0]),
                "a_final": float(a_diff[-1]),
                "H_mean": float(np.mean(H_diff)),
                "H_std": float(np.std(H_diff)),
            },
            {
                "model": "omm_like",
                "a_initial": float(a_omm[0]),
                "a_final": float(a_omm[-1]),
                "H_mean": float(np.mean(H_omm)),
                "H_std": float(np.std(H_omm)),
            },
        ]
    )
    summary.to_csv(OUT / "cosmology_vs_diffusion_summary.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(a_diff, label="diffusion")
    plt.plot(a_omm, label="omm_like")
    plt.xlabel("step")
    plt.ylabel("scale-factor proxy a(t)")
    plt.title("Expansion proxy comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "a_t_comparison.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(H_diff, label="diffusion")
    plt.plot(H_omm, label="omm_like")
    plt.xlabel("step")
    plt.ylabel("H(t) proxy")
    plt.title("Hubble-like proxy comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "H_t_comparison.png", dpi=220)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(diff_field, cmap="coolwarm")
    axes[0].set_title("Diffusion final field")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(omm_field, cmap="coolwarm")
    axes[1].set_title("OMM-like final field")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(OUT / "final_field_comparison.png", dpi=220)
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()