#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT = Path("results/reviewer_checks/explicit_potential")
OUT.mkdir(parents=True, exist_ok=True)


def potential_quadratic(psi: np.ndarray, m2: float = 1.0) -> np.ndarray:
    return 0.5 * m2 * psi**2


def force_quadratic(psi: np.ndarray, m2: float = 1.0) -> np.ndarray:
    return -m2 * psi


def potential_phi4(psi: np.ndarray, m2: float = 1.0, lam: float = 1.0) -> np.ndarray:
    return 0.5 * m2 * psi**2 + 0.25 * lam * psi**4


def force_phi4(psi: np.ndarray, m2: float = 1.0, lam: float = 1.0) -> np.ndarray:
    return -(m2 * psi + lam * psi**3)


def potential_double_well(psi: np.ndarray, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    return -0.5 * a * psi**2 + 0.25 * b * psi**4


def force_double_well(psi: np.ndarray, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    return a * psi - b * psi**3


def laplacian(field: np.ndarray) -> np.ndarray:
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    )


def run_simulation(
    force_fn,
    *,
    size: int = 128,
    dt: float = 0.05,
    n_steps: int = 220,
    wave_gain: float = 0.16,
    damp: float = 0.998,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    psi = 1e-3 * rng.normal(size=(size, size))
    vel = np.zeros_like(psi)

    snapshots = {}
    energy_trace = []

    for step in range(n_steps):
        lap = laplacian(psi)
        force = force_fn(psi)

        vel = damp * vel + dt * (wave_gain * lap + force)
        psi = psi + dt * vel

        grad_y, grad_x = np.gradient(psi)
        energy = (
            0.5 * np.mean(vel**2)
            + 0.5 * wave_gain * np.mean(grad_x**2 + grad_y**2)
            + np.mean(np.abs(force_fn(psi)))
        )
        energy_trace.append(float(energy))

        if step in {0, 40, 80, 120, 180, n_steps - 1}:
            snapshots[step] = psi.copy()

    return psi, snapshots, np.array(energy_trace, dtype=float)


def summarize_field(psi: np.ndarray) -> dict:
    flat = psi.ravel()
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    centered = flat - mean
    m2 = float(np.mean(centered**2)) + 1e-12
    m4 = float(np.mean(centered**4))
    kurtosis = m4 / (m2**2)

    threshold = 1.5 * std
    active_fraction = float(np.mean(np.abs(flat) > threshold))

    return {
        "mean": mean,
        "std": std,
        "kurtosis": kurtosis,
        "active_fraction": active_fraction,
        "max_abs": float(np.max(np.abs(flat))),
    }


def plot_snapshots(name: str, snapshots: dict[int, np.ndarray]) -> None:
    steps = sorted(snapshots.keys())
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.ravel()

    for ax, step in zip(axes, steps):
        im = ax.imshow(snapshots[step], cmap="coolwarm")
        ax.set_title(f"{name} | step={step}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)

    plt.tight_layout()
    plt.savefig(OUT / f"{name}_snapshots.png", dpi=220)
    plt.close(fig)


def main():
    print("=== Explicit Potential Checks ===")

    models = [
        ("quadratic", lambda x: force_quadratic(x, m2=1.0)),
        ("phi4", lambda x: force_phi4(x, m2=1.0, lam=1.0)),
        ("double_well", lambda x: force_double_well(x, a=1.0, b=1.0)),
    ]

    summary_rows = []

    for name, force_fn in models:
        final_field, snapshots, energy_trace = run_simulation(force_fn)
        metrics = summarize_field(final_field)
        metrics["model"] = name
        summary_rows.append(metrics)

        plot_snapshots(name, snapshots)

        plt.figure(figsize=(7, 4))
        plt.plot(energy_trace)
        plt.xlabel("step")
        plt.ylabel("energy proxy")
        plt.title(f"Energy trace | {name}")
        plt.tight_layout()
        plt.savefig(OUT / f"{name}_energy_trace.png", dpi=220)
        plt.close()

        np.save(OUT / f"{name}_final_field.npy", final_field)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT / "potential_model_summary.csv", index=False)

    print(summary_df.to_string(index=False))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()