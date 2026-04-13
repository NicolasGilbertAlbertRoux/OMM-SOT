#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT = Path("results/reviewer_checks/stability_seed_sensitivity")
OUT.mkdir(parents=True, exist_ok=True)


def laplacian(field: np.ndarray) -> np.ndarray:
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    )


def energy_proxy(psi: np.ndarray, vel: np.ndarray, wave_gain: float) -> float:
    gy, gx = np.gradient(psi)
    return float(
        0.5 * np.mean(vel**2) + 0.5 * wave_gain * np.mean(gx**2 + gy**2) + np.mean(np.abs(psi)**2)
    )


def run_case(
    *,
    seed: int,
    dt: float,
    do_clip: bool,
    size: int = 128,
    n_steps: int = 180,
    wave_gain: float = 0.18,
    damp: float = 0.998,
    clip_value: float = 10.0,
):
    rng = np.random.default_rng(seed)
    psi = 1e-3 * rng.normal(size=(size, size))
    vel = np.zeros_like(psi)

    energies = []
    max_abs_trace = []

    for step in range(n_steps):
        nonlinear = psi - psi**3
        vel = damp * vel + dt * (wave_gain * laplacian(psi) + nonlinear)
        psi = psi + dt * vel

        if do_clip:
            psi = np.clip(psi, -clip_value, clip_value)

        energies.append(energy_proxy(psi, vel, wave_gain))
        max_abs_trace.append(float(np.max(np.abs(psi))))

        if not np.isfinite(psi).all():
            return {
                "seed": seed,
                "dt": dt,
                "clip": do_clip,
                "status": "non_finite",
                "final_energy": np.nan,
                "final_max_abs": np.nan,
            }, np.array(energies), np.array(max_abs_trace)

    return {
        "seed": seed,
        "dt": dt,
        "clip": do_clip,
        "status": "ok",
        "final_energy": float(energies[-1]),
        "final_max_abs": float(max_abs_trace[-1]),
    }, np.array(energies), np.array(max_abs_trace)


def main():
    print("=== Stability and Seed Sensitivity ===")

    dts = [0.03, 0.05, 0.08, 0.12, 0.20, 0.50]
    seeds = list(range(20))

    rows = []

    for dt in dts:
        for do_clip in [False, True]:
            energy_bank = []
            amp_bank = []

            for seed in seeds:
                row, energies, amps = run_case(seed=seed, dt=dt, do_clip=do_clip)
                rows.append(row)

                if row["status"] == "ok":
                    energy_bank.append(energies)
                    amp_bank.append(amps)

            if energy_bank:
                energy_bank = np.array(energy_bank)
                amp_bank = np.array(amp_bank)

                mean_energy = np.mean(energy_bank, axis=0)
                std_energy = np.std(energy_bank, axis=0)

                plt.figure(figsize=(7, 4))
                plt.plot(mean_energy, label="mean energy")
                plt.fill_between(
                    np.arange(len(mean_energy)),
                    mean_energy - std_energy,
                    mean_energy + std_energy,
                    alpha=0.3,
                    label="±1 std",
                )
                plt.xlabel("step")
                plt.ylabel("energy proxy")
                plt.title(f"dt={dt:.2f} | clip={do_clip}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(OUT / f"energy_dt_{dt:.2f}_clip_{int(do_clip)}.png", dpi=220)
                plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "stability_seed_summary.csv", index=False)

    summary = (
        df.groupby(["dt", "clip", "status"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    summary.to_csv(OUT / "stability_seed_grouped_counts.csv", index=False)

    print(df.head().to_string(index=False))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()