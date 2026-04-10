#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import itertools

OUT = Path("results/bell/lorentz_pauli")
OUT.mkdir(parents=True, exist_ok=True)

SOURCE_V8 = Path("src/bell/bell_v8_anisotropic_preparation.py")


# ============================================================
# Dynamic loader
# ============================================================
def load_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location("v8_module", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# Local geometry
# ============================================================
def local_disk_mask(size: int, center_xy: np.ndarray, radius: float):
    y, x = np.indices((size, size))
    dx = x - center_xy[0]
    dy = y - center_xy[1]
    return (dx * dx + dy * dy) <= radius * radius


def weighted_centroid(phi, center_xy, radius):
    mask = local_disk_mask(phi.shape[0], center_xy, radius)

    y, x = np.indices(phi.shape)
    w = np.abs(phi[mask])
    total = np.sum(w)

    if total < 1e-12:
        return center_xy.copy()

    cx = np.sum(x[mask] * w) / total
    cy = np.sum(y[mask] * w) / total
    return np.array([cx, cy])


# ============================================================
# Lorentz diagnostic
# ============================================================
def lorentz_observable(saved_fields, center_xy, angle_deg, start, delta, dt):

    theta = np.radians(angle_deg)
    axis = np.array([np.cos(theta), np.sin(theta)])

    steps = [start + i * delta for i in range(5)]

    centroids = []

    for s in steps:
        phi = saved_fields[s]
        c = weighted_centroid(phi, center_xy, radius=10.0)
        centroids.append(c)

    centroids = np.array(centroids)

    rel = centroids - center_xy
    proj = rel @ axis

    velocities = np.diff(proj) / max(dt * delta, 1e-12)

    return {
        "mean_speed": float(np.mean(velocities)),
        "std_speed": float(np.std(velocities)),
        "mean_proj": float(np.mean(proj)),
    }


# ============================================================
# Pauli diagnostic
# ============================================================
def pauli_observable(saved_fields, center_xy, start, delta):

    axes = [0.0, 45.0, 90.0]

    results = []

    for angle in axes:
        theta = np.radians(angle)
        axis = np.array([np.cos(theta), np.sin(theta)])

        steps = [start + i * delta for i in range(5)]

        vals = []

        for s in steps:
            phi = saved_fields[s]
            c = weighted_centroid(phi, center_xy, 10.0)
            rel = c - center_xy
            vals.append(rel @ axis)

        vals = np.array(vals)
        results.append(np.mean(vals))

    # non-commutativity proxy
    return {
        "axis_0": results[0],
        "axis_45": results[1],
        "axis_90": results[2],
        "incompatibility": float(np.std(results)),
    }


# ============================================================
# Main
# ============================================================
def main():

    print("\n=== LORENTZ–PAULI DIAGNOSTICS ===")

    v8 = load_module(SOURCE_V8)

    cases = [
        {
            "label": "baseline",
            "separation": 20.0,
            "phase_offset": 0.0,
            "anisotropy": 0.0,
            "theta": 0.0,
        },
        {
            "label": "anisotropic",
            "separation": 20.0,
            "phase_offset": np.pi / 4,
            "anisotropy": 1.2,
            "theta": np.pi / 4,
        },
    ]

    rows = []

    for case in cases:

        print(f"\n--- {case['label']} ---")

        src_a, src_b = v8.source_positions(case["separation"])

        for seed in range(32):

            saved = v8.run_prepared_field_history(
                seed=seed,
                separation=case["separation"],
                phase_offset=case["phase_offset"],
                required_steps=set(range(100, 140)),
                aniso_strength=case["anisotropy"],
                source_orientation_offset=case["theta"],
            )

            lorentz_a = lorentz_observable(saved, src_a, 0.0, 120, 4, v8.DT)
            lorentz_b = lorentz_observable(saved, src_b, 45.0, 120, 4, v8.DT)

            pauli_a = pauli_observable(saved, src_a, 120, 4)
            pauli_b = pauli_observable(saved, src_b, 120, 4)

            row = {
                "case": case["label"],
                "seed": seed,

                "lorentz_speed_a": lorentz_a["mean_speed"],
                "lorentz_speed_b": lorentz_b["mean_speed"],

                "lorentz_proj_a": lorentz_a["mean_proj"],
                "lorentz_proj_b": lorentz_b["mean_proj"],

                "pauli_incompat_a": pauli_a["incompatibility"],
                "pauli_incompat_b": pauli_b["incompatibility"],
            }

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lorentz_pauli_results.csv", index=False)

    print("\n=== SUMMARY ===")
    print(df.groupby("case").mean())

    print("\n[OK] wrote results")
    print("[DONE]")


if __name__ == "__main__":
    main()