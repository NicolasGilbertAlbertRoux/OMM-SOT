#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path("results/bell/v10_1_normalized")
OUT.mkdir(parents=True, exist_ok=True)

SOURCE_V8 = Path("src/bell/bell_v8_anisotropic_preparation.py")


# ============================================================
# Dynamic loading of the V8 module
# ============================================================
def load_module_from_path(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")

    spec = importlib.util.spec_from_file_location("bell_v8_module", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# CHSH / normalization utilities
# ============================================================
A_ANGLES = [0.0, 45.0]
B_ANGLES = [22.5, 67.5]


def compute_chsh_from_map(corr_map: dict[tuple[float, float], float]) -> float:
    a, ap = A_ANGLES
    b, bp = B_ANGLES
    return corr_map[(a, b)] + corr_map[(a, bp)] + corr_map[(ap, b)] - corr_map[(ap, bp)]


def robust_scale(x: np.ndarray, q: float = 0.95) -> float:
    scale = float(np.quantile(np.abs(x), q))
    return max(scale, 1e-12)


def tanh_norm(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def ratio_norm(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.abs(x))


def clip_norm(x: np.ndarray, scale: float) -> np.ndarray:
    y = x / max(scale, 1e-12)
    return np.clip(y, -1.0, 1.0)


# ============================================================
# Local geometry / event tracking
# ============================================================
def local_disk_mask(size: int, center_xy: np.ndarray, radius: float) -> np.ndarray:
    y, x = np.indices((size, size))
    dx = x - center_xy[0]
    dy = y - center_xy[1]
    return (dx * dx + dy * dy) <= radius * radius


def weighted_centroid_abs(phi: np.ndarray, center_xy: np.ndarray, radius: float) -> tuple[np.ndarray, float]:
    """
    Weighted centroid computed from |phi| inside a local disk.
    Returns:
      - centroid (x, y)
      - total weight
    """
    size_y, size_x = phi.shape
    assert size_y == size_x

    mask = local_disk_mask(size_x, center_xy, radius)

    y, x = np.indices(phi.shape)
    w = np.abs(phi[mask]).astype(float)
    total = float(np.sum(w))

    if total <= 1e-15:
        return center_xy.copy(), 0.0

    xw = float(np.sum(x[mask] * w) / total)
    yw = float(np.sum(y[mask] * w) / total)
    return np.array([xw, yw], dtype=float), total


def event_tracking_observable(
    saved_fields: dict[int, np.ndarray],
    center_xy: np.ndarray,
    angle_deg: float,
    start_step: int,
    delta_step: int,
    *,
    dt: float,
    window_length: int = 5,
    track_radius: float = 10.0,
    amp_weight: float = 0.30,
    speed_weight: float = 1.00,
    drift_weight: float = 0.55,
):
    """
    Continuous observable inspired by event tracking:
    - local amplitude
    - projected drift along the readout axis
    - projected apparent velocity

    The construction is intentionally observer-sensitive:
    a local event is reconstructed and projected onto the chosen
    measurement axis. The resulting score is not bounded a priori.
    """
    theta = np.radians(angle_deg)
    axis_n = np.array([np.cos(theta), np.sin(theta)], dtype=float)

    steps = [start_step + k * delta_step for k in range(window_length)]

    centroids = []
    amps = []

    for step in steps:
        phi = saved_fields[step]
        centroid, amp = weighted_centroid_abs(phi, center_xy, track_radius)
        centroids.append(centroid)
        amps.append(amp)

    centroids = np.array(centroids, dtype=float)
    amps = np.array(amps, dtype=float)

    rel = centroids - center_xy[None, :]
    proj = rel @ axis_n

    if len(proj) >= 2:
        delta_t = delta_step * dt
        vel = np.diff(proj) / max(delta_t, 1e-12)
        mean_speed = float(np.mean(vel))
    else:
        mean_speed = 0.0

    mean_drift = float(np.mean(proj))
    mean_amp = float(np.mean(amps))

    amp_term = np.log1p(mean_amp)

    continuous_value = (
        speed_weight * mean_speed
        + drift_weight * mean_drift
        + amp_weight * amp_term
    )

    binary_value = 1 if continuous_value >= 0.0 else -1

    aux = {
        "continuous_value": float(continuous_value),
        "mean_speed": float(mean_speed),
        "mean_drift": float(mean_drift),
        "mean_amp": float(mean_amp),
        "amp_term": float(amp_term),
    }
    return binary_value, aux


# ============================================================
# Detailed case evaluation
# ============================================================
def evaluate_case_detailed(
    v8,
    separation: float,
    phase_offset: float,
    start_step: int,
    delta_step: int,
    anisotropy: float,
    src_theta: float,
    n_realizations: int | None = None,
):
    """
    Returns a detailed row list:
      - one row per seed and angle pair (a, b)
      - binary and continuous products
    """
    if n_realizations is None:
        n_realizations = int(v8.N_REALIZATIONS)

    required_steps = {
        start_step + k * delta_step
        for k in range(v8.WINDOW_LENGTH)
    }

    src_a, src_b = v8.source_positions(separation)
    rows = []

    for seed in range(n_realizations):
        saved_fields = v8.run_prepared_field_history(
            seed=seed,
            separation=separation,
            phase_offset=phase_offset,
            required_steps=required_steps,
            aniso_strength=anisotropy,
            source_orientation_offset=src_theta,
        )

        for a_angle, b_angle in itertools.product(A_ANGLES, B_ANGLES):
            a_sign, aux_a = event_tracking_observable(
                saved_fields=saved_fields,
                center_xy=src_a,
                angle_deg=a_angle,
                start_step=start_step,
                delta_step=delta_step,
                dt=v8.DT,
                window_length=v8.WINDOW_LENGTH,
            )

            b_sign, aux_b = event_tracking_observable(
                saved_fields=saved_fields,
                center_xy=src_b,
                angle_deg=b_angle,
                start_step=start_step,
                delta_step=delta_step,
                dt=v8.DT,
                window_length=v8.WINDOW_LENGTH,
            )

            a_cont = aux_a["continuous_value"]
            b_cont = aux_b["continuous_value"]

            rows.append(
                {
                    "seed": int(seed),
                    "a_angle": float(a_angle),
                    "b_angle": float(b_angle),
                    "A_sign": int(a_sign),
                    "B_sign": int(b_sign),
                    "sign_product": int(a_sign * b_sign),
                    "A_cont": float(a_cont),
                    "B_cont": float(b_cont),
                    "cont_product": float(a_cont * b_cont),
                    "A_mean_speed": float(aux_a["mean_speed"]),
                    "B_mean_speed": float(aux_b["mean_speed"]),
                    "A_mean_drift": float(aux_a["mean_drift"]),
                    "B_mean_drift": float(aux_b["mean_drift"]),
                    "A_mean_amp": float(aux_a["mean_amp"]),
                    "B_mean_amp": float(aux_b["mean_amp"]),
                }
            )

    return rows


def corr_map_from_column(df: pd.DataFrame, col: str) -> dict[tuple[float, float], float]:
    out = {}
    for a_angle, b_angle in itertools.product(A_ANGLES, B_ANGLES):
        sub = df[(df["a_angle"] == a_angle) & (df["b_angle"] == b_angle)]
        out[(a_angle, b_angle)] = float(np.mean(sub[col]))
    return out


# ============================================================
# Scan grid
# ============================================================
def case_grid(v8, quick: bool = False):
    separations = list(v8.SEPARATIONS)
    phases = list(v8.PHASE_OFFSETS)
    starts = list(v8.START_STEPS)
    deltas = list(v8.DELTA_STEPS)
    anisotropies = list(v8.ANISO_STRENGTHS)
    thetas = list(v8.SOURCE_ORIENTATION_OFFSETS)

    if quick:
        deltas = [2, 8]
        starts = [80, 120]

    for sep, phase, start, delta, aniso, theta in itertools.product(
        separations, phases, starts, deltas, anisotropies, thetas
    ):
        yield {
            "separation": float(sep),
            "phase_offset": float(phase),
            "start_step": int(start),
            "delta_step": int(delta),
            "anisotropy": float(aniso),
            "src_theta": float(theta),
        }


# ============================================================
# Main
# ============================================================
def main():
    print("\n=== BELL V10.1 NORMALIZED EVENT TRACKING ===")

    v8 = load_module_from_path(SOURCE_V8)

    QUICK = False
    n_realizations_override = 120 if QUICK else int(v8.N_REALIZATIONS)

    cases = list(case_grid(v8, quick=QUICK))
    total = len(cases)

    all_detail = []
    summary_rows = []

    for i, case in enumerate(cases, start=1):
        print(
            f"\n[{i}/{total}] "
            f"sep={case['separation']:.1f}, "
            f"phase={case['phase_offset']:.3f}, "
            f"start={case['start_step']}, "
            f"delta={case['delta_step']}, "
            f"aniso={case['anisotropy']:.2f}, "
            f"src_theta={case['src_theta']:.3f}"
        )

        rows = evaluate_case_detailed(
            v8=v8,
            separation=case["separation"],
            phase_offset=case["phase_offset"],
            start_step=case["start_step"],
            delta_step=case["delta_step"],
            anisotropy=case["anisotropy"],
            src_theta=case["src_theta"],
            n_realizations=n_realizations_override,
        )

        df = pd.DataFrame(rows)

        raw = df["cont_product"].to_numpy(dtype=float)
        scale95 = robust_scale(raw, q=0.95)

        df["cont_tanh"] = tanh_norm(raw)
        df["cont_ratio"] = ratio_norm(raw)
        df["cont_clip95"] = clip_norm(raw, scale95)

        sign_map = corr_map_from_column(df, "sign_product")
        raw_map = corr_map_from_column(df, "cont_product")
        tanh_map = corr_map_from_column(df, "cont_tanh")
        ratio_map = corr_map_from_column(df, "cont_ratio")
        clip95_map = corr_map_from_column(df, "cont_clip95")

        s_sign = compute_chsh_from_map(sign_map)
        s_raw = compute_chsh_from_map(raw_map)
        s_tanh = compute_chsh_from_map(tanh_map)
        s_ratio = compute_chsh_from_map(ratio_map)
        s_clip95 = compute_chsh_from_map(clip95_map)

        print(f"S_sign   = {s_sign: .6f}")
        print(f"S_raw    = {s_raw: .6f}")
        print(f"S_tanh   = {s_tanh: .6f}")
        print(f"S_ratio  = {s_ratio: .6f}")
        print(f"S_clip95 = {s_clip95: .6f}")

        for k, v in case.items():
            df[k] = v
        all_detail.append(df)

        summary_rows.append(
            {
                **case,
                "n_realizations": int(n_realizations_override),
                "raw_abs_q95": float(scale95),
                "raw_mean_abs": float(np.mean(np.abs(raw))),
                "raw_max_abs": float(np.max(np.abs(raw))),
                "S_sign": float(s_sign),
                "S_raw": float(s_raw),
                "S_tanh": float(s_tanh),
                "S_ratio": float(s_ratio),
                "S_clip95": float(s_clip95),
                "Craw_0_22.5": raw_map[(0.0, 22.5)],
                "Craw_0_67.5": raw_map[(0.0, 67.5)],
                "Craw_45_22.5": raw_map[(45.0, 22.5)],
                "Craw_45_67.5": raw_map[(45.0, 67.5)],
            }
        )

    detail_df = pd.concat(all_detail, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    detail_csv = OUT / "bell_v10_1_measurements.csv"
    summary_csv = OUT / "bell_v10_1_summary.csv"

    detail_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== BEST CASES BY |S_raw| ===")
    best_raw = summary_df.iloc[np.argsort(np.abs(summary_df["S_raw"]))[::-1]].head(20)
    print(
        best_raw[
            [
                "separation",
                "phase_offset",
                "start_step",
                "delta_step",
                "anisotropy",
                "src_theta",
                "S_sign",
                "S_raw",
                "S_tanh",
                "S_ratio",
                "S_clip95",
                "raw_mean_abs",
                "raw_max_abs",
            ]
        ].to_string(index=False)
    )

    print("\n=== BEST CASES BY |S_tanh| ===")
    best_tanh = summary_df.iloc[np.argsort(np.abs(summary_df["S_tanh"]))[::-1]].head(20)
    print(
        best_tanh[
            [
                "separation",
                "phase_offset",
                "start_step",
                "delta_step",
                "anisotropy",
                "src_theta",
                "S_sign",
                "S_raw",
                "S_tanh",
                "S_ratio",
                "S_clip95",
            ]
        ].to_string(index=False)
    )

    plt.figure(figsize=(7, 4))
    plt.hist(summary_df["S_raw"], bins=30, alpha=0.6, label="S_raw")
    plt.hist(summary_df["S_tanh"], bins=30, alpha=0.6, label="S_tanh")
    plt.hist(summary_df["S_clip95"], bins=30, alpha=0.6, label="S_clip95")
    plt.axvline(2.0, linestyle="--")
    plt.axvline(-2.0, linestyle="--")
    plt.xlabel("S")
    plt.ylabel("count")
    plt.title("Bell V10.1: raw vs normalized S")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "bell_v10_1_S_histograms.png", dpi=220)
    plt.close()

    print(f"\n[OK] wrote {detail_csv}")
    print(f"[OK] wrote {summary_csv}")
    print(f"[OK] wrote {OUT / 'bell_v10_1_S_histograms.png'}")
    print("[DONE] Bell V10.1 complete")


if __name__ == "__main__":
    main()