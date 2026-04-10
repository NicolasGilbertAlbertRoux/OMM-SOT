#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util
import itertools

import numpy as np
import pandas as pd


OUT = Path("results/bell/v10_2_targeted")
OUT.mkdir(parents=True, exist_ok=True)

SOURCE_V8 = Path("src/bell/bell_v8_anisotropic_preparation.py")

A_ANGLES = [0.0, 45.0]
B_ANGLES = [22.5, 67.5]


# ------------------------------------------------------------
# Dynamic loading
# ------------------------------------------------------------
def load_module_from_path(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")

    spec = importlib.util.spec_from_file_location("bell_v8_module", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------
# CHSH helpers
# ------------------------------------------------------------
def compute_chsh_from_map(corr_map: dict[tuple[float, float], float]) -> float:
    a, ap = A_ANGLES
    b, bp = B_ANGLES
    return corr_map[(a, b)] + corr_map[(a, bp)] + corr_map[(ap, b)] - corr_map[(ap, bp)]


def corr_map_from_column(df: pd.DataFrame, col: str) -> dict[tuple[float, float], float]:
    out: dict[tuple[float, float], float] = {}
    for a_angle, b_angle in itertools.product(A_ANGLES, B_ANGLES):
        sub = df[(df["a_angle"] == a_angle) & (df["b_angle"] == b_angle)]
        out[(a_angle, b_angle)] = float(np.mean(sub[col]))
    return out


def robust_scale(x: np.ndarray, q: float = 0.95) -> float:
    scale = float(np.quantile(np.abs(x), q))
    return max(scale, 1e-12)


def normalize_tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def normalize_ratio(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.abs(x))


def normalize_clip(x: np.ndarray, scale: float) -> np.ndarray:
    y = x / max(scale, 1e-12)
    return np.clip(y, -1.0, 1.0)


# ------------------------------------------------------------
# Local tracking observable
# ------------------------------------------------------------
def local_disk_mask(size: int, center_xy: np.ndarray, radius: float) -> np.ndarray:
    y, x = np.indices((size, size))
    dx = x - center_xy[0]
    dy = y - center_xy[1]
    return (dx * dx + dy * dy) <= radius * radius


def weighted_centroid_abs(phi: np.ndarray, center_xy: np.ndarray, radius: float) -> tuple[np.ndarray, float]:
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
) -> tuple[int, dict[str, float]]:
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

    delta_t = max(delta_step * dt, 1e-12)
    vel = np.diff(proj) / delta_t if len(proj) >= 2 else np.array([0.0])

    mean_speed = float(np.mean(vel))
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
    }
    return binary_value, aux


# ------------------------------------------------------------
# Case evaluation
# ------------------------------------------------------------
def evaluate_case(v8, case: dict, n_realizations: int) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    required_steps = {
        case["start_step"] + k * case["delta_step"]
        for k in range(v8.WINDOW_LENGTH)
    }

    src_a, src_b = v8.source_positions(case["separation"])
    rows: list[dict[str, float | int | str]] = []

    for seed in range(n_realizations):
        saved_fields = v8.run_prepared_field_history(
            seed=seed,
            separation=case["separation"],
            phase_offset=case["phase_offset"],
            required_steps=required_steps,
            aniso_strength=case["anisotropy"],
            source_orientation_offset=case["src_theta"],
        )

        for a_angle, b_angle in itertools.product(A_ANGLES, B_ANGLES):
            a_sign, aux_a = event_tracking_observable(
                saved_fields=saved_fields,
                center_xy=src_a,
                angle_deg=a_angle,
                start_step=case["start_step"],
                delta_step=case["delta_step"],
                dt=v8.DT,
                window_length=v8.WINDOW_LENGTH,
            )

            b_sign, aux_b = event_tracking_observable(
                saved_fields=saved_fields,
                center_xy=src_b,
                angle_deg=b_angle,
                start_step=case["start_step"],
                delta_step=case["delta_step"],
                dt=v8.DT,
                window_length=v8.WINDOW_LENGTH,
            )

            rows.append(
                {
                    "seed": int(seed),
                    "a_angle": float(a_angle),
                    "b_angle": float(b_angle),
                    "A_sign": int(a_sign),
                    "B_sign": int(b_sign),
                    "sign_product": int(a_sign * b_sign),
                    "A_cont": float(aux_a["continuous_value"]),
                    "B_cont": float(aux_b["continuous_value"]),
                    "cont_product": float(aux_a["continuous_value"] * aux_b["continuous_value"]),
                    **case,
                }
            )

    df = pd.DataFrame(rows)

    raw = df["cont_product"].to_numpy(dtype=float)
    q95 = robust_scale(raw, q=0.95)

    df["cont_tanh"] = normalize_tanh(raw)
    df["cont_ratio"] = normalize_ratio(raw)
    df["cont_clip95"] = normalize_clip(raw, q95)

    sign_map = corr_map_from_column(df, "sign_product")
    raw_map = corr_map_from_column(df, "cont_product")
    tanh_map = corr_map_from_column(df, "cont_tanh")
    ratio_map = corr_map_from_column(df, "cont_ratio")
    clip_map = corr_map_from_column(df, "cont_clip95")

    summary = {
        **case,
        "n_realizations": int(n_realizations),
        "raw_abs_q95": float(q95),
        "raw_mean_abs": float(np.mean(np.abs(raw))),
        "raw_max_abs": float(np.max(np.abs(raw))),
        "S_sign": float(compute_chsh_from_map(sign_map)),
        "S_raw": float(compute_chsh_from_map(raw_map)),
        "S_tanh": float(compute_chsh_from_map(tanh_map)),
        "S_ratio": float(compute_chsh_from_map(ratio_map)),
        "S_clip95": float(compute_chsh_from_map(clip_map)),
        "Craw_0_22.5": raw_map[(0.0, 22.5)],
        "Craw_0_67.5": raw_map[(0.0, 67.5)],
        "Craw_45_22.5": raw_map[(45.0, 22.5)],
        "Craw_45_67.5": raw_map[(45.0, 67.5)],
    }

    return df, summary


# ------------------------------------------------------------
# Targeted case list
# ------------------------------------------------------------
def targeted_cases() -> list[dict[str, float | int | str]]:
    return [
        {
            "label": "baseline_iso_phase0",
            "separation": 20.0,
            "phase_offset": 0.0,
            "start_step": 120,
            "delta_step": 4,
            "anisotropy": 0.0,
            "src_theta": 0.0,
        },
        {
            "label": "moderate_aniso_phase_pi_over_4",
            "separation": 20.0,
            "phase_offset": float(np.pi / 4.0),
            "start_step": 120,
            "delta_step": 4,
            "anisotropy": 0.6,
            "src_theta": float(np.pi / 4.0),
        },
        {
            "label": "strong_aniso_phase_pi_over_4",
            "separation": 20.0,
            "phase_offset": float(np.pi / 4.0),
            "start_step": 120,
            "delta_step": 4,
            "anisotropy": 1.2,
            "src_theta": float(np.pi / 4.0),
        },
        {
            "label": "moderate_aniso_phase_pi",
            "separation": 20.0,
            "phase_offset": float(np.pi),
            "start_step": 120,
            "delta_step": 4,
            "anisotropy": 0.6,
            "src_theta": 0.0,
        },
        {
            "label": "strong_aniso_phase_pi",
            "separation": 20.0,
            "phase_offset": float(np.pi),
            "start_step": 120,
            "delta_step": 4,
            "anisotropy": 1.2,
            "src_theta": 0.0,
        },
        {
            "label": "moderate_aniso_phase_3pi_over_4",
            "separation": 20.0,
            "phase_offset": float(3.0 * np.pi / 4.0),
            "start_step": 120,
            "delta_step": 4,
            "anisotropy": 0.6,
            "src_theta": float(np.pi / 4.0),
        },
    ]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    print("\n=== BELL V10.2 TARGETED EVENT-TRACKING VALIDATION ===")

    v8 = load_module_from_path(SOURCE_V8)
    n_realizations = 64

    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []

    cases = targeted_cases()

    for i, case in enumerate(cases, start=1):
        print(
            f"\n[{i}/{len(cases)}] {case['label']} | "
            f"sep={case['separation']:.1f}, "
            f"phase={case['phase_offset']:.3f}, "
            f"start={case['start_step']}, "
            f"delta={case['delta_step']}, "
            f"aniso={case['anisotropy']:.2f}, "
            f"src_theta={case['src_theta']:.3f}"
        )

        detail_df, summary = evaluate_case(v8, case, n_realizations=n_realizations)
        detail_frames.append(detail_df)
        summary_rows.append(summary)

        print(f"S_sign   = {summary['S_sign']:.6f}")
        print(f"S_raw    = {summary['S_raw']:.6f}")
        print(f"S_tanh   = {summary['S_tanh']:.6f}")
        print(f"S_ratio  = {summary['S_ratio']:.6f}")
        print(f"S_clip95 = {summary['S_clip95']:.6f}")

    detail_all = pd.concat(detail_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    detail_csv = OUT / "bell_v10_2_measurements.csv"
    summary_csv = OUT / "bell_v10_2_summary.csv"

    detail_all.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== SUMMARY ===")
    print(
        summary_df[
            [
                "label",
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

    print(f"\n[OK] wrote {detail_csv}")
    print(f"[OK] wrote {summary_csv}")
    print("[DONE] Bell V10.2 complete")


if __name__ == "__main__":
    main()