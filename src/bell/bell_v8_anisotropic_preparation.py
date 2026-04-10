#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path("results/bell/v8_anisotropic_preparation")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# Domain / dynamics
# ============================================================
SIZE = 180
CENTER = np.array([SIZE // 2, SIZE // 2], dtype=float)

DT = 0.08
N_STEPS = 260

WAVE_GAIN = 0.18
DAMP = 0.999
SOURCE_DURATION = 24
SIGMA = 6.0
OMEGA = 0.22

N_REALIZATIONS = 120

# CHSH angles
A_ANGLES = [0.0, 45.0]
B_ANGLES = [22.5, 67.5]

# Scan grid
SEPARATIONS = [20.0, 28.0, 36.0]
PHASE_OFFSETS = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
START_STEPS = [80, 120]
DELTA_STEPS = [2, 4, 8]
WINDOW_LENGTH = 5

# Imposed anisotropy
ANISO_STRENGTHS = [0.0, 0.6, 1.2]
SOURCE_ORIENTATION_OFFSETS = [0.0, np.pi / 4, np.pi / 2]

# Local readout
PATCH_RADIUS = 3.0
PROBE_OFFSET = 6.5

# Weights
W_ALIGN = 1.0
W_CROSS = 1.0
W_EVENT = 0.8
W_GLOBAL = 0.35


# ============================================================
# Utilities
# ============================================================
def evaluate_case(
    separation: float,
    phase_offset: float,
    start_step: int,
    delta_step: int,
    anisotropy: float,
    src_theta: float,
    n_realizations_override: int | None = None,
    window_length_override: int | None = None,
) -> dict[str, float]:
    """
    Evaluate a single V8 case.
    Returns a minimal dict compatible with lightweight screening scripts.
    """
    n_realizations = N_REALIZATIONS if n_realizations_override is None else int(n_realizations_override)
    window_length = WINDOW_LENGTH if window_length_override is None else int(window_length_override)

    required_steps = {start_step + k * delta_step for k in range(window_length)}
    local_rows: list[dict[str, float | int]] = []

    for seed in range(n_realizations):
        saved_fields = run_prepared_field_history(
            seed=seed,
            separation=separation,
            phase_offset=phase_offset,
            required_steps=required_steps,
            aniso_strength=anisotropy,
            source_orientation_offset=src_theta,
        )

        src_a, src_b = source_positions(separation)

        for a_angle, b_angle in itertools.product(A_ANGLES, B_ANGLES):
            a_bin, aux_a = anisotropic_binary_observable_custom_window(
                saved_fields=saved_fields,
                center_xy=src_a,
                angle_deg=a_angle,
                start_step=start_step,
                delta_step=delta_step,
                window_length=window_length,
            )
            b_bin, aux_b = anisotropic_binary_observable_custom_window(
                saved_fields=saved_fields,
                center_xy=src_b,
                angle_deg=b_angle,
                start_step=start_step,
                delta_step=delta_step,
                window_length=window_length,
            )

            local_rows.append(
                {
                    "a_angle": float(a_angle),
                    "b_angle": float(b_angle),
                    "A": int(a_bin),
                    "B": int(b_bin),
                    "AB": int(a_bin * b_bin),
                    "A_value": float(aux_a["continuous_value"]),
                    "B_value": float(aux_b["continuous_value"]),
                    "A_event_term": float(aux_a["event_term"]),
                    "B_event_term": float(aux_b["event_term"]),
                }
            )

    local_df = pd.DataFrame(local_rows)

    corr_map: dict[tuple[float, float], float] = {}
    for a_angle, b_angle in itertools.product(A_ANGLES, B_ANGLES):
        sub = local_df[(local_df["a_angle"] == a_angle) & (local_df["b_angle"] == b_angle)]
        corr_map[(a_angle, b_angle)] = float(np.mean(sub["AB"]))

    s_value = (
        corr_map[(0.0, 22.5)]
        + corr_map[(0.0, 67.5)]
        + corr_map[(45.0, 22.5)]
        - corr_map[(45.0, 67.5)]
    )

    return {
        "E_0_22.5": corr_map[(0.0, 22.5)],
        "E_0_67.5": corr_map[(0.0, 67.5)],
        "E_45_22.5": corr_map[(45.0, 22.5)],
        "E_45_67.5": corr_map[(45.0, 67.5)],
        "S": float(s_value),
        "mean_abs_A_value": float(np.mean(np.abs(local_df["A_value"]))),
        "mean_abs_B_value": float(np.mean(np.abs(local_df["B_value"]))),
        "mean_abs_A_event": float(np.mean(np.abs(local_df["A_event_term"]))),
        "mean_abs_B_event": float(np.mean(np.abs(local_df["B_event_term"]))),
        "n_realizations_used": float(n_realizations),
        "window_length_used": float(window_length),
    }


def laplacian(phi: np.ndarray) -> np.ndarray:
    return (
        np.roll(phi, 1, axis=0)
        + np.roll(phi, -1, axis=0)
        + np.roll(phi, 1, axis=1)
        + np.roll(phi, -1, axis=1)
        - 4.0 * phi
    )


def source_positions(separation: float) -> tuple[np.ndarray, np.ndarray]:
    offset = np.array([separation / 2.0, 0.0], dtype=float)
    src_a = CENTER - offset
    src_b = CENTER + offset
    return src_a, src_b


def local_disk_mask(center_xy: np.ndarray, radius: float) -> np.ndarray:
    y, x = np.indices((SIZE, SIZE))
    dx = x - center_xy[0]
    dy = y - center_xy[1]
    return (dx * dx + dy * dy) <= radius * radius


def local_mean(phi: np.ndarray, center_xy: np.ndarray, radius: float) -> float:
    mask = local_disk_mask(center_xy, radius)
    return float(np.mean(phi[mask]))


def global_phase_and_flux(phi: np.ndarray) -> tuple[float, float]:
    gy, gx = np.gradient(phi)
    gx_mean = float(np.mean(gx))
    gy_mean = float(np.mean(gy))
    phase_global = float(np.arctan2(gy_mean, gx_mean))
    flux_global = float(np.sqrt(gx_mean * gx_mean + gy_mean * gy_mean))
    return phase_global, flux_global


def anisotropic_source(
    center_xy: np.ndarray,
    amp: float,
    sigma: float,
    theta: float,
    anisotropy: float,
) -> np.ndarray:
    """
    Oriented elliptical source.
    anisotropy = 0 -> nearly isotropic
    anisotropy > 0 -> stronger long/short axis contrast
    """
    y, x = np.indices((SIZE, SIZE))
    dx = x - center_xy[0]
    dy = y - center_xy[1]

    ct = np.cos(theta)
    st = np.sin(theta)

    xp = ct * dx + st * dy
    yp = -st * dx + ct * dy

    sigma_long = sigma * (1.0 + 0.55 * anisotropy)
    sigma_short = sigma / (1.0 + 0.55 * anisotropy)

    expo = (xp**2) / (2.0 * sigma_long**2) + (yp**2) / (2.0 * sigma_short**2)
    return amp * np.exp(-expo)


def run_prepared_field_history(
    seed: int,
    separation: float,
    phase_offset: float,
    required_steps: set[int],
    aniso_strength: float,
    source_orientation_offset: float,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)

    phi = np.zeros((SIZE, SIZE), dtype=float)
    psi = np.zeros_like(phi)

    src_a, src_b = source_positions(separation)

    phi += 1e-4 * rng.normal(size=(SIZE, SIZE))
    global_phase = rng.uniform(0.0, 2.0 * np.pi)

    theta_a = source_orientation_offset
    theta_b = source_orientation_offset + np.pi / 2.0

    saved: dict[int, np.ndarray] = {}

    for step in range(N_STEPS):
        source = np.zeros_like(phi)

        if step < SOURCE_DURATION:
            amp_a = np.sin(OMEGA * step + global_phase)
            amp_b = np.sin(OMEGA * step + global_phase + phase_offset)

            source += anisotropic_source(src_a, amp_a, SIGMA, theta_a, aniso_strength)
            source += anisotropic_source(src_b, amp_b, SIGMA, theta_b, aniso_strength)

        lap = laplacian(phi)
        psi = DAMP * psi + DT * (WAVE_GAIN * lap + source)
        phi = phi + DT * psi

        if step in required_steps:
            saved[step] = phi.copy()

    return saved


# ============================================================
# Oriented readout
# ============================================================
def oriented_probe_sets(center_xy: np.ndarray, angle_deg: float) -> dict[str, list[np.ndarray]]:
    theta = np.radians(angle_deg)
    n = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    t = np.array([-np.sin(theta), np.cos(theta)], dtype=float)

    align_plus = center_xy + PROBE_OFFSET * n
    align_minus = center_xy - PROBE_OFFSET * n

    cross_plus = center_xy + PROBE_OFFSET * t
    cross_minus = center_xy - PROBE_OFFSET * t

    d = PROBE_OFFSET / np.sqrt(2.0)
    diag_ap = center_xy + d * (n + t)
    diag_am = center_xy + d * (n - t)
    diag_cp = center_xy + d * (-n + t)
    diag_cm = center_xy + d * (-n - t)

    return {
        "align": [align_plus, align_minus, diag_ap, diag_am],
        "cross": [cross_plus, cross_minus, diag_cp, diag_cm],
    }


def channel_score_over_time(
    saved_fields: dict[int, np.ndarray],
    center_xy: np.ndarray,
    angle_deg: float,
    start_step: int,
    delta_step: int,
) -> tuple[float, float, dict[str, float]]:
    return channel_score_over_time_custom_window(
        saved_fields=saved_fields,
        center_xy=center_xy,
        angle_deg=angle_deg,
        start_step=start_step,
        delta_step=delta_step,
        window_length=WINDOW_LENGTH,
    )


def channel_score_over_time_custom_window(
    saved_fields: dict[int, np.ndarray],
    center_xy: np.ndarray,
    angle_deg: float,
    start_step: int,
    delta_step: int,
    window_length: int,
) -> tuple[float, float, dict[str, float]]:
    steps = [start_step + k * delta_step for k in range(window_length)]
    probes = oriented_probe_sets(center_xy, angle_deg)

    align_series = []
    cross_series = []

    for step in steps:
        phi = saved_fields[step]

        align_vals = [local_mean(phi, p, PATCH_RADIUS) for p in probes["align"]]
        cross_vals = [local_mean(phi, p, PATCH_RADIUS) for p in probes["cross"]]

        align_score = 0.45 * (align_vals[0] - align_vals[1]) + 0.275 * (align_vals[2] - align_vals[3])
        cross_score = 0.45 * (cross_vals[0] - cross_vals[1]) + 0.275 * (cross_vals[2] - cross_vals[3])

        align_series.append(align_score)
        cross_series.append(cross_score)

    align_series = np.array(align_series, dtype=float)
    cross_series = np.array(cross_series, dtype=float)

    diff_series = align_series - cross_series
    sign_switches = np.sum(np.abs(np.diff(np.sign(diff_series + 1e-12))))
    event_term = float(np.mean(np.abs(np.diff(diff_series)))) + 0.25 * sign_switches

    phi_ref = saved_fields[steps[-1]]
    phase_global, flux_global = global_phase_and_flux(phi_ref)
    theta = np.radians(angle_deg)
    global_term = np.cos(phase_global - theta) * np.tanh(8.0 * flux_global)

    align_total = W_ALIGN * float(np.mean(align_series)) + W_EVENT * event_term + W_GLOBAL * global_term
    cross_total = W_CROSS * float(np.mean(cross_series)) - W_EVENT * event_term - W_GLOBAL * global_term

    aux = {
        "mean_align": float(np.mean(align_series)),
        "mean_cross": float(np.mean(cross_series)),
        "event_term": float(event_term),
        "global_term": float(global_term),
        "align_total": float(align_total),
        "cross_total": float(cross_total),
    }
    return align_total, cross_total, aux


def anisotropic_binary_observable(
    saved_fields: dict[int, np.ndarray],
    center_xy: np.ndarray,
    angle_deg: float,
    start_step: int,
    delta_step: int,
) -> tuple[int, dict[str, float]]:
    return anisotropic_binary_observable_custom_window(
        saved_fields=saved_fields,
        center_xy=center_xy,
        angle_deg=angle_deg,
        start_step=start_step,
        delta_step=delta_step,
        window_length=WINDOW_LENGTH,
    )


def anisotropic_binary_observable_custom_window(
    saved_fields: dict[int, np.ndarray],
    center_xy: np.ndarray,
    angle_deg: float,
    start_step: int,
    delta_step: int,
    window_length: int,
) -> tuple[int, dict[str, float]]:
    align_total, cross_total, aux = channel_score_over_time_custom_window(
        saved_fields=saved_fields,
        center_xy=center_xy,
        angle_deg=angle_deg,
        start_step=start_step,
        delta_step=delta_step,
        window_length=window_length,
    )

    value = align_total - cross_total
    binary = 1 if value >= 0.0 else -1
    aux["continuous_value"] = float(value)
    return binary, aux


# ============================================================
# CHSH
# ============================================================
def compute_chsh(df: pd.DataFrame) -> float:
    corr_map: dict[tuple[float, float], float] = {}
    for a_angle, b_angle in itertools.product(A_ANGLES, B_ANGLES):
        sub = df[(df["a_angle"] == a_angle) & (df["b_angle"] == b_angle)]
        corr_map[(a_angle, b_angle)] = float(np.mean(sub["AB"]))

    a, ap = A_ANGLES
    b, bp = B_ANGLES
    return corr_map[(a, b)] + corr_map[(a, bp)] + corr_map[(ap, b)] - corr_map[(ap, bp)]


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("\n=== BELL V8 ANISOTROPIC PREPARATION ===")

    required_steps: set[int] = set()
    for start_step, delta_step in itertools.product(START_STEPS, DELTA_STEPS):
        for k in range(WINDOW_LENGTH):
            required_steps.add(start_step + k * delta_step)

    all_rows: list[dict[str, float | int]] = []
    summary_rows: list[dict[str, float | int]] = []
    example_fields: list[np.ndarray] = []

    cases = list(
        itertools.product(
            SEPARATIONS,
            PHASE_OFFSETS,
            START_STEPS,
            DELTA_STEPS,
            ANISO_STRENGTHS,
            SOURCE_ORIENTATION_OFFSETS,
        )
    )

    total = len(cases)

    for i, (sep, phase_offset, start_step, delta_step, aniso_strength, source_orientation_offset) in enumerate(cases, start=1):
        print(
            f"\n[{i}/{total}] sep={sep:.1f}, phase_offset={phase_offset:.3f}, "
            f"start={start_step}, delta={delta_step}, aniso={aniso_strength:.2f}, "
            f"src_theta={source_orientation_offset:.3f}"
        )

        local_rows: list[dict[str, float | int]] = []

        for seed in range(N_REALIZATIONS):
            saved_fields = run_prepared_field_history(
                seed=seed,
                separation=sep,
                phase_offset=phase_offset,
                required_steps=required_steps,
                aniso_strength=aniso_strength,
                source_orientation_offset=source_orientation_offset,
            )

            src_a, src_b = source_positions(sep)

            if seed < 4 and i == 1 and start_step in saved_fields:
                example_fields.append(saved_fields[start_step].copy())

            for a_angle, b_angle in itertools.product(A_ANGLES, B_ANGLES):
                a_bin, aux_a = anisotropic_binary_observable(saved_fields, src_a, a_angle, start_step, delta_step)
                b_bin, aux_b = anisotropic_binary_observable(saved_fields, src_b, b_angle, start_step, delta_step)

                row = {
                    "separation": float(sep),
                    "phase_offset": float(phase_offset),
                    "start_step": int(start_step),
                    "delta_step": int(delta_step),
                    "aniso_strength": float(aniso_strength),
                    "source_orientation_offset": float(source_orientation_offset),
                    "seed": int(seed),
                    "a_angle": float(a_angle),
                    "b_angle": float(b_angle),
                    "A": int(a_bin),
                    "B": int(b_bin),
                    "AB": int(a_bin * b_bin),
                    "A_value": float(aux_a["continuous_value"]),
                    "B_value": float(aux_b["continuous_value"]),
                    "A_mean_align": float(aux_a["mean_align"]),
                    "A_mean_cross": float(aux_a["mean_cross"]),
                    "B_mean_align": float(aux_b["mean_align"]),
                    "B_mean_cross": float(aux_b["mean_cross"]),
                    "A_event_term": float(aux_a["event_term"]),
                    "B_event_term": float(aux_b["event_term"]),
                    "A_global_term": float(aux_a["global_term"]),
                    "B_global_term": float(aux_b["global_term"]),
                }
                local_rows.append(row)
                all_rows.append(row)

        local_df = pd.DataFrame(local_rows)

        corr_map: dict[tuple[float, float], float] = {}
        for a_angle, b_angle in itertools.product(A_ANGLES, B_ANGLES):
            sub = local_df[(local_df["a_angle"] == a_angle) & (local_df["b_angle"] == b_angle)]
            e_ab = float(np.mean(sub["AB"]))
            corr_map[(a_angle, b_angle)] = e_ab
            print(f"E({a_angle:4.1f}, {b_angle:4.1f}) = {e_ab:.6f}")

        s_value = compute_chsh(local_df)
        print(f"CHSH-like S = {s_value:.6f}")

        summary_rows.append(
            {
                "separation": float(sep),
                "phase_offset": float(phase_offset),
                "start_step": int(start_step),
                "delta_step": int(delta_step),
                "aniso_strength": float(aniso_strength),
                "source_orientation_offset": float(source_orientation_offset),
                "E_0_22.5": corr_map[(0.0, 22.5)],
                "E_0_67.5": corr_map[(0.0, 67.5)],
                "E_45_22.5": corr_map[(45.0, 22.5)],
                "E_45_67.5": corr_map[(45.0, 67.5)],
                "S": float(s_value),
                "mean_abs_A_value": float(np.mean(np.abs(local_df["A_value"]))),
                "mean_abs_B_value": float(np.mean(np.abs(local_df["B_value"]))),
                "mean_abs_A_event": float(np.mean(np.abs(local_df["A_event_term"]))),
                "mean_abs_B_event": float(np.mean(np.abs(local_df["B_event_term"]))),
            }
        )

    all_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)

    all_df.to_csv(OUT / "bell_v8_measurements.csv", index=False)
    summary_df.to_csv(OUT / "bell_v8_summary.csv", index=False)

    print("\n=== BEST CASES BY |S| ===")
    best = summary_df.iloc[np.argsort(np.abs(summary_df["S"]))[::-1]].head(20)
    print(best.to_string(index=False))

    subset = summary_df[
        (summary_df["aniso_strength"] == ANISO_STRENGTHS[-1])
        & (summary_df["source_orientation_offset"] == SOURCE_ORIENTATION_OFFSETS[1])
        & (summary_df["start_step"] == START_STEPS[0])
        & (summary_df["delta_step"] == DELTA_STEPS[0])
    ].copy()

    if len(subset) > 0:
        pivot = subset.pivot(index="separation", columns="phase_offset", values="S")
        plt.figure(figsize=(8, 4))
        plt.imshow(pivot.values, aspect="auto", origin="lower")
        plt.xticks(range(len(pivot.columns)), [f"{c:.2f}" for c in pivot.columns])
        plt.yticks(range(len(pivot.index)), [f"{v:.1f}" for v in pivot.index])
        plt.xlabel("phase_offset")
        plt.ylabel("separation")
        plt.title("Bell V8: S scan (anisotropic regime)")
        plt.colorbar(label="S")
        plt.tight_layout()
        plt.savefig(OUT / "bell_v8_S_heatmap.png", dpi=260)
        plt.close()

    best_row = best.iloc[0]
    best_sub = all_df[
        (all_df["separation"] == best_row["separation"])
        & (all_df["phase_offset"] == best_row["phase_offset"])
        & (all_df["start_step"] == best_row["start_step"])
        & (all_df["delta_step"] == best_row["delta_step"])
        & (all_df["aniso_strength"] == best_row["aniso_strength"])
        & (all_df["source_orientation_offset"] == best_row["source_orientation_offset"])
    ].copy()

    plt.figure(figsize=(6, 4))
    plt.hist(best_sub["AB"], bins=[-1.5, -0.5, 0.5, 1.5], rwidth=0.8)
    plt.xticks([-1, 1])
    plt.xlabel("A * B")
    plt.ylabel("count")
    plt.title("Bell V8: AB histogram (best case)")
    plt.tight_layout()
    plt.savefig(OUT / "bell_v8_best_histogram.png", dpi=260)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(best_sub["A_value"], bins=30, alpha=0.6, label="A_value")
    plt.hist(best_sub["B_value"], bins=30, alpha=0.6, label="B_value")
    plt.xlabel("continuous anisotropic observable")
    plt.ylabel("count")
    plt.title("Bell V8: anisotropic observable values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "bell_v8_continuous_values.png", dpi=260)
    plt.close()

    if len(example_fields) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        axes = axes.ravel()
        for ax, phi in zip(axes, example_fields[:4]):
            ax.imshow(phi, cmap="coolwarm")
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(OUT / "bell_v8_example_fields.png", dpi=260)
        plt.close(fig)

    print(f"\n[OK] wrote {OUT / 'bell_v8_measurements.csv'}")
    print(f"[OK] wrote {OUT / 'bell_v8_summary.csv'}")
    print("[DONE] Bell V8 complete")


if __name__ == "__main__":
    main()