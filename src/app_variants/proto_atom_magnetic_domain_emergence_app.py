#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("results/app_variants/proto_atom_magnetic_domain_emergence")
OUT.mkdir(parents=True, exist_ok=True)

FIG_OUT = Path("figures")
FIG_OUT.mkdir(parents=True, exist_ok=True)


def laplacian(arr):
    return (
        np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0)
        + np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1)
        - 4.0 * arr
    )


def gradient(arr):
    gy, gx = np.gradient(arr)
    return gx, gy


def divergence(fx, fy):
    return np.gradient(fx, axis=1) + np.gradient(fy, axis=0)


def curl_2d(fx, fy):
    return np.gradient(fy, axis=1) - np.gradient(fx, axis=0)


def local_loop_intensity(fx, fy):
    top = fx
    bottom = np.roll(fx, -1, axis=0)
    left = fy
    right = np.roll(fy, -1, axis=1)
    return top - right - bottom + left


def build_oriented_source(shape, pos, angle_deg, amplitude=1.0):
    angle = np.radians(angle_deg)
    ux = np.cos(angle)
    uy = np.sin(angle)

    y, x = np.indices(shape)
    x0, y0 = pos

    s_plus = np.exp(-((x - (x0 + 3.0 * ux))**2 + (y - (y0 + 3.0 * uy))**2) / 8.0)
    s_minus = np.exp(-((x - (x0 - 3.0 * ux))**2 + (y - (y0 - 3.0 * uy))**2) / 8.0)

    return amplitude * (s_plus - s_minus)


def structure_positions(size, n_structures, seed_spacing):
    cx = size // 2
    cy = size // 2

    offsets = [-1.5, -0.5, 0.5, 1.5]
    positions = []

    for iy in offsets:
        for ix in offsets:
            x = cx + int(ix * seed_spacing)
            y = cy + int(iy * seed_spacing)
            positions.append((x, y))

    return positions[:n_structures]


def structure_angles(mode, n_structures, base_angle_deg):
    if mode == "aligned":
        rng = np.random.default_rng(1234)
        return [base_angle_deg + rng.normal(0, 6) for _ in range(n_structures)]

    if mode == "random":
        rng = np.random.default_rng(5678)
        return list(rng.uniform(0, 180, size=n_structures))

    raise ValueError(f"Unknown mode: {mode}")


def global_flux_alignment(Fx, Fy, angles_deg):
    mean_angle = np.radians(np.mean(angles_deg))
    ux = np.cos(mean_angle)
    uy = np.sin(mean_angle)

    normF = np.sqrt(Fx**2 + Fy**2) + 1e-12
    proj = (Fx * ux + Fy * uy) / normF

    mask = normF > 1e-8
    return float(np.mean(np.abs(proj[mask]))) if np.any(mask) else 0.0


def run_case(
    case_name,
    mode,
    size,
    n_steps,
    dt,
    phi_gain,
    phi_damp,
    edge_damp,
    edge_drive,
    edge_diff,
    loop_gain,
    n_structures,
    seed_spacing,
    source_amplitude,
    omega,
    base_angle_deg,
    snap_steps,
):
    print(f"\n=== CASE: {case_name} ===")

    positions = structure_positions(size, n_structures, seed_spacing)
    angles = structure_angles(mode, n_structures, base_angle_deg)

    phi = np.zeros((size, size), dtype=float)
    psi = np.zeros_like(phi)

    Fx = np.zeros_like(phi)
    Fy = np.zeros_like(phi)

    div_hist = []
    curl_hist = []
    loop_hist = []
    align_hist = []

    snapshots = {}

    for step in range(n_steps):
        source = np.zeros_like(phi)

        for pos, ang in zip(positions, angles):
            source += build_oriented_source(
                phi.shape,
                pos,
                ang,
                amplitude=source_amplitude * np.sin(omega * step)
            )

        psi = phi_damp * psi + dt * (phi_gain * laplacian(phi) + source)
        phi = phi + dt * psi

        gx, gy = gradient(phi)

        loopF = local_loop_intensity(Fx, Fy)
        glx, gly = gradient(loopF)

        Lx = -gly
        Ly = glx

        Fx = (
            edge_damp * Fx
            + dt * (
                edge_drive * gx
                + edge_diff * laplacian(Fx)
                + loop_gain * Lx
            )
        )

        Fy = (
            edge_damp * Fy
            + dt * (
                edge_drive * gy
                + edge_diff * laplacian(Fy)
                + loop_gain * Ly
            )
        )

        divF = divergence(Fx, Fy)
        curlF = curl_2d(Fx, Fy)
        loopF = local_loop_intensity(Fx, Fy)

        div_mean = float(np.mean(np.abs(divF)))
        curl_mean = float(np.mean(np.abs(curlF)))
        loop_mean = float(np.mean(np.abs(loopF)))
        align = global_flux_alignment(Fx, Fy, angles)

        div_hist.append(div_mean)
        curl_hist.append(curl_mean)
        loop_hist.append(loop_mean)
        align_hist.append(align)

        if step in snap_steps:
            snapshots[step] = {
                "phi": phi.copy(),
                "curlF": curlF.copy(),
                "loopF": loopF.copy(),
                "Fx": Fx.copy(),
                "Fy": Fy.copy(),
            }

        if step % 40 == 0:
            print(
                f"step={step} div={div_mean:.6e} curl={curl_mean:.6e} "
                f"loop={loop_mean:.6e} align={align:.4f}"
            )

    summary = {
        "case": case_name,
        "mode": mode,
        "mean_div": float(np.mean(div_hist)),
        "max_div": float(np.max(div_hist)),
        "mean_curl": float(np.mean(curl_hist)),
        "max_curl": float(np.max(curl_hist)),
        "mean_loop": float(np.mean(loop_hist)),
        "max_loop": float(np.max(loop_hist)),
        "mean_alignment": float(np.mean(align_hist)),
        "final_alignment": float(align_hist[-1]),
        "final_div": float(div_hist[-1]),
        "final_curl": float(curl_hist[-1]),
        "final_loop": float(loop_hist[-1]),
        "snapshots": snapshots,
        "positions": positions,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Interactive magnetic domain emergence.")
    parser.add_argument("--size", type=int, default=220)
    parser.add_argument("--n_steps", type=int, default=320)
    parser.add_argument("--dt", type=float, default=0.08)

    parser.add_argument("--phi_gain", type=float, default=0.24)
    parser.add_argument("--phi_damp", type=float, default=0.996)

    parser.add_argument("--edge_damp", type=float, default=0.992)
    parser.add_argument("--edge_drive", type=float, default=0.22)
    parser.add_argument("--edge_diff", type=float, default=0.10)
    parser.add_argument("--loop_gain", type=float, default=0.32)

    parser.add_argument("--n_structures", type=int, default=16)
    parser.add_argument("--seed_spacing", type=int, default=28)
    parser.add_argument("--source_amplitude", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=0.12)
    parser.add_argument("--base_angle_deg", type=float, default=35.0)

    args = parser.parse_args()

    snap_steps = [0, 40, 80, 140, 220, args.n_steps - 1]

    cases = {
        "aligned_domain": "aligned",
        "random_domain": "random",
    }

    summaries = []
    results = {}

    print("\n=== PROTO ATOM MAGNETIC DOMAIN EMERGENCE ===")

    for case_name, mode in cases.items():
        result = run_case(
            case_name=case_name,
            mode=mode,
            size=args.size,
            n_steps=args.n_steps,
            dt=args.dt,
            phi_gain=args.phi_gain,
            phi_damp=args.phi_damp,
            edge_damp=args.edge_damp,
            edge_drive=args.edge_drive,
            edge_diff=args.edge_diff,
            loop_gain=args.loop_gain,
            n_structures=args.n_structures,
            seed_spacing=args.seed_spacing,
            source_amplitude=args.source_amplitude,
            omega=args.omega,
            base_angle_deg=args.base_angle_deg,
            snap_steps=snap_steps,
        )
        results[case_name] = result

        summary = {k: v for k, v in result.items() if k not in ["snapshots", "positions"]}
        summaries.append(summary)

    df = pd.DataFrame(summaries)
    out_csv = OUT / "magnetic_domain_summary.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== DOMAIN SUMMARY ===")
    print(df.to_string(index=False))

    # comparison figure
    plt.figure(figsize=(8, 5))
    x = np.arange(len(df))
    plt.bar(x - 0.2, df["final_curl"], width=0.2, label="final_curl")
    plt.bar(x, df["final_loop"], width=0.2, label="final_loop")
    plt.bar(x + 0.2, df["final_alignment"], width=0.2, label="final_alignment")
    plt.xticks(x, df["case"])
    plt.title("Magnetic domain comparison")
    plt.legend()
    plt.tight_layout()
    fig1 = FIG_OUT / "app_magnetic_domain_comparison.png"
    plt.savefig(fig1, dpi=220)
    plt.close()

    # aligned curl snapshots
    aligned = results["aligned_domain"]
    keys = sorted(aligned["snapshots"].keys())

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for ax, step in zip(axes, keys):
        ax.imshow(aligned["snapshots"][step]["curlF"], cmap="coolwarm")
        ax.set_title(f"aligned curl step={step}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig2 = FIG_OUT / "app_magnetic_domain_aligned_curl.png"
    plt.savefig(fig2, dpi=220)
    plt.close(fig)

    # aligned quiver final
    Fx_final = aligned["snapshots"][keys[-1]]["Fx"]
    Fy_final = aligned["snapshots"][keys[-1]]["Fy"]
    phi_final = aligned["snapshots"][keys[-1]]["phi"]

    step_q = 6
    yy, xx = np.mgrid[0:args.size:step_q, 0:args.size:step_q]

    plt.figure(figsize=(7, 7))
    plt.imshow(phi_final, cmap="coolwarm", alpha=0.65)
    plt.quiver(
        xx, yy,
        Fx_final[::step_q, ::step_q],
        Fy_final[::step_q, ::step_q],
        color="black",
        pivot="mid",
        scale=30,
    )
    for (x0, y0) in aligned["positions"]:
        plt.scatter([x0], [y0], s=20)
    plt.title("Aligned domain final flux")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig3 = FIG_OUT / "app_magnetic_domain_aligned_quiver.png"
    plt.savefig(fig3, dpi=220)
    plt.close()

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    print(f"[OK] wrote {fig3}")
    print("\n=== MAGNETIC DOMAIN REPORT ===")
    aligned_row = df[df["case"] == "aligned_domain"].iloc[0]
    random_row = df[df["case"] == "random_domain"].iloc[0]
    print(f"aligned_final_alignment={aligned_row['final_alignment']:.6f}")
    print(f"aligned_final_curl={aligned_row['final_curl']:.6e}")
    print(f"aligned_final_loop={aligned_row['final_loop']:.6e}")
    print(f"random_final_alignment={random_row['final_alignment']:.6f}")
    print(f"random_final_curl={random_row['final_curl']:.6e}")
    print(f"random_final_loop={random_row['final_loop']:.6e}")
    print("[DONE] proto atom magnetic domain emergence complete")


if __name__ == "__main__":
    main()