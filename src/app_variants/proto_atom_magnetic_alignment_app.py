#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

OUT = Path("results/app_variants/proto_atom_magnetic_alignment")
OUT.mkdir(parents=True, exist_ok=True)

FIG_OUT = Path("figures")
FIG_OUT.mkdir(parents=True, exist_ok=True)


def laplacian(a):
    return (
        np.roll(a, 1, 0) + np.roll(a, -1, 0) +
        np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4 * a
    )


def gradient(a):
    gy, gx = np.gradient(a)
    return gx, gy


def curl_2d(fx, fy):
    dFy_dx = np.gradient(fy, axis=1)
    dFx_dy = np.gradient(fx, axis=0)
    return dFy_dx - dFx_dy


def local_loop(fx, fy):
    return fx - np.roll(fy, -1, 1) - np.roll(fx, -1, 0) + fy


def oriented_seed_field(size, center, orientation_angle):
    y, x = np.indices((size, size))
    cx, cy = center

    dx = x - cx
    dy = y - cy
    r2 = dx**2 + dy**2

    ux = np.cos(orientation_angle)
    uy = np.sin(orientation_angle)
    projection = dx * ux + dy * uy

    return np.exp(-r2 / 120.0) * (1.0 + 0.6 * projection / (np.sqrt(r2) + 1e-6))


def main():
    parser = argparse.ArgumentParser(description="Interactive magnetic alignment test.")
    parser.add_argument("--size", type=int, default=180)
    parser.add_argument("--n_steps", type=int, default=260)
    parser.add_argument("--dt", type=float, default=0.08)

    parser.add_argument("--edge_damp", type=float, default=0.992)
    parser.add_argument("--edge_drive", type=float, default=0.20)
    parser.add_argument("--edge_diff", type=float, default=0.10)
    parser.add_argument("--loop_gain", type=float, default=0.30)

    parser.add_argument("--orientation_angle_deg", type=float, default=45.0)

    args = parser.parse_args()

    size = args.size
    n_steps = args.n_steps
    dt = args.dt

    edge_damp = args.edge_damp
    edge_drive = args.edge_drive
    edge_diff = args.edge_diff
    loop_gain = args.loop_gain

    center = (size // 2, size // 2)
    orientation_angle = np.radians(args.orientation_angle_deg)

    print("\n=== MAGNETIC ALIGNMENT TEST ===")

    phi = oriented_seed_field(size, center, orientation_angle)
    psi = np.zeros_like(phi)

    Fx = np.zeros_like(phi)
    Fy = np.zeros_like(phi)

    alignment_hist = []
    curl_hist = []

    for step in range(n_steps):
        psi = 0.996 * psi + dt * (0.25 * laplacian(phi))
        phi = phi + dt * psi

        gx, gy = gradient(phi)

        loopF = local_loop(Fx, Fy)
        glx, gly = gradient(loopF)

        Lx = -gly
        Ly = glx

        Fx = (
            edge_damp * Fx +
            dt * (edge_drive * gx + edge_diff * laplacian(Fx) + loop_gain * Lx)
        )

        Fy = (
            edge_damp * Fy +
            dt * (edge_drive * gy + edge_diff * laplacian(Fy) + loop_gain * Ly)
        )

        curlF = curl_2d(Fx, Fy)

        normF = np.sqrt(Fx**2 + Fy**2) + 1e-12
        normG = np.sqrt(gx**2 + gy**2) + 1e-12

        mask = (normF > 1e-10) & (normG > 1e-10)
        dot = np.zeros_like(Fx)
        dot[mask] = (Fx[mask] * gx[mask] + Fy[mask] * gy[mask]) / (normF[mask] * normG[mask])
        alignment = float(np.mean(dot[mask])) if np.any(mask) else 0.0

        alignment_hist.append(alignment)
        curl_hist.append(float(np.mean(np.abs(curlF))))

        if step % 40 == 0:
            print(f"step={step} align={alignment:.4f} curl={curl_hist[-1]:.6e}")

    final_alignment = alignment_hist[-1]
    mean_alignment = float(np.mean(alignment_hist))
    final_curl = curl_hist[-1]
    mean_curl = float(np.mean(curl_hist))

    # alignment history
    plt.figure(figsize=(7, 4))
    plt.plot(alignment_hist)
    plt.title("Flux / structure alignment")
    plt.xlabel("step")
    plt.ylabel("alignment")
    plt.tight_layout()
    fig1 = FIG_OUT / "app_magnetic_alignment_history.png"
    plt.savefig(fig1, dpi=220)
    plt.close()

    # curl history
    plt.figure(figsize=(7, 4))
    plt.plot(curl_hist)
    plt.title("Curl intensity")
    plt.xlabel("step")
    plt.ylabel("mean |curl|")
    plt.tight_layout()
    fig2 = FIG_OUT / "app_magnetic_alignment_curl.png"
    plt.savefig(fig2, dpi=220)
    plt.close()

    # final field
    step_q = 6
    yy, xx = np.mgrid[0:size:step_q, 0:size:step_q]

    plt.figure(figsize=(6, 6))
    plt.imshow(phi, cmap="coolwarm", alpha=0.6)
    plt.quiver(
        xx, yy,
        Fx[::step_q, ::step_q],
        Fy[::step_q, ::step_q],
        color="black"
    )
    plt.title("Magnetic alignment field")
    plt.tight_layout()
    fig3 = FIG_OUT / "app_magnetic_alignment_quiver.png"
    plt.savefig(fig3, dpi=220)
    plt.close()

    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    print(f"[OK] wrote {fig3}")
    print("\n=== MAGNETIC ALIGNMENT REPORT ===")
    print(f"orientation_angle_deg={args.orientation_angle_deg:.6f}")
    print(f"final_alignment={final_alignment:.6f}")
    print(f"mean_alignment={mean_alignment:.6f}")
    print(f"final_curl={final_curl:.6e}")
    print(f"mean_curl={mean_curl:.6e}")
    print("[DONE] magnetic alignment test complete")


if __name__ == "__main__":
    main()