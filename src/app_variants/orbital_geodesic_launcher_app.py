#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter

# ============================================================
# PARAMETERS (CLI)
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--size", type=int, default=200)
parser.add_argument("--n_steps", type=int, default=400)
parser.add_argument("--dt", type=float, default=0.1)

parser.add_argument("--x0", type=float, default=140.0)
parser.add_argument("--y0", type=float, default=100.0)
parser.add_argument("--vx0", type=float, default=0.0)
parser.add_argument("--vy0", type=float, default=0.6)

parser.add_argument("--source_sigma", type=float, default=6.0)

parser.add_argument("--epsilon_global", type=float, default=0.02)
parser.add_argument("--global_sigma", type=float, default=8.0)
parser.add_argument("--global_eta", type=float, default=0.02)
parser.add_argument("--global_mass", type=float, default=0.1)

args = parser.parse_args()

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ============================================================
# GRID
# ============================================================

N = args.size
CENTER = np.array([N / 2, N / 2])

def laplacian(phi):
    return (
        np.roll(phi, 1, 0)
        + np.roll(phi, -1, 0)
        + np.roll(phi, 1, 1)
        + np.roll(phi, -1, 1)
        - 4 * phi
    )

def gradient(arr):
    gy, gx = np.gradient(arr)
    return gx, gy

def gaussian(pos, sigma):
    y, x = np.indices((N, N))
    r2 = (x - pos[0])**2 + (y - pos[1])**2
    return np.exp(-r2 / (2 * sigma**2))

def solve_poisson(source, iters=60, mass=0.0):
    pot = np.zeros_like(source)
    for _ in range(iters):
        pot = (
            np.roll(pot, 1, 0)
            + np.roll(pot, -1, 0)
            + np.roll(pot, 1, 1)
            + np.roll(pot, -1, 1)
            - source
        ) / (4 + mass**2)
    return pot

# ============================================================
# FIELD
# ============================================================

phi = np.zeros((N, N))
pi = np.zeros_like(phi)

src = gaussian(CENTER, args.source_sigma)

psi_global_mem = np.zeros_like(phi)

# ============================================================
# TRAJECTORY
# ============================================================

pos = np.array([args.x0, args.y0], dtype=float)
vel = np.array([args.vx0, args.vy0], dtype=float)

traj = []

# ============================================================
# LOOP
# ============================================================

for step in range(args.n_steps):

    # field evolve
    pi += args.dt * (0.75**2 * laplacian(phi) + src - 0.001 * pi)
    phi += args.dt * pi

    gx, gy = gradient(phi)
    rho = 0.5 * (gx**2 + gy**2) + 0.5 * (pi**2)

    # geometry
    psi_local = solve_poisson(rho, mass=0.08)

    rho_g = gaussian_filter(rho, sigma=args.global_sigma)
    psi_g = solve_poisson(rho_g, mass=args.global_mass)

    psi_global_mem = (
        (1 - args.global_eta) * psi_global_mem
        + args.global_eta * psi_g
    )

    psi_total = psi_local + args.epsilon_global * psi_global_mem

    # force
    fx, fy = gradient(psi_total)

    ix = int(np.clip(pos[0], 0, N - 1))
    iy = int(np.clip(pos[1], 0, N - 1))

    acc = -np.array([fx[iy, ix], fy[iy, ix]])

    # integrate
    vel += args.dt * acc
    pos += args.dt * vel

    traj.append(pos.copy())

traj = np.array(traj)

# ============================================================
# OUTPUT
# ============================================================

plt.figure(figsize=(6,6))
plt.imshow(psi_total, cmap="coolwarm")
plt.plot(traj[:,0], traj[:,1], 'k-')
plt.scatter([CENTER[0]], [CENTER[1]], c='white')
plt.title("Emergent geometry + trajectory")
plt.xticks([])
plt.yticks([])

path = OUT / "app_orbital_launcher.png"
plt.savefig(path, dpi=200)
plt.close()

print(f"final_x={pos[0]:.4f}")
print(f"final_y={pos[1]:.4f}")
print(f"final_speed={np.linalg.norm(vel):.4f}")