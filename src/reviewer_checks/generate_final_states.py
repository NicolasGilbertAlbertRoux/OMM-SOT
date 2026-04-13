#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np

OUT = Path("results/final_states")
OUT.mkdir(parents=True, exist_ok=True)


def save_field(name, field):
    path = OUT / f"{name}.npy"
    np.save(path, field)
    print(f"[OK] saved {path}")


def generate_diffusion(size=128, steps=200, dt=0.1):
    field = np.zeros((size, size))
    field[size//2, size//2] = 1.0

    for _ in range(steps):
        field = field + dt * (
            np.roll(field,1,0) + np.roll(field,-1,0) +
            np.roll(field,1,1) + np.roll(field,-1,1) - 4*field
        )

    return field


def generate_omm_like(size=128, steps=200, dt=0.05):
    field = 0.01 * np.random.randn(size, size)

    for _ in range(steps):
        lap = (
            np.roll(field,1,0) + np.roll(field,-1,0) +
            np.roll(field,1,1) + np.roll(field,-1,1) - 4*field
        )

        nonlinear = field - field**3

        field = field + dt * (lap + nonlinear)

    return field


def main():
    print("=== Generating final states ===")

    diff = generate_diffusion()
    omm = generate_omm_like()

    save_field("diffusion_like", diff)
    save_field("omm_like", omm)


if __name__ == "__main__":
    main()