#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT = Path("results/reviewer_checks/regime_distinction")
OUT.mkdir(parents=True, exist_ok=True)

FINAL_STATES = Path("results/final_states")


def load_field(path: Path) -> np.ndarray:
    return np.load(path)


def kurtosis(field: np.ndarray) -> float:
    x = field.ravel().astype(float)
    x = x - np.mean(x)
    m2 = np.mean(x ** 2) + 1e-12
    m4 = np.mean(x ** 4)
    return float(m4 / (m2 ** 2))


def participation_ratio(field: np.ndarray) -> float:
    w = np.abs(field.ravel()) ** 2
    s1 = np.sum(w)
    s2 = np.sum(w ** 2) + 1e-12
    return float((s1 ** 2) / s2)


def active_fraction(field: np.ndarray, sigma_factor: float = 1.5) -> float:
    x = field.ravel().astype(float)
    thr = sigma_factor * np.std(x)
    return float(np.mean(np.abs(x) > thr))


def radial_concentration(field: np.ndarray) -> float:
    h, w = field.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices(field.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    weights = np.abs(field).astype(float)
    total = np.sum(weights) + 1e-12
    mean_r = np.sum(r * weights) / total
    max_r = np.max(r) + 1e-12
    return float(mean_r / max_r)


def angular_anisotropy(field: np.ndarray) -> float:
    h, w = field.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices(field.shape)
    theta = np.arctan2(y - cy, x - cx)

    weights = np.abs(field).astype(float)
    cos2 = np.sum(weights * np.cos(2.0 * theta))
    sin2 = np.sum(weights * np.sin(2.0 * theta))
    norm = np.sum(weights) + 1e-12
    return float(np.sqrt(cos2 ** 2 + sin2 ** 2) / norm)


def ring_contrast(field: np.ndarray) -> float:
    """
    Simple proxy for oscillatory radial structure.
    Higher values indicate stronger alternating shell / ring organization.
    """
    h, w = field.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices(field.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    r_int = np.floor(r).astype(int)
    max_bin = int(r_int.max())

    radial_profile = []
    for k in range(max_bin + 1):
        mask = r_int == k
        if np.any(mask):
            radial_profile.append(np.mean(field[mask]))
        else:
            radial_profile.append(0.0)

    radial_profile = np.asarray(radial_profile, dtype=float)

    if len(radial_profile) < 3:
        return 0.0

    second_diff = np.diff(radial_profile, n=2)
    return float(np.mean(np.abs(second_diff)))


def summarize_field(name: str, field: np.ndarray) -> dict:
    return {
        "regime": name,
        "kurtosis": kurtosis(field),
        "participation_ratio": participation_ratio(field),
        "active_fraction": active_fraction(field),
        "radial_concentration": radial_concentration(field),
        "angular_anisotropy": angular_anisotropy(field),
        "ring_contrast": ring_contrast(field),
        "mean_abs": float(np.mean(np.abs(field))),
        "max_abs": float(np.max(np.abs(field))),
    }


def bar_plot(df: pd.DataFrame, col: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(df["regime"], df[col])
    plt.xticks(rotation=15, ha="right")
    plt.ylabel(col.replace("_", " "))
    plt.title(f"Regime comparison: {col.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(OUT / f"{col}_comparison.png", dpi=220)
    plt.close()


def field_panel(fields: dict[str, np.ndarray]) -> None:
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))

    if n == 1:
        axes = [axes]

    for ax, (name, field) in zip(axes, fields.items()):
        im = ax.imshow(field, cmap="coolwarm")
        ax.set_title(name.replace("_", " "))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(OUT / "field_comparison.png", dpi=220)
    plt.close()


def main():
    print("=== Regime Distinction Metrics ===")

    regime_files = {
        "diffusion_like": FINAL_STATES / "diffusion_like.npy",
        "omm_like": FINAL_STATES / "omm_like.npy",
    }

    missing = [str(path) for path in regime_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing final-state files.\n"
            "Please generate them first, for example with:\n"
            "  python src/reviewer_checks/generate_final_states.py\n"
            f"Missing:\n- " + "\n- ".join(missing)
        )

    fields = {}
    rows = []

    for regime, path in regime_files.items():
        field = load_field(path)
        fields[regime] = field
        rows.append(summarize_field(regime, field))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "regime_metrics_summary.csv", index=False)

    for col in [
        "kurtosis",
        "participation_ratio",
        "active_fraction",
        "radial_concentration",
        "angular_anisotropy",
        "ring_contrast",
        "mean_abs",
        "max_abs",
    ]:
        bar_plot(df, col)

    field_panel(fields)

    print(df.to_string(index=False))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()