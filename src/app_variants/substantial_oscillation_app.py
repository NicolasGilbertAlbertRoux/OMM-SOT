#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Interactive substantial oscillation / single-pair Dirac beat explorer."
)
parser.add_argument(
    "--repo_root",
    type=str,
    default="../unified-emergent-framework",
    help="Path to the Research 1 repository root.",
)
parser.add_argument("--beta_index", type=int, default=0)
parser.add_argument("--seed_index", type=int, default=0)
parser.add_argument("--pair_index", type=int, default=1)
parser.add_argument("--t_max", type=float, default=200.0)
parser.add_argument("--n_steps", type=int, default=4000)
parser.add_argument("--max_pairs", type=int, default=8)
args = parser.parse_args()

# ------------------------------------------------------------
# Resolve Research 1 paths robustly
# ------------------------------------------------------------

REPO_ROOT = Path(args.repo_root).resolve()
CANDIDATES = [
    REPO_ROOT,
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
]

for p in CANDIDATES:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ------------------------------------------------------------
# Imports from Research 1
# ------------------------------------------------------------

try:
    from core.io import load_nodes_edges
    from core.graph import build_incidence
    from core.operators import build_dirac_like
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import Research 1 modules. "
        f"Checked repo_root={REPO_ROOT}. "
        "Make sure the path points to the unified-emergent-framework repository "
        "and that the expected core/ package is available there."
    ) from exc

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

OUT = Path("results/app_variants/substantial_oscillation")
OUT.mkdir(parents=True, exist_ok=True)

FIG_OUT = Path("figures")
FIG_OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_config(repo_root: Path, rel_path: str = "configs/default.yaml") -> dict:
    path = repo_root / rel_path
    if not path.exists():
        raise FileNotFoundError(
            f"Missing config file: {path}. "
            "Check that --repo_root points to the root of unified-emergent-framework."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def pair_modes(evals: np.ndarray, zero_tol: float = 1e-8, max_pairs: int = 12):
    pos_idx = np.where(evals > zero_tol)[0]
    neg_idx = np.where(evals < -zero_tol)[0]

    pos_sorted = pos_idx[np.argsort(evals[pos_idx])]
    neg_sorted = neg_idx[np.argsort(np.abs(evals[neg_idx]))]

    used_neg = set()
    pairs = []

    for ip in pos_sorted:
        target = abs(evals[ip])
        best_j = None
        best_err = None

        for jn in neg_sorted:
            if jn in used_neg:
                continue
            err = abs(abs(evals[jn]) - target)
            if best_err is None or err < best_err:
                best_err = err
                best_j = jn

        if best_j is not None:
            used_neg.add(best_j)
            pairs.append((best_j, ip, float(abs(evals[ip])), float(best_err)))

        if len(pairs) >= max_pairs:
            break

    return pairs


def evolve_pair_with_overlap(
    evals: np.ndarray,
    evecs: np.ndarray,
    jn: int,
    ip: int,
    t_max: float,
    n_steps: int,
):
    psi0 = (evecs[:, jn] + evecs[:, ip]).astype(np.complex128)
    psi0 /= np.linalg.norm(psi0)

    coeffs = evecs.conj().T @ psi0
    times = np.linspace(0.0, t_max, n_steps + 1)

    autocorr = []
    overlap_re = []
    overlap_im = []

    for t in times:
        phase = np.exp(-1j * evals * t)
        psi_t = evecs @ (phase * coeffs)
        overlap = np.vdot(psi0, psi_t)

        autocorr.append(float(np.abs(overlap) ** 2))
        overlap_re.append(float(np.real(overlap)))
        overlap_im.append(float(np.imag(overlap)))

    return (
        times,
        np.asarray(autocorr, dtype=float),
        np.asarray(overlap_re, dtype=float),
        np.asarray(overlap_im, dtype=float),
    )


def dominant_frequency(times: np.ndarray, values: np.ndarray):
    if len(times) < 8:
        return float("nan"), float("nan"), np.array([]), np.array([])

    dt = float(times[1] - times[0])
    x = values - np.mean(values)

    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=dt)
    amps = np.abs(fft)

    if len(amps) > 0:
        amps[0] = 0.0

    idx = int(np.argmax(amps))
    peak_freq = float(freqs[idx])
    rel_strength = float(amps[idx] / (np.sum(amps) + 1e-12))

    return peak_freq, rel_strength, freqs, amps


def recurrence_quality(values: np.ndarray) -> float:
    vmax = float(np.max(values))
    vmin = float(np.min(values))
    if vmax <= 0:
        return float("nan")
    return (vmax - vmin) / vmax


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    cfg = load_config(REPO_ROOT)

    betas = list(cfg["betas"])
    seeds = list(cfg["seeds"])

    beta_index = max(0, min(args.beta_index, len(betas) - 1))
    seed_index = max(0, min(args.seed_index, len(seeds) - 1))

    beta = betas[beta_index]
    seed = seeds[seed_index]

    detail_dir = REPO_ROOT / cfg["paths"]["filament_graph_dir"]

    nodes, edges = load_nodes_edges(detail_dir, beta, seed)
    if nodes is None or edges is None:
        raise FileNotFoundError(
            f"Missing input for beta={beta:.2f}, seed={seed} in {detail_dir}"
        )

    incidence, edge_list, node_ids = build_incidence(nodes, edges)
    D = build_dirac_like(incidence)

    evals, evecs = np.linalg.eigh(D)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    pairs = pair_modes(evals, zero_tol=1e-8, max_pairs=args.max_pairs)
    if not pairs:
        raise RuntimeError("No valid mode pairs found.")

    pair_index = max(1, min(args.pair_index, len(pairs)))
    jn, ip, energy_abs, pair_err = pairs[pair_index - 1]

    times, autocorr, overlap_re, overlap_im = evolve_pair_with_overlap(
        evals=evals,
        evecs=evecs,
        jn=jn,
        ip=ip,
        t_max=args.t_max,
        n_steps=args.n_steps,
    )

    peak_freq, rel_strength, freqs, amps = dominant_frequency(times, autocorr)
    rq = recurrence_quality(autocorr)
    expected_pair_freq = float(abs(evals[ip] - evals[jn]) / (2.0 * np.pi))
    freq_ratio = float(peak_freq / expected_pair_freq) if expected_pair_freq > 0 else float("nan")

    ts = pd.DataFrame(
        {
            "t": times,
            "autocorr": autocorr,
            "overlap_re": overlap_re,
            "overlap_im": overlap_im,
            "beta": beta,
            "seed": seed,
            "pair_index": pair_index,
        }
    )
    ts.to_csv(OUT / "substantial_oscillation_timeseries.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(times, autocorr)
    axes[0].set_title("Autocorrelation C(t)")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("C(t)")

    axes[1].plot(freqs, amps)
    axes[1].axvline(expected_pair_freq, linestyle="--")
    axes[1].set_title("Frequency spectrum")
    axes[1].set_xlabel("frequency")
    axes[1].set_ylabel("amplitude")

    plt.tight_layout()
    fig1 = FIG_OUT / "app_substantial_oscillation.png"
    plt.savefig(fig1, dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(times, overlap_re, label="Re")
    axes[0].plot(times, overlap_im, label="Im")
    axes[0].legend()
    axes[0].set_title("Overlap components")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("overlap")

    axes[1].plot(times, np.sqrt(np.clip(autocorr, 0.0, None)))
    axes[1].set_title("Beat amplitude")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("|overlap|")

    plt.tight_layout()
    fig2 = FIG_OUT / "app_substantial_oscillation_components.png"
    plt.savefig(fig2, dpi=220)
    plt.close(fig)

    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    print("\n=== SUBSTANTIAL OSCILLATION REPORT ===")
    print(f"beta={beta}")
    print(f"seed={seed}")
    print(f"pair_index={pair_index}")
    print(f"abs_energy={energy_abs:.6f}")
    print(f"pairing_error={pair_err:.6e}")
    print(f"dominant_frequency={peak_freq:.6f}")
    print(f"expected_pair_frequency={expected_pair_freq:.6f}")
    print(f"frequency_ratio={freq_ratio:.6f}")
    print(f"relative_peak_strength={rel_strength:.6f}")
    print(f"recurrence_quality={rq:.6f}")
    print(f"dirac_dim={int(D.shape[0])}")
    print(f"n_nodes={int(len(node_ids))}")
    print(f"n_edges={int(len(edge_list))}")


if __name__ == "__main__":
    main()