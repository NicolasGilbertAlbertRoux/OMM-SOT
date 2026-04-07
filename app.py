#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
from pathlib import Path

import streamlit as st

PYTHON = sys.executable

DOMAINS = {
    "Proto-atom (interactive)": {
        "script": "src/app_variants/proto_atom_render_app.py",
        "description": "Interactive proto-atomic render using the real research code adapted for parameterized launch.",
        "params": {
            "size": {"type": "int", "min": 64, "max": 256, "default": 128, "step": 32},
            "n_steps": {"type": "int", "min": 50, "max": 300, "default": 180, "step": 10},
            "seed": {"type": "int", "min": 1, "max": 20, "default": 3, "step": 1},
            "beta": {"type": "float", "min": 5.0, "max": 12.0, "default": 8.75},
            "center_gain": {"type": "float", "min": 0.0, "max": 0.05, "default": 0.012},
            "node_gain": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.100},
            "matter_gain": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.098},
            "omega_bg": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.22},
            "background_gain": {"type": "float", "min": 0.0, "max": 0.1, "default": 0.035},
            "omega_local": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.47},
            "local_beat_gain": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.085},
            "flux_gain": {"type": "float", "min": 0.0, "max": 0.1, "default": 0.045},
            "edge_penalty": {"type": "float", "min": 0.0, "max": 0.3, "default": 0.10},
        },
        "figures": [
            "figures/app_proto_atom_render.png",
            "figures/app_proto_atom_render_diagnostics.png",
        ],
    },
}

st.set_page_config(page_title="OMM-SOT Explorer", layout="wide")

st.title("OMM-SOT Interactive Explorer")
st.markdown(
    "This interface launches interactive variants of selected research scripts, "
    "so that users can modify parameters and regenerate actual outputs."
)

domain = st.selectbox("Scientific domain", list(DOMAINS.keys()))
entry = DOMAINS[domain]

st.subheader(domain)
st.write(entry["description"])

st.markdown("**Executed script**")
st.code(f"{PYTHON} {entry['script']} ...")

st.markdown("**Parameters**")
values = {}
for key, spec in entry["params"].items():
    step = spec.get("step", None)
    if spec["type"] == "int":
        values[key] = st.slider(
            key,
            int(spec["min"]),
            int(spec["max"]),
            int(spec["default"]),
            step=int(step) if step is not None else 1,
        )
    else:
        values[key] = st.slider(
            key,
            float(spec["min"]),
            float(spec["max"]),
            float(spec["default"]),
        )

if st.button("Run simulation", width="stretch"):
    cmd = [PYTHON, entry["script"]]
    for key, value in values.items():
        cmd.extend([f"--{key}", str(value)])

    st.info("Running interactive research variant...")
    try:
        subprocess.run(cmd, check=True)
        st.success("Simulation completed.")
    except subprocess.CalledProcessError as exc:
        st.error(f"Simulation failed with return code {exc.returncode}.")

st.subheader("Expected output figures")
for fig in entry["figures"]:
    st.code(fig)

st.subheader("Preview")
existing = [Path(fig) for fig in entry["figures"] if Path(fig).exists()]
if not existing:
    st.warning("No preview figure currently found.")
else:
    cols = st.columns(2)
    for i, path in enumerate(existing):
        with cols[i % 2]:
            st.image(str(path), caption=path.name, width="stretch")

st.markdown("---")
st.caption(
    "These interactive variants should remain faithful to the original research scripts. "
    "They are intended for controlled parameter exploration, not for replacing the reference paper pipeline."
)        "figures/aligned_domain_curl_snapshots.png",
    ],
    "Orbital regime": [
        "figures/full_dynamics_v2_trajectories.png",
    ],
    "Two-scale geometry": [
        "figures/two_scale_total_geometry.png",
        "figures/two_scale_trajectories.png",
        "figures/two_scale_tail_fit.png",
    ],
    "Cosmology": [
        "figures/cosmic_scale_factor.png",
        "figures/cosmic_hubble_rate.png",
        "figures/cosmic_final_total_geometry.png",
    ],
}

DOMAIN_DESCRIPTIONS = {
    "Proto-atom (stable structure)": "Generate the canonical stable proto-atomic render used as the main atom-like reference.",
    "Proto-periodic classification": "Generate the emergent family / periodic-style classification outputs.",
    "Dipole / binding regime": "Generate dipolar and binding-oriented interaction outputs between structured configurations.",
    "Magnetic alignment": "Generate alignment-sensitive magnetic-like domain outputs.",
    "Orbital regime": "Generate orbital or quasi-orbital structured trajectories.",
    "Two-scale geometry": "Generate effective geometry outputs linking local and global structure.",
    "Cosmology": "Generate the large-scale expansion scan and cosmological outputs.",
}

st.set_page_config(page_title="OMM-SOT Explorer", layout="wide")

st.title("OMM-SOT Interactive Explorer")
st.markdown(
    "A lightweight interface for exploring representative regimes of the Oscillatory Mantle Model."
)

with st.sidebar:
    st.header("Explorer")
    domain = st.selectbox("Scientific domain", list(DOMAIN_COMMANDS.keys()))
    st.caption(DOMAIN_DESCRIPTIONS[domain])

    if domain == "Magnetic alignment":
        angle = st.slider("Initial alignment angle (preview parameter)", 0, 180, 45)
        st.caption("Current version: UI preset only. Full parameter wiring can be added in a dedicated wrapper script.")
    elif domain == "Dipole / binding regime":
        rot_x = st.slider("Render rotation X (preview)", 0, 360, 20)
        rot_y = st.slider("Render rotation Y (preview)", 0, 360, 35)
        rot_z = st.slider("Render rotation Z (preview)", 0, 360, 0)
        st.caption("Current version: UI preset only. Full parameter wiring can be added in a dedicated wrapper script.")
    elif domain == "Orbital regime":
        orbit_scale = st.slider("Orbit scale (preview)", 1, 10, 5)
        st.caption("Current version: UI preset only. Full parameter wiring can be added in a dedicated wrapper script.")
    elif domain == "Cosmology":
        complexity = st.slider("Scan scope (preview)", 1, 10, 5)
        st.caption("Current version: UI preset only. Full parameter wiring can be added in a dedicated wrapper script.")

    run = st.button("Run simulation", use_container_width=True)

st.subheader(domain)
st.write(DOMAIN_DESCRIPTIONS[domain])

st.subheader("Expected outputs")
for fig in DOMAIN_HINTS.get(domain, []):
    st.code(fig)

if run:
    st.info("Running simulation. Depending on the selected domain, this may take some time.")
    try:
        subprocess.run(DOMAIN_COMMANDS[domain], check=True)
        st.success("Simulation completed.")
    except subprocess.CalledProcessError as exc:
        st.error(f"Simulation failed with return code {exc.returncode}.")

st.subheader("Preview")
figures = DOMAIN_HINTS.get(domain, [])
cols = st.columns(2)

for i, fig in enumerate(figures[:2]):
    path = Path(fig)
    if path.exists():
        with cols[i % 2]:
            st.image(str(path), caption=path.name, use_container_width=True)

if len(figures) > 2:
    for fig in figures[2:]:
        path = Path(fig)
        if path.exists():
            st.image(str(path), caption=path.name, use_container_width=True)

st.markdown("---")
st.caption(
    "This explorer currently launches existing simulation scripts. Domain-specific parameter controls can be wired more deeply by introducing lightweight wrapper scripts for each regime."
)
