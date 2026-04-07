#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--size", type=int, default=128)
parser.add_argument("--distance", type=float, default=20.0)
parser.add_argument("--angle", type=float, default=0.0)

args = parser.parse_args()

SIZE = args.size
DIST = args.distance
ANGLE = args.angle

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ===============================
# BUILD 2 ATOMS
# ===============================

def gaussian_blob(cx, cy, sigma=5):
    y, x = np.indices((SIZE, SIZE))
    return np.exp(-((x-cx)**2 + (y-cy)**2)/(2*sigma**2))

cx = SIZE/2
cy = SIZE/2

dx = DIST * np.cos(ANGLE)
dy = DIST * np.sin(ANGLE)

atom1 = gaussian_blob(cx - dx/2, cy - dy/2)
atom2 = gaussian_blob(cx + dx/2, cy + dy/2)

field = atom1 - atom2  # dipole structure

# ===============================
# METRICS
# ===============================

interaction_strength = np.sum(np.abs(atom1 * atom2))

if interaction_strength < 0.01:
    regime = "independent"
elif interaction_strength < 0.1:
    regime = "weak coupling"
else:
    regime = "dipole formed"

# ===============================
# SAVE
# ===============================

plt.imshow(field, cmap="coolwarm")
plt.axis("off")
plt.savefig(OUT / "app_proto_dipole.png", dpi=300)
plt.close()

# ===============================
# PRINT
# ===============================

print(f"interaction_strength={interaction_strength}")
print(f"regime={regime}")
