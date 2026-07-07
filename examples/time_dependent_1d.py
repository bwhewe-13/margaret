"""Generate a manufactured 1D time-dependent multigroup scalar flux.

Builds a synthetic ``phi(x, g, t)`` on a slab of width ``L``: the steady bump of
``steady_state_1d`` is set in motion, with each group's Gaussian sloshing at its
own frequency on top of a global "power breathing" transient. The flux is stored
in MARGARET's canonical ``(I, G, T)`` order (I = spatial cell, G = energy group,
T = time step), so load it in the GUI as the "1D - Time Dependent Scalar Flux"
type with the ``I x G x T`` dimension ordering.

Alongside the flux it writes the energy grids (group boundaries and centres) and
the spatial and time grids. Choose the output layout with ``--format``:

    python examples/time_dependent_1d.py out.npz            # single .npz
    python examples/time_dependent_1d.py out.h5             # single HDF5 file
    python examples/time_dependent_1d.py out_dir --format npy  # one .npy per array

Run with ``--help`` for the full list of options.
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from margaret.io.arrays import save_arrays


def phi(x, g, t, L=10.0, G=6, sigma=0.8, A=0.75, eps=0.4, T=5.0, d=1.0, omega0=2.0):
    """Manufactured scalar flux ``phi(x, g, t)`` for time-dependent 1D multigroup S_N.

    x      : position, 0 <= x <= L
    g      : energy group, 1 <= g <= G (1 = fastest)
    t      : time
    eps, T : amplitude and period of the global "power breathing" mode
    d      : sloshing distance of each group's bump (keep d + 3*sigma inside domain)
    omega0 : base sloshing frequency; group g sloshes at omega0 / 2**((g-1)/2)
    """
    omega_g = omega0 / 2.0 ** ((g - 1) / 2)  # fast groups oscillate faster
    mu_gt = L * g / (G + 1) + d * np.sin(omega_g * t)  # bump center sloshes in time
    spectrum = 2.0 ** (-(g - 1) / 2)
    breathing = 1.0 + eps * np.sin(2 * np.pi * t / T)  # global amplitude transient
    return (
        spectrum
        * breathing
        * (np.sin(np.pi * x / L) + A * np.exp(-((x - mu_gt) ** 2) / (2 * sigma**2)))
    )


def energy_grid(G=6, E_max=2.0e7, E_min=1.0e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Return the energy grid edges and centres for a multigroup structure.

    G     : number of energy groups
    E_max : maximum energy (eV)
    E_min : minimum energy (eV)

    Returns:
        E_edges : shape (G+1,) group edges (descending)
        E_mid   : shape (G,) representative energy per group (geometric mean)
    """
    E_edges = np.logspace(np.log10(E_max), np.log10(E_min), G + 1)
    E_mid = np.sqrt(E_edges[:-1] * E_edges[1:])
    return E_edges, E_mid


def build_dataset(
    cells: int = 100, steps: int = 50, L: float = 10.0, G: int = 6, T: float = 5.0
) -> Dict[str, np.ndarray]:
    """Assemble the canonical ``(I, G, T)`` flux plus its energy, space, time grids."""
    x = np.linspace(0.0, L, cells)
    t = np.linspace(0.0, T, steps)
    # Fill (I, G, T): phi is vectorized over x, so one column per (group, step).
    flux = np.empty((cells, G, steps))
    for gi, g in enumerate(range(1, G + 1)):
        for ti, tv in enumerate(t):
            flux[:, gi, ti] = phi(x, g, tv, L=L, G=G, T=T)
    E_edges, E_mid = energy_grid(G=G)
    return {
        "flux": flux,                # (I, G, T) canonical scalar flux
        "energy_edges": E_edges,     # (G+1,) group boundaries (descending)
        "energy_centers": E_mid,     # (G,) representative energy per group
        "x_grid": x,                 # (I,) spatial cell positions
        "t_grid": t,                 # (T,) time-step values
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "output", help="output file (.npz/.h5/.hdf5) or directory (for --format npy)"
    )
    parser.add_argument(
        "--format",
        choices=("npz", "h5", "npy"),
        default=None,
        help="output format; inferred from the output extension when omitted",
    )
    parser.add_argument("--groups", type=int, default=6, help="number of energy groups G")
    parser.add_argument("--cells", type=int, default=100, help="number of spatial cells I")
    parser.add_argument("--steps", type=int, default=50, help="number of time steps T")
    parser.add_argument("--length", type=float, default=10.0, help="slab width L (cm)")
    parser.add_argument("--time", type=float, default=5.0, help="total transient time T")
    args = parser.parse_args(argv)

    data = build_dataset(
        cells=args.cells, steps=args.steps, L=args.length, G=args.groups, T=args.time
    )
    written = save_arrays(args.output, data, fmt=args.format)
    print(f"flux {data['flux'].shape} (I, G, T) written to:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
