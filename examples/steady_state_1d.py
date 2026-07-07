"""Generate a manufactured 1D steady-state multigroup scalar flux.

Builds a synthetic ``phi(x, g)`` on a slab of width ``L`` for ``G`` energy
groups: a sine fundamental mode plus a group-dependent Gaussian bump whose
centre migrates with energy. The flux is stored in MARGARET's canonical
``(I, G)`` order (I = spatial cell, G = energy group), so load it in the GUI as
the "1D - Scalar Flux" type with the ``I x G`` dimension ordering.

Alongside the flux it writes the energy grids (group boundaries and centres) and
the spatial grid. Choose the output layout with ``--format``:

    python examples/steady_state_1d.py out.npz            # single .npz
    python examples/steady_state_1d.py out.h5             # single HDF5 file
    python examples/steady_state_1d.py out_dir --format npy  # one .npy per array

Run with ``--help`` for the full list of options.
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from margaret.io.arrays import save_arrays


def group_flux(x, g, L=10.0, G=6, sigma=0.8, A=0.75):
    """Manufactured scalar flux ``phi(x, g)`` for 1D multigroup S_N.

    x     : position (scalar or array), 0 <= x <= L
    g     : energy group index, 1 <= g <= G (1 = fastest)
    L     : slab width (cm)
    G     : number of energy groups
    sigma : width of the migrating Gaussian bump (cm)
    A     : amplitude of the bump relative to the sine mode
    """
    mu_g = L * g / (G + 1)  # bump center migrates with group
    spectrum = 2.0 ** (-(g - 1) / 2)  # decaying group spectrum
    return spectrum * (
        np.sin(np.pi * x / L) + A * np.exp(-((x - mu_g) ** 2) / (2 * sigma**2))
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


def build_dataset(cells: int = 100, L: float = 10.0, G: int = 6) -> Dict[str, np.ndarray]:
    """Assemble the canonical ``(I, G)`` flux plus its energy and spatial grids."""
    x = np.linspace(0.0, L, cells)
    # Stack the per-group profiles into (I, G): group_flux is vectorized over x.
    flux = np.stack([group_flux(x, g, L=L, G=G) for g in range(1, G + 1)], axis=1)
    E_edges, E_mid = energy_grid(G=G)
    return {
        "flux": flux,                # (I, G) canonical scalar flux
        "energy_edges": E_edges,     # (G+1,) group boundaries (descending)
        "energy_centers": E_mid,     # (G,) representative energy per group
        "x_grid": x,                 # (I,) spatial cell positions
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
    parser.add_argument("--length", type=float, default=10.0, help="slab width L (cm)")
    args = parser.parse_args(argv)

    data = build_dataset(cells=args.cells, L=args.length, G=args.groups)
    written = save_arrays(args.output, data, fmt=args.format)
    print(f"flux {data['flux'].shape} (I, G) written to:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
