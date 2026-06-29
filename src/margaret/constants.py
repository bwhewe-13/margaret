"""Application-wide constants and registries.

This module holds the small pieces of configuration the rest of MARGARET keys
off - most importantly the ``FLUX_TYPES`` registry. Adding a new flux type (and
therefore a new set of dimension-ordering options) is a one-line edit here.
"""

from __future__ import annotations

APP_NAME = "MARGARET"
APP_SUBTITLE = (
    "Modular Analysis and Reactor Graphics Application "
    "for Radiation and Energy Tallies"
)

# Flux types known to the app. Each entry's ``axes`` tuple is the canonical
# internal axis order; the dimension-ordering dropdown offers every permutation
# of these labels. Add a new type by adding one entry here.
#   I = spatial cell / interval index, G = energy group, T = time step
FLUX_TYPES = {
    "1D - Scalar Flux": {"axes": ("I", "G")},
    "1D - Time Dependent Scalar Flux": {"axes": ("I", "G", "T")},
}

# File-dialog filter covering every supported array format (flux + grids).
FILE_FILTER = "Array data (*.npz *.npy *.csv *.txt *.h5 *.hdf5);;All files (*)"

# Preferred dataset/array keys to look for in container formats (.npz/.h5).
PREFERRED_KEYS = ("flux", "values", "data", "grid")

# Extensions whose files can hold several named arrays the user may choose from.
CONTAINER_EXTS = (".npz", ".h5", ".hdf5")

# Separator used when presenting/parsing an axis ordering, e.g. "I x G".
ORDER_SEP = " x "
