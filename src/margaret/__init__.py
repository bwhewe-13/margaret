"""MARGARET - Modular Analysis and Reactor Graphics Application for Radiation
and Energy Tallies.

A PyQt-based tool for loading and visualizing neutron flux data.

Only the Qt-free surface (the flux model and array IO) is exported here, so
``import margaret`` never pulls in PyQt6. The GUI lives in ``margaret.gui`` and
is launched via ``margaret.app.main``.
"""

from margaret.core.flux_model import FluxModel
from margaret.io.arrays import ArrayInfo, list_arrays, load_array

__version__ = "0.1.0"

__all__ = ["FluxModel", "ArrayInfo", "list_arrays", "load_array", "__version__"]
