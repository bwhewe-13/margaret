"""The flux domain model - all numpy, no Qt.

:class:`FluxModel` holds the loaded flux array (in a canonical axis order), the
optional coordinate grids, and every pure-data operation the GUI needs:
canonicalizing input, reducing to a 1-D line, labelling groups/time steps, and
computing a time-independent y-range. Keeping this Qt-free makes it unit-testable
without a display and gives new options a clean place to live.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from margaret.constants import ORDER_SEP

# Index of each axis label within a canonical array shape.
_AXIS_INDEX = {"I": 0, "G": 1, "T": 2}


class FluxModel:
    """Loaded flux plus its coordinate grids and the operations over them."""

    def __init__(self) -> None:
        self.flux: Optional[np.ndarray] = None       # canonical (I, G[, T])
        self.name: str = ""
        self.axes: Tuple[str, ...] = ()              # canonical axis labels
        self.energy_grid: Optional[np.ndarray] = None  # length G or G+1
        self.x_grid: Optional[np.ndarray] = None       # spatial, length I
        self.t_grid: Optional[np.ndarray] = None       # time, length T

    # -- state -------------------------------------------------------------- #
    @property
    def is_loaded(self) -> bool:
        return self.flux is not None

    @property
    def has_time(self) -> bool:
        return "T" in self.axes

    def set_flux(self, array: np.ndarray, name: str) -> None:
        self.flux = array
        self.name = name

    def clear(self) -> None:
        """Drop the flux and its dimension-specific grids (keep the energy grid)."""
        self.flux = None
        self.name = ""
        self.x_grid = None
        self.t_grid = None

    # -- input handling ----------------------------------------------------- #
    def canonicalize(self, raw: np.ndarray, input_order) -> np.ndarray:
        """Transpose ``raw`` into this model's canonical axis order.

        ``input_order`` lists how the file's axes are laid out (e.g. ``["G",
        "I"]``); the result always matches ``self.axes``. A genuinely 1-D input
        is treated as a single energy group. Raises ``ValueError`` if the array's
        dimensionality doesn't match the expected number of axes.
        """
        axes = self.axes
        n = len(axes)

        if raw.ndim == 1 and n >= 2:
            raw = raw[:, np.newaxis]

        if raw.ndim != n:
            raise ValueError(
                f"This flux type expects {n} axes ({ORDER_SEP.join(axes)}), "
                f"but the file has {raw.ndim} dimension(s) with shape {raw.shape}."
            )

        input_order = list(input_order)
        # For each canonical axis, find where it sits in the input ordering.
        perm = [input_order.index(axis) for axis in axes]
        return np.transpose(raw, perm)

    # -- coordinate grids --------------------------------------------------- #
    def axis_size(self, axis: str) -> Optional[int]:
        if self.flux is None:
            return None
        index = _AXIS_INDEX[axis]
        if index >= self.flux.ndim:
            return None
        return self.flux.shape[index]

    def set_grid(self, axis: str, grid: np.ndarray) -> None:
        if axis == "I":
            self.x_grid = grid
        elif axis == "T":
            self.t_grid = grid

    def time_grid_from_range(self, start: float, stop: float) -> np.ndarray:
        """Time grid that reserves index 0 as the t=0 initial step.

        The flux's first time step (index 0) is the pre-transient initial
        condition (commonly zero flux) and is fixed at t=0; ``start``/``stop``
        describe the remaining real steps, spread uniformly over indices
        1..T-1. Edge cases: T=1 -> ``[0.0]``; T=2 -> ``[0.0, start]``.
        """
        n = self.axis_size("T")            # T length of the loaded cube
        grid = np.zeros(n)                 # index 0 -> t = 0 (initial)
        if n > 1:
            grid[1:] = np.linspace(start, stop, n - 1)
        return grid

    # -- labels ------------------------------------------------------------- #
    def group_count(self) -> int:
        return self.flux.shape[1]

    def group_label(self, g: int) -> str:
        n = self.flux.shape[1]
        e = self.energy_grid
        if e is not None and e.size == n + 1:      # group boundaries
            return f"Group {g} ({e[g]:.3g}-{e[g + 1]:.3g})"
        if e is not None and e.size == n:          # group-center energies
            return f"Group {g} (E={e[g]:.3g})"
        return f"Group {g}"

    def group_from_energy(self, energy: float) -> Optional[Tuple[int, bool]]:
        """Resolve ``energy`` to ``(group_index, exact)`` using the energy grid.

        G+1 entries are band boundaries -> the containing band (clamped to the
        ends; handles ascending or descending grids); ``exact`` is False only
        when the energy falls outside the grid's range (so it was clamped). G
        entries are band centers -> the nearest center; ``exact`` is False when
        the energy is not that center. Returns ``None`` when no usable energy
        grid is loaded.
        """
        e = self.energy_grid
        if e is None or self.flux is None:
            return None
        n = self.group_count()
        if e.size == n + 1:                      # boundaries -> containing band
            lo, hi = min(e[0], e[-1]), max(e[0], e[-1])
            asc = e[-1] >= e[0]
            arr = e if asc else e[::-1]
            idx = int(np.clip(np.searchsorted(arr, energy) - 1, 0, n - 1))
            g = idx if asc else (n - 1 - idx)
            return g, bool(lo <= energy <= hi)
        if e.size == n:                          # centers -> nearest
            g = int(np.argmin(np.abs(e - energy)))
            return g, bool(np.isclose(e[g], energy))
        return None

    def time_label(self, t: int) -> str:
        suffix = " (initial)" if t == 0 else ""
        if self.t_grid is not None and t < self.t_grid.size:
            return f"t = {self.t_grid[t]:.3g}{suffix}"
        return f"t = {t}{suffix}"

    # -- reductions for plotting -------------------------------------------- #
    def slice(self, group, t):
        """Reduce the cube to a 1-D line for the given group and time step.

        ``group`` is an energy-group index, or ``-1``/``None`` to sum over all
        groups. ``t`` is the time-step index (ignored for non-time-dependent
        flux). Returns ``(x, y, xlabel, line_label)``.
        """
        flux = self.flux
        if self.has_time:
            flux = flux[:, :, t]

        if group == -1 or group is None:
            y = flux.sum(axis=1)
            line_label = "Sum over all groups"
        else:
            y = flux[:, group]
            line_label = self.group_label(group)

        n_cells = flux.shape[0]
        if self.x_grid is not None and self.x_grid.size == n_cells:
            x, xlabel = self.x_grid, "Position"
        else:
            x, xlabel = np.arange(n_cells), "Spatial cell (I)"
        return x, y, xlabel, line_label

    def spectrum(self, ix: int, t):
        """Reduce the cube to a 1-D energy spectrum at a single spatial cell.

        The inverse of :meth:`slice`: fix the spatial cell ``ix`` (and time step
        ``t``, ignored for non-time-dependent flux) and keep the energy axis, so
        ``y`` is flux vs. energy group. ``x`` is the energy grid when one is
        loaded (band centers, or midpoints of ``G+1`` boundaries), otherwise the
        group index. Returns ``(x, y, xlabel, line_label)``.
        """
        flux = self.flux
        if self.has_time:
            flux = flux[:, :, t]

        y = flux[ix, :]

        n = self.group_count()
        e = self.energy_grid
        if e is not None and e.size == n:            # group-center energies
            x, xlabel = e, "Energy"
        elif e is not None and e.size == n + 1:      # group boundaries -> centers
            x, xlabel = (e[:-1] + e[1:]) / 2, "Energy"
        else:
            x, xlabel = np.arange(n), "Energy group (G)"

        if self.x_grid is not None and self.x_grid.size == flux.shape[0]:
            line_label = f"Position {self.x_grid[ix]:.3g}"
        else:
            line_label = f"Spatial cell {ix}"
        return x, y, xlabel, line_label

    def constant_ylim(self, group) -> Tuple[float, float]:
        """Y-range over *all* time steps for ``group`` (so the slider can't rescale)."""
        cube = self.flux  # canonical (I, G, T)
        if group == -1 or group is None:
            vals = cube.sum(axis=1)     # (I, T)
        else:
            vals = cube[:, group, :]    # (I, T)
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if hi > lo:
            pad = 0.05 * (hi - lo)
            return lo - pad, hi + pad
        delta = abs(hi) * 0.05 or 1.0   # flat data - give it a small window
        return lo - delta, hi + delta
