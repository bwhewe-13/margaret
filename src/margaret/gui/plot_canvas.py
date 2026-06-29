"""Matplotlib canvas embedded in Qt for rendering flux.

:class:`FluxCanvas` is a dumb view: it knows how to show a placeholder and how to
draw a single 1-D line from primitives handed to it. All the data reduction lives
in :class:`~margaret.core.flux_model.FluxModel`, so new plot kinds can be added
here without touching the model or the window.
"""

from __future__ import annotations

from typing import Optional, Tuple

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class FluxCanvas(FigureCanvasQTAgg):
    """A Qt widget that draws a flux line plot."""

    def __init__(self, parent=None, width: float = 7.0, height: float = 4.5):
        self.figure = Figure(figsize=(width, height), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)
        self.show_placeholder()

    def show_placeholder(self) -> None:
        self.axes.clear()
        self.axes.text(
            0.5, 0.5, "Load a flux file to display it here.",
            ha="center", va="center", color="gray",
            transform=self.axes.transAxes,
        )
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.draw_idle()

    def draw_flux(self, x, y, xlabel: str, line_label: str, title: str,
                  ylim: Optional[Tuple[float, float]] = None) -> None:
        self.axes.clear()
        self.axes.plot(x, y, marker=".", color="tab:blue", label=line_label)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel("Scalar flux")
        self.axes.set_title(title)
        self.axes.grid(True, alpha=0.3)
        self.axes.legend(loc="best")
        # A fixed y-range keeps the scale comparable while scrubbing time.
        if ylim is not None:
            self.axes.set_ylim(*ylim)
        self.draw_idle()
