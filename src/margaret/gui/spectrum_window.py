"""A pop-out window showing the energy spectrum at a single spatial cell.

:class:`SpectrumWindow` is opened by :class:`~margaret.gui.start_page.StartPage`
when the user clicks a point on the spatial flux plot. It reuses
:class:`~margaret.gui.plot_canvas.FluxCanvas` so its styling matches the main
plot, and adds a "Log-log axes" toggle (energy spectra often span many decades).
The window is reusable: each new click redraws it via :meth:`show_spectrum`.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QVBoxLayout, QWidget

from margaret.gui.plot_canvas import FluxCanvas


class SpectrumWindow(QWidget):
    """Reusable pop-out plotting flux vs. energy at one spatial cell."""

    def __init__(self, parent=None):
        # Pass parent for ownership but keep this a top-level window.
        super().__init__(parent)
        self.setWindowTitle("Energy spectrum")
        self.resize(600, 450)
        # Show as its own top-level window rather than embedded in the parent.
        self.setWindowFlag(Qt.WindowType.Window, True)

        self.canvas = FluxCanvas(self)
        self.log_check = QCheckBox("Log-log axes")
        self.log_check.toggled.connect(self._render)

        layout = QVBoxLayout(self)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.log_check)

        # Latest spectrum primitives, so the log toggle can redraw without a click.
        self._primitives = None

    def show_spectrum(self, x, y, xlabel: str, line_label: str, title: str) -> None:
        """Install a new spectrum and draw it."""
        self._primitives = (x, y, xlabel, line_label, title)
        self._render()

    def _render(self, *_args) -> None:
        if self._primitives is None:
            return
        x, y, xlabel, line_label, title = self._primitives
        self.canvas.draw_flux(x, y, xlabel, line_label, title)
        if self.log_check.isChecked():
            # matplotlib masks non-positive values on a log axis, so zero-flux
            # groups simply drop out.
            self.canvas.axes.set_xscale("log")
            self.canvas.axes.set_yscale("log")
            self.canvas.draw_idle()
