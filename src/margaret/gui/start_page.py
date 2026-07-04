"""The MARGARET start-page window.

``StartPage`` builds the UI, owns a :class:`~margaret.core.flux_model.FluxModel`
and a :class:`~margaret.gui.plot_canvas.FluxCanvas`, and wires Qt signals to
them. It holds only widget state - every data operation is delegated to the
model, and all drawing to the canvas.
"""

from __future__ import annotations

import itertools
import os
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator, QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from margaret.constants import (
    APP_NAME,
    APP_SUBTITLE,
    FILE_FILTER,
    FLUX_TYPES,
    ORDER_SEP,
)
from margaret.core.flux_model import FluxModel
from margaret.gui.array_picker import ArrayPickerDialog
from margaret.gui.plot_canvas import FluxCanvas
from margaret.io.arrays import list_arrays, load_array


class StartPage(QMainWindow):
    """Top-level MARGARET start page."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} - Neutron Flux Viewer")
        self.resize(1100, 800)

        self.model = FluxModel()

        self._build_ui()
        self._on_flux_type_changed(self.type_combo.currentText())

    # -- construction ------------------------------------------------------- #
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        root.addWidget(self._build_header())

        top_row = QHBoxLayout()
        top_row.addWidget(self._build_flux_controls(), stretch=1)
        top_row.addWidget(self._build_view_controls(), stretch=1)
        root.addLayout(top_row)
        root.addWidget(self._build_grid_controls())

        self.canvas = FluxCanvas(self)
        root.addWidget(NavigationToolbar2QT(self.canvas, self))
        root.addWidget(self.canvas, stretch=1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Pick a flux type and load a file to begin.")

    def _build_header(self) -> QWidget:
        header = QWidget()
        layout = QVBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel(APP_NAME)
        title.setObjectName("appTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # QSS has no letter-spacing, so widen the tracking on the font itself
        # to give the (already uppercase) wordmark an instrument-panel feel.
        font = title.font()
        font.setPointSize(26)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 135)
        title.setFont(font)

        # A thin accent rule under the wordmark.
        rule = QFrame()
        rule.setObjectName("titleRule")
        rule.setFixedHeight(2)
        rule.setFixedWidth(220)

        subtitle = QLabel(APP_SUBTITLE)
        subtitle.setObjectName("appSubtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(rule, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(subtitle)
        return header

    def _build_flux_controls(self) -> QWidget:
        panel = QGroupBox("Flux")
        form = QFormLayout(panel)

        self.type_combo = QComboBox()
        self.type_combo.addItems(FLUX_TYPES.keys())
        self.type_combo.currentTextChanged.connect(self._on_flux_type_changed)
        form.addRow("Flux type", self.type_combo)

        self.order_combo = QComboBox()
        form.addRow("Dimension ordering", self.order_combo)

        self.load_button = QPushButton("Load flux...")
        self.load_button.setObjectName("primaryButton")
        self.load_button.clicked.connect(self._on_load_clicked)
        form.addRow(self.load_button)

        self.info_label = QLabel("No flux loaded")
        self.info_label.setWordWrap(True)
        self.info_label.setProperty("role", "muted")
        form.addRow(self.info_label)

        return panel

    def _build_view_controls(self) -> QWidget:
        panel = QGroupBox("View")
        form = QFormLayout(panel)

        self.group_combo = QComboBox()
        self.group_combo.setEnabled(False)
        self.group_combo.currentIndexChanged.connect(self._redraw)
        form.addRow("Energy group", self.group_combo)

        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.setEnabled(False)
        self.time_slider.valueChanged.connect(self._on_time_changed)
        self.time_value_label = QLabel("t = 0")
        slider_row = QHBoxLayout()
        slider_row.addWidget(self.time_slider, stretch=1)
        slider_row.addWidget(self.time_value_label)
        self.time_row = QWidget()
        self.time_row.setLayout(slider_row)
        self.time_form_label = QLabel("Time step")
        form.addRow(self.time_form_label, self.time_row)

        return panel

    def _build_grid_controls(self) -> QWidget:
        panel = QGroupBox("Grids")
        grid = QGridLayout(panel)
        grid.setColumnStretch(6, 1)

        # Energy grid: loaded from a file only.
        grid.addWidget(QLabel("Energy grid"), 0, 0)
        self.energy_load_btn = QPushButton("Load...")
        self.energy_load_btn.clicked.connect(self._load_energy_grid)
        grid.addWidget(self.energy_load_btn, 0, 5)
        self.energy_status = self._status_label("not loaded")
        grid.addWidget(self.energy_status, 0, 6)

        # Spatial and time grids: generated from a range or loaded from a file.
        (self.x_start, self.x_stop, self.x_gen_btn,
         self.x_load_btn, self.x_status) = self._grid_row(
            grid, 1, "Spatial grid",
            self._generate_spatial_grid, self._load_spatial_grid,
        )
        (self.t_start, self.t_stop, self.t_gen_btn,
         self.t_load_btn, self.t_status) = self._grid_row(
            grid, 2, "Time grid",
            self._generate_time_grid, self._load_time_grid,
        )

        return panel

    def _grid_row(self, layout: QGridLayout, row: int, title: str,
                  on_generate, on_load):
        layout.addWidget(QLabel(title), row, 0)
        start = QLineEdit()
        start.setPlaceholderText("start")
        start.setValidator(QDoubleValidator())
        start.setMaximumWidth(70)
        stop = QLineEdit()
        stop.setPlaceholderText("stop")
        stop.setValidator(QDoubleValidator())
        stop.setMaximumWidth(70)
        gen = QPushButton("Generate")
        gen.clicked.connect(on_generate)
        load = QPushButton("Load...")
        load.clicked.connect(on_load)
        status = self._status_label("not set")
        layout.addWidget(start, row, 1)
        layout.addWidget(stop, row, 2)
        layout.addWidget(gen, row, 3)
        layout.addWidget(load, row, 5)
        layout.addWidget(status, row, 6)
        return start, stop, gen, load, status

    @staticmethod
    def _status_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("role", "muted")
        return label

    # -- reactive UI -------------------------------------------------------- #
    def _on_flux_type_changed(self, type_name: str) -> None:
        """Set the canonical axes and repopulate the ordering dropdown."""
        axes = FLUX_TYPES[type_name]["axes"]
        self.model.axes = axes

        self.order_combo.blockSignals(True)
        self.order_combo.clear()
        for perm in itertools.permutations(axes):
            self.order_combo.addItem(ORDER_SEP.join(perm))
        self.order_combo.blockSignals(False)

        has_time = "T" in axes
        self.time_form_label.setVisible(has_time)
        self.time_row.setVisible(has_time)
        for w in (self.t_start, self.t_stop, self.t_gen_btn, self.t_load_btn):
            w.setEnabled(has_time)

        # A previously loaded flux belongs to the old type's axis layout, so
        # drop it (and its dimension-specific grids) to avoid a shape mismatch.
        self._clear_flux()

    def _clear_flux(self) -> None:
        self.model.clear()
        self.info_label.setText("No flux loaded")
        self.x_status.setText("not set")
        self.t_status.setText("not set")
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        self.group_combo.setEnabled(False)
        self.group_combo.blockSignals(False)
        self.time_slider.blockSignals(True)
        self.time_slider.setMaximum(0)
        self.time_slider.setEnabled(False)
        self.time_slider.blockSignals(False)
        self.canvas.show_placeholder()
        self.statusBar().showMessage("Pick a flux type and load a file to begin.")

    # -- flux loading ------------------------------------------------------- #
    def _read_array(self, path: str) -> Optional[np.ndarray]:
        """Read an array from ``path``, prompting to choose when a container
        (``.npz``/``.h5``) holds more than one array.

        Returns ``None`` if the user cancels the picker or a load error was
        already surfaced to them.
        """
        try:
            infos = list_arrays(path)
        except Exception as exc:  # e.g. missing h5py, unreadable container
            QMessageBox.critical(self, "Load failed", str(exc))
            return None

        key = None
        if infos is not None:
            if not infos:
                QMessageBox.critical(
                    self, "Load failed",
                    f"{os.path.basename(path)} contains no arrays.",
                )
                return None
            if len(infos) > 1:
                key = ArrayPickerDialog.choose(self, os.path.basename(path), infos)
                if key is None:
                    return None  # cancelled
            else:
                key = infos[0].name

        try:
            return load_array(path, key=key)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return None

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open flux file", "", FILE_FILTER
        )
        if not path:
            return
        raw = self._read_array(path)
        if raw is None:
            return
        order = self.order_combo.currentText().split(ORDER_SEP)
        try:
            flux = self.model.canonicalize(raw, order)
        except Exception as exc:  # surface any shape error to the user
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self.model.set_flux(flux, os.path.basename(path))
        self.info_label.setText(
            f"{self.model.name}\nshape {flux.shape} "
            f"({ORDER_SEP.join(self.model.axes)})"
        )
        # Drop coordinate grids that no longer match the new flux dimensions.
        if self.model.x_grid is not None and self.model.x_grid.size != flux.shape[0]:
            self.model.x_grid = None
            self.x_status.setText("cleared (size changed)")
        if self.model.has_time:
            if self.model.t_grid is not None and self.model.t_grid.size != flux.shape[2]:
                self.model.t_grid = None
                self.t_status.setText("cleared (size changed)")

        self._populate_selectors()
        self._redraw()
        self.statusBar().showMessage(
            f"Loaded {self.model.name}  shape={flux.shape}"
        )

    def _populate_selectors(self) -> None:
        self._refresh_group_labels()

        if self.model.has_time:
            n_times = self.model.axis_size("T")
            self.time_slider.blockSignals(True)
            self.time_slider.setMaximum(max(n_times - 1, 0))
            self.time_slider.setValue(0)
            self.time_slider.setEnabled(True)
            self.time_slider.blockSignals(False)
            self.time_value_label.setText(self.model.time_label(0))

    # -- energy / coordinate grids ----------------------------------------- #
    def _load_energy_grid(self) -> None:
        grid = self._load_1d_grid("energy grid")
        if grid is None:
            return
        self.model.energy_grid = grid
        self.energy_status.setText(
            f"{grid.size} pts, {grid[0]:.3g}-{grid[-1]:.3g}"
        )
        self._refresh_group_labels()
        self._redraw()

    def _generate_spatial_grid(self) -> None:
        self._generate_grid(axis="I", start=self.x_start, stop=self.x_stop,
                             status=self.x_status, name="spatial")

    def _load_spatial_grid(self) -> None:
        self._set_loaded_grid(axis="I", status=self.x_status, name="spatial")

    def _generate_time_grid(self) -> None:
        self._generate_grid(axis="T", start=self.t_start, stop=self.t_stop,
                             status=self.t_status, name="time")

    def _load_time_grid(self) -> None:
        self._set_loaded_grid(axis="T", status=self.t_status, name="time")

    def _generate_grid(self, axis: str, start: QLineEdit, stop: QLineEdit,
                       status: QLabel, name: str) -> None:
        n = self.model.axis_size(axis)
        if n is None:
            QMessageBox.information(
                self, "Load flux first",
                f"Load a flux file first so the {name} grid length is known.",
            )
            return
        try:
            a, b = float(start.text()), float(stop.text())
        except ValueError:
            QMessageBox.warning(
                self, "Invalid range",
                f"Enter numeric start and stop values for the {name} grid.",
            )
            return
        grid = np.linspace(a, b, n)
        self.model.set_grid(axis, grid)
        status.setText(f"{n} pts, {a:.3g}-{b:.3g} (generated)")
        self._redraw()

    def _set_loaded_grid(self, axis: str, status: QLabel, name: str) -> None:
        grid = self._load_1d_grid(f"{name} grid")
        if grid is None:
            return
        n = self.model.axis_size(axis)
        if n is not None and grid.size != n:
            QMessageBox.critical(
                self, "Grid size mismatch",
                f"The {name} grid has {grid.size} points but the flux needs "
                f"{n} ({axis}).",
            )
            return
        self.model.set_grid(axis, grid)
        status.setText(f"{grid.size} pts, {grid[0]:.3g}-{grid[-1]:.3g} (loaded)")
        if axis == "T":
            self.time_value_label.setText(
                self.model.time_label(self.time_slider.value())
            )
        self._redraw()

    def _load_1d_grid(self, label: str) -> Optional[np.ndarray]:
        path, _ = QFileDialog.getOpenFileName(
            self, f"Open {label}", "", FILE_FILTER
        )
        if not path:
            return None
        arr = self._read_array(path)
        if arr is None:
            return None  # cancelled or error already surfaced
        grid = np.ravel(arr)
        if grid.size == 0:
            QMessageBox.critical(self, "Load failed", f"The {label} is empty.")
            return None
        return grid

    # -- labels / plotting -------------------------------------------------- #
    def _refresh_group_labels(self) -> None:
        if not self.model.is_loaded:
            return
        current = self.group_combo.currentData()
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        for g in range(self.model.group_count()):
            self.group_combo.addItem(self.model.group_label(g), g)
        self.group_combo.addItem("Sum over all groups", -1)
        if current is not None:
            i = self.group_combo.findData(current)
            if i >= 0:
                self.group_combo.setCurrentIndex(i)
        self.group_combo.setEnabled(True)
        self.group_combo.blockSignals(False)

    def _on_time_changed(self, value: int) -> None:
        self.time_value_label.setText(self.model.time_label(value))
        self._redraw()

    def _redraw(self) -> None:
        if not self.model.is_loaded:
            return
        t = self.time_slider.value() if self.model.has_time else None
        group = self.group_combo.currentData()
        x, y, xlabel, line_label = self.model.slice(group, t)

        title = self.model.name or "Neutron flux"
        ylim = None
        if self.model.has_time:
            title += f"  ({self.model.time_label(t)})"
            # Hold the y-axis fixed across time so the scale stays comparable.
            ylim = self.model.constant_ylim(group)

        self.canvas.draw_flux(x, y, xlabel, line_label, title, ylim=ylim)
