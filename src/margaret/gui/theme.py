"""Central theme for MARGARET: colour palette and the app-wide Qt Style Sheet.

Qt Style Sheets (QSS) are Qt's CSS dialect. Keeping the whole look in one place
means every window shares one coherent, light, single-accent aesthetic - the UI
stays quiet so the flux plot carries the colour. The palette constants are plain
strings, so they can be imported anywhere (e.g. by the matplotlib canvas) without
requiring a display.
"""

from __future__ import annotations

# -- palette --------------------------------------------------------------- #
# A light, instrument-panel look with a single Cherenkov-blue accent.
ACCENT = "#2d7ff9"          # reactor-pool blue; UI accent + flux line
ACCENT_HOVER = "#1e6fe6"
ACCENT_PRESSED = "#155ec4"

WINDOW_BG = "#f5f6f8"       # app background
SURFACE = "#ffffff"         # cards, inputs, buttons
BORDER = "#d5d9e0"
TEXT = "#1c2330"
MUTED = "#6b7280"           # secondary labels (subtitle, status, hints)
WARN = "#b45309"            # amber; soft warnings (e.g. snapped-to-nearest hints)
SELECTION_TINT = "#e4efff"  # list/selection highlight

# -- stylesheet ------------------------------------------------------------ #
STYLESHEET = f"""
QMainWindow, QDialog {{
    background: {WINDOW_BG};
}}
QWidget {{
    color: {TEXT};
    font-size: 13px;
}}

/* Grouped control panels read as cards. */
QGroupBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 12px;
    padding: 10px 12px 12px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
    color: {ACCENT};
}}

/* Buttons: light surface by default, accent-filled when marked primary. */
QPushButton {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 12px;
}}
QPushButton:hover {{
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background: {SELECTION_TINT};
}}
QPushButton:disabled {{
    color: {MUTED};
    background: {WINDOW_BG};
}}
QPushButton#primaryButton {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: white;
    font-weight: 600;
}}
QPushButton#primaryButton:hover {{
    background: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
}}
QPushButton#primaryButton:pressed {{
    background: {ACCENT_PRESSED};
    border-color: {ACCENT_PRESSED};
}}

/* Inputs: subtle border, accent focus ring. */
QComboBox, QLineEdit {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 8px;
    selection-background-color: {ACCENT};
    selection-color: white;
}}
QComboBox:focus, QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    selection-background-color: {SELECTION_TINT};
    selection-color: {TEXT};
    outline: none;
}}

/* Time-step / value sliders in the accent colour. */
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background: {ACCENT_HOVER};
}}

/* Array-picker list. */
QListWidget {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    outline: none;
}}
QListWidget::item:selected {{
    background: {SELECTION_TINT};
    color: {TEXT};
}}

/* Chrome. */
QStatusBar {{
    background: {WINDOW_BG};
    color: {MUTED};
    border-top: 1px solid {BORDER};
}}
QToolBar {{
    background: {WINDOW_BG};
    border: none;
    spacing: 2px;
}}

/* Header wordmark + shared muted-label role. */
QLabel[role="muted"] {{
    color: {MUTED};
}}
QLabel[role="warn"] {{
    color: {WARN};
}}
#appTitle {{
    color: {ACCENT};
}}
#appSubtitle {{
    color: {MUTED};
    font-size: 13px;
}}
#titleRule {{
    background: {ACCENT};
    border: none;
    border-radius: 1px;
}}
"""


def apply_theme(app) -> None:
    """Install the MARGARET stylesheet on a ``QApplication``."""
    app.setStyleSheet(STYLESHEET)
