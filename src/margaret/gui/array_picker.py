"""Dialog for choosing which array to load from a multi-array container file."""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from margaret.constants import PREFERRED_KEYS
from margaret.io.arrays import ArrayInfo


class ArrayPickerDialog(QDialog):
    """Pop-up listing the arrays in a container file so the user can choose one."""

    def __init__(self, parent, filename: str, infos: List[ArrayInfo]):
        super().__init__(parent)
        self.setWindowTitle("Select an array")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<b>{filename}</b> contains several arrays - "
                                "choose one to load:"))

        self.list = QListWidget()
        for info in infos:
            item = QListWidgetItem(info.describe())
            item.setData(Qt.ItemDataRole.UserRole, info.name)
            self.list.addItem(item)
        # Preselect a preferred key if present, otherwise the first entry.
        preferred = next(
            (i for i, info in enumerate(infos) if info.name in PREFERRED_KEYS), 0
        )
        self.list.setCurrentRow(preferred)
        self.list.itemDoubleClicked.connect(lambda _item: self.accept())
        layout.addWidget(self.list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_name(self) -> Optional[str]:
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    @classmethod
    def choose(cls, parent, filename: str,
               infos: List[ArrayInfo]) -> Optional[str]:
        """Show the dialog; return the chosen array name, or ``None`` if cancelled."""
        dialog = cls(parent, filename, infos)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.selected_name()
        return None
