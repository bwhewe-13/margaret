"""Application entry point.

Launch with the installed ``margaret`` command or ``python -m margaret``.
"""

from __future__ import annotations

import sys


def main() -> int:
    # Import Qt lazily so importing this module (e.g. for ``--help`` tooling or
    # tests) doesn't require a display.
    from PyQt6.QtWidgets import QApplication

    from margaret.gui.start_page import StartPage

    app = QApplication(sys.argv[:1])
    window = StartPage()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
