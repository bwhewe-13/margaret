"""PyInstaller entry script.

PyInstaller needs a plain script rather than the ``margaret.app:main``
console-script entry point declared in ``pyproject.toml``.
"""

import sys

from margaret.app import main

if __name__ == "__main__":
    sys.exit(main())
