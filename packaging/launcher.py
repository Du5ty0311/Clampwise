"""The frozen build's entry point (PyInstaller needs a script, not a module).

Pins pyqtgraph to PyQt5 before it looks for a binding, and lets the batch's
spawned workers run the work they are sent instead of opening a second window.
"""

import multiprocessing
import os
import sys

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from clampwise.app.main import main

    sys.exit(main())
