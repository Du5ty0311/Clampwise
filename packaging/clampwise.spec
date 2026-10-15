# PyInstaller spec: the one-folder windowed build, dist\Clampwise\Clampwise.exe.
# One folder, not one file: a one-file build unpacks itself on every start.
# Run by packaging\build.ps1, which sets CLAMPWISE_DEMOS.

import os
import sys
from pathlib import Path

PROJECT = Path(SPECPATH).parent
sys.path.insert(0, str(PROJECT / "src"))
from clampwise.analysis.registry import MODULES  # noqa: E402  (imported by name at run time)

DEMOS = Path(os.environ["CLAMPWISE_DEMOS"])

a = Analysis(
    [str(PROJECT / "packaging" / "launcher.py")],
    pathex=[str(PROJECT / "src")],
    datas=[(str(PROJECT / "src" / "clampwise" / "app" / "resources" / name), "clampwise/app/resources")
           for name in ("icon.png", "reading.md")]
          # NeuroExpress's test files: the demos and the self-test (app/main.py DEMO_FILES)
          + [(str(DEMOS / n), "demos") for n in ("Test_IV_1.abf", "Test_mEPSC_1.abf", "Test_SpT.ASC")]
          # the digitizer's helper, run as a file by the 32-bit Python (rig.recorder), so kept as one
          + [(str(PROJECT / "src" / "clampwise" / "rig" / "dd1440_helper.py"), "clampwise/rig")],
    # the analyses are imported by name; pandas loads openpyxl only when writing; the figure builder
    # draws with matplotlib's Qt canvas and exports with its PDF, SVG and PostScript backends
    hiddenimports=list(MODULES) + ["openpyxl", "pandas.io.excel._openpyxl",
                                   "matplotlib.backends.backend_qt5agg", "matplotlib.backends.backend_agg",
                                   "matplotlib.backends.backend_pdf", "matplotlib.backends.backend_svg",
                                   "matplotlib.backends.backend_ps", "statsmodels.formula.api",
                                   "scipy.cluster.hierarchy", "scipy.spatial.distance"],
    excludes=["tkinter", "matplotlib.backends.backend_tkagg", "h5py", "tables", "IPython", "pytest", "PyQt6",
              "PySide2", "PySide6",
              "PyQt5.QtWebEngineCore", "PyQt5.QtWebEngineWidgets", "PyQt5.QtQuick", "PyQt5.QtQml",
              "PyQt5.QtMultimedia", "PyQt5.QtBluetooth", "PyQt5.QtDesigner", "PyQt5.Qt3DCore"],
)
# Qt's software OpenGL (pyqtgraph draws without OpenGL) and Qt's own translations
# (the app is in English): 26 MB nothing uses.
unused = ("opengl32sw.dll", "Qt5/translations")
a.binaries = [b for b in a.binaries if not any(u in b[0].replace("\\", "/") for u in unused)]
a.datas = [d for d in a.datas if not any(u in d[0].replace("\\", "/") for u in unused)]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Clampwise", console=False, upx=False,
          icon=str(PROJECT / "packaging" / "clampwise.ico"))
coll = COLLECT(exe, a.binaries, a.datas, upx=False, name="Clampwise")
