"""Clampwise: analysis for slice electrophysiology.

Whole-cell patch clamp (current and voltage clamp) and extracellular field
recordings, from Clampex ``.abf`` files and WinLTP files. Every analysis
follows a published method and names it; flags are evidence, never an edit;
nothing is chosen silently.

Layers (each a package here; see docs/development/PLAN_CLAMPWISE.md):
``io`` reads files, ``core`` holds the data model, ``project`` keeps an
experiment's store, ``analysis`` measures (no Qt), ``export`` writes,
``protocol`` designs stimuli, ``app`` is the window.
"""

__version__ = "1.4.0"
APP_NAME = "Clampwise"
