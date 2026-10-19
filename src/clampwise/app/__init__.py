"""The window (B0): one theme module, one skeleton, seven destinations.

``python -m clampwise.app`` starts it. Nothing in here measures anything; it
asks the pipeline and the project store and shows what they say.

Before numpy and scipy load, their matrix library (OpenBLAS, one copy in each) is
held to 2 threads: with one thread a core it reserved 570 MB the window never
uses (200 MB at 2), and Clampwise's fits and filters gain nothing from more; the
batch's worker processes inherit it, so they do not compete for the cores.
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "2")
