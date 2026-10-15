"""The shared tools every analysis uses (B21).

- ``filters``: Bessel, Butterworth and Gaussian filters, the notch, with a
  spec that describes itself;
- ``baselines``: named windows, levels, drift removal;
- ``fits``: every curve fit, with confidence intervals, residuals, goodness
  of fit, the window and flags; one vs two exponentials chosen and shown;
- ``kinetics``: rise, decay start, onset and half-width, defined once;
- ``averaging``: aligned averages that say what went in;
- ``ledger``: what each analysis applied, listed with its result.

Nothing here imports Qt. Each tool is tested against analytic answers in
``tests/test_tools_*.py``.
"""

from . import averaging, baselines, filters, fits, kinetics, ledger, spikes

__all__ = ["averaging", "baselines", "filters", "fits", "kinetics", "ledger", "spikes"]
