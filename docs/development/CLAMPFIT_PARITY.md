# The Workbench against Clampfit

What a person does by hand in Clampfit (pCLAMP 10 and 11), item by item, and where the Workbench stands:
**done**, **partly** (what is missing is said) or **missing** (with why, when it is left out on purpose).

Clampfit's side of this list was written from the program's menus as the lab uses them (Analyze > Statistics,
Fit, Arithmetic, the cursors and the sweep navigation), from memory of versions 10.7 and 11.2, not from the
manual page by page: an entry that names a Clampfit option the installed version does not have, or misses one
it has, is a mistake in this file and should be corrected here.

Every Workbench item marked done is pressed by `tests/test_every_control.py` and, for its numbers, checked
against known waveforms in `tests/test_workbench.py`.

## Sweeps and the view

| Clampfit | Workbench | |
|---|---|---|
| Select sweeps (by number, by range) | Sweeps box: `1-10, 15`, `every 2nd`, `odd`, `last 5`, `all` | done |
| Next / previous sweep, one sweep shown | Sweep ▶ and ◀ (Page Down, Page Up); Alone draws it without the others; a click on a table row steps to its sweep | done |
| Sweeps overlaid | Every chosen sweep is drawn (60 at most on screen; all are measured) | done |
| Continuous (concatenated) view of the sweeps | | missing: the Workbench measures sweep by sweep |
| Show or hide signals | Channel list; Stimulus and Command drawn on the trace | done |
| Average the selected traces | Sweeps measured as: their average, their median, or the sweep closest to their mean | done |
| Exclude a sweep | Leave out a sweep: a click on it, with a typed reason kept with the measurement | done |

## Cursors

| Clampfit | Workbench | |
|---|---|---|
| Cursors 1 and 2 with time, amplitude and their differences | Cursors: two lines to drag; each one's time and value, Δt, the difference, the slope and the mean between them, on the sweep stepped to (else the average, else the first sweep) | done |
| Cursors 3 and 4 (a second pair) | | partly: one pair. Regions do the second pair's work for measuring (any number of them) |
| Set a search region from the cursors | Region to the cursors | done |
| Move a cursor or region by the keyboard | The arrow keys move the selected region 0.1 ms (Shift: 1 ms; Alt: its end alone) | done for regions; cursors move by the mouse |
| Write the cursor readings to the Results sheet | The readout can be selected and copied | partly |

## Analyze > Statistics

| Clampfit | Workbench | |
|---|---|---|
| Search regions (up to 8), a baseline region | Regions: any number, each a baseline, a measure or a fit region; each can repeat on every stimulus pulse, against its own baseline | done |
| Peak amplitude, time of peak | Peak (signed as recorded), Time of peak | done |
| Antipeak amplitude, time of antipeak | Antipeak, Time of antipeak | done |
| Peak polarity: positive, negative, absolute | Direction: up, down, auto (the larger departure) | done |
| Smoothing points for the peak | Smoothing points | done |
| Mean, standard deviation | Mean, SD (and Minimum, Maximum, Peak to peak) | done |
| Area | Area (units × ms) | done |
| Half-width | Half-width | done |
| Rise time, decay time between two percentages | Rise time (10-90% unless set), Decay time (90-10% unless set) | done |
| Maximum rise slope, maximum decay slope, and their times | Max rise slope, Max decay slope, their times | done |
| Rise slope, decay slope (a line between the two percentages) | Slope (a line through the whole region) | partly: the line is over the region, not between the rise percentages |
| | Onset latency (threshold, extrapolated foot, or maximal curvature; or set by a click) | more than Clampfit |
| | Decay tau: one exponential from the peak to the region's end, for every sweep and pulse | more than Clampfit's Statistics (there it needs a fit per trace) |
| | Value at a time, threshold crossings, spike count, instantaneous frequency | more than Clampfit's Statistics |
| Results in a sheet, one row a trace | Table: one row per sweep, region and pulse, columns with units; Summary: mean, SD, n per measure | done |
| Plot a column of the sheet | Plot tab: a measure against the sweep, the time in the recording, the command level, or another measure | done |

## Analyze > Fit

| Clampfit | Workbench | |
|---|---|---|
| Exponential, standard, 1 to 6 terms | One exponential (offset free or fixed at the baseline), two exponentials | partly: two terms at most |
| Exponential, product (rise and decay) | Synaptic (rise and decay exponentials); Exponential rise; Alpha function | done |
| Straight line, polynomial | Straight line; polynomial of 2nd and 3rd order | partly: up to 3rd order (Clampfit: 6th) |
| Gaussian, Boltzmann, Hill, power | Gaussian, Boltzmann, Hill (logistic), Power | done |
| Custom function | Typed expression in t with a, b, c, d | done |
| Lorentzian, Nernst, Goldman-Hodgkin-Katz, Michaelis-Menten, Lineweaver-Burk, probability-density fits | | missing: not measures of the lab's slice recordings |
| Fit between the cursors | Fit on a region (dragged, typed, or set to the cursors) | done |
| Fitted curve on the trace, residuals, parameters | The curve on the trace; the Fit tab: the parameters with their 95% intervals, r², the residuals | done |
| Fit every selected trace | The fit is of one trace: the first sweep, or the average, median or typical trace. Decay tau (a measure) gives a time constant for every sweep and pulse | partly |
| Compare models (F test), fix a parameter, choose the search method, seed values | | missing. The evoked analysis compares one against two exponentials by an F test |

## Analyze > Arithmetic, and the trace's processing

| Clampfit | Workbench | |
|---|---|---|
| Adjust baseline: subtract the mean of a region, or a slope | Baseline: the mean of a region taken off; a sloping line through two regions taken off | done |
| Filter: low-pass Bessel or Gaussian, high-pass, notch | Low-pass (Bessel, 4th order), High-pass (Butterworth, 2nd order), Notch (the mains and its harmonics), Gaussian smoothing | done |
| Filter: Butterworth and Chebyshev low-pass, RC, electrical interference | | partly: the types above only |
| Subtract a trace (a control, a leak) | Subtract a trace (another sweep, the average, a file's) | done |
| Scale, offset | Scale, Offset | done |
| Force values (blank a stretch) | Blank a stretch after each stimulus | partly: after a stimulus only |
| Arithmetic with a typed expression over traces | | missing |
| Normalize, time shift, data reduction (decimate), interpolation | | missing |
| The data file is changed, or saved as a new file | The recording is never changed: transforms are a list kept with the measurement, applied in order each time | by design |

## Other entries of the Analyze menu

| Clampfit | Clampwise | |
|---|---|---|
| Event Detection (template, threshold) | The Analyze screen's spontaneous-event analysis (deconvolution by a template, or a template fitted at every point; each event measured) | elsewhere in Clampwise, not in the Workbench |
| Power Spectrum | Time-frequency power and band power (Figures, heat maps) | elsewhere |
| Quick I-V graph, resistance from a step | Intrinsic properties (Analyze); the Workbench's Plot tab against the command level | elsewhere, and partly here |
| Burst analysis, histograms | Spike trains (Analyze) | elsewhere |
| Single-channel search, P(open), nonstationary fluctuation analysis, V-M analysis | | missing: outside the lab's recordings |

## What Clampfit does not do

- Every measurement is kept with how it was made (the regions, the measures, the options, the transforms, the
  clicks, the sweeps left out and why) and is made again from that description: the Results screen lists it as
  measured by hand, and nothing is a number typed from a screen.
- A measurement saved as a template runs on every file of its protocol in one step.
- Every step is undone by Ctrl+Z, kept measurements and batch runs included.
- **Send to a figure**: the traces in view (with the light's bars and scale bars) or the table become a panel of
  the figure builder.
