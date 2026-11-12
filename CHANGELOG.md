# Changelog

## 1.8.1 — 2026-11-12

- **A cleaner look, light and dark.** The light theme is white panels on a light grey page (it was warm beige);
  the dark theme a neutral charcoal (it was brown). Drop-down lists show their arrow, spin boxes have clean up and
  down arrows, check boxes a tick; the selected screen in the sidebar is marked by a green bar, without the focus
  box a click left on it. In the dark theme the blue, brown and grey series are lifted so no curve sinks into the
  plot. Every text colour is checked against the WCAG AA contrast ratios, as before.
- Fixed: a figure's logarithmic axis wrote its labels as 4 × 10⁻¹ (now 0.4, 2, 10); a panel of traces was
  wrongly warned that its rows "look like a spreadsheet's summaries" (a trace named "average"); in the Workbench a
  click on overlapping sweeps could take the sample after the click, and so another sweep.
- **The Workbench works the way Clampfit is used.**
  - **Cursors.** Tick Cursors for two lines to drag over the trace; the line above the trace reads each one's
    time and value, Δt, the difference, the slope and the mean between them, on the sweep stepped to (else the
    average, else the first sweep). **Region to the cursors** sets the selected region to span them.
  - **One sweep at a time.** Sweep ▶ and ◀ (Page Down, Page Up) step through the chosen sweeps: the one stepped
    to is drawn in bold and named, its row is selected in the table, and Alone draws it without the others. A
    click on a row of the table steps to its sweep.
  - **The arrow keys move the selected region**, 0.1 ms a press (with Shift, 1 ms; with Alt, its end alone),
    once the trace has been clicked. Each press is undone by Ctrl+Z.
  - **Decay tau** is a new measure: one exponential fitted from the peak to the region's end, for every sweep
    and every pulse (the Fit tab fits one trace).
  - **Send to a figure** puts what is on the screen into the figure builder as a new panel: the traces in view
    (with the light's bars and scale bars) or the measurement's table (the first measure against the sweep).
  - **A downward response was measured wrongly**: its Peak came out positive, and its rise time, half-width and
    decay time were left blank. The peak is now signed as recorded (as Clampfit gives it) and the kinetics are
    found whichever way the response goes.
  - `docs/development/CLAMPFIT_PARITY.md` lists Clampfit's Statistics, Fit, Arithmetic, cursors and sweep
    navigation against the Workbench: done, partly (a second pair of cursors, more than two exponential terms,
    a fit on every trace) or missing (typed arithmetic over traces, a concatenated view).
  - Checked on a copy of a real light recording, by the screen's own controls: peak, onset latency and decay
    time for both pulses of every sweep, a sweep left out with its reason, the decay fitted on the average, the
    amplitude plotted against the sweep, kept, saved as a template and run on a second file of the same
    protocol; then every step taken back by Undo.

- **The figure builder is rebuilt: the figure itself is what you work on.** It opens on an empty page with one
  button, **Add panel...**, which asks two things: what the panel shows (traces of a recording, a graph of the
  Figures screen, a measurement kept in the Workbench, a table of the project, a spreadsheet, a digitized plot,
  another saved figure) and how it is drawn (every plot as a small picture). Then:
  - **Click anything to select it** (a panel, an axis, a title, a group's points, the legend, a bracket, the
    scale bars, a stimulus bar, a text, a letter): its settings, and only its, show beside the figure. **Drag** to
    move it (or the arrow keys, half a millimetre a press), **double-click** text to rewrite it in place,
    **Delete** removes it, **right-click** lists what else can be done. Every step is undone by Ctrl+Z.
  - **The data are beside the graph.** The tab *Data and statistics* shows the table the selected panel is drawn
    from, with the columns it uses named (Y, Groups, Cell). Rows can be left out by hand (they stay in the table,
    greyed, and the legend says how many); the table can be copied or saved.
  - **The statistics are yours to choose and to overwrite.** The test is a choice (Welch's or Student's t,
    Mann-Whitney, paired t, Wilcoxon), with each group against the first or every pair; the result is drawn as
    brackets when asked. A bracket can be dragged, hidden, and its text double-clicked and replaced with the
    result of a test done elsewhere ("t(22) = 5.1, p < 0.001").
  - **A proportion has its own plot**: the share of each group in one category (connected, a success) as a bar
    with Wilson's 95% interval and "k/n" over it, groups compared by Fisher's exact test (Holm over the pairs),
    the difference with Newcombe's interval. The caption warns when the trials are windows of one recording, not
    cells.
  - **Panels go where you put them.** Templates of common figures (wide on top with three below, traces on the
    left with graphs on the right...), panels spanning columns and rows, insets, and free placement: every panel
    dragged and resized by its handles, to the millimetre, with aligning, even spacing and a snap grid.
  - **What a trace panel needs is drawn from the recording**: the stimulus as bars at its recorded time (light in
    the light's colour, a shock as a tick), the potential each trace sat at (or was held at) written at its start,
    scale bars with round sizes that can be dragged and set, the command under the traces when it steps, traces
    stacked or overlaid, spikes cut at a level. Names and labels that would overlap are moved apart.
  - **Text sizes are separate** (ticks, axis titles, panel titles, letters, annotations, legend), a group keeps
    one colour in every panel of the figure and in the next figure of the project, and the palettes are
    colour-blind safe.
  - **No journal is needed.** The page is 85, 114 or 174 mm wide, or any width typed. Choosing a journal (J
    Neurosci, eLife, Nature, Nat Neurosci, Neuron, J Physiol, Cereb Cortex) sets its column widths, font and text
    sizes, and the export then checks the figure against its rules.
  - **Export for Illustrator**: PDF, SVG and EPS keep text as text (Arial, editable) at the figure's exact size
    in millimetres; TIFF and PNG at 600 dpi or what is typed; text as shapes only when asked. The dialog first
    says what is too small to print (text under 5 pt, lines under 0.5 pt, a page wider than the journal's).
    The exported file still opens in Clampwise as the editable figure.
  - Fixed on the way: a pick list in the settings beside the figure could close the program by removing itself
    while it was still in use (choosing "No legend"); the test now presses every one. Matching stimulus bars to
    one of them lost the light's colour. The Add panel button filled the first panel again instead of adding one.
- **The Figures screen opens a light recording on what the experiment asks.** For the lab's connectivity runs
  (current clamp, light through the Polygon) the four graphs are now: **the sweeps around the light** (every
  sweep against its own level before it, their average, the light shaded), **the response windows against the
  sham windows** they are judged by (the sham windows as bars on a logarithmic axis, each response window a
  circle, the threshold, and the verdict in words, e.g. on a simulated cell: "Connected: 16 of 20 response windows
  beat the sham threshold (chance alone gives 1.0); p < 0.001; false-positive rate 5%"), **the patterns of the cell
  side by side** (each
  file's success rate with its 95% interval; one file alone: each pulse's) and **the cell's health sweep by
  sweep** (the potential before each sweep against the -55 and -85 mV limits; the input resistance is the next
  graph in the menu). Each pulse's success rate, latency and jitter are in the readout.
  - **Show lists the graphs by analysis** (Example traces, Evoked responses, Connected or not, Health sweep by
    sweep, ...) instead of one list of forty; choosing a graph that another panel shows makes the two change
    places. A graph's readout takes two lines.
  - **A recording whose kind could not be told shows its sweeps as recorded**, one graph, and a line with the
    reason (for a file stopped after its first sweep: "one sweep with one command step of -100 pA for 0.50 s: a
    step family needs at least two step sizes") and what to do.
  - **Edit opens each of these as a figure made for it**: the sweeps as a trace panel with the light's bars,
    scale bars and the potential; the sham test as two cumulative distributions with the threshold; the patterns
    as proportions with Fisher's exact test, from a table of every window of the cell's files. Every graph of
    every recording type is opened in the builder and drawn by a test.
  - The connectivity analysis keeps what these graphs need: every sham window, each pulse's jitter, each window
    as a multiple of the sham threshold, and, for a cell, one table of every window of its files (version 3:
    recordings are analyzed again).
- **Starters ("a whole figure in one step") rebuilt for the lab's recordings.**
  - **Light or stimulus responses: connected or not** (new): example traces of each pattern stacked with the
    light's bars, every response window over its file's sham threshold, the success rate with its interval and
    Fisher's exact test, the successes' latency with its jitter, and the potential before every sweep of every
    file, so a connection is never shown apart from the health of the cell it was read in.
  - **Current-clamp step family**: a cell that fires on no step used to get one trace and an f-I curve drawn as a
    flat line on an axis from -0.04 to +0.04 Hz. It now gets the whole family with the command under it, and a
    rate axis from 0 to 10 Hz with "The cell fired on no step" in the legend. A cell that fires gets the most
    hyperpolarizing step, rheobase and the step nearest twice rheobase.
  - **Evoked PSC** is no longer offered for a step family played under the light (there is no one response to
    average): it says so and names the starter to use.
- **Buttons that were broken in the Workbench, Figures and the figure builder.** Every button, checkbox, pick
  list and menu action of the three screens (and of the trace picker and the digitizer) is now pressed by a test,
  on a simulated recording of every recording type and on copies of real light and step recordings; it fails on
  a control that raises, does nothing, cannot be undone, or is greyed without saying why
  (`tests/test_every_control.py`; `docs/development/CONTROLS.md` lists each control with what was checked). What
  it found, fixed:
  - **The Workbench's table could crash the screen.** Removing the only measure region, or repeating a region on
    the pulses of a recording with no stimulus, left a table with no columns whose header still asked for the old
    ones (an IndexError on every redraw). The table now keeps one model and is told when its columns change.
  - **The table's Sweep column read one too high** (2 to n + 1), and its headings had no units; they now read
    Peak (mV), Time of peak (ms), Area (mV·ms), Command (pA). The Summary tab gave n only for measures with
    nothing measured; every line has its n.
  - **"Choose traces..." and "Raw traces from the recordings" did nothing** in the figure builder on any
    recording with more than one sweep (the dialog raised while listing the sweeps). They open.
  - **The Workbench's side panel was cut off at the right**: Remove (region), the hint lines and the template
    buttons lay past the edge. Its rows now wrap to the panel's width, and each region's name is on its own line
    of the trace (two names sat on top of each other).
  - **Keep, Delete (a kept measurement), Save as template, Delete (a template) and a template's batch run could
    not be undone.** They go through Undo like every other step.
  - **Buttons that did nothing now say why they are off**: Remove and Clear with no transform, Load and Delete
    with nothing kept, Apply here and Run with no template, Stimulus and "Repeat on each stimulus pulse" on a
    recording with no stimulus, the Groups tab's Up, Down, Hide and Show with no group selected, the trace
    picker's Up, Down and "Remove selected sweeps from it", the digitizer's Trace and Click before calibration,
    and the builder's Undo, Redo, Remove panel and right-click entries.
  - **Figures showed a graph twice.** A light-evoked recording opened on "Amplitude vs. time" in two panels (one
    per pulse position, one per sweep: they are now named for what a row is), a voltage-step family on the slope
    resistance twice, and a file with one graph to give showed it four times. A panel never repeats another's
    graph; with fewer than four to give, the unused panels give up their room and a line says why (for a file
    whose type is unknown: that only its baseline was measured, and what to do).
  - **Edit said "nothing drawn to edit"** on a graph that drew nothing (the cell-health Rs with no test step).
    The Edit button is then off, with the graph's own reason. The Show menu ticks the graph shown.
  - The Workbench measured everything twice on each change (once for the change, once for its place in Undo).
- **Three more ways to read a response's onset, decay and paired-pulse ratio, beside the lab's.** In Analyze,
  Evoked responses has **Onset (latency) read as**: the lab's rule (the first departure past twice the noise that
  holds for 0.5 ms, the default), 10% of the peak, or the foot (a line through 20% and 80% of the rise, read at
  the baseline; Jonas, Major & Sakmann 1993), each on the average beside the others whichever is chosen. The
  decay is also fitted with two exponentials, and when two fit better than one (an F test) the amplitude-weighted
  tau is given. The mean of the sweeps' own pulse 2 / pulse 1 ratios is shown beside the paired-pulse ratio of
  the means, marked as biased upward (Kim & Alger 2001): it is never the headline.
- **A success rate has its interval, a "not connected" its detection limit, and two patterns are compared with
  each other.** Connected or not gives the success rate's 95% interval (Wilson's, sound for 20 windows) and the
  smallest response the recording could have detected (the sham threshold times the noise): "not connected"
  means "none this large". Over a cell's files, each pair of patterns is compared directly (Fisher's exact test
  on successes over windows, Holm over the pairs): one pattern passing its own test and another failing is not
  a difference between them.
- **Connected or not, with the false-positive rate stated.** A connectivity experiment (the light aimed at
  labelled neighbours, one region at a time, with the patched cell opsin-negative) asks whether the recorded cell
  answers more often than chance. The new **Connected or not** analysis takes every light pulse of every sweep as
  a response window and the same window at 40 sham times in the same sweeps (clear of every stimulus, command
  change and test pulse), each window's largest departure from its own baseline in units of the sweep's noise, so
  a sweep at -70 mV and one at -30 mV count alike. The pooled windows are tested against the pooled shams
  (p from 2000 draws); a window is a success when it beats the 95% quantile of the shams, so the false-positive
  rate (5% of empty windows) and the successes chance alone would give are printed beside the success rate. It
  gives the successes' latency and jitter (the lab's onset: past twice the noise, then growing to the peak),
  the amplitude with failures (efficacy) and without (potency), a spike in a window as a success, and which
  pattern lit the cell (the light's log, else the file's name: Cell_Only, ROI1, Full_Field). Over a cell's files,
  **Connected or not, over a cell's files** puts the patterns side by side.
- **A direct photocurrent is told from a synaptic response.** When the response begins sooner after the light than
  a synapse can answer (under 1.5 ms), is still at half its peak or more when the light goes off and peaks there
  or within 3 ms after, the recorded cell is answering the light itself: it expresses the opsin, or its membrane
  is lit (Turesson et al. 2013's criteria). It is flagged as such, with its reversal potential when a step family
  gives one, the 1.5 ms rule keeps its latency unmeasured, and Connected or
  not says the file cannot test a connection. The rule that a "synaptic" onset under 1.5 ms is not measured
  stays.
- **The light's own mark on the trace is recognised and blanked.** A square deflection held while the light is on
  and gone when it goes off, or a sharp jump at its onset and another at its end, the same in every sweep
  whatever the cell is doing (a photoelectric potential at the electrode, pickup from the LED driver), is flagged
  as a light artifact; the light's pulse plus 0.5 ms is blanked and the response measured after it. A response
  already under way where the blank ends is said to have its onset hidden, not read at the blank's end.
- **A mapping protocol with no record of its regions is not averaged.** A grid map or a connectivity run lights a
  different region on each sweep. When no Polygon log covers the recording (or none is sure), Clampwise flags
  "region unknown", gives no averaged number for the recording (each sweep's own response stays in the sweeps
  table) and says what to do: record the light's log with the recording, or type each sweep's region.
- **The lab's photostim pulses on AM_Stim are light.** AM_Stim is Clampex's name for a stimulator input; the lab's
  Polygon rig records the LED's command on it, and the protocol's name (Photostim, ChroME, Chrimson) says what was
  done. Such pulses were taken as electrical, so none of the light rules (the 1.5 ms minimum latency, the
  pattern lit) applied to them. A rig profile naming AM_Stim as a light input still comes first.
- **The Workbench: measure anything by hand, as in Clampfit, and keep how it was measured.** A new screen
  (Ctrl+8). The chosen sweeps (typed: 1-10, 15; every 2nd; odd; last 5) overlaid, averaged, as their median or
  the one nearest their mean, on any channel, with the stimulus and the command marked. Named regions dragged on
  the trace (a baseline, a measure, a fit), each able to repeat relative to every stimulus pulse. Clampfit's
  Statistics in a region: mean, SD, minimum, maximum, peak and antipeak (direction and smoothing points as
  options), peak to peak, times of peak, area, half-width, rise and decay times with set percentages, the
  steepest rise and fall and their times, the onset latency (by threshold with the lab's 2 x noise, by the
  extrapolated foot, or by maximal curvature), a slope, the value at a time, threshold crossings, spike count and
  instantaneous frequency. Fits on a region, drawn with their residuals and 95% intervals: one or two
  exponentials (offset free or fixed), an exponential rise, the synaptic waveform, alpha, a line, polynomials,
  power, Gaussian, Boltzmann, Hill, or a typed expression in t with a, b, c, d. A stack of transforms that
  never touch the raw data: a baseline by region or by a sloping line, Bessel low-pass, high-pass, notch,
  Gaussian smoothing, blanking after each stimulus, subtracting a trace (the sweeps' average, one sweep, or
  another file's average: trace arithmetic between sweeps and files), scaling, an offset. The table of every
  measure per sweep and region, and a plot of any measure against sweep, time, command level or another
  measure. Clicks set an onset, a peak or a spike threshold on a sweep; a click leaves a sweep out, with a
  reason. Every change goes through Undo (Ctrl+Z). **Keep** saves the measurement with the recording: Results
  shows it under "Measured by hand" with its mean, SD, median and n per measure and the full description of how
  it was made, and it is made again from that description. **Save as template** keeps the regions, measures,
  fit and transforms for every file of a protocol, and **Run on every file of its protocol** measures and keeps
  them all, as a Clampfit batch.
- **The cell's class, by the lab's rule.** Intrinsic properties names the cell an interneuron when its first
  spike at rheobase is narrower than 0.6 ms at half-amplitude from the threshold (Ahmed 2021 after Woodruff &
  Sah 2007), a principal cell otherwise, with the half-width it rests on; the cut-off is a setting. The pattern
  lit in Connected or not is read from the condition typed in Files when there is one, before the file's name.
- **The lab's definitions are settings, the Pare lab's first.** In Analyze, Intrinsic properties now has
  **Input resistance** (the lab's: a line over the hyperpolarizing steps, the linear portion of the V-I;
  NeuroExpress's line read at 0 pA; IPFX's slope over -100 to 0 pA), **Sag** (Ahmed 2021: the most negative Vm in
  the last 100 ms over that in the first 100 ms of the -200 pA step; or IPFX's), **AHP** (Ahmed 2021: 1 ms before
  the peak minus the trough within 10 ms; or IPFX's), **Maximal firing rate** (Ahmed 2021: 1 / mean interspike
  interval at 200 pA; or IPFX's), **Spike threshold** (IPFX's 5% of the steepest rise, the 20 mV/ms crossing, or
  the kink) and **Spike half-width** (Ahmed 2021: at half the amplitude from the threshold; or IPFX's from the
  fast trough). The headline number follows the setting, every other definition's number is reported beside it,
  and the choice is written with the result. New beside them: Amir 2011's adaptation (% ISI increase on the
  first step with 6-10 spikes), and the liquid junction potential typed for a recording shown beside the
  voltages (never subtracted: the lab has not corrected it since 2014). On real step families the lab's
  half-width reads about a quarter narrower than IPFX's: the likely reason for the difference with Ahmed 2021
  in the corpus check. Evoked responses: the onset is now the
  lab's (a departure past **twice** the noise, sustained; it was 3 times), and each pulse gets the **time to half
  peak** (Turesson 2013), the **80-20% decay time** (Ahmed & Pare 2023) and the **initial slope** of the rise
  (Popescu 2007, the LTP readout).
- **Light inside a step family stays out of the step's own measures.** The lab's photostim protocols light the
  cell during every step. The 100 ms after each pulse is now kept out of the step's spike counts (a light-evoked
  spike gave a false rheobase), its peak (a light response was read as the sag) and its steady state; the
  stimulus's spikes are counted apart.
- **Health, sweep by sweep.** In current clamp, each sweep's membrane potential before the protocol, its own input
  resistance from the test pulse (the lab's photostim protocols carry a -20 pA, 100 ms step at 766 ms; without
  one, the sweep's own step gives it from its steady state) and whether the cell fired on its own, each judged
  and said in words ("depolarized: -26 mV before the protocol, above -55 mV"; "input resistance moved +35% from
  the first sweeps' 110 MOhm"). A file is flagged unhealthy when a quarter of its sweeps are: a cell resting
  far above -55 mV is called unhealthy in every sweep, and its numbers are said not to be a healthy cell's.

- **Light (or a stimulus) inside a step family is measured step by step, with the response's reversal potential.**
  The lab's photostim protocols play the light inside each step of a -200 to +160 pA family. Clampwise used to
  average the responses of every step together, so a PSP that is 4 mV at -56 mV and 0.6 mV at -13 mV came out as
  one meaningless number, and often as "no response". Now each step level is measured on its own (the table
  **by_command**: the potential before the light, the response at its own peak and at one fixed latency, latency,
  success, spike probability, PPR). The recording's numbers are those of the step at the holding (0 pA). A straight
  line through the response at the fixed latency against the potential gives the **reversal potential** (the lab's
  fixed-latency method; liquid junction potential not corrected), said as extrapolated when it lies beyond the
  potentials recorded. In voltage clamp the command's own level is the potential.
- **An intensity series is measured level by level, with its threshold.** When the light command (AM_Stim) or
  another recorded stimulus grows from sweep to sweep (ChroME_threshold_intensity_CC), each size is measured on its
  own (the table **by_intensity**). The recording's numbers are the strongest stimulus's, and the **stimulus
  threshold** is the weakest level from which every stronger one gives a time-locked response above the detection
  level.
- **"Full field" in a file's name no longer makes it a field recording.** Full_Field_PPulse_Good_Response.abf (a
  whole-cell recording lit by the Polygon's full field) was analysed as an fEPSP.
- **A current-clamp protocol played in voltage clamp is said, not misread.** The protocol's steps in pA were read as
  mV steps: a membrane test of 1638 MOhm and an I-V against "-100 mV" that was in fact a few mV. Clampwise now says
  the voltage the cell was stepped to is not known, reads no steps from it, and keeps only what does not need them
  (an evoked response, with no reversal potential). The same for a voltage-clamp protocol played in current clamp.

- **The acquisition's trace can keep a fixed scale.** Beside the trace, the up and down arrows set its vertical
  scale about 0 a step at a time (+-500 pA, +-1000, +-1500...); **Auto** follows the trace again, as before. Fixed,
  a seal's flat trace stays flat instead of being blown up to fill the plot, so breaking in shows plainly. It
  applies to the protocol's traces and the membrane test's pulse; the step (500 pA, 50 mV at first) can be changed,
  and each unit's scale is kept between sessions.

- **With two screens, the amplifier goes with the acquisition, compact as Commander's own window.** In Views > Two
  screens, the acquisition's window has the amplifier beside the trace in one panel: the channel's V and I meters,
  VC / I=0 / IC, then every control on one **Clamp** page (holding, Zap and Buzz, seal test, pipette offset, Cp fast
  and slow, whole cell, Rs compensation, leak subtraction; in current clamp the bridge) with the outputs on a second
  tab. Values that belong together share a line (Cp fast and its tau, Cm and Rs), each Auto at its line's end. The
  window fits any screen: the lines under the trace go one under the other when it is narrow, and the amplifier
  panel scrolls rather than being cut off, so every Auto stays in reach. Two screens is now kept when Clampwise
  closes (it was switched off each time).

- **The membrane test follows the amplifier while it runs.** It read the amplifier's output scale once, when it
  started: a gain or feedback resistor changed on Commander during the test went on being read with the old one, and
  Rt came out wrong by the same factor (x5 to x1: a 7.5 MOhm pipette read 37.5). The telegraph is now read again
  twice a second, as Clampex follows it: a new scale is used at once (the pulses before it left out of the average),
  and the status line says the scale in use (0.05 V/nA). A new mode or command sensitivity ends the test, saying
  why. `tools/rigtest/membrane_test_check.py` measures, with the secondary output, the step that reached the pipette.

- **The patterned light can step on with each sweep, with no trigger cable.** **Next step with each sweep** (the
  patterned light's More options) shows the first step before a run's first sweep, and the next one in the gap
  after each sweep, round again after the last. Grid maps and cell-by-cell connectivity protocols then light one
  region a sweep without wiring a digital output to the Polygon's Trig In.

- **A grid's cells can be sized in um.** The Grid's **Each cell** sets each cell's width and height (34 x 24 um,
  say); the grid is centred on the picture, where the recorded cell is put. Left at 0 the grid tiles the whole
  picture, as before.

- **Photostimulation protocols for single-cell mapping.** `tools/make_photostim_protocols.py` writes, from a rig's
  own photostim protocol, four current-clamp protocols: the light's level or its pulse length stepped sweep by sweep
  (to find the cell's spike threshold), a 49-sweep grid map, and a 50-sweep connectivity run; a 5 ms pulse at 310 ms
  and a -20 pA step at the end of each sweep for the input resistance.

- **What a protocol sends can be hidden.** Acquisition's **Commands** lists each output the chosen protocol sends
  (the amplifier's command, a light, digital outputs), each ticked when drawn under the inputs. Unticked, its plot
  is gone and the inputs take its room; the protocol's description still names it, as hidden. Kept between
  sessions; changed during a run, from the next run on.

- **Protocol buttons, as Clampex's sequencing keys.** A row of buttons under Acquisition's transport, set up with
  **Buttons...**: each one gets a protocol, a name, and what pressing it does (Record, Play, or choose the protocol
  only). Ctrl+1 to Ctrl+8 press them from the keyboard. A membrane test running is stopped first; a run going is
  never interrupted.

- **Two screens.** Live's **Views** menu has **Two screens**: the acquisition (the sweeps and the membrane test)
  fills the other screen, the camera takes the room it leaves in Live. Kept, and set up again at the next start;
  closing the acquisition's window puts it back.

- **The camera catches light pulses.** A photostimulation pulse of a few ms mostly fell between two pictures, so
  the camera seldom showed it. **Catch light pulses** (in the Camera section, off by default) sets the exposure to
  fill each frame's time (15 ms at the camera's 62.5 frames a second), raises the gain when the picture is dark
  between pulses, and looks at every frame, not only those drawn. The picture that caught a pulse stays on screen
  for a second, where the light landed in colour, with a count. Checked at the rig with the 375/475 ms photostim
  protocol: 22 of 22 pulses caught, none falsely, each on screen 4 to 19 ms after it fired, inside the region drawn.

- **Patterned-light calibrations can be saved and called back for each objective.** Calibrate... starts with
  **Saved calibrations**. **Save the one in use** writes the calibration to a file of its own, named by its objective
  and when it was measured, in `%LOCALAPPDATA%\Clampwise\polygon calibrations`. Pick one from the list (the
  objective in use's first, each with its date and how close its check came) and **Use this one**: it is kept again
  for its objective, and used at once when that is the objective in use. No camera or light is needed to call one
  back. A saved calibration holds only while neither the camera nor the patterned light has moved since.

- **The patterned light's calibration is there again after a restart.** It was saved, but at start Clampwise looked
  for it before the camera's panel was in the window, found nowhere to read it from, and said "Not calibrated". The
  next calibration then saved over the ones kept for other objectives. Now each objective's calibration is read at
  start and kept until it is measured again.

- **A calibration made after placing the field by hand is called a calibration.** It kept the "Placed by hand"
  label. A calibration stopped before it finishes now leaves the one kept in use, unchanged.

- **Acquisition shows everything a protocol sends.** The protocol was drawn as one output only, the one Clampex
  calls active. In the photostim protocols that is the LED's, so the current step injected on the amplifier's
  command never showed. Now every output a protocol changes is drawn: the amplifier's command in its present unit,
  the LED, any other analog output, and the digital outputs, each named and in its unit as the protocol has it.
  Before a run they are drawn for every sweep. While it plays or records they sit, smaller, under the recorded
  inputs on the same time axis, the sweep under way strong and the one before it faint.

- **The patterned light's field can be placed by hand.** Under Calibrate..., **Place by hand** lights a target on
  the mirrors (the field's edge, a ring of spots, and an L at its centre whose arms show which way round the field
  lies) and draws the same target in orange where Clampwise puts it. Drag the orange frame on the camera's picture
  until the two sit on each other: inside it to move, the round handle to turn, the square ones to size. Buttons
  move, turn and size it in small steps, turn it a quarter, or mirror it. It starts from the calibration in use, so
  a calibration that has drifted since is corrected in seconds; with none, it starts from the middle of the picture.
  **Keep** replaces that objective's calibration (noted as placed by hand) and **Cancel** puts the old one back.
- **The calibration fits the field to every spot it finds, not just the first three.** Three spots fixed the map
  exactly, so any error in finding one of them went straight into it, and runs at the rig came out 5.7 to 7 um off.
  The map is now fitted to the first spots and the check grid together. Each spot is then checked against a map
  fitted to all the others, so no spot passes by having been part of the fit. A spot found far further off than
  the rest (one cut by the picture's edge, or a speck beside it) is left out, and the result says so. At the rig
  (20x), this brought the check from 5.7–7.2 um to 2.3 um.
- **Step on each trigger was checked on the Polygon itself.** Two patterns were loaded into it and set to step on
  their own every 0.5 s, and the camera saw them alternate every 0.5 s: patterns are stored in the device, the shown
  time is in microseconds, and the first pattern is shown as soon as they are loaded. Once stored, each TTL at the
  Polygon's Trig In shows the next one with no PC in the way (Mightex gives 50 us from the pulse). On the rig no
  Digidata digital output reaches Trig In yet: a BNC from one to it is needed, and a digital pulse in the protocol
  shortly before each LED pulse. `tools/rigtest/polygon_trigger_timing.py` finds which output is wired (`--find`)
  and measures how soon after the pulse the new pattern takes the light (`--timing`).

## 1.8.0 — 2026-10-23

- **The patterned light now shows the pattern drawn.** Clampwise sent each pattern as a byte a mirror, but the
  Polygon reads a bit a mirror, eight to a byte. So only the first eighth of each image reached it, and every pattern
  came out as fine stripes over the whole field: an even glow, no spot to find, and the calibration failing. Measured
  on the rig with a mirror (the first eighth of an image carried 97% of the light, the rest none) and fixed: patterns
  are packed as the device reads them, for single patterns and trigger sequences alike.
- **The camera's controls are in Live's side panel.** Exposure, gain, frames averaged, binning, the objective,
  Auto exposure, Snapshot, Adjust... and the histogram now sit in a **Camera** section beside the views. The picture
  fills the camera's view alone, and the controls stay in reach while it is enlarged or in a window of its own on
  another screen.
- **The patterned light's controls are right under the camera's**, in a **Patterned light** section, so the light
  and the picture it is drawn on are set up together. Its view in the grid says where its controls went.
- **The side panel folds.** Hardware, Camera, Patterned light, Amplifier, Amplifier controls, Cell health and Recorded
  are sections: a click on a header folds or unfolds it, and a right-click folds the others or moves it. The folds
  and the order are kept for next time. A folded header still says how its section stands: the camera's exposure,
  frame rate and objective; what the patterned light shows, or that it is not calibrated at this objective; the
  amplifier's mode and holding; and the cell's readings outside their limits, so a fold never hides a warning.
- **Use here, Let go and the light's Off are on the section headers**, usable with the section folded.
- **Each patterned-light calibration says when it was measured.** From two days old, it says to check it again
  before relying on regions of one cell: the map holds only while the camera and the Polygon stay where they were.
- The side panel is never narrower than its controls need, so nothing is cut off at its right edge.
- **The patterned light says what its mirrors do.** A line at the top of its section always reads **Mirrors ON**
  (and which steps) or **Mirrors off**, beside the two buttons that change it: **Mirrors on** (was Project) and
  **All mirrors off** (was Off). Under it, **1 · Set up** shows whether the camera is held and whether the
  objective in use is calibrated (when, how closely), each with its button; then **2 · Pattern** and **3 · Steps**.
  Rarely changed options (Next every, after each recording, Trig In stepping, outlines, lit by 0) are under
  **More options**. Use here on the patterned light takes the camera too.
- **Calibrate... is a guided window.** It covers what to put under the objective (a mirror or a fluorescent
  slide) and how the LED is on: **Automatic** works out steady or pulsed by itself. Then Start and Stop, a progress
  bar, and the outcome in a word. A new half-field test picture catches a camera that is out of focus where the
  pattern forms, and says so, instead of only "no spots found".
- **The camera is simple and fast.** The picture is shown raw and grey, exactly as the camera gives it, nothing
  computed on it. Display processing was costing frames: on the rig, saved display settings (auto levels and a colour
  balance on the grey camera) held it to 1 to 1.5 frames a second; raw, it takes 61 at 10 ms and 27 at 37 ms. The
  controls left:
  - The frame rate the camera gives now, live, and a **Frame rate** box: type the pictures a second you want, and
    the exposure is kept short enough, with the gain raised to keep the picture as bright.
  - Exposure, gain, and **Auto exposure**, which now follows the light twice a second, as the camera's own program
    does: more light, a shorter exposure, more frames a second. It stays within the frame rate typed, and holds still
    during the patterned light's calibration.
  - Snapshot (the picture as taken), Measure, the slice **Compass** (switched beside Measure), Fit, the Objective
    slider, and **Scale & slice...** (the length calibration, and the slice's plane and orientation for the compass).
  - **Picture display**, folded: brightness, contrast and gamma, Auto and Raw. They change only how the screen draws
    the picture, so they cost the camera no frames, and snapshots stay raw.
  - **Full screen**, the picture alone (Esc leaves), and **Picture only** for the docked view: no title, status line
    or border. The picture itself has no frame or margin, and no empty band under it any more; the **Margin** round
    it is chosen, the same on every side (0: the picture alone, edge to edge).
- **Removed from the camera:** the histogram; Adjust... (sharpness, clarity, saturation, colour balance, auto
  levels, presets, colour scales, clipped pixels in red, crosshair);
  background and dark-frame corrections; the fluorescence overlay; frame averaging; binning. A colour camera's picture is shown grey.
- Hardware's Detect button is gone (it looks every 5 s anyway), as are the patterned light's second Off and its
  "Nothing shown" line.
- **The patterned light's window shows its field as the mirrors show it**: what is lit now in white, the steps
  outlined and numbered, a click to choose one, a double-click to light it, with Mirrors ON/off and the two buttons
  above. It works without the calibration.
- **The amplifier's meters are steady and show the real zero.** Each meter is read ten times a tenth of a second
  apart and averaged, and the large V and I show the mean of the last 2 seconds. Before, one instant's reading jumped
  by 10 to 20 pA with an open pipette, so a zeroed pipette never looked zero. The mean begins again on any change at
  the amplifier (an Auto pipette offset, the mode, the holding, the seal test), so the new zero shows at once. The
  **pipette offset** is shown among the amplifier's readings.
- **The membrane test's plot marks 0 and the holding.** A dashed 0 line and a line at the holding measured: an open,
  zeroed pipette's square wave runs as far below 0 as above it (-500 and +500 pA for a 10 mV step through 10 MOhm),
  which the axis, fitted to the trace, could make look like an offset.
- **Time never starts before 0** on the membrane test's plots and the acquisition's scope, however they are zoomed
  or dragged. The current or voltage axis stays free.
- **Live laid out by use.** The camera on the left, the whole height and half the width; the acquisition at the
  top right, the amplifier's and the patterned light's windows under it. The side panel in the order things are
  used: Camera, Cell health, Amplifier, Amplifier controls (open), then Patterned light, Hardware and Recorded.
- **Compact controls, as on the amplifier's own panel.** The amplifier's controls as Commander's: the mode as three
  buttons, the groups in the order a cell is patched, each switch beside its value and each Auto on its value's
  line. The acquisition's controls in two lines under the scope instead of five. Live's buttons and boxes closer
  together.
- **Function keys in Live:** F2 snapshot, F5 Play, F6 Record, F7 Pause, F8 Stop, F9 membrane test, F11 camera full
  screen, F12 all mirrors off. None for the mode or Zap.
- **Sturdier with the hardware.** The patterned light is put off and let go even when Python is already closing
  (it once stayed lit after a script ended); a camera that refuses to open (Lumenera's error 10 after a program ended
  without letting it go) is asked once more.
- A window that keeps nothing (the tests, the rig scripts) no longer reads or writes the person's own settings.
- Dead code removed: the camera's old display pipeline (presets, corrections, overlay, colour tables, binning).
- `tools/rigtest/polygon_calibrate.py`: calibrates at the rig from a console (`--keep` keeps it), with
  `--focus` (a checkerboard and a sharpness reading to focus by) and `--spots-only`.

## 1.7.1 — 2026-10-20

- **When the camera cannot be taken, the camera's view says why.** In 1.7.0, a camera that Use here could not take
  went back to "Not open", and the reason was shown only for a moment in the status bar. If the camera's library
  failed outright, no reason was given at all. The reason now stays under "Not open" until the camera is taken,
  and any failure is written to `clampwise-error.log`.
- A pylon library that cannot be loaded, or that lacks a call Clampwise needs, is named with what is wrong.

## 1.7.0 — 2026-10-20

- **Basler cameras.** Live's camera view takes a Basler camera (an ace such as the acA1920-40um, USB 3.0 or GigE)
  as it takes the Lumenera INFINITY: **Use here** with pylon Viewer closed, the live picture at the camera's full
  frame rate, exposure (ms) and gain (dB, as Basler counts it), frames averaged, every adjustment, snapshot, scale
  bar and the patterned light's calibration. It goes through Basler's own pylon library, installed with pylon
  Viewer (nothing is added to Clampwise). Which camera is plugged in decides which library is used. What Clampwise
  changes on the camera to take it (8-bit frames, its automatic exposure and gain off, its trigger off) is put back
  when it lets go, so pylon Viewer finds the camera as it left it.
- The camera's pixel is filled in from its model (5.86 um on the acA1920-40), so the scale bar is nominal until you
  calibrate. Hardware detection finds a Basler camera, and pylon Viewer counts as the camera's program.
- **Clampwise.exe --camera-check** takes the camera, ten pictures from it and lets it go, and writes each step to
  `camera-check.txt` in `%LOCALAPPDATA%\Clampwise`: the file to send if a camera will not start.
- A gain of 0 is set when asked (it was taken as "unchanged", and Basler cameras count their gain from 0 dB).

## 1.6.1 — 2026-10-19

- **The patterned light lights only the steps you chose.** In 1.6.0, after several steps were chosen with
  Ctrl+click, a plain click, a double-click, Previous, Next or Next every kept the earlier ones chosen and lit them
  too (cells 1 and 57 chosen, Next lit 1 and 58). Each now chooses that step alone; Ctrl+click still adds.
- **The steps stay drawn while the camera's picture is enlarged** (the patterned light's view hidden) as long as
  the patterned light is held, and clicking a cell on the picture still chooses it.
- **Fit fits the camera's picture**, not the patterned light's field when it is larger than the picture.
- A grid's numbers are drawn for every cell only up to 25 cells; beyond that, for the chosen and lit ones (100
  numbers ran together). The field's edge reads on light and dark backgrounds; the patterned light's temperature
  is given in degrees C.

## 1.6.0 — 2026-10-19

- **The Polygon, from Clampwise.** Live's patterned light view, **Use here**, sets where the light goes on the
  slice and holds it, while Clampex switches the light as your protocol says (PolyScan closed: one program holds
  the Polygon).
  Patterns: a full field, a spot, a numbered grid (1 at the top left along each row, as PolyScan numbers its
  grids), a line of spots, an image file, PolyScan's own sequences (both its earlier and later files: grids, cells,
  rectangles, ellipses and drawn polygons), or shapes drawn on the camera's picture. Its steps are shown one at a
  time (chosen, previous and next, or the next every few seconds), each written to polygon_log.csv in the cell's
  folder. **Test and calibrate** learns which byte lights a mirror and where the Polygon's field lies on the
  camera's picture, from the picture itself; the steps' outlines are then drawn over it. Spots are round on the
  slice: the mirrors are diamonds in rows half a mirror apart (a 16:10 field).
  PolyScan4's driver is used (the newest installed; PolyScan2's older one finds no Polygon1000), and every
  call to it runs on a thread of its own: finding the Polygon takes it about 25 s, shown with a bar and the
  seconds while the window goes on. **Lit by 0** swaps lit and unlit by hand. The light test and calibration work
  with the light in pulses too (the protocol playing): each test picture waits for a pulse and reads the frame it
  lit. A region drawn over a neuron is lit alone, and dragged onto another cell it is shown there at once.
- **The patterned light's calibration, checked and kept per objective.** A map is kept for each objective (and
  camera adapter) and refused at another; after the fit a check grid of 8 spots it did not use reports how far each
  lands, in um (target under 5 um), drawn on the picture.
- **One cell of a grid, or several, lit alone.** A grid can tile the camera's picture rather than the whole field
  (the default once calibrated), so a field larger than the picture still gives cells you can see. Choose cells in
  the list or by clicking them on the picture (Ctrl+click for more); Project lights them together and nothing else,
  logged as one step (`1 + 57`) with each cell's shape. The picture shows the chosen steps (yellow), what the
  patterned light shows now (filled blue), the field's edge (dashed) and an image file's mirrors. The calibration
  finds a field larger than the picture (spots nearer the centre when the outer ones fall off it; the check grid
  laid over the picture).
- **Regions for one cell, and their controls.** Click the cell's centre to place a round region of a diameter typed
  in um; name it; add a control region of the same size a set distance away (for light scattered onto the cell);
  keep the regions with the cell and draw them again later.
- **The pattern belongs with the recording.** Each step shown is logged to the millisecond with its shapes (field,
  picture pixels and um), area in um^2, fraction of the field lit, objective and byte. A recording made in
  Clampwise gets `<file>.light.json`: each sweep's and each light pulse's pattern and how sure the match is. The
  evoked analysis measures each region's sweeps on their own (table `by_pattern`) and notes what differs between a
  full field and a region (the same irradiance at one LED command; the power and the scatter go with the area).
- **Nothing is left lit.** Off after each recording (on by default), and Off before Let go and when Clampwise
  closes (the Polygon keeps showing its last image otherwise).
- **Per-pulse patterns on the Polygon's own trigger.** Step on each trigger loads every step into the Polygon and
  each TTL at its Trig In shows the next: pulse 1 on one region, pulse 2 on another; the analysis labels each pulse
  with its region. Needs a digital output wired to Trig In.
- **Lighter on the PC.** Measured with `tools/rigtest/profile_live.py` on the rig's PC: Live idle 16.5% of one
  core before, 4.7% now (badges no longer restyled when unchanged; the camera's pictures read once, not every
  2 s; the data folder listed without a call per file; the telegraph once a second; no 20 ms polling between
  looks); the window's longest pause 50-90 ms before, under 50 ms now. The camera at its full 60 frames a second:
  28% of one core (its thread half what it was: frame memory no longer zeroed), 30 drawn a second (the eye sees
  no more), every frame still taken. Memory reserved 611 MB before, 225 MB now (numpy's and scipy's matrix
  libraries held to 2 threads each). On 2 cores at low priority every budget holds in low-power mode.
- **Low-power mode** (Settings, Performance): the picture drawn 15 times a second, every 2nd pixel while shown
  smaller, its histogram once a second, the rig's programs looked at once a second. On by default on a PC with
  fewer than 4 cores or under 8 GB. The frame rate's tooltip says what sets it (the exposure, the camera's
  readout, or the drawing).
- **Never a frozen window**: opening a large project (5,000 recordings take 2 s to find) and Send to Excel / CSV
  run in the background, the status line counting the seconds.
- **Live's views at 1366 x 768**: a narrow view keeps its name and Use here / Let go readable (its status word,
  then enlarge and hide, give way); the patterned light's controls fit a narrow window.
- **`tools/rigtest/`**: scripts that drive Clampwise from source as a person would (the window, the camera, the
  patterned light), take screenshots and print what they measured: `profile_live.py` (CPU per thread, memory,
  the event loop's pauses, frames; `--cores 2 --low-priority` for a cheaper PC), `camera_fps.py`,
  `small_screen.py`, and, only with `--at-the-rig`, `polygon_light_test.py` and `polygon_roi.py`. Each ends with
  nothing lit and every device let go.
- **An adjustable histogram under the camera's picture**, set by hand as ImageJ's Brightness & Contrast: the
  picture's histogram (log counts, so the few bright pixels of dim fluorescence show) with a band whose edges are
  the grey levels shown black and white (contrast), the band itself dragged to move both (brightness), and a line
  inside it for the level shown mid-grey (gamma: dim parts lifted); the curve over it is how each level is shown.
  Each can be typed; Auto fits black and white to the picture, Reset shows every level as taken. The buttons
  under the picture stay in view however small the controls are made. They are kept
  with the other adjustments (presets, the snapshot as shown); the picture saved as taken never changes.
- **Binning, when chosen (off by default).** Binning 2 x 2 or 4 x 4 adds that many pixels into one: a dim
  picture (a thick slice) 4 or 16 times brighter, so a quarter or a sixteenth of the exposure gives the same picture
  and that many more pictures a second (on the rig's camera: 100 ms, 10 a second, becomes 25 ms and about 40 a
  second), at half or a quarter of the resolution each way. Adding them costs no frame rate (62 a second, the
  camera's most, at 10 ms). The scale bar, measuring, snapshots and the Polygon's outlines follow it.
- **Auto exposure no longer freezes Clampwise in the dark.** On a dark picture (the light off, a filter cube being
  changed) it multiplied the exposure up to the camera's maximum (88 s a frame on the rig's camera) and every button
  then waited for those frames. It now changes the exposure at most 4 times a step, goes no longer than 1 s, stops
  after about 3 s, keeps the picture moving meanwhile, and says when the picture is still dark ("is the light
  on?"); a longer exposure can still be typed.
- **Buttons answer within one frame while averaging.** An averaged picture is taken a frame at a time, so a change
  or a button pressed meanwhile is served at once instead of after all the frames.
- **Every pop-up window can be sized freely.** Adjust..., the protocol editor, the trace picker, the naming rules,
  the finder, the digitizer and the figure builder's small windows have a grip in their corner, scroll their content
  when made smaller than it, and open again at the size and place they were left.
- **The camera's controls get the room you give them.** Under the picture, the controls (exposure, gain, average,
  objective, the buttons) sit below a divider to drag; they scroll when small, and the split is kept, in Live and in
  the camera's own window.

## 1.5.0 — 2026-10-18

- **The compass follows the slice's plane.** Horizontal (rostral-caudal and medial-lateral in the picture), coronal
  (dorsal-ventral and medial-lateral) or sagittal (dorsal-ventral and rostral-caudal); the axis cut across is named
  beside it. Rostral-caudal can be named anterior-posterior, and every direction by its letter (R C, or A P; D V;
  M L). Turned by the plane's first direction (rostral in a horizontal slice, dorsal in the others). Snapshots
  record the plane.

## 1.4.0 — 2026-10-16

- **A compass for the slice.** Rostral, caudal, medial and lateral marked over the camera's picture (Adjust...,
  Compass), turned to where rostral points on the screen (degrees clockwise from up; a quarter turn either way with
  one click) with medial on the side chosen; the rostral arm drawn thicker. Kept between sessions; snapshots record
  the orientation.

## 1.3.0 — 2026-10-16

- **Each pixel's size in um, without a micrometer.** The camera's own pixel (4.5 um on the INFINITY8-3M, filled in
  from its model) over the objective's and the camera adapter's magnifications gives every objective's scale at
  once (20x with a 1x adapter: 0.225 um a pixel), shown beside the Objective slider and used by Measure and the scale
  bar; marked nominal until a stage micrometer is measured, which then takes over. The adapter is set under Adjust,
  Length calibration.

## 1.2.0 — 2026-10-16

- **The camera at its full frame rate.** The live picture is now the camera's video stream (27 pictures a second at
  a 37 ms exposure, as Infinity Analyze shows; it was 1). It used single "fast frames" (160 ms each), a picture
  statistic that sorted every pixel on each frame (217 ms; now a count of the 256 levels, twice a second), colour
  processing on a monochrome camera's frames, and a fixed redraw gate that skipped every other frame. A monochrome
  camera's frames stay one grey channel; the picture's adjustments are a lookup of each level; every frame the screen
  can draw is drawn, never a queue behind the picture.
- **A scale bar for each objective.** An Objective slider on the camera's panel (2.5x to 63x; the list can be set):
  calibrate once at any objective (type the um a pixel, or drag the Measure line along a stage micrometer), and every
  other objective's scale follows by the ratio of magnifications, marked estimated until it is calibrated itself.
  Snapshots record the objective and its scale.
- **The rig's hardware found by itself.** Live's Hardware section lists the Digidata 1440A, the MultiClamp 700B
  (with its serial and each channel's mode from Commander's telegraph), the Lumenera camera (serial and model) and
  the Polygon's IOBox, from Windows' device list without opening any of them, and says who is using each.

## 1.1.0 — 2026-10-15

- **Updates from inside Clampwise.** When it starts, Clampwise asks GitHub for the newest release (Settings, Updates;
  the check can be turned off). A newer version is offered, never installed unasked: **Update to** downloads it,
  checks it against the SHA-256 GitHub lists, closes Clampwise, replaces the program's folder (the old copy kept
  until the new one is in place) and opens the new copy. Projects and settings are kept; no update while a recording
  or membrane test runs. `Clampwise.exe --update` does the same without a window.
- `packaging\release.ps1` publishes a new version in one step (version, build and checks, tag, release).

Copies of 1.0.0 have no updater: download this version once by hand.

## 1.0.0 — 2026-10-15

The first public release of the whole suite, under the GNU General Public License, version 3.

**Analysis**
- Readers for Axon ABF (ABF1, ABF2) and ATF, WinLTP sweep files and spike-time files; headers first, samples on
  demand, so large folders open quickly.
- Each recording's type found from its content, with the reason shown and an override.
- Analyses: intrinsic properties and firing (NeuroExpress's measures and IPFX's spike detection), I–V curves,
  evoked responses pulse by pulse, spontaneous events (template detection and review), DSI/DSE (evoked and
  spontaneous, per cell across files), tonic current, membrane test, series-resistance error, bridge balance, field
  potentials and LTP, short-square and ramp thresholds, spike trains, voltage-step currents, recording quality.
- Every measurement with its definition, window, source and flags; artifacts and test pulses identified.
- Projects saved beside the recordings, with notes, sessions, rig profiles, decisions and undo; Synaptoscope
  projects and presets imported.

**Figures and export**
- Standard graphs for each analysis, heat maps, comparisons across recordings and groups with statistics, a figure
  builder with presets; Excel and CSV export with each column's definition.

**Live, on the rig**
- The amplifier's readings and MultiClamp Commander's telegraph, the cell's health, each new recording analysed as
  it is written, and the rig's programs shown in place.
- Recording without Clampex on a Digidata 1440A: Clampex protocols played as Clampex plays them (Play, Pause,
  Record, Stop), each sweep drawn live, written as ABF files Clampex, Clampfit and pyabf open; the protocol drawn
  before it plays.
- A membrane test following Clampex's algorithm; the amplifier's controls; a simple stimulus editor; the camera
  without its own program, with tools for patching.
- Any MultiClamp 700B is found from Commander (no serial number to set); the protocols folder is Clampex's own.
- The 32-bit Python the Digidata's library needs ships with the program.

See [docs/VALIDATION.md](docs/VALIDATION.md) for how this release was checked.
