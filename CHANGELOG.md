# Changelog

## 1.8.0 — not released yet

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
