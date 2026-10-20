# Changelog

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
