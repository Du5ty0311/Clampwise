# Changelog

## 1.5.1 — 2026-10-19

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
