# Changelog

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
