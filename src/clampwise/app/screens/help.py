"""Help: the quick start, the screens and shortcuts, a glossary, one entry per
analysis and further reading. The analysis entries are written from the
registry (what each measures, its reference, every setting and every number
with its definition), so they never fall behind the code."""

from __future__ import annotations

import html
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets

from ...analysis import registry
from .. import skeleton, widgets
from ..widgets import box
from .base import Screen

READING = Path(__file__).resolve().parents[1] / "resources" / "reading.md"

QUICK_START = """
<h2>Quick start</h2>
<ol>
<li><b>Open a folder</b> of recordings (Files, Ctrl+O). Clampwise saves the analysis in a project file in that
folder, so work resumes where it was left. Supported formats: Axon ABF and ATF, WinLTP sweep files and spike-time
files. To explore the program first, choose <b>Open the demos</b> on the Files screen.</li>
<li><b>Identify each recording</b>: animal, slice, cell and condition are read from folder names using naming rules
(Files, Details) or entered manually.</li>
<li><b>Analyze</b>. Each recording's type is determined from its content (a step family, a stimulus, spontaneous
events, etc.), and the analyses for that type run in the background. The Check screen shows the reason for the
assigned type and allows it to be changed.</li>
<li><b>Check</b> each recording: the trace, the stimulus, the measurement points marked on it, and the quality flags.
Drag a step's baseline, peak, steady state or rebound marker to remeasure it, and review detected events one at a
time. Shortcuts: J and K move to the previous and next file, the arrow keys step through sweeps, and F fits the trace
to the view. G and B mark the file as good or bad (with an optional reason); N moves to the next unmarked file. The
review marks can be disabled under Settings, Features.</li>
<li><b>Results</b> lists every measurement with its definition and method. <b>Figures</b> plots them four graphs at a
time, or compares all recordings of a type. Any graph can be maximized (corner button or double-click on its title;
Esc restores) or opened in a separate window.</li>
<li><b>Export</b>: Send to Excel writes one workbook per project, with each column's definition attached to its
header.</li>
</ol>
<p>Quality flags are informational: a value is never altered or excluded because of a flag. Every analysis reports
which pulse, window or baseline was used for the recording, and why.</p>
<p><b>Updates.</b> When Clampwise starts it asks GitHub whether a newer version has been released (Settings,
Updates: the check can be turned off, and Check for updates asks at any time). A newer version is only announced;
<b>Update to</b> downloads it, checks it against the checksum GitHub lists for it, closes Clampwise, replaces the
program's folder and opens the new copy. Projects, settings and recordings are kept; an update is refused while a
recording or membrane test runs. If the program's folder cannot be written (installed for all users), the release's
page is opened instead. Whoever looks after a lab's PCs can run <i>Clampwise.exe --update</i>.</p>
"""

HEAT_MAPS = """<h2>Heat maps</h2>
<p>A heat map displays a matrix as colour, with rows and columns on axes in physical units, a colour bar giving
the quantity and its units, and markers for the stimulus, step or induction. Each graph's <b>Show</b> menu lists the
maps available for the recording's results; the adjacent <b>Colour</b> menu sets the colour map, the limits and dB
scaling, and can hold a map's limits across recordings so that recordings compared side by side share one scale.</p>
<ul>
<li><b>Missing values</b> are left blank, never drawn as zero (for example, a bin that was not analyzed, or the
step period in a DSI episode).</li>
<li><b>Sequential</b> data (power, rate, counts, the recorded signal) use cividis, a perceptually uniform map that is
legible with colour-vision deficiency. Limits are the 1st to 99th percentile, the full range, or fixed values. When
most of the map is at its minimum (e.g. event rates in short bins), percentiles are computed over the values above
it so that the few active bins do not saturate.</li>
<li><b>Diverging</b> data (% of baseline, ratios to pulse 1, dB relative to a median, correlations) are centred on
their reference value: 100%, 1 or 0.</li>
<li><b>Time-frequency power</b> (spontaneous and field recordings): a short-time Fourier transform with 1 s Hann
windows and 50% overlap (1 Hz resolution), or complex Morlet wavelets of 3 to 10 cycles across the frequency range.
Power is shown in its native units; dB (10 log10 of a power, 20 log10 of an amplitude) can be selected in the Colour
menu. Band power is expressed for each band in dB relative to its own median.</li>
<li><b>Correlations</b> (Figures, Across recordings): Spearman's r between the headline measures of a recording
type, from -1 to 1; dots mark pairs that remain significant after Benjamini-Hochberg correction (q = 0.05).</li>
</ul>
<p>Heat maps can be disabled under Settings, Features.</p>
"""

def _figures_page() -> str:
    """The figure builder's guide (B12 item 5), from the builder's own words."""
    from ...figure.recipe import PLOTS, QUESTIONS, STANDARD, SUMMARIES

    e = html.escape
    out = ["<h2>Making figures</h2><p>Figures, <b>Make a figure</b>. Begin with what the figure should show; the "
           "builder sets up the standard plot for it and writes a description beneath the figure (suitable for a "
           "figure legend), together with any conditions under which it could mislead. Unless otherwise selected, "
           "each value is one value per cell, and the caption states the unit of analysis.</p>"
           "<p><b>Editing a graph from the Figures screen.</b> Every graph on the Figures screen (including heat "
           "maps and Across recordings) has an <b>Edit</b> button that opens it here with its data, series, colours, "
           "error bars, reference lines and axis titles. Everything can then be changed like any other figure, "
           "combined with other panels, saved and exported.</p>"
           "<p><b>Raw traces.</b> For an example panel, <b>Choose traces...</b> (Data tab, or Open data) picks the "
           "recording and the sweeps to show. Select sweeps by hand (Ctrl or Shift + click), by the time they "
           "start, or, for DSI, as an episode's last baseline responses or its first responses after the step; "
           "add each selection as a named trace (for example 'Before DSI' and 'After DSI') drawn as the average "
           "of its sweeps, every sweep, or the sweep closest to their mean. The window around time 0, the baseline "
           "subtraction and the artifact blanking apply to every trace alike. The DSI layout's example panel opens "
           "in the picker with its own picks.</p>"
           "<p><b>Applying changes.</b> Lists, check boxes and numbers apply at once; text boxes apply a moment "
           "after typing stops or at Enter. <b>Apply</b> (Ctrl+Enter) applies everything and redraws now.</p>"
           "<p><b>Adding and removing.</b> Everything that can be added can be removed, by a button beside it and "
           "by right-clicking it: panels (Remove panel, the × on a panel's tab, or right-click the panel in the "
           "preview to duplicate, move or remove it), reference lines, marks and text (Remove selected, or the "
           "Delete key), row conditions, traces and their sweeps in Choose traces, saved styles (Styles, Delete a "
           "saved style), a panel's data (Open data, Remove this panel's data) and the digitizer's points. A "
           "group comes from the data, so it is hidden rather than removed (Groups, Hide from the figure). Undo "
           "(Ctrl+Z) brings back anything removed.</p>"
           "<h3>What should the figure show?</h3><dl>"]
    for key, title, why, plot, plots in QUESTIONS:
        out.append(f"<dt><b>{e(title)}</b></dt><dd>{e(why)} Other plots: "
                   f"{e(', '.join(PLOTS[p][0] for p in plots if p != plot)) or 'none'}.</dd>")
    out.append("</dl><p>Any other plot type can be selected on the <b>Plot</b> tab; the question updates to match. "
               "<b>Groups</b> sets each group's colour (click the swatch), marker, line style and width, filled or "
               "hollow points, legend name, visibility and order. Only modified properties are stored, so applying a "
               "new theme recolours the rest. <b>Theme</b> applies a complete style in one step (journal, minimal, "
               "boxed, grey grid, bold presentation, dark slide, greyscale print, high contrast), and exposes each of "
               "its properties: colours, font and sizes, axis lines, grid, ticks, background (or transparent), text "
               "colour, panel letters, legend frame and columns, point transparency and jitter, and error-bar caps. "
               "<b>Axes</b> sets titles, ranges, log scales, label angle, legend position (inside or outside) and "
               "reference lines (e.g. 'y=100 baseline; x=0').</p><h3>Central tendency and spread</h3><ul>")
    out += [f"<li><b>{e(t)}</b>: {e(m)}.</li>" for t, m in SUMMARIES.values()]
    out.append("</ul><p>SEM, SD and the 95% CI convey different information; the caption states which one is shown "
               "(Cumming 2014).</p><h3>Warnings</h3><p>The builder warns when a figure could mislead:</p><ul>"
               "<li>Points are rows (events, sweeps) and cells contribute several: n and the spread then count rows "
               "rather than cells (pseudoreplication; Aarts et al. 2014).</li><li>A group has n &lt; 3.</li>"
               "<li>A bar chart's axis does not start at zero. Bar charts always show the individual points "
               "(Weissgerber et al. 2015).</li><li>A distribution pools events from several cells.</li>"
               "<li>Summary rows of a spreadsheet (AVERAGE, SEM, n) remain in the table.</li><li>The cells of a "
               "time course are not sampled at the same times (use time bins).</li></ul>"
               "<h3>Statistics</h3><p>Computed on one value per cell: the difference between groups with its 95% "
               "CI by bootstrap over cells, or hierarchically over animals and then cells when animals are "
               "identified (Saravanan et al. 2020), and Hedges' g (Ho et al. 2019); Welch's t-test and Mann-Whitney "
               "U, or paired t-test and Wilcoxon signed-rank for repeated measures on the same cells, with Holm "
               "correction for multiple groups; and a mixed model with animal as a random intercept when "
               "statsmodels is installed.</p><h3>Standard figure for each analysis</h3><table cellpadding='4'>"
               "<tr><th align='left'>Analysis</th><th align='left'>Figure</th><th align='left'>Reference</th></tr>")
    out += [f"<tr><td>{e(a)}</td><td>{e(f)}</td><td>{e(s)}</td></tr>" for a, _q, f, s in STANDARD]
    out.append("</table><h3>Data sources</h3><p>A table from the current project; a CSV or Excel file (the header "
               "row is detected, and side-by-side tables can be stacked with 'Conditions are in separate "
               "columns'); a saved figure (every export embeds its figure definition, so it reopens for editing); "
               "or an image of a plot, digitized. Digitized values are estimates, limited by the image "
               "resolution.</p>")
    return "".join(out)


SESSIONS = """<h2>Notes, episodes and rigs</h2>
<p><b>Notes.txt</b>: a text file in a cell's folder is read (never modified). Its first lines give the age, sex,
region, cell type and bath drugs; each <i>label: number</i> line identifies a file, for example:</p>
<pre>5-6 wk adolescent male rat
Recording CA1 pyramidal neuron
Drugs:
DAP5 50 uM
File (baseline): 24125268
Depol (5s): 24125269
5s: 24125270</pre>
<p>Each named file receives its label (Files, Label), and the details are filled in wherever none were entered. A
number that matches no file is assigned to the single file one digit away, and the uncertainty is reported. A file
that disagrees with its note (for example, a "5 s" step that lasts 2.5 s) is flagged. This feature can be disabled
under Settings, Features.</p>
<p><b>Episodes</b>: each cell's files are grouped by experimental episode: the recording before a step (or an
induction), the step, and the recording after it; or a change of condition between two files. Episodes are taken from
the notes when they name the files, and otherwise inferred from the files' protocols, timing and content. The Files
screen shows each file's episode (01 before, 01 step, 01 after, ...) and any uncertainty; sorting by Episode keeps
each episode together.</p>
<p><b>Rigs</b>: Files, More, Name the rig of the selected recordings. A rig is recognized from its files' metadata
(the digitizer and amplifier in the header, the primary channel's name and units, and the protocols). A preset
assigned to a rig is applied to its files in any project, and so are the <b>inputs that carry light</b> on it
(asked last; a rig's wiring that a line's name cannot tell: on one Polygon rig, AM_Stim records analog output 2,
which commands the 470 nm LED feeding the Polygon 1000). Files from unrecognized hardware are not assigned a rig;
when two rigs match equally, the ambiguity is reported rather than resolved automatically.</p>
<p><b>Cell health</b> (Figures, Show): series resistance, holding current and noise across a cell's files on a common
time axis, with the drift limits shown as dashed lines (Rs change above 20%, holding-current change above 100 pA, or
noise more than doubling). Limits are set under Settings, Health.</p>
<p><b>Overview</b> (Files): the status of every animal, slice and cell (analyzed, flagged, reviewed, exported, or
missing), and the <b>audit trail</b>: every edit and decision with its time and origin (user, Notes.txt or undo).
Each entry can be undone, and the trail can be exported as CSV.</p>
"""

STIMULUS = """<h2>Where the stimulus is read from</h2>
<p>Each recording's stimulus times come from the first of these that applies, and the reason lists every channel
considered: a time typed on Check; the file's own record (a WinLTP stimulation setting); a recorded trigger or light
channel; a digital output; a second analog output (a light driver's command); brief pulses in the command itself; and,
when nothing records it, the stimulator's own artifact in the signal.</p>
<p>A recorded line is a trigger when it is near flat in volts and rises by at least 0.1 V in brief square pulses
(a TTL read through an attenuator reads 0.45 V and counts). It is rejected, with the reason shown, when it is a
monitor (pA, nA or mV), is held high longer than 50 ms without being light, toggles faster than 500 Hz or more
than 200 times in a sweep (a clock or frame sync), pulses once per sweep where Clampex's held first 1/64 ends
(an acquisition trigger), or lines up with the command's own steps.</p>
<p>A channel carries <b>light</b> when its name says so (LED, light, laser, Polygon, DMD, opto), when it is one of
the inputs the lab's rig records the LED driver's trigger on (<b>IN 8</b>, <b>IN 14</b>), when the file's rig names it
among its inputs that carry light (Rigs, under Notes, episodes and rigs; this comes before the name's words), or when
the protocol's name does; otherwise it is electrical when its name or the protocol says so (stim, isolator, shock), else of unknown
kind. Light may be held on for seconds (a 1 s pulse is not a gate), and a light pulse that climbs slowly is a ramp,
timed from its foot. A protocol mapping (Settings) can name the stimulus channel and its kind for every file of a
protocol.</p>
<p>A file whose protocol is named for a stimulus but whose trigger line never rises stays without a type and says so:
nothing in the file records when the stimulus came.</p>
"""

LIVE = """<h2>The Live screen</h2>
<p>Live puts the cell you are recording and the rig's programs on one screen. The screen names the rig's roles;
these are the programs behind them on this rig, and what is read from each. Nothing on the rig is changed unless you
do it from Clampwise (record, the amplifier's controls, the camera); everything else is <b>read</b>.</p>
<p><b>Hardware.</b> At the top of the cell's panel, the rig's devices as Windows lists them, found by themselves
every few seconds (Detect looks now) and never opened to do it: the digitizer (Digidata 1440A), the amplifier
(MultiClamp 700B, with its serial, and each channel's mode from MultiClamp Commander's telegraph when it is open),
the camera (Lumenera INFINITY, with its serial and model) and, when connected, the Polygon's IOBox (Mightex). Each
says who is using it, in the screen's words: the acquisition program (Clampex), the camera's program (Infinity
Analyze), Clampwise; or that it is free.</p>
<table cellpadding='4'><tr><th align='left'>Role</th><th align='left'>Program</th><th align='left'>What is read, and how</th></tr>
<tr><td>Amplifier</td><td>MultiClamp 700B Commander</td><td>The amplifier itself, twice a second, through Commander's
message interface (its <i>AxMultiClampMsg</i> library): the mode, the holding and whether it is on, the primary and
secondary outputs with their gain and filters, the pipette offset, the seal test, and in voltage clamp the whole-cell
compensation (the amplifier's own Rs and Cm), Rs compensation, pipette capacitance and leak subtraction; in current
clamp the bridge balance, capacitance neutralization and slow injection; and the four meters as Commander shows
them. The readout reaches only the interface's reading functions; the amplifier's controls (below, when turned on)
are a separate, listed set of its setters and buttons. Commander must be open, since it is the amplifier's
driver.</td></tr>
<tr><td>Acquisition</td><td>Clampex (pCLAMP)</td><td>Its Lab Book, the log Clampex keeps of what it does: the folder it
writes to, each protocol opened, each file started and whether it completed or was aborted, and every Membrane Test
with the values it logged (Rt, Ra, Rm, Cm, Tau, the holding current). The Membrane Test logged before a file is its
start, the first one after it its end. The file being written is read sweep by sweep as Clampex writes it (see
The cell), and each finished recording is added and analyzed (Start watching).</td></tr>
<tr><td>Patterned light</td><td>PolyScan (Mightex Polygon)</td><td>The session open (PolyScan keeps each session in its
own folder): the experiment, the pattern sequence loaded and its regions, the objective, the calibration, and how the
Polygon is triggered (an external trigger from the acquisition, on a rising or falling edge). When a PolyScan workflow
logs each trigger, the triggers are read too; where the light is fired by Clampex itself (analog output 2
to the LED's light source), its triggers are in Clampex's files, not PolyScan's log. Personal details in its files (an account's e-mail, the computer's
identifiers) are not kept.</td></tr>
<tr><td>Camera</td><td>Infinity Analyze (Teledyne Lumenera)</td><td>Its settings (the camera, where it saves) and the
pictures it saves: the latest is shown with its exposure, gain and length calibration. Save a cell's pictures in the
cell's folder and each of its recordings finds them. The camera is not opened by Clampwise: Infinity Analyze has
it.</td></tr></table>
<p><b>The programs' windows.</b> Each role's window is shown live on the right, drawn by Windows itself (as in the
taskbar's previews): nothing is copied and the program is not touched. A minimized window has nothing to show until
it is restored. A click on a window brings that program to the front, to be worked in directly. The acquisition and
the camera have the wide column. Each view's <b>enlarge</b> button (as on a window's title bar) lets it fill the views alone (again
brings the others back) and its <b>close</b> button hides one not in use; <b>Views</b> above them (or a right-click beside them) shows it
again. A right-click on a view can show <b>only part of the window</b>: drag over the part to keep, which then follows
the window's edges when it is resized; <b>Show the whole window</b> undoes it. The camera's view shows only Infinity
Analyze's picture panel (below its menu, left of its Control Panel) unless another part is chosen. These choices are
kept for the next time Clampwise opens.</p>
<p><b>The cell.</b> The amplifier's readings are shown large; its Rs and Cm are the whole-cell compensation's
settings, marked "compensation off" when it is off (then they estimate nothing). The four meters are channel 1's
potential and current, then channel 2's, as Commander shows them. <b>From the recorded trace:</b> the newest file in the
folder Clampex writes to (or the watched project's folder) is read while it is written, each sweep once complete
(opened with read, write and delete sharing and closed at once, never mapped; a partial last sweep waits), and its
test step measured as the Membrane test analysis measures one: Rs, Rm, Cm and the holding current, with the noise;
a gap-free file in 5 s stretches (its holding or resting level and noise). Each sweep's Rs is drawn beside the
amplifier's, and the badge says how far Rs has moved from the cell's first three sweeps, flagging more than 20% (a
new cell folder begins again). <b>Replaying</b> a recorded folder as if live: <i>python -m clampwise.rig.replay
&lt;recorded folder&gt; &lt;watched folder&gt; [--speed 10]</i> writes its files as Clampex would, with their original
timing. A note typed at the bottom is kept with the project, with the time and the
last file Clampex logged.</p>
<p><b>The camera in Clampwise.</b> The camera view's <b>Use here</b> takes the camera itself, through Lumenera's
own camera library (<i>lucamapi.dll</i>, installed with Infinity Analyze; its layouts as Christoph Gohlke's
<i>lucam</i> declares them, checked on the camera). Only one program can hold the camera, so Infinity Analyze must be
closed first; <b>Let go</b> (or closing Clampwise) gives it back. Below the picture: the exposure, the gain and the
frames averaged into each picture (more for dim fluorescence). <b>Adjust...</b> changes what is shown, never the
picture saved: brightness, contrast, gamma (below 1 lifts dim parts), saturation, sharpness, clarity (local
contrast), auto levels and the colour balance (White balance on an empty, bright field), each with its reset; a set
of them can be kept by name as a preset. With the um per pixel from a stage micrometer's picture, a scale bar is
drawn. <b>Snapshot</b> saves the picture as taken, a TIFF in the folder the acquisition writes to (else the open
project's), with the exposure, gain, frames averaged, objective and adjustments in its description and the scale
as its resolution (ImageJ and Fiji read it); when anything was changed on screen, the picture as shown is saved beside
it (<i>_as_shown.png</i>): the original is never altered (Rossner &amp; Yamada 2004).</p>
<p><b>For patching</b> (Adjust...): <b>Take background</b> on a defocused or empty field and subtract it, the
video-enhanced contrast that makes cells stand out under infrared contrast optics (Allen, Allen &amp; Travis 1981;
Inoue 1981; Stuart, Dodt &amp; Sakmann 1993), or divide by it to even out the illumination of fluorescence (flat
field); <b>Take dark frame</b> with the light blocked to take out the camera's own offset. <b>Keep this picture</b>
holds a fluorescence picture and lays it, in a colour of your choice and as strongly as you like, over the live
infrared one, so the labelled cells stay marked while the pipette approaches. <b>Auto exposure</b> brings the
brightest 0.5% of the picture to 80% of the camera's full scale; the <b>histogram</b> shows every pixel's brightness
and <b>clipped pixels</b> are painted red, since a pixel at full scale is no longer a measure (Waters 2009; Lambert
&amp; Waters 2014; Murphy &amp; Davidson 2012). Grey, green (GFP), magenta and fire colour scales; a crosshair;
<b>Measure</b> drags a line whose length is given in um once the camera is calibrated; the wheel zooms, a drag pans,
<b>Fit</b> shows the whole picture. The <b>Objective</b> slider says which objective is in use (2.5x to 63x; the list
is set in Adjust...), and the scale bar and Measure follow it; the size of one pixel in um is shown beside it. Before
any calibration it is nominal: the camera's own pixel (4.5 um on the INFINITY8-3M, filled in from its model) over the
objective's magnification times the camera adapter's (Adjust..., Length calibration; the adapter is printed on it,
often 1x, 0.63x or 0.5x). To measure it instead: under Adjust..., Length calibration,
type the um a pixel for the objective in use, or drag the Measure line along a stage micrometer and choose From the
Measure line; every other objective's scale is then that one's times the ratio of their magnifications, said as
estimated, until it is calibrated itself. A snapshot records the objective and its scale. The <b>compass</b>
(Adjust..., Compass) marks the slice's four directions in its plane over the picture. Choose the plane it was cut in:
horizontal (rostral-caudal and medial-lateral in the picture, dorsal-ventral cut across), coronal (dorsal-ventral and
medial-lateral; rostral-caudal cut across) or sagittal (dorsal-ventral and rostral-caudal; medial-lateral cut across);
and the names: rostral and caudal, or anterior and posterior, in words or by their letters. Then set where the plane's
first direction points on the screen (rostral in a horizontal slice, dorsal in a coronal or a sagittal one; degrees
clockwise from up, Turn left and Turn right by a quarter) and which side of it the other axis's first end is on
(medial; rostral in a sagittal slice); the first direction's arm is the thicker. A snapshot records the plane and the
orientation too. The picture is the
camera's video stream at its fastest frame rate, so the exposure alone sets the pace (37 ms: 27 pictures a second);
the frame rate is shown beside the buttons, as taken and, when fewer, as drawn. While Clampwise holds the camera, the camera's own processing (the contrast and
brightness its program leaves in it) is made neutral, and put back when it lets go.</p>
<p><b>Recording in Clampwise.</b> The acquisition view's <b>Use here</b> records without Clampex (Clampex must be
closed: one program at a time holds the Digidata). Choose one of your protocols (<b>Protocols...</b> picks their
folder) and the folder to record into. Before it plays, the protocol is drawn in the scope (each sweep's command in the amplifier's present unit), and a line above the scope says what it is doing or waiting for (opening the digitizer, the sweep under way, the seconds to the next). <b>Play</b> runs the protocol, drawn and measured live, keeping nothing (as Clampex's View); <b>Record</b> runs it into a file; <b>Pause</b> ends the run once the sweep under way is complete (the digitizer queues its outputs seconds ahead, so a sweep is never cut; a gap-free run pauses at once), the file then whole, and <b>Resume</b> goes on with the next sweep in the same file, each sweep at its real time in the synch array (a gap-free run resumes into a new file, so no gap is hidden in a continuous one); <b>Stop</b> ends it. The protocol is played as Clampex plays it: its epochs,
sweep by sweep, at its start-to-start interval, the holding levels between; the command's numbers are read in the
amplifier's unit (mV in voltage clamp, pA in current clamp) and its scale, like each input's scale, gain, filter and
unit, comes from MultiClamp Commander's telegraph, never from the scale the protocol was saved with. Each input is
drawn live as it is recorded (the sweep under way grows across the plot, the earlier ones fainter; gap-free scrolls),
about a tenth of a second behind the cell, and each sweep is measured as Clampex's are (see The cell). The file is an
ABF2 file Clampex, Clampfit and pyabf read, named with Clampex's date code and <i>cw</i> (26o15cw000.abf), written
as Clampex writes one: the header first, each sweep as it completes, the counts at the end. A stopped run keeps every complete sweep. The amplifier's telegraph is watched while a run goes: a change of mode, gain or scale pauses it after the sweep under way, and Resume goes on into a new file with the new scales, so a file's numbers are always true. The digitizer is driven through pCLAMP's own library,
<i>AxDD1400.dll</i>, from a small helper run by a 32-bit Python (python.org's embeddable package,
shipped with Clampwise), the method of John Dempster's WinWCP (University of Strathclyde, <i>dd1440.pas</i>);
thiago4455's notes on the Digidata 1440A were read too. The telegraph follows Molecular Devices'
<i>MultiClampBroadcastMsg.hpp</i>, with ACQ4's <i>MultiClampTelegraph.py</i> (Campagnola et al., MIT licence) read for
the method. On the rig, a protocol recorded here and by Clampex gave the same files (the same header, epochs, timing
and scaling; the command within one sample count) and the same Rs, Rm and Cm on a model cell.</p>
<p><b>The amplifier's controls.</b> <b>Controls</b> under the channel shows the amplifier's settings for its mode: the
mode and holding, the seal test, the pipette offset and capacitance, whole-cell compensation, Rs compensation, bridge
balance and neutralization, leak subtraction, the outputs and their gains and filters, and the Auto, Zap and Buzz
buttons. Each shows the amplifier's value as read; a change is sent through Commander's own interface, so Commander
shows it too and can be used as before. Reset is not offered.</p>
<p><b>The stimulus editor.</b> <b>Edit...</b> beside the protocol opens a copy of it: each output's epochs (a step, a
ramp or a pulse train; the first level and its change from sweep to sweep, the first duration and its change, a train's
period and pulse width; the digital outputs of each epoch), each output's holding, the number of sweeps, their
start-to-start interval and length, with every sweep drawn as edited. It is saved under a new name beside the
original, which is never changed, as a protocol Clampex opens too.</p>
<p><b>The membrane test.</b> <b>Membrane test</b> in the acquisition view is Clampex's Membrane Test: a square wave
centred on the holding (the pulse peak to peak; 10 mV at 50 Hz in voltage clamp, Clampex's Cell stage, and 50 pA at
5 Hz in current clamp, each mode's own kept), 500 samples a period as Clampex's (so 25 kHz at 50 Hz). Twenty times a
second the newest pulses are averaged and measured as pCLAMP measures them (the pCLAMP 10 User Guide, Membrane Test):
I1 and I2 the mean current over the last 20% of each half, Rt = the pulse / (I1 - I2), the holding (I1 + I2) / 2;
each edge's transient fitted from 80% to 20% of its peak (drawn in red), tau; the charge Q the transient carries
above the new steady state plus (I1 - I2) tau; Rs (pCLAMP's Ra) = tau x pulse / Q, Rm = Rt - Rs, Cm = Q Rt / (pulse
x Rm); both edges averaged. What it is reading is said (a cell; a seal, Rt over 1 GOhm; the pipette alone, under
20 MOhm). In current clamp, which Clampex's test does not do, the input resistance, tau and Cm come from the charging
fitted. Rt and Rs are drawn over the test. Nothing is saved. One sample count of the digitizer is said beside the
numbers (6.1 pA at the amplifier's 0.05 V/nA): the means let the recording's noise resolve Rt below a count.</p>
<p><b>The telegraph.</b> The channel's <b>Telegraph</b> line is what MultiClamp Commander telegraphs to an
acquisition program, as Clampex receives it: the mode, the primary output's scale and gain, its filter and the
command's sensitivity. Clampwise records with these scales.</p>
<p>Live, recording in Clampwise and the amplifier's controls can each be turned off in Settings, Features.</p>
"""

GLOSSARY = {
    "Sweep": "One execution of the protocol: the recording from one trigger to the end of that trial.",
    "Epoch": "One segment of a protocol's command waveform (a holding level, a step, a ramp or a train).",
    "Pulse 1": "The response to the first stimulus of each sweep, measured on the sweep average; the only response "
               "not influenced by preceding pulses.",
    "PPR": "Paired-pulse ratio: the pulse 2 amplitude divided by the pulse 1 amplitude, measured on the averages, "
           "after subtracting the decay of pulse 1.",
    "Rs": "Series (access) resistance: the resistance of the pipette and the patch opening, in series with the "
          "membrane.",
    "Rheobase": "The smallest current step that evokes a full-sized action potential.",
    "Short square": "A brief current pulse (typically 3 ms); its threshold is the smallest pulse amplitude that evokes "
                    "an action potential.",
    "Ramp": "A linearly increasing current; its threshold is the current at the threshold of the first action "
            "potential.",
    "Sag": "The recovery of membrane voltage from its peak during a hyperpolarizing step, mediated mainly by Ih.",
    "DSI": "Depolarization-induced suppression of inhibition: the transient reduction of inhibitory synaptic "
           "transmission following a depolarizing step.",
    "Sham test": "A test of whether a response is time-locked to the stimulus, comparing it with the same measurement "
                 "at times without stimulus, command change or artifact; p > 0.05 indicates no time-locked "
                 "response.",
    "Stimulus artifact": "The capacitive transient produced by a stimulator: a jump between consecutive samples that "
                         "reverses within 0.5 ms and occurs at the same time in every sweep.",
    "Test pulse": "A small square step used to monitor recording parameters (Rs, input resistance), delivered by the "
                  "acquisition protocol or the amplifier.",
    "Flag": "A quality warning on a recording or a measurement, with the limit exceeded and its reference. Flags "
            "never alter or exclude data.",
    "Project": "The analysis state stored in the experiment folder: the recordings and their identification, the "
               "settings, the review decisions (events reviewed, points moved) and the results.",
}


def _analysis_entry(key: str, a: registry.Analysis) -> str:
    out = [f"<h2>{html.escape(a.label)}</h2>", f"<p>{html.escape(a.description)}</p>",
           f"<p><i>Reference:</i> {html.escape(a.source or 'Clampwise')}</p>",
           "<p><i>Recording types:</i> " + ", ".join(registry.TYPE_WORDS.get(t, t) for t in sorted(a.types)) + "</p>"]
    visible = [s for s in a.settings if not s.hidden]
    if visible:
        out.append("<h3>Settings</h3><table cellpadding='4'><tr><th align='left'>Setting</th><th align='left'>Default"
                   "</th><th align='left'>Description</th><th align='left'>Reference</th></tr>")
        for s in visible:
            out.append(f"<tr><td>{html.escape(s.label)}</td><td>{html.escape(str(s.default))} {html.escape(s.units)}"
                       f"</td><td>{html.escape(s.help)}</td><td>{html.escape(s.source)}</td></tr>")
        out.append("</table>")
    for level in dict.fromkeys(["recording"] + [o.level for o in a.outputs]):
        outs = [o for o in a.outputs if o.level == level]
        if not outs:
            continue
        out.append(f"<h3>Measurements ({'per ' + level if level != 'recording' else 'per recording'})</h3>"
                   "<table cellpadding='4'><tr><th align='left'>Measurement</th><th align='left'>Units</th>"
                   "<th align='left'>Definition</th><th align='left'>Reference</th></tr>")
        for o in outs:
            out.append(f"<tr><td>{html.escape(o.title)}</td><td>{html.escape(o.units)}</td>"
                       f"<td>{html.escape(o.definition)}</td><td>{html.escape(o.source)}</td></tr>")
        out.append("</table>")
    from ...analysis import columns

    own = columns.TABLES.get(key, ())
    if own:
        out.append("<h3>Table columns</h3><table cellpadding='4'><tr><th align='left'>Column</th>"
                   "<th align='left'>Units</th><th align='left'>Definition</th><th align='left'>Reference</th></tr>")
        for o in own:
            out.append(f"<tr><td>{html.escape(o.title)}</td><td>{html.escape(o.units)}</td>"
                       f"<td>{html.escape(o.definition)}</td><td>{html.escape(o.source)}</td></tr>")
        out.append("</table>")
    return "".join(out)


def entries() -> list[tuple[str, str, str]]:
    """(key, title, HTML) for every page of the manual."""
    keys = [f"<tr><td>{html.escape(d.title)}</td><td>{html.escape(d.shortcut)}</td><td>{html.escape(d.tip)}</td></tr>"
            for d in skeleton.DESTINATIONS + skeleton.APART]
    keys += [f"<tr><td>{a}</td><td>{k}</td><td>{b}</td></tr>" for a, k, b in (
        ("Find", "Ctrl+K", "Search for a screen, an action or a recording"),
        ("Open folder", "Ctrl+O", "Open a folder of recordings as a project"),
        ("Back", "Alt+Left", "Return to the previous screen"), ("Details", "Ctrl+D", "Show or hide the details panel"),
        ("Theme", "Ctrl+Shift+L", "Switch between light and dark themes"),
        ("Undo, redo", "Ctrl+Z, Ctrl+Y", "Undo or redo an event review or a moved point"),
        ("Sweeps", "Left, Right", "Check: previous or next sweep"),
        ("Events", "[ ] R A", "Check: previous event, next event, reject, add at the cursor"),
        ("Files", "J, K", "Check: previous or next file"), ("Fit", "F", "Check: fit the whole sweep to the view"),
        ("Review", "G, B, N", "Check: mark the file good or bad; go to the next unmarked file"),
        ("Restore", "Esc", "Figures: return a maximized graph to the grid"))]
    pages = [("quick start", "Quick start", QUICK_START),
             ("screens", "Screens and shortcuts", "<h2>Screens and shortcuts</h2><table cellpadding='4'><tr><th align='left'>"
              "Screen or action</th><th align='left'>Shortcut</th><th align='left'>Description</th></tr>" + "".join(keys) + "</table>"),
             ("glossary", "Glossary", "<h2>Glossary</h2><dl>" + "".join(
                 f"<dt><b>{html.escape(k)}</b></dt><dd>{html.escape(v)}</dd>" for k, v in GLOSSARY.items()) + "</dl>"),
             ("stimulus", "Where the stimulus is read from", STIMULUS),
             ("live", "The Live screen", LIVE),
             ("heat maps", "Heat maps", HEAT_MAPS), ("figures", "Making figures", _figures_page()),
             ("sessions", "Notes, episodes and rigs", SESSIONS)]
    pages += [(key, a.label, _analysis_entry(key, a)) for key, a in registry.load().items()]
    try:
        reading = READING.read_text(encoding="utf-8")
    except OSError:
        reading = "# Further reading\n\nThe reading list is not included in this installation."
    pages.append(("reading", "Further reading", "markdown:" + reading))
    return pages


class HelpScreen(Screen):
    key = "help"
    title = "Help"
    purpose = "The Clampwise user guide: quick start, screens and shortcuts, glossary, a reference entry for " \
              "every analysis, and further reading."

    def __init__(self, window) -> None:
        super().__init__(window)
        row = box("h", "s")
        side = box("v", "s")
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search the user guide")
        self.search.textChanged.connect(self._filter)
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumWidth(260)
        self.list.currentRowChanged.connect(self._show)
        side.addWidget(self.search)
        side.addWidget(self.list, 1)
        self.page = QtWidgets.QTextBrowser()
        self.page.setOpenExternalLinks(True)
        row.addLayout(side)
        row.addWidget(self.page, 1)
        holder = QtWidgets.QWidget()
        holder.setLayout(row)
        self.body(holder)
        self._pages = entries()
        for key, title, _ in self._pages:
            item = QtWidgets.QListWidgetItem(title)
            item.setData(QtCore.Qt.UserRole, key)
            self.list.addItem(item)
        self.list.setCurrentRow(0)
        widgets.HelpButton.open_manual = self.open

    def open(self, key: str) -> None:
        """The entry for ``key`` (an analysis's key, or a page's), from a "?" anywhere."""
        self.window.go("help")
        for i in range(self.list.count()):
            if self.list.item(i).data(QtCore.Qt.UserRole) == key:
                self.list.setCurrentRow(i)
                return

    def _show(self, row: int) -> None:
        if not 0 <= row < len(self._pages):
            return
        text = self._pages[row][2]
        if text.startswith("markdown:"):
            self.page.setMarkdown(text[len("markdown:"):])
        else:
            self.page.setHtml(text)

    def _filter(self, text: str) -> None:
        words = text.lower().split()
        for i, (_, title, body) in enumerate(self._pages):
            hay = (title + " " + body).lower()
            self.list.item(i).setHidden(bool(words) and not all(w in hay for w in words))
