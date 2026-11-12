# Every control, and what was checked

Written by `tools/audit/controls_doc.py` from the runs of `tests/test_every_control.py`; do not edit by hand.

The test presses every button, checkbox, pick list and menu action of the Workbench, the Figures screen, the
figure builder (with its choosers of data and plot and its export dialog), the trace picker and the digitizer,
each from the same starting state. In the builder the figure itself is a control too: each kind of thing drawn on
it (a panel, a letter, a title, an axis and its title, the legend, a group, the scale bars, a stimulus bar, the
command, a text, a bracket, a reference line, a mark along x, a potential label, a text on the page) is clicked,
dragged, moved with an arrow key, double-clicked (and rewritten), deleted and right-clicked (each entry of its
menu chosen). All of it on a simulated recording
of every recording type (cc_evoked, cc_ramp, cc_short_square, cc_steps, dsi, membrane_test, vc_evoked, vc_spontaneous, vc_steps) and, when the lab's own recordings are named on the machine
(`tests/labdata.py`), on copies of three of them. It
fails when a control raises, changes nothing, makes an edit that Undo does not put back, or is disabled without
saying why. A control found on a screen and missing from this list fails the test too.

Columns: **What changed** is what the harness saw (the thing being edited, the picture, the rest of the screen, a
dialog, a file, the clipboard, a message, a window), over every state the control was pressed in. **Checked by
name** is what the test asserts for that control beyond a change. **Undo** says whether Undo put an edit back
(a dash: the control edits nothing, so there is nothing to undo). **Off when** is what a disabled control's
tooltip said.

535 controls, pressed 2210 times in 74 states. Last written from the simulated recordings and copies of the lab's own files.


## Workbench

| Screen | Control | Kind | What changed | Checked by name | Undo | Off when |
|---|---|---|---|---|---|---|
| Workbench | Send to a figure > The traces shown, as a panel of example traces | menu action | opened or closed a window; said something | the builder opens with a trace panel of the chosen sweeps over the time in view, the light's bars at their recorded times | – |  |
| Workbench | Send to a figure > This measurement's table, as a graph | menu action | opened or closed a window; said something | a second panel: the table of every row measured, the first measure against the sweep | – |  |
| Workbench | Keep this measurement | button | said something; the screen changed; what is edited changed | the measurement is in the project's kept list and in Results; Undo takes it out, Redo puts it back | restored |  |
| Workbench | Save as template... | button | asked in a dialog; said something; the screen changed; what is edited changed | the template is in the project and in the Templates list; Undo removes it | restored |  |
| Workbench | Plot: Measure | pick list | the screen changed |  | – |  |
| Workbench | Plot: against | pick list | the screen changed |  | – |  |
| Workbench | Sweeps shown and measured as | pick list | the screen changed; what is edited changed | "average": one trace named "average of n" is measured | restored |  |
| Workbench | Channel | pick list | the screen changed; what is edited changed | the table's units become the channel's (Baseline (V) on AM_Stim) | restored |  |
| Workbench | Stimulus | checkbox | the screen changed | a dashed line at each pulse (2), none when unticked | – | No stimulus was found in this recording, so there is none to mark. |
| Workbench | Command | checkbox | the screen changed | one more curve (the command) under the trace | – |  |
| Workbench | Previous sweep | button | the screen changed | the sweep before; from the first, round to the last | – | Only one sweep is chosen. |
| Workbench | Next sweep | button | the screen changed | the next sweep is drawn in bold and named ("sweep 1 (1 of n)"), its row selected; Page Down does the same | – | Only one sweep is chosen. |
| Workbench | Show the sweep alone | checkbox | the screen changed | the other sweeps are not drawn | – | Only one sweep is chosen. / Step to a sweep first (Sweep ▶). |
| Workbench | All sweeps | button | the screen changed | no sweep singled out; every sweep drawn again | – | No sweep is singled out. / Only one sweep is chosen. |
| Workbench | Cursors | checkbox | the screen changed | two lines on the trace; the readout gives each one's time and value (checked against the samples), their difference and the slope, and follows a cursor that is dragged | – |  |
| Workbench | Region to the cursors | button | the screen changed; what is edited changed | the selected region spans the cursors (from the pulse when it repeats on each pulse); Undo puts it back | restored | Tick Cursors first. |

## Workbench details

| Screen | Control | Kind | What changed | Checked by name | Undo | Off when |
|---|---|---|---|---|---|---|
| Workbench details | Add region | button | the screen changed; what is edited changed | one more region, named "region n", in the list and on the trace; Undo removes it | restored |  |
| Workbench details | Remove region | button | the screen changed; what is edited changed | the selected region is gone; the table keeps working when no measure region is left (it raised IndexError); Undo brings it back | restored |  |
| Workbench details | Region kind | pick list | the screen changed; what is edited changed |  | restored |  |
| Workbench details | Repeat on each stimulus pulse | checkbox | the screen changed; what is edited changed | the table has one row per pulse per sweep (2 x the sweeps), one per sweep when a single pulse is asked | restored | No stimulus was found in this recording, so a region cannot repeat on its pulses. |
| Workbench details | Options | checkbox | the screen changed |  | – |  |
| Workbench details | Direction of the peak | pick list | the screen changed; what is edited changed |  | restored |  |
| Workbench details | Onset by | pick list | the screen changed; what is edited changed |  | restored |  |
| Workbench details | Fit model | pick list | the screen changed; what is edited changed | the fit's words name tau, its residuals are drawn | restored |  |
| Workbench details | Fit on region | pick list | the screen changed; what is edited changed |  | restored |  |
| Workbench details | Add transform > Baseline: the mean of a region taken off | menu action | asked in a dialog; the screen changed; what is edited changed |  | restored |  |
| Workbench details | Add transform > Baseline: a sloping line through two regions taken off | menu action | asked in a dialog; the screen changed; what is edited changed |  | restored |  |
| Workbench details | Add transform > Low-pass (Bessel, 4th order) | menu action | asked in a dialog; the screen changed; what is edited changed | "low-pass 2000 Hz" in the list and in the measurement | restored |  |
| Workbench details | Add transform > High-pass (Butterworth, 2nd order) | menu action | asked in a dialog; the screen changed; what is edited changed |  | restored |  |
| Workbench details | Add transform > Notch (the mains and its harmonics) | menu action | asked in a dialog; the screen changed; what is edited changed |  | restored |  |
| Workbench details | Add transform > Gaussian smoothing | menu action | asked in a dialog; the screen changed; what is edited changed |  | restored |  |
| Workbench details | Add transform > Blank a stretch after each stimulus | menu action | asked in a dialog; the screen changed; what is edited changed |  | restored |  |
| Workbench details | Add transform > Subtract a trace | menu action | asked in a dialog; the screen changed; what is edited changed |  | restored |  |
| Workbench details | Add transform > Scale | menu action | asked in a dialog; the screen changed; what is edited changed |  | restored |  |
| Workbench details | Add transform > Offset | menu action | asked in a dialog; the screen changed; what is edited changed |  | restored |  |
| Workbench details | Remove transform | button | the screen changed; what is edited changed | the selected transform is gone, the other stays | restored | There is no transform to remove. |
| Workbench details | Clear transforms | button | the screen changed; what is edited changed | no transform left; the button then says there is none to clear | restored | There is no transform to clear. |
| Workbench details | Set onset | checkbox | said something; the screen changed | the mode is on (its button pressed, the line above the trace says so); a click on a sweep sets that sweep's onset, and its latency reads 6.0 ms | – |  |
| Workbench details | Set peak | checkbox | said something; the screen changed |  | – |  |
| Workbench details | Set threshold | checkbox | said something; the screen changed |  | – |  |
| Workbench details | Leave out a sweep | checkbox | said something; the screen changed | the clicked sweep leaves the table with the reason typed; pressed again the mode ends | – |  |
| Workbench details | Clear clicks and exclusions | button | the screen changed; what is edited changed | no onset set and no sweep left out; Undo restores them | restored | Nothing was set by a click and no sweep is left out. |
| Workbench details | Load kept | button | the screen changed | the kept measurement's fit and regions are back on the screen | – | Nothing is kept with this recording yet (Keep this measurement). |
| Workbench details | Delete kept | button | said something; the screen changed; what is edited changed | gone from the project; Undo brings it back | restored | Nothing is kept with this recording yet (Keep this measurement). |
| Workbench details | Apply here | button | the screen changed; what is edited changed | the template's fit and regions on this recording, marked as from the template | restored | No template is saved yet (Save as template...). |
| Workbench details | Delete template | button | said something; the screen changed; what is edited changed | gone from the project; Apply here then says no template is saved; Undo brings it back | restored | No template is saved yet (Save as template...). |
| Workbench details | Run on its protocol's files | button | said something; the screen changed; what is edited changed | every file of the protocol has the measurement kept; Undo takes the whole batch back | restored | No template is saved yet (Save as template...). |

## Figures

| Screen | Control | Kind | What changed | Checked by name | Undo | Off when |
|---|---|---|---|---|---|---|
| Figures | Across recordings | checkbox | the screen changed; what is edited changed | one point per analyzed recording; pressed again, the four graphs are back | – |  |
| Figures | Make a figure | button | opened or closed a window | the builder opens, empty | – |  |
| Figures | Show > (each graph the results give) | menu action | the screen changed; what is edited changed | that graph is drawn, named in the title and ticked in the menu; no two panels show the same graph | – |  |
| Figures | Edit | button | opened or closed a window | the builder opens on this graph (its table, its title); off, saying why, when nothing is drawn | – |  |
| Figures | ⤢ | button | the screen changed; what is edited changed | the graph fills the area; pressed again, the four are back | – |  |
| Figures | ⧉ | button | opened or closed a window | a window of its own, named after the graph | – |  |
| Figures | (QComboBox) | pick list | the screen changed; what is edited changed |  | – |  |
| Figures | Correlations | checkbox | the screen changed; what is edited changed |  | – |  |

## Figure builder

| Screen | Control | Kind | What changed | Checked by name | Undo | Off when |
|---|---|---|---|---|---|---|
| Figure builder | Add panel... | button | asked in a dialog | asks for the data, then for the plot; fills the empty panel, then adds a panel beside it (it filled the first panel again) | – |  |
| Figure builder | Open > A saved figure, to work on it again... | menu action | asked in a dialog |  | – |  |
| Figure builder | Open > Data for the selected panel... | menu action | asked in a dialog |  | – |  |
| Figure builder | Open > Raw traces from the recordings (pick sweeps, average them)... | menu action | asked in a dialog |  | – |  |
| Figure builder | Open > A table of this project... | menu action | asked in a dialog; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Open > A CSV or Excel file... | menu action | asked in a dialog; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Open > A picture of a plot (digitize it)... | menu action | asked in a dialog |  | – |  |
| Figure builder | Open > Remove this panel's data | menu action | the picture changed; the screen changed; what is edited changed |  | restored | This panel has no data to remove. |
| Figure builder | Undo | button | the picture changed; the screen changed; what is edited changed |  | restored | Nothing to undo yet. |
| Figure builder | Redo | button | the picture changed; the screen changed; what is edited changed |  | restored | Nothing to redo: no change was undone. |
| Figure builder | Arrange > A template > One panel (1 panel) | menu action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > A template > Two side by side (2 panels) | menu action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > A template > Two stacked (2 panels) | menu action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > A template > Three in a row (3 panels) | menu action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > A template > Two by two (4 panels) | menu action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > A template > Three by two (6 panels) | menu action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > A template > Wide on top, two below (3 panels) | menu action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > A template > Wide on top, three below (4 panels) | menu action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > A template > Tall on the left, two on the right (3 panels) | menu action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > A template > Traces on the left, three graphs on the right (4 panels) | menu action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > A template > One wide, then two by two (5 panels) | menu action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > A template > Three above, two below (5 panels) | menu action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > Columns of the grid > 1 column | menu action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > Columns of the grid > 2 columns | menu action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > Columns of the grid > 3 columns | menu action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > Columns of the grid > 4 columns | menu action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > Columns of the grid > 5 columns | menu action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > Columns of the grid > 6 columns | menu action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > Place every panel freely | menu action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Arrange > Put every panel back in the grid | menu action | the picture changed; the screen changed; what is edited changed |  | restored | No panel is placed freely. |
| Figure builder | Arrange > Align and space the freely placed panels > Align left edges | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs 2 freely placed panels (Place every panel freely). |
| Figure builder | Arrange > Align and space the freely placed panels > Align right edges | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs 2 freely placed panels (Place every panel freely). |
| Figure builder | Arrange > Align and space the freely placed panels > Align top edges | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs 2 freely placed panels (Place every panel freely). |
| Figure builder | Arrange > Align and space the freely placed panels > Align bottom edges | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs 2 freely placed panels (Place every panel freely). |
| Figure builder | Arrange > Align and space the freely placed panels > Centre on one vertical line | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs 2 freely placed panels (Place every panel freely). |
| Figure builder | Arrange > Align and space the freely placed panels > Centre on one horizontal line | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs 2 freely placed panels (Place every panel freely). |
| Figure builder | Arrange > Align and space the freely placed panels > Make the same width | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs 2 freely placed panels (Place every panel freely). |
| Figure builder | Arrange > Align and space the freely placed panels > Make the same height | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs 2 freely placed panels (Place every panel freely). |
| Figure builder | Arrange > Align and space the freely placed panels > Space evenly across | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs 3 freely placed panels (Place every panel freely). |
| Figure builder | Arrange > Align and space the freely placed panels > Space evenly down | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs 3 freely placed panels (Place every panel freely). |
| Figure builder | Arrange > Letter the panels A, B, C | menu action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Styles > Save this look as a style... | menu action | asked in a dialog; said something |  | – |  |
| Figure builder | Styles > Export the saved styles to a file... | menu action | asked in a dialog; said something; wrote a file |  | – |  |
| Figure builder | Styles > Import styles from a file... | menu action | asked in a dialog; said something |  | – |  |
| Figure builder | Styles > Use (a saved style) | menu action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Styles > Delete a saved style > (a saved style) | menu action | asked in a dialog; said something |  | – |  |
| Figure builder | Starters > Current-clamp step family | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs a current-step recording analyzed for intrinsic properties. |
| Figure builder | Starters > Light or stimulus responses: connected or not | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs a recording with a stimulus, tested for a connection (Connected or not). |
| Figure builder | Starters > Evoked PSC | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Its sweeps are a step family played under the stimulus, each at another level: there is no one response to average. Use “Light or stimulus responses: connected or not”. / Needs an evoked-response recording. |
| Figure builder | Starters > Paired pulses and trains | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs an evoked-response recording. |
| Figure builder | Starters > Spontaneous and miniature events | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs detected spontaneous or miniature events. |
| Figure builder | Starters > DSI | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs a DSI recording or a DSI cell. |
| Figure builder | Starters > Group comparison | menu action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Methods text | button | asked in a dialog |  | – |  |
| Figure builder | Save figure... | button | asked in a dialog; said something; wrote a file |  | – |  |
| Figure builder | Export... | button | asked in a dialog |  | – |  |
| Figure builder | + Copy as a new panel | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Remove panel | button | said something; the picture changed; the screen changed; what is edited changed |  | restored | A figure keeps at least one panel. |
| Figure builder | Help: Marks along x | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | + Add mark | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Remove selected | button | the picture changed; the screen changed; what is edited changed |  | restored | There is no mark to remove. / There is no reference line to remove. / There is no text to remove. |
| Figure builder | Help: Text | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | + Add text | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Theme | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Colours | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Font | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Axis lines | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Grid | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Background | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Background | button | asked in a dialog; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Text and axis colour | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Text and axis colour | button | asked in a dialog; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Panel letters | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: X axis title | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Y axis title | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: X range | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Y range | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Log axes | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Y axis starts at zero | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Reference lines | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | + Add reference line | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Groups' look | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Up | button | the picture changed; the screen changed; what is edited changed |  | restored | Select a group in the table first. / The selected group is already drawn first. / This panel draws no groups yet. |
| Figure builder | Down | button | the picture changed; the screen changed; what is edited changed |  | restored | Select a group in the table first. / This panel draws no groups yet. |
| Figure builder | Hide from the figure | button | the picture changed; the screen changed; what is edited changed |  | restored | Select a group in the table first. / This panel draws no groups yet. |
| Figure builder | Show | button | the picture changed; the screen changed; what is edited changed |  | restored | Select a group in the table first. / The selected group is already shown. / This panel draws no groups yet. |
| Figure builder | Reset to the theme | button | the picture changed; the screen changed; what is edited changed |  | restored | Every group's look already comes from the theme. |
| Figure builder | Help: Plot | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Group centre and spread | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Show every point | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Connect values from the same cell (repeated measures) | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Statistics | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Split into side-by-side panels by | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Y | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Groups | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Cell | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Each replicate is a | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Animal | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Keep only rows where... | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Add a condition | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Each point is | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Apply | button | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Change data... | button | asked in a dialog |  | – |  |
| Figure builder | Columns and rows... | button | the screen changed |  | – | This panel has no table yet: Change data... |
| Figure builder | Leave the selected rows out | button | the picture changed; the screen changed; what is edited changed | the rows are greyed in the table, out of the figure and of its n, and counted in the legend; Undo puts them back | restored | Select rows of the table first (rows already left out are greyed). / This panel has no table yet: Change data... |
| Figure builder | Put them back | button | the picture changed; the screen changed; what is edited changed |  | restored | Select rows that were left out (greyed) first. / This panel has no table yet: Change data... |
| Figure builder | Copy the table | button | put something on the clipboard; said something |  | – | This panel has no table yet: Change data... |
| Figure builder | Save the table... | button | asked in a dialog; said something; wrote a file |  | – | This panel has no table yet: Change data... |
| Figure builder | Figure: Add text on the page | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Figure: More of the look (axes, grid, background)... | button | the screen changed |  | – |  |
| Figure builder | Copy legend | button | put something on the clipboard |  | – |  |
| Figure builder | Figure size | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | The panel's table | right-click | showed a menu; the screen changed |  | – |  |
| Figure builder | The panel's table > Leave the selected rows out | right-click action | the picture changed; the screen changed; what is edited changed |  | restored | Select rows of the table first (rows already left out are greyed). / This panel has no table yet: Change data... |
| Figure builder | The panel's table > Put them back | right-click action | the picture changed; the screen changed; what is edited changed |  | restored | None of the selected rows is left out. / This panel has no table yet: Change data... |
| Figure builder | The panel's table > Copy the table | right-click action | put something on the clipboard; said something; the screen changed |  | – | This panel has no table yet: Change data... |
| Figure builder | Figure: Journal | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Figure: Width | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Figure: Letter the panels | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Figure: Letters as | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Figure: Colours | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Figure: Theme | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | What do you want to show? | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | The panel the controls below change | right-click | showed a menu; the screen changed |  | – |  |
| Figure builder | The panel the controls below change > Choose its data... | right-click action | asked in a dialog; the screen changed |  | – |  |
| Figure builder | The panel the controls below change > Choose how it is drawn... | right-click action | asked in a dialog; the screen changed |  | – | This panel has no data yet. |
| Figure builder | The panel the controls below change > See its data and statistics | right-click action | the screen changed |  | – | This panel has no data yet. |
| Figure builder | The panel the controls below change > Add text | right-click action | the picture changed; the screen changed; what is edited changed |  | restored | This panel has no data yet. |
| Figure builder | The panel the controls below change > Add an arrow with text | right-click action | the picture changed; the screen changed; what is edited changed |  | restored | This panel has no data yet. |
| Figure builder | The panel the controls below change > Add a stimulus bar | right-click action | the picture changed; the screen changed; what is edited changed |  | restored | This panel has no data yet. |
| Figure builder | The panel the controls below change > Add an inset | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | The panel the controls below change > Place the panels freely (drag and resize) | right-click action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | The panel the controls below change > Duplicate this panel | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | The panel the controls below change > Remove this panel | right-click action | said something; the picture changed; the screen changed; what is edited changed |  | restored | A figure keeps at least one panel. |
| Figure builder | The panel the controls below change > Move it earlier | right-click action | the picture changed; the screen changed; what is edited changed |  | restored | It is the first panel already. |
| Figure builder | The panel the controls below change > Move it later | right-click action | the picture changed; the screen changed; what is edited changed |  | restored | It is the last panel already. |
| Figure builder | (QTableWidget) | right-click | showed a menu; the screen changed |  | – |  |
| Figure builder | (QTableWidget) > Add a mark | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | (QTableWidget) > Add a text | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Theme | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Colours | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Bold titles | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | A different marker for each group | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Axis lines | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Grid | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Ticks | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Minor ticks | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Transparent background | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Frame around the legend | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel letters | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Log x axis | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Log y axis | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Y axis starts at zero | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Legend | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | (QTableWidget) > Add a reference line | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Plot | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Group centre and spread | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Spread drawn as | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Show every point | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Connect values from the same cell (repeated measures) | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Statistics | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Each replicate is a | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Each point is | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Zoom | pick list | the picture changed; the screen changed |  | – |  |
| Figure builder | Test | pick list | the picture changed; the screen changed; what is edited changed | the brackets and the legend carry the chosen test's p (Mann-Whitney); a paired test pairs the cells | restored |  |
| Figure builder | Compare | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Brackets say | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Paired design | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Draw the result on the graph | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Copy the result | button | put something on the clipboard |  | – |  |
| Figure builder | A group's shown in the figure | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | A group's colour | button | asked in a dialog; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | A group's marker | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | A group's line | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | A group's filled markers | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Split into side-by-side panels by | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Conditions are in separate columns... | button | asked in a dialog |  | – |  |
| Figure builder | Y | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Groups | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Cell | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Animal | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: X | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | X | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Time window | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Average into time bins of | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Show y relative to a baseline | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Fitted curve | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Fitted curve | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Second Y (right axis) | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Second Y (right axis) | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Error bars from | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Error bars from | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Show percentages | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Scale bars instead of axes | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Scale bars instead of axes | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Use the selected recording's typical response | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Choose traces... | button | asked in a dialog |  | – |  |
| Figure builder | The panel the controls below change > Change its data... | right-click action | asked in a dialog; the screen changed |  | – |  |
| Figure builder | (QTableWidget) > Duplicate this mark | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | (QTableWidget) > Remove the selected mark(s) | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | (QTableWidget) > Remove every mark | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | (QTableWidget) > Duplicate this text | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | (QTableWidget) > Remove the selected text(s) | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | (QTableWidget) > Remove every text | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | (QComboBox) | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Group: Colour | button | asked in a dialog; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Group: Marker | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Group: Line | pick list | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Group: Filled markers | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Group: Hide this group | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Group: Every group's look, and their order... | button | the screen changed |  | – |  |
| Figure builder | Comparison bracket: Every bracket says | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Comparison bracket: Hide this bracket | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Comparison bracket: Put every bracket back | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Reference line: Line | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Reference line: Remove it | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | X axis title: Logarithmic | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | X axis title: Axes cross at zero | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Y axis title: Logarithmic | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Y axis title: Starts at zero | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Y axis title: Axes cross at zero | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel title: Bold | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Mark along x: Remove it | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Text: Bold | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Text: Colour | button | asked in a dialog; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Text: Remove it | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Legend: Place | pick list | the picture changed; the screen changed; what is edited changed | the legend moves or goes ("No legend" closed the program: the pick list was deleted while still in use) | restored |  |
| Figure builder | Legend: Frame | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Legend: Remove the legend | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Potential label: Hide these labels | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Scale bars: Show the time bar | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Scale bars: Show the vertical bar | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Scale bars: The bars open to the | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Scale bars: Put them back at the corner | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Scale bars: Use axes instead of scale bars | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Stimulus bar: Stimulus | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Stimulus bar: Colour | button | asked in a dialog; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Stimulus bar: Same height and colour for every bar | button | the screen changed |  | – |  |
| Figure builder | Stimulus bar: Remove this bar | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Command trace: Show the command | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel letter: Bold | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel letter: Put it back at the corner | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Text on the page: Bold | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Text on the page: Remove it | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | X axis: Logarithmic | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | X axis: Axes cross at zero | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Y axis: Logarithmic | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Y axis: Starts at zero | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Y axis: Axes cross at zero | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel: Plot: (how the panel is drawn)... | button | asked in a dialog; the screen changed |  | – | This panel has no data yet: choose its data first. |
| Figure builder | Panel: Change its data... | button | asked in a dialog |  | – |  |
| Figure builder | Panel: Place the panels freely (drag and resize) | button | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel: Add text | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel: Add an arrow with text | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel: Add an inset to this panel | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel: Duplicate this panel | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel: Remove this panel | button | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel: All of this panel's settings... | button | the screen changed |  | – |  |
| Figure builder | Group | click | the screen changed |  | – |  |
| Figure builder | Group | double-click | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Group | Delete key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Group > Rewrite the group... | right-click action | the screen changed |  | – |  |
| Figure builder | Group > Hide the group | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Group > Copy the figure as a picture | right-click action | put something on the clipboard; said something; the screen changed |  | – |  |
| Figure builder | Group > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | Comparison bracket | click | the screen changed |  | – |  |
| Figure builder | Comparison bracket | drag | the picture changed; the screen changed; what is edited changed | the bracket stays at the height it was dragged to; its text can be replaced with a typed result | restored |  |
| Figure builder | Comparison bracket | arrow key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Comparison bracket | double-click | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Comparison bracket | Delete key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Comparison bracket > Rewrite the comparison bracket... | right-click action | the screen changed |  | – |  |
| Figure builder | Comparison bracket > Hide the comparison bracket | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Comparison bracket > Copy the figure as a picture | right-click action | put something on the clipboard; said something |  | – |  |
| Figure builder | Comparison bracket > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | Reference line | click | the screen changed |  | – |  |
| Figure builder | Reference line | double-click | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Reference line | Delete key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Reference line > Rewrite the reference line... | right-click action | the screen changed |  | – |  |
| Figure builder | Reference line > Remove the reference line | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Reference line > Copy the figure as a picture | right-click action | put something on the clipboard; said something |  | – |  |
| Figure builder | Reference line > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | X axis title | click | the screen changed |  | – |  |
| Figure builder | X axis title | double-click | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | X axis title | Delete key | said something; the screen changed |  | – |  |
| Figure builder | X axis title > Rewrite the x axis title... | right-click action | the screen changed |  | – |  |
| Figure builder | X axis title > Copy the figure as a picture | right-click action | put something on the clipboard; said something |  | – |  |
| Figure builder | X axis title > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | Y axis title | click | the screen changed |  | – |  |
| Figure builder | Y axis title | double-click | the picture changed; the screen changed; what is edited changed | a box opens over it with its words; Enter keeps the new words, Esc leaves the old; Undo takes them back | restored |  |
| Figure builder | Y axis title | Delete key | said something; the screen changed |  | – |  |
| Figure builder | Y axis title > Rewrite the y axis title... | right-click action | the screen changed |  | – |  |
| Figure builder | Y axis title > Copy the figure as a picture | right-click action | put something on the clipboard; said something |  | – |  |
| Figure builder | Y axis title > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | Panel title | click | the screen changed |  | – |  |
| Figure builder | Panel title | double-click | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel title | Delete key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel title > Rewrite the panel title... | right-click action | the screen changed |  | – |  |
| Figure builder | Panel title > Remove the panel title | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel title > Copy the figure as a picture | right-click action | put something on the clipboard; said something |  | – |  |
| Figure builder | Panel title > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | Mark along x | click | the screen changed |  | – |  |
| Figure builder | Mark along x | double-click | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Mark along x | Delete key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Mark along x > Rewrite the mark along x... | right-click action | the screen changed |  | – |  |
| Figure builder | Mark along x > Remove the mark along x | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Mark along x > Copy the figure as a picture | right-click action | put something on the clipboard; said something |  | – |  |
| Figure builder | Mark along x > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | Text | click | the screen changed |  | – |  |
| Figure builder | Text | drag | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Text | arrow key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Text | double-click | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Text | Delete key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Text > Rewrite the text... | right-click action | the screen changed |  | – |  |
| Figure builder | Text > Remove the text | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Text > Copy the figure as a picture | right-click action | put something on the clipboard; said something |  | – |  |
| Figure builder | Text > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | Legend | click | the screen changed |  | – |  |
| Figure builder | Legend | drag | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Legend | arrow key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Legend | double-click | the screen changed |  | – |  |
| Figure builder | Legend | Delete key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Legend > Remove the legend | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Legend > Copy the figure as a picture | right-click action | put something on the clipboard; said something |  | – |  |
| Figure builder | Legend > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | Potential label | click | the screen changed |  | – |  |
| Figure builder | Potential label | double-click | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Potential label | Delete key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Potential label > Rewrite the potential label... | right-click action | the screen changed |  | – |  |
| Figure builder | Potential label > Hide the potential label | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Potential label > Copy the figure as a picture | right-click action | put something on the clipboard; said something |  | – |  |
| Figure builder | Potential label > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | Scale bars | click | the screen changed |  | – |  |
| Figure builder | Scale bars | drag | the picture changed; the screen changed; what is edited changed | the corner where the two bars meet moves with the drag | restored |  |
| Figure builder | Scale bars | arrow key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Scale bars | double-click | the screen changed |  | – |  |
| Figure builder | Scale bars | Delete key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Scale bars > Remove the scale bars | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Scale bars > Copy the figure as a picture | right-click action | put something on the clipboard; said something |  | – |  |
| Figure builder | Scale bars > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | Stimulus bar | click | the screen changed |  | – |  |
| Figure builder | Stimulus bar | drag | the picture changed; the screen changed; what is edited changed | it moves up and down only: its time is the recording's | restored |  |
| Figure builder | Stimulus bar | arrow key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Stimulus bar | double-click | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Stimulus bar | Delete key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Stimulus bar > Rewrite the stimulus bar... | right-click action | the screen changed |  | – |  |
| Figure builder | Stimulus bar > Remove the stimulus bar | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Stimulus bar > Copy the figure as a picture | right-click action | put something on the clipboard; said something |  | – |  |
| Figure builder | Stimulus bar > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | Command trace | click | the screen changed |  | – |  |
| Figure builder | Command trace | double-click | the screen changed |  | – |  |
| Figure builder | Command trace | Delete key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Command trace > Remove the command trace | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Command trace > Copy the figure as a picture | right-click action | put something on the clipboard; said something |  | – |  |
| Figure builder | Command trace > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | Panel letter | click | the screen changed |  | – |  |
| Figure builder | Panel letter | drag | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel letter | arrow key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel letter | double-click | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel letter | Delete key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel letter > Rewrite the panel letter... | right-click action | the screen changed |  | – |  |
| Figure builder | Panel letter > Remove the panel letter | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel letter > Copy the figure as a picture | right-click action | put something on the clipboard; said something |  | – |  |
| Figure builder | Panel letter > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | Text on the page | click | the screen changed |  | – |  |
| Figure builder | Text on the page | drag | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Text on the page | arrow key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Text on the page | double-click | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Text on the page | Delete key | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Text on the page > Rewrite the text on the page... | right-click action | the screen changed |  | – |  |
| Figure builder | Text on the page > Remove the text on the page | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Text on the page > Copy the figure as a picture | right-click action | put something on the clipboard; said something |  | – |  |
| Figure builder | Text on the page > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | X axis | click | the screen changed |  | – |  |
| Figure builder | X axis | double-click | the screen changed |  | – |  |
| Figure builder | X axis | Delete key | said something; the screen changed |  | – |  |
| Figure builder | X axis > Copy the figure as a picture | right-click action | put something on the clipboard; said something |  | – |  |
| Figure builder | X axis > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | Y axis | click | the screen changed |  | – |  |
| Figure builder | Y axis | double-click | the screen changed |  | – |  |
| Figure builder | Y axis | Delete key | said something; the screen changed |  | – |  |
| Figure builder | Y axis > Copy the figure as a picture | right-click action | put something on the clipboard; said something |  | – |  |
| Figure builder | Y axis > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | Panel | click | the screen changed |  | – |  |
| Figure builder | Panel | drag | said something; the picture changed; the screen changed; what is edited changed | in the grid, it changes places with the panel it is dropped on; placed freely, it moves on the page (on the snap grid when one is set) | restored |  |
| Figure builder | Panel | arrow key | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel | double-click | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Panel | Delete key | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel > Change its data... | right-click action | asked in a dialog |  | – |  |
| Figure builder | Panel > Choose how it is drawn... | right-click action | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Panel > See its data and statistics | right-click action | the screen changed |  | – |  |
| Figure builder | Panel > Add text | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel > Add an arrow with text | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel > Add a stimulus bar | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel > Add an inset | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel > Place the panels freely (drag and resize) | right-click action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel > Duplicate this panel | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel > Remove this panel | right-click action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel > Move it earlier | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel > Move it later | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel > Copy the figure as a picture | right-click action | put something on the clipboard; said something |  | – |  |
| Figure builder | Panel > Export... | right-click action | asked in a dialog |  | – |  |
| Figure builder | Panel > Put the panels back in the grid | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel's handle (nw) | drag | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel's handle (n) | drag | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel's handle (ne) | drag | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel's handle (e) | drag | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel's handle (se) | drag | the picture changed; the screen changed; what is edited changed | a freely placed panel (or an inset) is resized | restored |  |
| Figure builder | Panel's handle (s) | drag | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel's handle (sw) | drag | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel's handle (w) | drag | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel: Inset of | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panel: Choose its data... | button | asked in a dialog |  | – |  |
| Figure builder | Panel: Make it a panel of its own | button | the picture changed; the screen changed; what is edited changed |  | restored |  |

## Data chooser

| Screen | Control | Kind | What changed | Checked by name | Undo | Off when |
|---|---|---|---|---|---|---|
| Data chooser | Cancel | button | opened or closed a window; what is edited changed |  | – |  |
| Data chooser | Use it | button | opened or closed a window; what is edited changed | the kind of data picked (a source that cannot be used now is greyed with its reason) | – |  |

## Plot chooser

| Screen | Control | Kind | What changed | Checked by name | Undo | Off when |
|---|---|---|---|---|---|---|
| Plot chooser | Plot: Every point with the centre and spread | checkbox | already chosen: choosing it again changes nothing |  | – |  |
| Plot chooser | Plot: Bar with every point | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Box plot with every point | checkbox | the screen changed; what is edited changed | the plot is named under the heading and Use it draws the panel that way | – |  |
| Plot chooser | Plot: Violin with every point | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Raincloud | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Centre and spread only | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Horizontal bars with every point | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Grouped bars | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Estimation plot | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: SuperPlot | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Time course | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Every cell over time | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Area under the time course | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Pulse profile | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Scatter | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Regression with its 95% band | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Line | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Step line | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: XY with error bars | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Bubble chart | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Hexagonal bins | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Heat map | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Cumulative distribution | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Histogram | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Density | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Q-Q plot | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Bars of counts | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Stacked bars | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Proportion with its 95% interval | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Pie | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Donut | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Trace | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Plot: Raster with its histogram | checkbox | the screen changed; what is edited changed |  | – |  |
| Plot chooser | Cancel | button | opened or closed a window; what is edited changed |  | – |  |
| Plot chooser | Use it | button | opened or closed a window; what is edited changed |  | – |  |

## Export

| Screen | Control | Kind | What changed | Checked by name | Undo | Off when |
|---|---|---|---|---|---|---|
| Export | Format | pick list | the screen changed; what is edited changed | a pixel format turns the resolution on and the text-as-shapes box off, each saying why | – |  |
| Export | Turn the text into shapes | checkbox | what is edited changed |  | – |  |
| Export | Cancel | button | opened or closed a window; what is edited changed |  | – |  |
| Export | Choose where to save... | button | opened or closed a window; what is edited changed |  | – |  |

## Trace picker

| Screen | Control | Kind | What changed | Checked by name | Undo | Off when |
|---|---|---|---|---|---|---|
| Trace picker | (QTableWidget) | right-click | showed a menu; the screen changed |  | – |  |
| Trace picker | A trace is drawn as | pick list | the screen changed; what is edited changed |  | – |  |
| Trace picker | Remove trace | button | the screen changed; what is edited changed |  | – |  |
| Trace picker | Up | button | the screen changed; what is edited changed |  | – |  |
| Trace picker | Down | button | the screen changed; what is edited changed |  | – | The selected trace is already drawn last. |
| Trace picker | Show its sweeps | button | the screen changed |  | – |  |
| Trace picker | + Add selected sweeps to it | button | the screen changed; what is edited changed |  | – |  |
| Trace picker | Remove selected sweeps from it | button | the screen changed; what is edited changed |  | – | None of the selected sweeps is in this trace. |
| Trace picker | Subtract each trace's baseline | checkbox | the screen changed; what is edited changed |  | – |  |
| Trace picker | (QTableWidget) > Add the selected sweeps as a new trace | right-click action | the screen changed; what is edited changed |  | – |  |
| Trace picker | (QTableWidget) > Select every sweep | right-click action | the screen changed |  | – |  |
| Trace picker | (QTableWidget) > Clear the selection | right-click action | the screen changed |  | – |  |
| Trace picker | Select | button | the screen changed |  | – |  |
| Trace picker | Align each sweep on | pick list | the screen changed; what is edited changed |  | – |  |
| Trace picker | Draw the new trace as | pick list | the screen changed; what is edited changed |  | – |  |
| Trace picker | Add as a trace | button | the screen changed; what is edited changed | the selected sweeps become a named trace; the dialog opens on a recording with several sweeps (it raised on every such recording) | – |  |
| Trace picker | Cancel | button | opened or closed a window; what is edited changed |  | – |  |
| Trace picker | Use these traces | button | opened or closed a window; what is edited changed |  | – |  |

## Digitizer

| Screen | Control | Kind | What changed | Checked by name | Undo | Off when |
|---|---|---|---|---|---|---|
| Digitizer | Log x axis | checkbox | the screen changed; what is edited changed |  | – |  |
| Digitizer | Log y axis | checkbox | the screen changed; what is edited changed |  | – |  |
| Digitizer | Trace a curve by its colour | checkbox | said something; the screen changed; what is edited changed |  | – |  |
| Digitizer | Click points | checkbox | already chosen: choosing it again changes nothing |  | – |  |
| Digitizer | Use this table | button | opened or closed a window; what is edited changed |  | – |  |
| Digitizer | Undo last click | button | the screen changed; what is edited changed |  | – |  |
| Digitizer | Clear points | button | the screen changed; what is edited changed |  | – |  |
| Digitizer | Calibrate again | button | said something; the screen changed; what is edited changed |  | – |  |
