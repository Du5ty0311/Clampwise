# Every control, and what was checked

Written by `tools/audit/controls_doc.py` from the runs of `tests/test_every_control.py`; do not edit by hand.

The test presses every button, checkbox, pick list and menu action of the Workbench, the Figures screen, the
figure builder, the trace picker and the digitizer, each from the same starting state, on a simulated recording
of every recording type (cc_evoked, cc_ramp, cc_short_square, cc_steps, membrane_test, vc_evoked, vc_spontaneous, vc_steps) and, when `E:\Daniel` is on the machine, on copies of three of his files. It
fails when a control raises, changes nothing, makes an edit that Undo does not put back, or is disabled without
saying why. A control found on a screen and missing from this list fails the test too.

Columns: **What changed** is what the harness saw (the thing being edited, the picture, the rest of the screen, a
dialog, a file, the clipboard, a message, a window), over every state the control was pressed in. **Checked by
name** is what the test asserts for that control beyond a change. **Undo** says whether Undo put an edit back
(a dash: the control edits nothing, so there is nothing to undo). **Off when** is what a disabled control's
tooltip said.

222 controls, pressed 1637 times in 42 states. Last written from the simulated recordings and copies of Daniel's files.


## Workbench

| Screen | Control | Kind | What changed | Checked by name | Undo | Off when |
|---|---|---|---|---|---|---|
| Workbench | Keep this measurement | button | said something; the screen changed; what is edited changed | the measurement is in the project's kept list and in Results; Undo takes it out, Redo puts it back | restored |  |
| Workbench | Save as template... | button | asked in a dialog; said something; the screen changed; what is edited changed | the template is in the project and in the Templates list; Undo removes it | restored |  |
| Workbench | Plot: Measure | pick list | the screen changed |  | – |  |
| Workbench | Plot: against | pick list | the screen changed |  | – |  |
| Workbench | Sweeps shown and measured as | pick list | the screen changed; what is edited changed | "average": one trace named "average of n" is measured | restored |  |
| Workbench | Channel | pick list | the screen changed; what is edited changed | the table's units become the channel's (Baseline (V) on AM_Stim) | restored |  |
| Workbench | Stimulus | checkbox | the screen changed | a dashed line at each pulse (2), none when unticked | – | No stimulus was found in this recording, so there is none to mark. |
| Workbench | Command | checkbox | the screen changed | one more curve (the command) under the trace | – |  |

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
| Figure builder | Open data > Raw traces from the recordings (pick sweeps, average them)... | menu action | asked in a dialog |  | – |  |
| Figure builder | Open data > A table of this project... | menu action | asked in a dialog; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Open data > A CSV or Excel file... | menu action | asked in a dialog; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Open data > A saved figure (to edit it again)... | menu action | asked in a dialog |  | – |  |
| Figure builder | Open data > A picture of a plot (digitize it)... | menu action | asked in a dialog |  | – |  |
| Figure builder | Open data > Remove this panel's data | menu action | the picture changed; the screen changed; what is edited changed |  | restored | This panel has no data to remove. |
| Figure builder | Undo | button | never enabled |  | – | Nothing to undo yet. |
| Figure builder | Redo | button | never enabled |  | – | Nothing to redo: no change was undone. |
| Figure builder | Panels > 1 panel | menu action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panels > 2 panels | menu action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panels > 3 panels | menu action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panels > 4 panels | menu action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panels > 6 panels | menu action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panels > In 1 column | menu action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panels > In 2 columns | menu action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panels > In 3 columns | menu action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Panels > Label the panels A, B, C | menu action | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Figure size | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Styles > Save this look as a style... | menu action | asked in a dialog; said something |  | – |  |
| Figure builder | Styles > Export the saved styles to a file... | menu action | asked in a dialog; said something; wrote a file |  | – |  |
| Figure builder | Styles > Import styles from a file... | menu action | asked in a dialog; said something |  | – |  |
| Figure builder | Styles > Use (a saved style) | menu action | said something; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Styles > Delete a saved style > (a saved style) | menu action | asked in a dialog; said something |  | – |  |
| Figure builder | Presets > Current-clamp step family | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs a current-step recording analyzed for intrinsic properties. |
| Figure builder | Presets > Evoked PSC | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs an evoked-response recording. |
| Figure builder | Presets > Paired pulses and trains | menu action | the picture changed; the screen changed; what is edited changed |  | restored | Needs an evoked-response recording. |
| Figure builder | Presets > Spontaneous and miniature events | menu action | never enabled |  | – | Needs detected spontaneous or miniature events. |
| Figure builder | Presets > DSI | menu action | never enabled |  | – | Needs a DSI recording or a DSI cell. |
| Figure builder | Presets > Group comparison | menu action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Methods text | button | asked in a dialog |  | – |  |
| Figure builder | Save figure... | button | asked in a dialog; said something; wrote a file |  | – |  |
| Figure builder | Export... | button | asked in a dialog; said something; wrote a file |  | – |  |
| Figure builder | Copy caption | button | put something on the clipboard |  | – |  |
| Figure builder | Zoom | pick list | the picture changed |  | – |  |
| Figure builder | (FigureCanvasQTAgg) | right-click | showed a menu; the screen changed |  | – |  |
| Figure builder | (FigureCanvasQTAgg) > Duplicate this panel | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | (FigureCanvasQTAgg) > Remove this panel | right-click action | never enabled |  | – | A figure keeps at least one panel. |
| Figure builder | (FigureCanvasQTAgg) > Move it left | right-click action | never enabled |  | – | It is the first panel already. |
| Figure builder | (FigureCanvasQTAgg) > Move it right | right-click action | never enabled |  | – | It is the last panel already. |
| Figure builder | (FigureCanvasQTAgg) > Add a new panel | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | (FigureCanvasQTAgg) > Copy the figure as a picture | right-click action | put something on the clipboard; said something; the screen changed |  | – |  |
| Figure builder | (FigureCanvasQTAgg) > Export... | right-click action | asked in a dialog; said something; the screen changed |  | – |  |
| Figure builder | What do you want to show? | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | The panel the controls below change | right-click | showed a menu; the screen changed |  | – |  |
| Figure builder | The panel the controls below change > Duplicate this panel | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | The panel the controls below change > Remove this panel | right-click action | never enabled |  | – | A figure keeps at least one panel. |
| Figure builder | The panel the controls below change > Move it left | right-click action | never enabled |  | – | It is the first panel already. |
| Figure builder | The panel the controls below change > Move it right | right-click action | never enabled |  | – | It is the last panel already. |
| Figure builder | The panel the controls below change > Add a new panel | right-click action | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | + Add panel | button | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Remove panel | button | said something; the picture changed; the screen changed; what is edited changed |  | restored | A figure keeps at least one panel. |
| Figure builder | Help: Marks along x | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | (QTableWidget) | right-click | showed a menu; the screen changed |  | – |  |
| Figure builder | (QTableWidget) > Add a mark | right-click action | the screen changed; what is edited changed |  | restored |  |
| Figure builder | + Add mark | button | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Remove selected | button | never enabled |  | – | Remove the selected mark(s) from the figure (Delete) / Remove the selected reference line(s) from the figure (Delete) / Remove the selected text(s) from the figure (Delete) |
| Figure builder | Help: Text | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | (QTableWidget) > Add a text | right-click action | the screen changed; what is edited changed |  | restored |  |
| Figure builder | + Add text | button | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Theme | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Theme | pick list | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Colours | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Colours | pick list | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Font | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Bold titles | checkbox | the screen changed; what is edited changed |  | restored |  |
| Figure builder | A different marker for each group | checkbox | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Axis lines | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Axis lines | pick list | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Grid | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Grid | pick list | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Ticks | pick list | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Minor ticks | checkbox | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Background | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Background | button | asked in a dialog; the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Text and axis colour | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Text and axis colour | button | asked in a dialog; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Transparent background | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Frame around the legend | checkbox | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Panel letters | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Panel letters | pick list | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: X axis title | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Y axis title | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: X range | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Y range | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Log axes | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Log x axis | checkbox | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Log y axis | checkbox | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Y axis starts at zero | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Y axis starts at zero | checkbox | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Legend | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Reference lines | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | (QTableWidget) > Add a reference line | right-click action | the screen changed; what is edited changed |  | restored |  |
| Figure builder | + Add reference line | button | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Groups' look | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Up | button | never enabled |  | – | Select a group in the table first. / This panel draws no groups yet. |
| Figure builder | Down | button | never enabled |  | – | Select a group in the table first. / This panel draws no groups yet. |
| Figure builder | Hide from the figure | button | never enabled |  | – | Select a group in the table first. / This panel draws no groups yet. |
| Figure builder | Show | button | never enabled |  | – | Select a group in the table first. / This panel draws no groups yet. |
| Figure builder | Reset to the theme | button | never enabled |  | – | Every group's look already comes from the theme. |
| Figure builder | Help: Plot | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Plot | pick list | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Group centre and spread | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Group centre and spread | pick list | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Spread drawn as | pick list | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Show every point | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Show every point | checkbox | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Connect values from the same cell (repeated measures) | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Connect values from the same cell (repeated measures) | checkbox | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Statistics | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Statistics | checkbox | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Split into side-by-side panels by | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Y | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Groups | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Cell | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Each replicate is a | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Each replicate is a | pick list | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Animal | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Help: Keep only rows where... | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Add a condition | button | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Each point is | button | asked in a dialog; the screen changed |  | – |  |
| Figure builder | Each point is | pick list | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Apply | button | said something; the screen changed; what is edited changed |  | restored |  |
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
| Figure builder | Show percentages | checkbox | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Help: Scale bars instead of axes | button | asked in a dialog; the picture changed; the screen changed |  | – |  |
| Figure builder | Scale bars instead of axes | checkbox | the picture changed; the screen changed; what is edited changed |  | restored |  |
| Figure builder | Use the selected recording's typical response | button | the screen changed; what is edited changed |  | restored |  |
| Figure builder | Choose traces... | button | asked in a dialog; the picture changed |  | – |  |

## Trace picker

| Screen | Control | Kind | What changed | Checked by name | Undo | Off when |
|---|---|---|---|---|---|---|
| Trace picker | (QTableWidget) | right-click | showed a menu; the screen changed |  | – |  |
| Trace picker | A trace is drawn as | pick list | the screen changed; what is edited changed |  | – |  |
| Trace picker | Remove trace | button | the screen changed; what is edited changed |  | – |  |
| Trace picker | Up | button | the screen changed; what is edited changed |  | – |  |
| Trace picker | Down | button | never enabled |  | – | The selected trace is already drawn last. |
| Trace picker | Show its sweeps | button | the screen changed |  | – |  |
| Trace picker | + Add selected sweeps to it | button | the screen changed; what is edited changed |  | – |  |
| Trace picker | Remove selected sweeps from it | button | never enabled |  | – | None of the selected sweeps is in this trace. |
| Trace picker | Subtract each trace's baseline | checkbox | the screen changed |  | – |  |
| Trace picker | (QTableWidget) > Add the selected sweeps as a new trace | right-click action | the screen changed; what is edited changed |  | – |  |
| Trace picker | (QTableWidget) > Select every sweep | right-click action | the screen changed |  | – |  |
| Trace picker | (QTableWidget) > Clear the selection | right-click action | the screen changed |  | – |  |
| Trace picker | Select | button | the screen changed |  | – |  |
| Trace picker | Align each sweep on | pick list | the screen changed |  | – |  |
| Trace picker | Draw the new trace as | pick list | the screen changed |  | – |  |
| Trace picker | Add as a trace | button | the screen changed; what is edited changed | the selected sweeps become a named trace; the dialog opens on a recording with several sweeps (it raised on every such recording) | – |  |
| Trace picker | Cancel | button | opened or closed a window; what is edited changed |  | – |  |
| Trace picker | Use these traces | button | opened or closed a window; what is edited changed |  | – |  |

## Digitizer

| Screen | Control | Kind | What changed | Checked by name | Undo | Off when |
|---|---|---|---|---|---|---|
| Digitizer | Log x axis | checkbox | the screen changed |  | – |  |
| Digitizer | Log y axis | checkbox | the screen changed |  | – |  |
| Digitizer | Trace a curve by its colour | checkbox | said something; the screen changed; what is edited changed |  | – |  |
| Digitizer | Click points | checkbox | already chosen: choosing it again changes nothing |  | – |  |
| Digitizer | Use this table | button | opened or closed a window; what is edited changed |  | – |  |
| Digitizer | Undo last click | button | the screen changed; what is edited changed |  | – |  |
| Digitizer | Clear points | button | the screen changed; what is edited changed |  | – |  |
| Digitizer | Calibrate again | button | said something; the screen changed; what is edited changed |  | – |  |
