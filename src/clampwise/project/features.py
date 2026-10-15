"""The optional features (every one can be turned on and off). Each is declared once here; Settings > Features lists
them with the line saying what it does, the project keeps the choice, and
every export prints it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    key: str
    title: str
    what: str
    default: bool = True


FEATURES = (
    Feature("file_review", "File review on Check",
            "G and B mark a recording good or bad, with an optional reason; shown in Files and exported as a "
            "column. N goes to the next recording not reviewed."),
    Feature("notes", "Read each cell's Notes.txt",
            "Labels each file from the cell folder's Notes.txt (Depol (5s): 24125269...) and fills in the age, sex, "
            "region, cell type and drugs where nothing was typed; flags where a file disagrees with its note. The "
            "notes are only read, never written."),
    Feature("session_grouping", "Episodes (auto session grouping)",
            "Groups each cell's files into episodes (the recording before a step or induction, the step, the "
            "recording after; a condition change), from the notes or from the files' protocols, timing and "
            "content; Files shows each file's episode and any doubt."),
    Feature("rig_presets", "Rig-aware presets",
            "Recognizes the rig a file was recorded on from its header (digitizer, amplifier, main channel, "
            "protocols); a rig named once keeps its preset, applied to its files after, in any project."),
    Feature("heat_maps", "Heat maps",
            "Each graph's Show menu also offers the heat maps its results can make (every sweep against time, "
            "DSI episodes, event rates, time-frequency power, pulse ratios), and Across recordings the "
            "correlations between measures."),
    Feature("figure_layouts", "Figure presets",
            "The figure builder's Presets: the figures patch-clamp papers use (step family, evoked PSC, paired "
            "pulses, spontaneous events, DSI, group comparison), filled from the selected recording or cells at a "
            "journal's size, every panel editable."),
    Feature("example_traces", "Example-trace picker",
            "For a raw example, the response closest to its set's mean, the stimulus artifact blanked and marked, "
            "with scale bars."),
    Feature("figure_statistics", "Statistics in figures",
            "Comparisons on one value per cell: the difference with its 95% CI (bootstrap over cells, or animals "
            "then cells), Hedges' g, the tests reviewers ask for, a plain sentence each, and the p on the figure."),
    Feature("rig_records", "Read the rig's records",
            "Each recording is matched to what the rig's programs recorded of it: the acquisition's Lab Book (a file "
            "with no test step and nothing typed gets its Rs from the Membrane Tests logged before and after it) and "
            "the patterned light's session running when it was made (a line of unknown kind that fired while the "
            "light waited on its external trigger is light). Said in the results' notes; never instead of what the "
            "file itself records."),
    Feature("live_experiment", "Live (the rig in one place)",
            "A Live destination: the amplifier read directly while you patch, what the acquisition last did and its "
            "Membrane Tests, the patterned light's session, the camera's last picture, and the rig programs' own "
            "windows shown live, each brought forward with a click. It only reads; nothing on the rig is changed."),
    Feature("record", "Record in Clampwise",
            "Live's acquisition view can record by itself while the acquisition program is closed: one of your "
            "protocols played on the digitizer, the amplifier's mode and scales from its own program, each sweep drawn "
            "and measured as it comes, into an ABF file the acquisition program and Clampfit read."),
    Feature("amplifier_controls", "The amplifier's controls in Clampwise",
            "Live's channel panel gets the amplifier's settings and buttons (mode, holding, seal test, the Auto "
            "buttons, whole-cell, Rs compensation, bridge balance, outputs and filters, Zap), sent through its own "
            "program, which stays open and shows each change; that program's own panel can be used as before."),
    Feature("methods_text", "Auto-written Methods",
            "A Methods paragraph for a figure: the analyses its numbers come from, their settings and sources, "
            "what the figure shows, and the statistics."),
)


def describe(states: dict[str, bool]) -> str:
    """The features' states as one line, for exports."""
    return "; ".join(f"{f.title}: {'on' if states.get(f.key, f.default) else 'off'}" for f in FEATURES)
