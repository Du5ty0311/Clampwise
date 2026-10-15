"""Presets (B18, item 10): named, versioned sets of settings for an experiment
type, stored in the project and printed in every export. A preset holds the
settings of each analysis that differ from the defaults, the shared settings,
and the protocol mappings; applying one makes them the project's settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..analysis import routing

FORMAT = "clampwise-preset"
VERSION = 1


@dataclass
class Preset:
    name: str
    description: str = ""
    settings: dict[str, dict] = field(default_factory=dict)       # analysis -> {key: value}
    shared: dict = field(default_factory=dict)
    mappings: list[routing.Mapping] = field(default_factory=list)
    version: int = 1
    builtin: bool = False
    source: str = ""                    # where the settings come from (a paper, a decision)
    legacy: dict = field(default_factory=dict)       # an imported Synaptoscope preset's own settings, kept as they were

    def to_dict(self) -> dict:
        return {"format": FORMAT, "format_version": VERSION, "name": self.name, "description": self.description,
                "settings": self.settings, "shared": self.shared, "version": self.version, "source": self.source,
                "mappings": [{"pattern": m.pattern, "recording_type": m.recording_type, "analyses": list(m.analyses),
                              "stimulus_channel": m.stimulus_channel, "stimulus_kind": m.stimulus_kind}
                             for m in self.mappings],
                "legacy": self.legacy}

    @classmethod
    def from_dict(cls, d: dict) -> "Preset":
        if d.get("format") not in (None, FORMAT):
            raise ValueError("not a Clampwise preset")
        return cls(d["name"], d.get("description", ""), dict(d.get("settings", {})), dict(d.get("shared", {})),
                   [routing.Mapping(m["pattern"], m["recording_type"], tuple(m.get("analyses", ())),
                                    m.get("stimulus_channel", ""), m.get("stimulus_kind", ""))
                    for m in d.get("mappings", [])], int(d.get("version", 1)), False, d.get("source", ""),
                   dict(d.get("legacy", {})))

    def describe(self) -> str:
        n = sum(len(v) for v in self.settings.values()) + len(self.shared)
        return f"{self.name} (v{self.version}): {n} setting(s), {len(self.mappings)} mapping(s)"


# Shipped with the app. Settings not named keep the analyses' defaults; each
# built-in preset gains its settings as the analyses it concerns are built.
BUILTIN: dict[str, Preset] = {
    "Defaults": Preset("Defaults", "Every analysis at its default settings.", builtin=True,
                       source="the registry's defaults"),
    "CA3 sIPSC DSI (carbachol)": Preset(
        "CA3 sIPSC DSI (carbachol)",
        "CA3 pyramidal cells in voltage clamp at -70 mV with a chloride-rich internal, carbachol-driven sIPSCs "
        "(no TTX) and DSI steps to 0 mV. Holding current is flagged beyond 500 pA (Hartzell et al. 2018, CsCl "
        "internal). Event template: rise 0.5 ms, decay 10 ms (matching an averaged sIPSC with a 10-90% rise of "
        "0.65 ms and a decay of 9-11 ms), applied to the whole experiment.",
        {"quality": {"max_holding_pa": 500.0},
         "events": {"template_rise_ms": 0.5, "template_decay_ms": 10.0,
                    "template_origin": "averaged sIPSCs (CA3 sIPSC DSI preset)"}},
        builtin=True, source=""),
    "Evoked IPSC DSI": Preset(
        "Evoked IPSC DSI",
        "Evoked-IPSC DSI: the last 4 responses before each step (4 sweeps) form the baseline; the first 10 "
        "responses after each step (40 s) are expressed as % of that baseline and averaged over the steps. Also "
        "applies when the step and the responses are recorded in separate files.",
        {"dsi": {"baseline_responses": 4, "recovery_responses": 10},
         "dsi_cell": {"baseline_responses": 4, "recovery_responses": 10}},
        builtin=True, source=""),
}
