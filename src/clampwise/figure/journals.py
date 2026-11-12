"""Journals' artwork rules as the figure builder's size presets, and the check an export runs against them.

Each rule was read from the journal's own author pages on 2026-11-11 (the vault note "Field standards for slice
electrophysiology figures and analysis" quotes them); a value a journal does not state is None, and then
Clampwise's own default stands (Arial 7 pt, lines at least 0.5 pt). ``verified`` is False where the journal's own
page could not be read and the numbers come from secondary guides: the check says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Journal:
    key: str
    name: str
    widths_mm: dict                       # "single", "one and a half", "double" -> mm (those the journal names)
    max_height_mm: float | None = None
    font: str = "Arial"
    min_pt: float | None = None           # text at final size
    max_pt: float | None = None
    text_pt: float = 7.0                  # what Clampwise sets the text to
    letter_pt: float = 9.0                # panel letters
    letter_style: str = "A"               # "A" or "a" (Style.label_style)
    letter_bold: bool = True
    min_line_pt: float | None = None
    formats: tuple = ("pdf", "eps", "tiff")
    dpi_line: int = 1200                  # line art exported as pixels
    dpi_halftone: int = 300
    colour: str = "RGB"
    outline_text: bool = False            # the journal wants text converted to outlines in vector files
    notes: tuple = ()                     # what it asks for that no check can see
    source: str = ""
    verified: bool = True


JOURNALS = {j.key: j for j in (
    Journal("jneurosci", "J Neurosci", {"single": 85.0, "one and a half": 116.0, "double": 176.0},
            formats=("tiff", "eps", "pdf"), dpi_line=1200, dpi_halftone=300, outline_text=True,
            notes=("Figures are submitted at the size they are to appear, as TIFF or EPS.",
                   "In EPS files the journal asks for fonts converted to outlines (Export: outline the text).",
                   "Remove top and right border lines that carry no scale (the Open axis lines do).",
                   "No two-bar graphs: state those two values in the text.",
                   "Report exact p values; give n as animals, slices and cells.",
                   "Magenta and green instead of red and green."),
            source="jneurosci.org/content/information-authors"),
    Journal("elife", "eLife", {"single": 100.0, "double": 200.0}, formats=("pdf", "eps", "tiff"), dpi_line=600,
            notes=("At least 10 cm wide at 300 dpi (20 cm for a full-page figure); RGB.",
                   "Legends give the tests, the exact n and the p values; source data go with each figure."),
            source="elife-rp.msubmit.net author instructions"),
    Journal("nature", "Nature", {"single": 89.0, "one and a half": 120.0, "double": 183.0}, max_height_mm=170.0,
            font="Arial", min_pt=5.0, max_pt=7.0, text_pt=6.5, letter_pt=8.0, letter_style="a",
            formats=("pdf", "eps"), dpi_line=600,
            notes=("Text stays editable: never outlined or rasterized.", "Helvetica or Arial, 5 to 7 pt; panel "
                   "letters 8 pt bold lower case."),
            source="research-figure-guide.nature.com; Nature's guide to preparing final artwork"),
    Journal("natneuro", "Nature Neuroscience", {"single": 88.0, "double": 180.0}, min_pt=5.0, max_pt=7.0,
            text_pt=6.5, letter_pt=8.0, letter_style="a", formats=("pdf", "eps"), dpi_line=600,
            notes=("At most 180 mm wide; 5 to 7 pt sans serif; at least 300 dpi.",
                   "The one-column width (88 mm) is not on the page read: check it before submitting.",
                   "Use scale bars; legends state the centre, the error bars, n, the test and P."),
            source="nature.com/neuro/submission-guidelines/aip-and-formatting"),
    Journal("neuron", "Neuron (Cell Press)", {"single": 85.0, "one and a half": 114.0, "double": 174.0},
            min_pt=6.0, max_pt=8.0, formats=("pdf", "eps", "tiff"), dpi_line=1000,
            notes=("These sizes come from secondary guides (cell.com would not serve its own page): check "
                   "cell.com/figureguidelines before submitting.",),
            source="secondary guides to cell.com/figureguidelines", verified=False),
    Journal("jphysiol", "J Physiol", {"single": 85.0, "double": 170.0}, max_height_mm=230.0, min_pt=8.0, max_pt=8.0,
            text_pt=8.0, letter_pt=10.0, formats=("pdf", "eps", "tiff"), dpi_line=600, dpi_halftone=600,
            notes=("Labelling 8 pt at final size, sans serif; panels A, B, C.",
                   "Every data point plotted when n is 30 or fewer; SD, not SEM; exact p to three significant "
                   "figures; asterisks alone are not accepted.",
                   "No information by colour only: vary markers or line types too."),
            source="jp.msubmit.net author requirements"),
    Journal("cercor", "Cerebral Cortex", {"single": 86.0, "double": 180.0}, max_height_mm=240.0,
            formats=("tiff", "eps"), dpi_line=1200, colour="CMYK",
            notes=("TIFF or EPS; colour as CMYK (Clampwise writes RGB: convert in the journal's tool or in "
                   "Illustrator).", "No faint or grey lines; alt text is required for every figure."),
            source="academic.oup.com/cercor General Instructions"),
)}
DEFAULT = "jneurosci"                      # Daniel's default until he says otherwise (2026-11-10)
MIN_LINE_PT = 0.5                          # Clampwise's own floor where a journal states none
MIN_TEXT_PT = 5.0


@dataclass
class Finding:
    """One thing an export would get wrong for the journal (``serious``), or should know (a note)."""
    text: str
    serious: bool = True
    where: str = ""                        # the panel's letter, when it is one panel's


def width_mm(journal: str, columns: str) -> float | None:
    j = JOURNALS.get(journal)
    return j.widths_mm.get(columns) if j else None


def check(journal: str, width_mm_: float, height_mm: float, texts: list[tuple[str, float, str]],
          lines: list[tuple[float, str]], fmt: str = "") -> list[Finding]:
    """What is wrong with a figure of this size for ``journal``. ``texts``: (the words, their size in pt, the
    panel) of every text drawn; ``lines``: (width in pt, the panel) of every line drawn; ``fmt``: the export's
    format, when one is chosen."""
    j = JOURNALS.get(journal)
    if j is None:                              # no journal chosen: only what does not print
        out = []
        small = [(w, s, p) for w, s, p in texts if w.strip() and s < MIN_TEXT_PT - 0.05]
        if small:
            w, s, p = min(small, key=lambda x: x[1])
            out.append(Finding(f"{len(small)} piece(s) of text are under {MIN_TEXT_PT:g} pt at final size (the "
                               f"smallest: “{w[:30]}” at {s:.1f} pt): too small to print legibly.", where=p))
        thin = [(w, p) for w, p in lines if 0 < w < MIN_LINE_PT - 0.01]
        if thin:
            out.append(Finding(f"{len(thin)} line(s) are thinner than {MIN_LINE_PT:g} pt: hairlines can vanish in "
                               "print.", serious=False, where=min(thin)[1]))
        return out
    out: list[Finding] = []
    allowed = sorted(j.widths_mm.values())
    if not any(abs(width_mm_ - w) <= 0.6 for w in allowed) and j.key not in ("elife",):
        if width_mm_ > allowed[-1] + 0.6:
            out.append(Finding(f"The figure is {width_mm_:.0f} mm wide: {j.name}'s widest is {allowed[-1]:g} mm."))
        else:
            names = ", ".join(f"{w:g} mm ({k})" for k, w in j.widths_mm.items())
            out.append(Finding(f"The figure is {width_mm_:.0f} mm wide: {j.name} sets figures at {names}.",
                               serious=False))
    if j.key == "elife" and width_mm_ < 100.0 - 0.6:
        out.append(Finding(f"The figure is {width_mm_:.0f} mm wide: eLife asks for at least 100 mm."))
    if j.max_height_mm and height_mm > j.max_height_mm + 0.6:
        out.append(Finding(f"The figure is {height_mm:.0f} mm high: {j.name}'s limit is {j.max_height_mm:g} mm."))
    lo = j.min_pt or MIN_TEXT_PT
    small = [(w, s, p) for w, s, p in texts if w.strip() and s < lo - 0.05]
    if small:
        w, s, p = min(small, key=lambda x: x[1])
        out.append(Finding(f"{len(small)} piece(s) of text are under {lo:g} pt at final size (the smallest: "
                           f"“{w[:30]}” at {s:.1f} pt)" + ("." if j.min_pt else f": {j.name} states no least "
                                                           "size; under 5 pt does not print legibly."), where=p))
    if j.max_pt:
        big = [(w, s, p) for w, s, p in texts if w.strip() and s > j.max_pt + 0.05 and s > j.letter_pt + 0.05]
        if big:
            w, s, p = max(big, key=lambda x: x[1])
            out.append(Finding(f"{len(big)} piece(s) of text are over {j.name}'s {j.max_pt:g} pt (the largest: "
                               f"“{w[:30]}” at {s:.1f} pt).", serious=False, where=p))
    floor = j.min_line_pt or MIN_LINE_PT
    thin = [(w, p) for w, p in lines if 0 < w < floor - 0.01]
    if thin:
        out.append(Finding(f"{len(thin)} line(s) are thinner than {floor:g} pt (the thinnest {min(thin)[0]:.2f} pt): "
                           "hairlines can vanish in print.", serious=False, where=min(thin)[1]))
    if fmt and fmt.lower().lstrip(".") not in j.formats + (("tif",) if "tiff" in j.formats else ()):
        out.append(Finding(f"{j.name} takes {', '.join(f.upper() for f in j.formats)}, not {fmt.upper().lstrip('.')}.",
                           serious=False))
    if not j.verified:
        out.append(Finding(f"{j.name}'s numbers here are from secondary guides, not its own page: check them.",
                           serious=False))
    return out
