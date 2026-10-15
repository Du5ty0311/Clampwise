"""The liquid junction potential between the pipette solution and the bath
(B20, item 3), by the generalized Henderson equation, as JPCalc computes it
(Barry & Lynch 1991, J Membr Biol 121:101; Barry 1994, J Neurosci Methods
51:107; Neher 1992, Methods Enzymol 207:123).

    V_LJP = (RT/F) * S_F / S_G * ln( G_pipette / G_bath )

    S_F = sum_i  z_i u_i (c_i,bath - c_i,pipette)
    S_G = sum_i  z_i^2 u_i (c_i,bath - c_i,pipette)
    G_x = sum_i  z_i^2 u_i c_i,x

with u_i the ion's mobility relative to K+ (its limiting equivalent
conductivity over K+'s) and concentrations in mM (activities are taken as
concentrations, as JPCalc does by default). The sign follows JPCalc: the bath
relative to the pipette; a membrane potential is corrected as
V_true = V_recorded - V_LJP (e.g. about +15 mV for a K-gluconate internal
against ACSF, so -60 mV recorded is -75 mV).

Mobilities: limiting equivalent conductivities at 25 C (CRC Handbook of
Chemistry and Physics, "Equivalent conductivity of electrolytes in aqueous
solution") for the inorganic ions, and JPCalc's published values for the
organic ones (Barry 1994 table; marked "verify"). Until a JPCalc number for the
internal is checked against this, the value is
shown as "computed by Clampwise (Henderson), not yet checked against JPCalc".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

R = 8.314462618        # J / (mol K)
F = 96485.33212        # C / mol

# ion -> (valence, mobility relative to K+). K+ limiting equivalent conductivity: 73.5 S cm2/eq.
_LAMBDA = {        # limiting equivalent conductivity, S cm2 / equivalent, 25 C (CRC)
    "Na": (1, 50.08), "K": (1, 73.48), "Li": (1, 38.66), "Cs": (1, 77.2), "Rb": (1, 77.8), "NH4": (1, 73.5),
    "H": (1, 349.65), "Ca": (2, 59.47), "Mg": (2, 53.0), "Ba": (2, 63.6), "Sr": (2, 59.4),
    "Cl": (-1, 76.31), "Br": (-1, 78.1), "F": (-1, 55.4), "HCO3": (-1, 44.5), "H2PO4": (-1, 36.0),
    "HPO4": (-2, 57.0), "SO4": (-2, 80.0), "OH": (-1, 198.0), "acetate": (-1, 40.9),
    # organic ions: JPCalc's relative mobilities x K+'s conductivity (Barry 1994; verify)
    "gluconate": (-1, 0.33 * 73.48), "methanesulfonate": (-1, 0.56 * 73.48), "MeSO3": (-1, 0.56 * 73.48),
    "HEPES": (-1, 0.30 * 73.48), "NMDG": (1, 0.38 * 73.48), "TEA": (1, 32.6), "QX314": (1, 0.24 * 73.48),
    "phosphocreatine": (-2, 0.36 * 73.48), "ATP": (-4, 0.16 * 73.48), "GTP": (-3, 0.16 * 73.48),
    "EGTA": (-2, 0.25 * 73.48), "BAPTA": (-2, 0.25 * 73.48), "glutamate": (-1, 0.26 * 73.48),
}
ALIASES = {"sodium": "Na", "potassium": "K", "chloride": "Cl", "calcium": "Ca", "magnesium": "Mg",
           "caesium": "Cs", "cesium": "Cs", "bicarbonate": "HCO3", "glu": "glutamate", "gluc": "gluconate",
           "ms": "methanesulfonate", "pcr": "phosphocreatine"}


def ion(name: str) -> tuple[int, float]:
    """(valence, mobility relative to K+) of an ion by name."""
    key = ALIASES.get(name.strip().lower(), name.strip())
    for k, v in _LAMBDA.items():
        if k.lower() == key.lower():
            z, lam = v
            return z, lam / _LAMBDA["K"][1]
    raise KeyError(f"no mobility for '{name}'; known ions: {', '.join(sorted(_LAMBDA))}")


@dataclass
class JunctionPotential:
    mv: float
    temperature_c: float
    pipette: dict[str, float]
    bath: dict[str, float]
    method: str = "generalized Henderson equation (Barry & Lynch 1991), as JPCalc"
    checked_against_jpcalc: bool = False

    def describe(self) -> str:
        check = "checked against JPCalc" if self.checked_against_jpcalc else "not yet checked against JPCalc"
        return (f"{self.mv:+.1f} mV (bath relative to pipette; a membrane potential is corrected as V - "
                f"{self.mv:.1f} mV), {self.method} at {self.temperature_c:g} C; {check}")


def henderson(pipette: dict[str, float], bath: dict[str, float], temperature_c: float = 25.0) -> JunctionPotential:
    """The junction potential of ``pipette`` against ``bath`` (ion -> mM)."""
    ions = set(pipette) | set(bath)
    sf = sg = gp = gb = 0.0
    for name in ions:
        z, u = ion(name)
        cp, cb = float(pipette.get(name, 0.0)), float(bath.get(name, 0.0))
        sf += z * u * (cb - cp)
        sg += z * z * u * (cb - cp)
        gp += z * z * u * cp
        gb += z * z * u * cb
    if gp <= 0 or gb <= 0:
        raise ValueError("both solutions need ions")
    rt_f = R * (temperature_c + 273.15) / F * 1e3          # mV
    if abs(sg) < 1e-12:
        return JunctionPotential(0.0, temperature_c, dict(pipette), dict(bath))
    v = rt_f * (sf / sg) * math.log(gp / gb)
    return JunctionPotential(v, temperature_c, dict(pipette), dict(bath))


# Typical solutions, as starting points to edit (mM).
ACSF = {"Na": 151.25, "K": 2.5, "Cl": 133.5, "Ca": 2.0, "Mg": 1.0, "HCO3": 26.0, "H2PO4": 1.25}
K_GLUCONATE_INTERNAL = {"K": 140.0, "gluconate": 130.0, "Cl": 10.0, "HEPES": 10.0, "Mg": 2.0, "Na": 2.0}
