# How Clampwise 1.0 was validated

What was checked, against what, and what has not been checked yet. The numbers are from the checks run for this
release (October 2026).

## The analyses: known answers

`tools/audit/known_answers.py` generates recordings whose true values are known (simulated cells with set input
resistance, capacitance, time constants, spike thresholds, synaptic events and responses with known amplitudes and
kinetics, field responses, DSI suppression) and compares what Clampwise measures with the truth, each measure with
its own tolerance.

- **186 of 186 measures within tolerance.**
- The same harness runs as part of the test suite (`tests/`), with the rest of the unit tests: 338 passing.

Each analysis also states, in Help, the published method it follows (IPFX, NeuroExpress, Clampex/pCLAMP, and the
papers each cites).

## Reading files: Axon's own library

Every ABF file written during validation, by Clampex (7) and by Clampwise (16, episodic and gap-free up to 10
minutes, including a paused and resumed recording), was read by Axon's ABFFIO library (installed with pCLAMP) and by
Clampwise: **every value equal (0 digitizer counts different), every sweep start within 0.01 ms.** Reading is also
checked against pyabf.

## Recording: Clampwise against Clampex on a model cell

An Axon PATCH-1U model cell (CELL position) on a MultiClamp 700B and Digidata 1440A. The same protocol (an 11-sweep
I–V) was recorded 5 times by Clampex and 5 times by Clampwise, in current and in voltage clamp, and each file measured
the same way (`tools/rig/agreement.py`; differences of means with bootstrap 95% intervals):

| Mode | Measure | Clampwise | Clampex | Difference [95% interval] |
|---|---|---|---|---|
| Current clamp | Resistance from the I–V | 426.73 MΩ | 426.70 MΩ | +0.01% [−0.03, +0.04] |
| Current clamp | Input resistance / tau / Cm | 430.62 MΩ / 18.106 ms / 36.98 pF | 430.68 / 18.112 / 36.99 | all within 0.05%, every interval includes 0 |
| Voltage clamp | Resistance from the I–V | 534.05 MΩ | 533.73 MΩ | +0.06% [−0.09, +0.22] |

The two programs' averaged capacitive transients are equal sample by sample, and each step's edge falls on the same
sample.

## The membrane test: against Clampex's own

Clampwise's membrane test follows the pCLAMP 10 User Guide (pp. 191–193). On the same model cell, with the same pulse
(10 mV, 100 Hz), against Clampex's Membrane Test read from its window:

| | Clampwise | Clampex |
|---|---|---|
| Access resistance (Ra) | 12.25 MΩ (12.07–12.43) | 11.7–12.3 MΩ |
| Capacitance (Cm) | 29.0 pF (28.9–29.2) | 29.0–30.6 pF |
| Time constant | 0.35 ms | 0.346–0.352 ms |
| Membrane resistance (Rm) | 523 MΩ | 484–595 MΩ |

One difference is deliberate: Clampex keeps the steady currents in whole digitizer counts, so at a coarse amplifier
gain its Rt jumps between values (397 and 808 MΩ on this cell at 6.1 pA a count); Clampwise keeps the fraction.

## The amplifier's own readings

The 700B digitizes its own signals: its current meter against the Digidata's recording of the same current gave a
gain of 1.011 and an offset of −8.6 pA (1.4 counts). Commander's Auto whole-cell compensation estimated Rs 10.8 MΩ and
Cm 31.8 pF on the model cell (nominal 10 MΩ, 33 pF).

## Not yet verified

- The absolute voltage and current scale against an independent instrument (a multimeter): the amplifier's meter and
  the Digidata share the headstage's feedback resistor, so an error there would show in neither.
- Timing against an oscilloscope, and the input noise floor with the inputs terminated.
- Runs interleaved between the programs (the runs above were in blocks, so slow drift is not separated).
- Recording from real cells with Clampwise (the analyses have been used on many real recordings; Clampwise's own
  recording has been used on the model cell).
- Other digitizers and amplifiers: recording supports the Digidata 1440A and MultiClamp 700B only.
