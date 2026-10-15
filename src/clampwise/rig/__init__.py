"""The rig's own programs and hardware, read by Clampwise: the whole experiment's workflow on one screen, and what
the programs applied detected rather than typed in.

Each module reads one part of the rig, and only reads:

- ``multiclamp``: the amplifier, live, through MultiClamp 700B Commander's message library (getters only);
- ``labbook``: what Clampex did, from its Lab Book (protocols opened, files started and stopped, Membrane Tests);
- ``polyscan``: the Polygon's sessions, from PolyScan's own folders (patterns, triggers, light sources);
- ``camera``: Infinity Analyze's settings and captures (camera, calibration, the images saved with a cell);
- ``windows``: the rig programs' windows, found and shown inside Clampwise without touching them.

Nothing here writes to another program's files, opens a device another program owns, or calls anything that
changes the hardware.
"""
