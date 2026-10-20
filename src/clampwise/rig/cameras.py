"""Which camera the rig has, and so which library takes it: a Basler camera through pylon (rig.basler), a Lumenera
INFINITY through Lumenera's library (rig.lucam). Told by the USB devices Windows lists (nothing is opened); with no
known camera listed (a GigE camera is on the network, not USB), by the library this PC has."""

from __future__ import annotations

from . import basler, hardware, lucam


def kind(devices=None) -> str:
    """'basler' or 'lumenera'."""
    try:
        listed = (devices or hardware.usb_devices)()
    except Exception:
        listed = []
    for instance, description in listed:
        d = hardware.identify(instance, description)
        if d is not None and d.role == "camera":
            return "basler" if d.name.startswith("Basler") else "lumenera"
    if lucam.find_library() is None and basler.find_library() is not None:
        return "basler"
    return "lumenera"


def make():
    """The camera, not yet opened (CameraThread opens it on its own thread)."""
    return basler.Camera() if kind() == "basler" else lucam.Camera()
