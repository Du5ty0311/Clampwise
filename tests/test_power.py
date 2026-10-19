"""Low-power mode: on by default on a cheaper PC, chosen in Settings, and what it changes."""

from __future__ import annotations

import pytest

from clampwise.app import power


@pytest.mark.parametrize("cores, gb, cheap", [(2, 16, True), (4, 4, True), (4, 7.9, False), (8, 32, False),
                                              (4, 0, False)])
def test_a_pc_with_few_cores_or_little_memory_is_a_cheaper_one(cores, gb, cheap):
    assert power.cheap_pc(cores, gb) is cheap


def test_the_choice_kept_wins_over_the_pc_s_size():
    class S:
        def __init__(self, v):
            self.v = v

        def value(self, key, default=None):
            return self.v

    assert power.enabled(S("true")) and not power.enabled(S("false"))
    assert power.enabled(S(None)) == power.cheap_pc()
    assert power.memory_gb() > 0.5


def test_low_power_draws_less_and_looks_less_often():
    low, normal = power.limits(True), power.limits(False)
    assert low["draw_fps"] < normal["draw_fps"] and low["stats_s"] > normal["stats_s"]
    assert low["min_step"] == 2 and low["rig_s"] > normal["rig_s"]


def test_the_frame_rate_says_what_limits_it():
    from clampwise.app.screens.camera_panel import fps_limit

    assert "exposure sets it" in fps_limit(9.8, 9.8, 100.0, 1, 60.0)                 # 100 ms: 10 a second
    assert "x 4 averaged" in fps_limit(2.4, 2.4, 100.0, 4, 60.0)
    assert "readout sets it" in fps_limit(27.0, 27.0, 5.0, 1, 60.0)                   # 5 ms allows 200
    assert "capped at 15 a second" in fps_limit(60.0, 15.0, 10.0, 1, 15.0)
    assert "capped at 30 a second" in fps_limit(60.0, 30.0, 10.0, 1, 30.0)
    assert "this PC draws no faster" in fps_limit(60.0, 20.0, 10.0, 1, 30.0)


def test_settings_applies_it_to_the_camera_at_once(monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtWidgets

    from clampwise.app import theme
    from clampwise.app.screens import camera_panel
    from clampwise.app.window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(app, theme.LIGHT)
    w = MainWindow(theme.LIGHT, persist=False)
    cam = w.screens["live"].camera
    cam._thread = camera_panel.CameraThread()                  # not started: its limits only
    s = w.screens["settings"]
    s.low_power.setChecked(False)
    s._keep_low_power(False)
    assert cam._thread.show_every_s == pytest.approx(1 / 30) and cam._min_step == 1
    s.low_power.setChecked(True)
    assert cam._thread.show_every_s == pytest.approx(1 / 15) and cam._thread.stats_every_s == 1.0
    assert cam._min_step == 2
    cam._thread = None
    w.close()
