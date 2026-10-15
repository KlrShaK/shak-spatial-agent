"""Metric-speed regressions: calibrated selection, noise rejection and observed scope."""

import numpy as np
import pytest

from d4rt_agent.agent_v5 import measure, synthetic
from d4rt_agent.agent_v5.program import Measurement
from d4rt_agent.agent_v5.research.counterexamples import track, workspace
from d4rt_agent.agent_v5.tools import TOOLS


def speed_workspace(x, *, fps=20, noise=0, present=None):
    x = np.asarray(x, float)
    scene, _ = synthetic.build_scene(len(x), fps, metres_per_unit=2)
    tr = track(np.column_stack([x, np.zeros(len(x)), np.full(len(x), 4)]), present=present)
    tr.noise = noise
    return workspace(scene, {"e1": [tr]}), tr


def measure_speed(ws):
    return measure.measure_metric_speed(ws.scene(), ws.program,
        Measurement("m", "speed_metric", ["e1"], "value"), ws._tracks)


@pytest.mark.parametrize("mode", ("net", "path"))
def test_metric_speed_mode_selects_the_calibrated_quantity(monkeypatch, mode):
    t = np.arange(81) / 20
    ws, _ = speed_workspace(np.minimum(t, 4 - t))
    monkeypatch.setitem(measure.TUNING, "speed_mode", mode)
    result = measure_speed(ws)
    assert result.available and result.value["mode"] == mode
    assert result.value["net_mps"] == pytest.approx(0, abs=1e-10)
    assert result.value["path_mps"] == pytest.approx(2)
    assert result.value["mps"] == pytest.approx(result.value[f"{mode}_mps"])


@pytest.mark.parametrize("mode", ("net", "path"))
def test_metric_speed_scale_multiplier_affects_both_metric_readings(monkeypatch, mode):
    ws, _ = speed_workspace(np.arange(41) / 20)
    monkeypatch.setitem(measure.TUNING, "speed_mode", mode)
    monkeypatch.setitem(measure.TUNING, "speed_scale_mult", 1)
    original = measure_speed(ws).value
    monkeypatch.setitem(measure.TUNING, "speed_scale_mult", 1.5)
    scaled = measure_speed(ws).value
    for key in ("mps", "net_mps", "path_mps", "metres_per_unit"):
        assert scaled[key] == pytest.approx(1.5 * original[key])
    for key in ("units_per_s", "net_units_per_s", "path_units_per_s"):
        assert scaled[key] == pytest.approx(original[key])


def test_metric_path_speed_uses_coarse_steps_and_subtracts_tracking_noise(monkeypatch):
    # This stationary track has 0.1-unit per-frame jitter and a large false raw path speed.
    x = 0.05 * (-1.0) ** np.arange(61)
    ws, _ = speed_workspace(x, noise=0.06)
    monkeypatch.setitem(measure.TUNING, "speed_mode", "path")
    monkeypatch.setitem(measure.TUNING, "speed_step_s", 0.5)
    coarse = measure_speed(ws)
    raw_rate = np.abs(np.diff(x)).sum() / 3
    assert raw_rate == pytest.approx(2)
    assert coarse.value["path_units_per_s"] == pytest.approx(0)
    monkeypatch.setitem(measure.TUNING, "speed_step_s", 0.05)
    monkeypatch.setitem(measure.TUNING, "speed_noise_mult", 0)
    uncorrected = measure_speed(ws)
    assert uncorrected.value["path_units_per_s"] == pytest.approx(raw_rate)
    monkeypatch.setitem(measure.TUNING, "speed_noise_mult", 2)
    corrected = measure_speed(ws)
    assert corrected.value["path_units_per_s"] == pytest.approx(0)


def test_metric_path_speed_never_bridges_visibility_or_depth_gaps(monkeypatch):
    t = np.arange(61) / 20
    x = np.where(t <= 1, 0, 100)
    present = (t <= 1) | (t >= 2)
    ws, tr = speed_workspace(x, present=present)
    monkeypatch.setitem(measure.TUNING, "speed_mode", "path")
    value = measure_speed(ws).value
    assert value["path_units_per_s"] == 0
    assert value["path_duration_s"] == pytest.approx(2)
    assert value["observed_duration_s"] == pytest.approx(2)
    missing = ~tr.present.copy()
    tr.present[:] = True
    tr.smooth[missing] = np.nan
    assert measure_speed(ws).value["path_units_per_s"] == 0


def test_metric_speed_uses_actual_timestamps_on_sparse_geometry(monkeypatch):
    ws, tr = speed_workspace(np.zeros(11))
    scene = ws.scene()
    scene.gf_index = np.array([0, 1, 2, 10, 20, 30, 50, 70, 90, 110, 130])
    tr.smooth[:, 0] = scene.t_s
    monkeypatch.setitem(measure.TUNING, "speed_mode", "path")
    value = measure_speed(ws).value
    assert value["path_units_per_s"] == pytest.approx(1)
    assert value["net_units_per_s"] == pytest.approx(1)


def test_disconnected_samples_do_not_establish_zero_path_speed(monkeypatch):
    present = np.arange(11) % 2 == 0
    ws, _ = speed_workspace(np.arange(11), present=present)
    monkeypatch.setitem(measure.TUNING, "speed_mode", "path")
    result = measure_speed(ws)
    assert not result.available
    monkeypatch.setitem(measure.TUNING, "speed_mode", "net")
    result = measure_speed(ws)
    assert result.available and result.value["path_units_per_s"] is None
    assert result.value["path_mps"] is None
    report = TOOLS["speed"](ws, {"entity": "e1"})
    assert "path rate unavailable" in report.text


def test_speed_tool_labels_the_selected_estimator_and_reports_both_rates(monkeypatch):
    t = np.arange(81) / 20
    ws, _ = speed_workspace(np.minimum(t, 4 - t))
    monkeypatch.setitem(measure.TUNING, "speed_mode", "net")
    report = TOOLS["speed"](ws, {"entity": "e1"})
    assert "selected net estimate" in report.text
    assert "displacement rate" in report.text and "noise-corrected path rate" in report.text
