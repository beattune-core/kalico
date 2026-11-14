# Tests for automatic calibration of input shapers
#
# Copyright (C) 2026  Beattune
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import numpy as np
import pytest

from klippy.extras import shaper_defs
from klippy.extras.shaper_calibrate import (
    CalibrationData,
    MAX_FREQ,
    MAX_SHAPER_FREQ,
    ShaperCalibrate,
)


def _make_calibration_data(peak_freq, peak_width=3.0, n_points=2000):
    """Synthetic CalibrationData with a Lorentzian peak at peak_freq Hz."""
    freq_bins = np.linspace(0.0, MAX_FREQ, n_points)
    psd = (peak_width / 2.0) ** 2 / (
        (freq_bins - peak_freq) ** 2 + (peak_width / 2.0) ** 2
    )
    cd = CalibrationData(
        freq_bins.copy(),
        psd.copy(),
        psd.copy() / 3.0,
        psd.copy() / 3.0,
        psd.copy() / 3.0,
    )
    cd.set_numpy(np)
    cd.normalize_to_frequencies()
    return cd


def test_ceiling_extends_when_peak_above_150():
    """Peak at 160 Hz: ceiling must be extended and result freq > MAX_SHAPER_FREQ."""
    sc = ShaperCalibrate(printer=None)
    cd = _make_calibration_data(peak_freq=160.0)
    best, _ = sc.find_best_shaper(cd)
    assert best.freq > MAX_SHAPER_FREQ


def test_no_extension_when_peak_well_below_ceiling():
    """Peak at 100 Hz: no extension, result freq stays within MAX_SHAPER_FREQ."""
    sc = ShaperCalibrate(printer=None)
    cd = _make_calibration_data(peak_freq=100.0)
    best, _ = sc.find_best_shaper(cd)
    assert best.freq <= MAX_SHAPER_FREQ


def test_explicit_shaper_freqs_not_extended():
    """Caller-supplied shaper_freqs ceiling is never overridden by the retry logic."""
    sc = ShaperCalibrate(printer=None)
    cd = _make_calibration_data(peak_freq=160.0)
    # Explicit ceiling of 145 Hz — should not be extended.
    best, _ = sc.find_best_shaper(cd, shaper_freqs=(None, 145.0, None))
    assert best.freq <= 145.0


def _cfg(name):
    return next(c for c in shaper_defs.INPUT_SHAPERS if c.name == name)


def _fit(
    sc,
    cd,
    name,
    shaper_freqs=None,
    max_smoothing=None,
    max_freq=None,
    excited_freq=None,
):
    return sc.fit_shaper(
        _cfg(name),
        cd,
        shaper_freqs,
        None,
        None,
        max_smoothing,
        None,
        max_freq,
        excited_freq,
    )


def test_fit_shaper_no_extension_when_excited_range_below_ceiling():
    sc = ShaperCalibrate(printer=None)
    res = _fit(sc, _make_calibration_data(160.0), "zv", excited_freq=135.0)
    assert res.freq < MAX_SHAPER_FREQ


def test_fit_shaper_extension_capped_by_excited_freq():
    sc = ShaperCalibrate(printer=None)
    cd = _make_calibration_data(160.0)
    res = _fit(sc, cd, "zv", max_freq=202.5, excited_freq=155.0)
    assert MAX_SHAPER_FREQ < res.freq < 155.0
    assert len(res.vals) == int((cd.freq_bins <= 202.5).sum())


def test_find_best_shaper_forwards_excited_freq():
    sc = ShaperCalibrate(printer=None)
    _, all_shapers = sc.find_best_shaper(
        _make_calibration_data(160.0), max_freq=202.5, excited_freq=135.0
    )
    assert all(s.freq < MAX_SHAPER_FREQ for s in all_shapers)


@pytest.mark.parametrize("name", ["zv", "mzv"])
def test_fit_shaper_extends_each_shaper_past_ceiling(name):
    sc = ShaperCalibrate(printer=None)
    res = _fit(sc, _make_calibration_data(160.0), name)
    assert MAX_SHAPER_FREQ < res.freq < MAX_FREQ


def test_fit_shaper_unchanged_when_peak_well_below_ceiling():
    sc = ShaperCalibrate(printer=None)
    cd = _make_calibration_data(100.0)
    auto = _fit(sc, cd, "mzv")
    base = _fit(sc, cd, "mzv", shaper_freqs=(None, MAX_SHAPER_FREQ, None))
    assert auto.freq == base.freq and auto.score == base.score


def test_fit_shaper_explicit_end_disables_extension():
    sc = ShaperCalibrate(printer=None)
    res = _fit(
        sc,
        _make_calibration_data(160.0),
        "zv",
        shaper_freqs=(None, 145.0, None),
    )
    assert res.freq < 145.0


def test_fit_shaper_explicit_list_disables_extension():
    sc = ShaperCalibrate(printer=None)
    res = _fit(
        sc, _make_calibration_data(160.0), "zv", shaper_freqs=[120.0, 140.0]
    )
    assert res.freq in (120.0, 140.0)


def test_fit_shaper_step_only_tuple_still_extends():
    sc = ShaperCalibrate(printer=None)
    res = _fit(
        sc, _make_calibration_data(160.0), "zv", shaper_freqs=(None, None, 0.5)
    )
    assert res.freq > MAX_SHAPER_FREQ


@pytest.mark.parametrize("max_freq", [152.0, 170.0])
def test_fit_shaper_extension_capped_by_caller_max_freq(max_freq):
    sc = ShaperCalibrate(printer=None)
    cd = _make_calibration_data(160.0)
    res = _fit(sc, cd, "zv", max_freq=max_freq)
    assert res.freq < max_freq
    assert res.freq > MAX_SHAPER_FREQ
    assert len(res.vals) == int((cd.freq_bins <= max_freq).sum())


def test_fit_shaper_extension_respects_max_smoothing():
    sc = ShaperCalibrate(printer=None)
    res = _fit(sc, _make_calibration_data(160.0), "mzv", max_smoothing=0.05)
    assert res.freq > MAX_SHAPER_FREQ
    assert res.smoothing <= 0.05


def test_find_best_shaper_extends_non_winning_shapers():
    sc = ShaperCalibrate(printer=None)
    _, all_shapers = sc.find_best_shaper(_make_calibration_data(160.0))
    by_name = {s.name: s for s in all_shapers}
    assert by_name["zv"].freq > MAX_SHAPER_FREQ
    assert by_name["mzv"].freq > MAX_SHAPER_FREQ


@pytest.mark.parametrize("max_freq", [152.0, 170.0])
def test_find_best_shaper_vals_length_follows_max_freq(max_freq):
    sc = ShaperCalibrate(printer=None)
    cd = _make_calibration_data(160.0)
    _, all_shapers = sc.find_best_shaper(cd, max_freq=max_freq)
    n = int((cd.freq_bins <= max_freq).sum())
    assert all(len(s.vals) == n for s in all_shapers)
