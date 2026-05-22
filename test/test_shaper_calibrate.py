# Tests for automatic calibration of input shapers
#
# Copyright (C) 2026  Beattune
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import numpy as np
import pytest

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
