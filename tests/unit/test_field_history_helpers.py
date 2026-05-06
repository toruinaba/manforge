"""Unit tests for FieldHistory cyclic-loading constructors."""

import numpy as np
import pytest

from manforge.simulation.types import FieldHistory, FieldType


# ---------------------------------------------------------------------------
# cyclic_strain
# ---------------------------------------------------------------------------

def test_cyclic_strain_shape():
    peaks = [5e-3, -5e-3, 3e-3]
    fh = FieldHistory.cyclic_strain(peaks, n_per_segment=10, ntens=6)
    assert fh.data.shape == (len(peaks) * 10, 6)


def test_cyclic_strain_peaks_match():
    peaks = [5e-3, -5e-3, 3e-3]
    nps = 10
    fh = FieldHistory.cyclic_strain(peaks, n_per_segment=nps, ntens=6, component=0)
    for k, p in enumerate(peaks):
        assert fh.data[(k + 1) * nps - 1, 0] == pytest.approx(p)


def test_cyclic_strain_other_components_zero():
    fh = FieldHistory.cyclic_strain([5e-3, -5e-3], n_per_segment=10, ntens=6, component=0)
    assert np.all(fh.data[:, 1:] == 0.0)


def test_cyclic_strain_increment_consistency():
    """Each segment should have a constant increment (uniform linspace)."""
    nps = 10
    fh = FieldHistory.cyclic_strain([5e-3, -5e-3], n_per_segment=nps, ntens=6)
    for seg in range(2):
        diff = np.diff(fh.data[seg * nps:(seg + 1) * nps, 0])
        assert np.allclose(diff, diff[0])


def test_cyclic_strain_field_type():
    fh = FieldHistory.cyclic_strain([5e-3], n_per_segment=5, ntens=1)
    assert fh.type == FieldType.STRAIN


def test_cyclic_strain_returns_field_history_instance():
    fh = FieldHistory.cyclic_strain([5e-3], n_per_segment=5, ntens=1)
    assert isinstance(fh, FieldHistory)


def test_cyclic_strain_default_ntens_is_one():
    fh = FieldHistory.cyclic_strain([5e-3, -5e-3], n_per_segment=10)
    assert fh.data.shape == (20, 1)


def test_cyclic_strain_strain_increment_matches_driver_view():
    """Cumsum of increments should reconstruct the original data."""
    fh = FieldHistory.cyclic_strain([5e-3, -5e-3, 5e-3], n_per_segment=15, ntens=1)
    data = fh.data[:, 0]
    # driver computes data[i] - data[i-1] with data[-1]=0 for i=0
    increments = np.diff(data, prepend=0.0)
    reconstructed = np.cumsum(increments)
    assert np.allclose(reconstructed, data)


def test_component_out_of_bounds():
    with pytest.raises(ValueError, match="component"):
        FieldHistory.cyclic_strain([5e-3], n_per_segment=5, ntens=3, component=3)


# ---------------------------------------------------------------------------
# triangular_strain
# ---------------------------------------------------------------------------

def test_triangular_strain_shape():
    fh = FieldHistory.triangular_strain(5e-3, n_cycles=3, n_per_segment=20, ntens=1)
    # 3 cycles × 2 half-cycles × 20 steps
    assert fh.data.shape == (3 * 2 * 20, 1)


def test_triangular_strain_n_cycles():
    amplitude = 5e-3
    fh = FieldHistory.triangular_strain(amplitude, n_cycles=3, n_per_segment=20, ntens=1)
    axial = fh.data[:, 0]
    # Count sign changes in the derivative (reversals)
    diff = np.diff(axial)
    sign_changes = np.sum(diff[:-1] * diff[1:] < 0)
    assert sign_changes == 3 * 2 - 1  # 5 reversals for 3 cycles


# ---------------------------------------------------------------------------
# sine_strain
# ---------------------------------------------------------------------------

def test_sine_strain_amplitude():
    amp = 5e-3
    fh = FieldHistory.sine_strain(amp, n_cycles=3, n_per_cycle=80, ntens=1)
    axial = fh.data[:, 0]
    assert np.max(axial) == pytest.approx(amp, rel=1e-3)
    assert np.min(axial) == pytest.approx(-amp, rel=1e-3)


def test_sine_strain_zero_crossings():
    n_cycles = 3
    fh = FieldHistory.sine_strain(5e-3, n_cycles=n_cycles, n_per_cycle=80, ntens=1)
    axial = fh.data[:, 0]
    crossings = np.sum(axial[:-1] * axial[1:] < 0)
    # 2 zero crossings per cycle expected (rough check)
    assert crossings >= 2 * n_cycles - 2


def test_sine_stress_field_type():
    fh = FieldHistory.sine_stress(250.0, n_cycles=2, n_per_cycle=40, ntens=6)
    assert fh.type == FieldType.STRESS


# ---------------------------------------------------------------------------
# decaying_cyclic_strain
# ---------------------------------------------------------------------------

def test_decaying_cyclic_decay():
    amp = 5e-3
    decay = 0.7
    nps = 10
    fh = FieldHistory.decaying_cyclic_strain(amp, n_cycles=4, decay=decay, n_per_segment=nps, ntens=1)
    axial = fh.data[:, 0]
    for k in range(4):
        expected_peak = amp * (decay ** k)
        actual_peak = axial[(2 * k) * nps + nps - 1]
        assert actual_peak == pytest.approx(expected_peak, rel=1e-10)


def test_decaying_cyclic_shape():
    fh = FieldHistory.decaying_cyclic_strain(5e-3, n_cycles=3, n_per_segment=10, ntens=1)
    assert fh.data.shape == (3 * 2 * 10, 1)


# ---------------------------------------------------------------------------
# stress variants
# ---------------------------------------------------------------------------

def test_cyclic_stress_field_type():
    fh = FieldHistory.cyclic_stress([250.0, -250.0], n_per_segment=10, ntens=6)
    assert fh.type == FieldType.STRESS


def test_triangular_stress_field_type():
    fh = FieldHistory.triangular_stress(250.0, n_cycles=2, n_per_segment=10, ntens=6)
    assert fh.type == FieldType.STRESS


def test_decaying_cyclic_stress_field_type():
    fh = FieldHistory.decaying_cyclic_stress(250.0, n_cycles=2, n_per_segment=10, ntens=6)
    assert fh.type == FieldType.STRESS
