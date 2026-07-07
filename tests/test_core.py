"""Headless tests for the Qt-free core: FluxModel and the array loaders.

These import no PyQt6 and need no display.
"""

import os

import numpy as np
import pytest

from margaret.core.flux_model import FluxModel
from margaret.io.arrays import list_arrays, load_array, save_arrays


# --------------------------------------------------------------------------- #
# FluxModel.canonicalize
# --------------------------------------------------------------------------- #
def test_canonicalize_orderings_agree():
    base = np.arange(50 * 8, dtype=float).reshape(50, 8)  # (I, G)
    m = FluxModel()
    m.axes = ("I", "G")
    from_ig = m.canonicalize(base, ["I", "G"])
    from_gi = m.canonicalize(base.T.copy(), ["G", "I"])
    assert np.array_equal(from_ig, base)
    assert np.array_equal(from_gi, base)
    assert np.array_equal(from_ig, from_gi)


def test_canonicalize_1d_becomes_single_group():
    m = FluxModel()
    m.axes = ("I", "G")
    out = m.canonicalize(np.arange(10, dtype=float), ["I", "G"])
    assert out.shape == (10, 1)


def test_canonicalize_ndim_mismatch_raises():
    m = FluxModel()
    m.axes = ("I", "G")
    with pytest.raises(ValueError):
        m.canonicalize(np.zeros((4, 5, 6)), ["I", "G"])


def test_canonicalize_time_permutation():
    cube = np.arange(50 * 8 * 5, dtype=float).reshape(50, 8, 5)  # (I, G, T)
    m = FluxModel()
    m.axes = ("I", "G", "T")
    # Stored as (T, G, I); declaring that order must recover the canonical cube.
    permuted = np.transpose(cube, (2, 1, 0))
    out = m.canonicalize(permuted, ["T", "G", "I"])
    assert out.shape == (50, 8, 5)
    assert np.array_equal(out, cube)


# --------------------------------------------------------------------------- #
# FluxModel.slice / labels / ylim
# --------------------------------------------------------------------------- #
def _loaded_scalar_model():
    base = np.arange(40 * 4, dtype=float).reshape(40, 4)
    m = FluxModel()
    m.axes = ("I", "G")
    m.set_flux(base, "f")
    return m, base


def test_slice_group_and_sum():
    m, base = _loaded_scalar_model()
    x, y, xlabel, label = m.slice(2, None)
    assert np.array_equal(y, base[:, 2])
    assert xlabel == "Spatial cell (I)" and label == "Group 2"
    _, ysum, _, sumlabel = m.slice(-1, None)
    assert np.array_equal(ysum, base.sum(axis=1))
    assert sumlabel == "Sum over all groups"


def test_slice_uses_position_grid_when_sized():
    m, _ = _loaded_scalar_model()
    m.set_grid("I", np.linspace(0, 10, 40))
    x, _, xlabel, _ = m.slice(0, None)
    assert xlabel == "Position" and x[0] == 0 and x[-1] == 10
    # A wrong-sized grid is ignored.
    m.set_grid("I", np.linspace(0, 1, 7))
    _, _, xlabel2, _ = m.slice(0, None)
    assert xlabel2 == "Spatial cell (I)"


# --------------------------------------------------------------------------- #
# FluxModel.spectrum
# --------------------------------------------------------------------------- #
def test_spectrum_static_returns_energy_series():
    m, base = _loaded_scalar_model()  # (40, 4)
    x, y, xlabel, label = m.spectrum(3, None)
    assert np.array_equal(y, base[3, :])          # energy series at cell 3
    assert xlabel == "Energy group (G)"           # no energy grid loaded
    assert np.array_equal(x, np.arange(4))
    assert label == "Spatial cell 3"


def test_spectrum_time_dependent_selects_step():
    cube = np.arange(20 * 4 * 5, dtype=float).reshape(20, 4, 5)  # (I, G, T)
    m = FluxModel()
    m.axes = ("I", "G", "T")
    m.set_flux(cube, "f")
    _, y, _, _ = m.spectrum(2, 3)
    assert np.array_equal(y, cube[2, :, 3])


def test_spectrum_energy_axis_from_grid():
    m, _ = _loaded_scalar_model()  # G = 4
    # Group-center energies (size G) are used as-is.
    centers = np.array([1.0, 2.0, 4.0, 8.0])
    m.energy_grid = centers
    x, _, xlabel, _ = m.spectrum(0, None)
    assert xlabel == "Energy" and np.array_equal(x, centers)
    # Group boundaries (size G+1) collapse to band midpoints.
    m.energy_grid = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    x2, _, xlabel2, _ = m.spectrum(0, None)
    assert xlabel2 == "Energy" and np.array_equal(x2, np.array([1.0, 3.0, 5.0, 7.0]))


def test_spectrum_uses_position_label_when_grid_sized():
    m, _ = _loaded_scalar_model()  # I = 40
    m.set_grid("I", np.linspace(0, 10, 40))
    _, _, _, label = m.spectrum(0, None)
    assert label == "Position 0"


def test_group_and_time_labels():
    m, _ = _loaded_scalar_model()  # G = 4
    assert m.group_label(1) == "Group 1"
    m.energy_grid = np.linspace(0, 20, 5)   # G+1 boundaries
    assert "-" in m.group_label(1)
    m.energy_grid = np.linspace(1, 4, 4)    # G centers
    assert m.group_label(1).startswith("Group 1 (E=")
    m.t_grid = np.array([0.0, 2.0, 4.0])
    assert m.time_label(1) == "t = 2"
    m.t_grid = None
    assert m.time_label(1) == "t = 1"


def _group_model(n_groups=3):
    m = FluxModel()
    m.axes = ("I", "G")
    m.set_flux(np.zeros((5, n_groups)), "f")
    return m


def test_group_from_energy_boundaries():
    m = _group_model(3)
    m.energy_grid = np.array([1.0, 2.0, 5.0, 10.0])  # G+1 ascending boundaries
    assert [m.group_from_energy(v) for v in (1.5, 3.0, 7.0)] == [
        (0, True), (1, True), (2, True)
    ]
    assert m.group_from_energy(0.1) == (0, False)   # clamped low -> warn
    assert m.group_from_energy(99.0) == (2, False)  # clamped high -> warn


def test_group_from_energy_descending_boundaries():
    m = _group_model(3)
    m.energy_grid = np.array([10.0, 5.0, 2.0, 1.0])  # G+1 descending boundaries
    assert m.group_from_energy(7.0) == (0, True)
    assert m.group_from_energy(3.0) == (1, True)
    assert m.group_from_energy(1.5) == (2, True)


def test_group_from_energy_centers():
    m = _group_model(3)
    m.energy_grid = np.array([1.0, 2.0, 3.0])  # G center energies
    assert m.group_from_energy(2.0) == (1, True)    # exact center
    assert m.group_from_energy(2.4) == (1, False)   # snapped -> warn
    assert m.group_from_energy(2.6) == (2, False)


def test_group_from_energy_no_grid_returns_none():
    assert _group_model(3).group_from_energy(2.0) is None


def test_time_grid_reserves_initial_step():
    m = FluxModel()
    m.axes = ("I", "G", "T")
    m.set_flux(np.zeros((4, 3, 5)), "f")  # T = 5
    grid = m.time_grid_from_range(2.0, 8.0)
    assert grid.shape == (5,)
    assert grid[0] == 0.0  # index 0 reserved as the t=0 initial step
    assert np.allclose(grid[1:], [2, 4, 6, 8])


def test_time_grid_edge_cases():
    m = FluxModel()
    m.axes = ("I", "G", "T")
    m.set_flux(np.zeros((4, 3, 1)), "f")  # T = 1: only the initial step
    assert np.array_equal(m.time_grid_from_range(2.0, 8.0), [0.0])
    m.set_flux(np.zeros((4, 3, 2)), "f")  # T = 2: initial + one real step
    assert np.array_equal(m.time_grid_from_range(2.0, 8.0), [0.0, 2.0])


def test_time_label_marks_initial_step():
    m, _ = _loaded_scalar_model()
    m.t_grid = np.array([0.0, 2.0, 4.0])
    assert m.time_label(0) == "t = 0 (initial)"
    m.t_grid = None
    assert m.time_label(0) == "t = 0 (initial)"


def test_constant_ylim_spans_all_time():
    cube = np.empty((20, 3, 4))
    for t in range(4):
        cube[:, :, t] = (t + 1) * np.arange(1, 21)[:, None]
    m = FluxModel()
    m.axes = ("I", "G", "T")
    m.set_flux(cube, "c")
    lo, hi = m.constant_ylim(0)
    vals = cube[:, 0, :]
    assert lo <= vals.min() and hi >= vals.max()


def test_clear_keeps_energy_grid_only():
    m, _ = _loaded_scalar_model()
    m.energy_grid = np.linspace(0, 1, 5)
    m.set_grid("I", np.arange(40.0))
    m.clear()
    assert not m.is_loaded
    assert m.x_grid is None
    assert m.energy_grid is not None  # energy grid survives a type switch


# --------------------------------------------------------------------------- #
# io.arrays loaders + introspection
# --------------------------------------------------------------------------- #
def test_list_and_load_npz(tmp_path):
    base = np.arange(6.0).reshape(2, 3)
    path = tmp_path / "multi.npz"
    np.savez(path, flux=base, energy=np.arange(4.0))
    infos = list_arrays(str(path))
    assert {i.name for i in infos} == {"flux", "energy"}
    assert next(i for i in infos if i.name == "flux").shape == (2, 3)
    np.testing.assert_array_equal(load_array(str(path), key="flux"), base)


def test_load_csv_and_non_container(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("1,2,3\n4,5,6\n")
    np.testing.assert_array_equal(
        load_array(str(path)), np.array([[1.0, 2, 3], [4, 5, 6]])
    )
    assert list_arrays(str(path)) is None  # csv is not a container


# --------------------------------------------------------------------------- #
# io.arrays.save_arrays
# --------------------------------------------------------------------------- #
def test_save_arrays_npz_roundtrip(tmp_path):
    flux = np.arange(12.0).reshape(3, 4)
    grid = np.arange(5.0)
    written = save_arrays(str(tmp_path / "out.npz"), {"flux": flux, "energy": grid})
    assert written == [str(tmp_path / "out.npz")]
    assert {i.name for i in list_arrays(written[0])} == {"flux", "energy"}
    np.testing.assert_array_equal(load_array(written[0], key="flux"), flux)
    np.testing.assert_array_equal(load_array(written[0], key="energy"), grid)


def test_save_arrays_npy_series(tmp_path):
    flux = np.arange(6.0).reshape(2, 3)
    grid = np.arange(4.0)
    out_dir = tmp_path / "series"
    written = save_arrays(str(out_dir), {"flux": flux, "energy": grid}, fmt="npy")
    assert sorted(os.path.basename(p) for p in written) == ["energy.npy", "flux.npy"]
    np.testing.assert_array_equal(load_array(str(out_dir / "flux.npy")), flux)


def test_save_arrays_format_inference_defaults_to_npy_dir(tmp_path):
    out = tmp_path / "no_ext"
    written = save_arrays(str(out), {"flux": np.zeros(3)})
    assert written == [str(out / "flux.npy")]
