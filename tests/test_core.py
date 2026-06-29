"""Headless tests for the Qt-free core: FluxModel and the array loaders.

These import no PyQt6 and need no display.
"""

import numpy as np
import pytest

from margaret.core.flux_model import FluxModel
from margaret.io.arrays import list_arrays, load_array


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
