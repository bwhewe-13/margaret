"""Reading numpy arrays from disk, and introspecting container files.

Supports ``.npy``, ``.npz``, ``.csv``/``.txt``, and ``.h5``/``.hdf5``. Container
formats (``.npz``/``.h5``) can hold several named arrays; :func:`list_arrays`
reports them (metadata only) so the GUI can let the user choose which to load.

Add a new format by handling its extension in :func:`load_array` (and, if it is
a multi-array container, in :func:`list_arrays`).
"""

from __future__ import annotations

import os
import zipfile
from typing import List, Mapping, NamedTuple, Optional

import numpy as np
from numpy.lib import format as npy_format

from margaret.constants import PREFERRED_KEYS


class ArrayInfo(NamedTuple):
    """Lightweight description of one array inside a container file."""

    name: str
    shape: tuple
    dtype: str

    def describe(self) -> str:
        shape = "x".join(str(d) for d in self.shape) if self.shape else "scalar"
        return f"{self.name}    [{shape}]  {self.dtype}"


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ValueError(
            "Reading HDF5 files needs the 'h5py' package "
            "(install it with: pip install h5py)."
        ) from exc
    return h5py


# --------------------------------------------------------------------------- #
# Introspection: what arrays does a container file hold?
# --------------------------------------------------------------------------- #
def list_arrays(path: str) -> Optional[List[ArrayInfo]]:
    """List the arrays in a container file, or ``None`` for single-array formats.

    Reads only metadata (names/shapes/dtypes), not the array data, so it stays
    cheap even for large files.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        return _npz_infos(path)
    if ext in (".h5", ".hdf5"):
        return _h5_infos(path)
    return None


# Public per-version header readers; format 3.0 (non-latin1 array names) has no
# public reader, so such entries fall back to the unreadable-header path below.
_NPY_HEADER_READERS = {
    (1, 0): npy_format.read_array_header_1_0,
    (2, 0): npy_format.read_array_header_2_0,
}


def _npz_infos(path: str) -> List[ArrayInfo]:
    infos: List[ArrayInfo] = []
    with zipfile.ZipFile(path) as archive:
        for entry in archive.namelist():
            if not entry.endswith(".npy"):
                continue
            with archive.open(entry) as handle:
                try:
                    version = npy_format.read_magic(handle)
                    shape, _, dtype = _NPY_HEADER_READERS[version](handle)
                    shape, dtype = tuple(shape), str(dtype)
                except Exception:  # unreadable header - still offer the name
                    shape, dtype = (), "?"
            infos.append(ArrayInfo(entry[:-4], shape, dtype))
    return infos


def _h5_infos(path: str) -> List[ArrayInfo]:
    h5py = _import_h5py()
    infos: List[ArrayInfo] = []
    with h5py.File(path, "r") as handle:

        def _collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                infos.append(ArrayInfo(name, tuple(obj.shape), str(obj.dtype)))
            return None

        handle.visititems(_collect)
    return infos


# --------------------------------------------------------------------------- #
# Loading: file on disk -> raw numpy array (no axis interpretation yet).
# --------------------------------------------------------------------------- #
def load_array(path: str, key: Optional[str] = None) -> np.ndarray:
    """Load an array from ``path``, dispatching on file extension.

    For container formats (``.npz``/``.h5``) ``key`` selects a specific array;
    when ``key`` is ``None`` a preferred/first array is chosen automatically.
    Returns the raw array exactly as stored; any axis ordering is applied later
    by the caller based on the user's selection.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.asarray(np.load(path))
    if ext == ".npz":
        with np.load(path) as archive:
            return _array_from_mapping(archive, path, key)
    if ext in (".csv", ".txt"):
        delimiter = "," if ext == ".csv" else None
        return np.atleast_1d(np.loadtxt(path, delimiter=delimiter, comments="#"))
    if ext in (".h5", ".hdf5"):
        return _load_hdf5(path, key)
    raise ValueError(f"Unsupported file type: {ext!r}")


def _array_from_mapping(mapping, path: str, key: Optional[str]) -> np.ndarray:
    """Pick one array from a dict-like container (.npz archive)."""
    if key is not None:
        if key not in mapping:
            raise ValueError(f"{os.path.basename(path)} has no array named {key!r}")
        return np.asarray(mapping[key])
    keys = list(mapping.keys())
    if not keys:
        raise ValueError(f"{os.path.basename(path)} contains no arrays")
    for preferred in PREFERRED_KEYS:
        if preferred in mapping:
            return np.asarray(mapping[preferred])
    return np.asarray(mapping[keys[0]])


def _load_hdf5(path: str, key: Optional[str] = None) -> np.ndarray:
    h5py = _import_h5py()
    with h5py.File(path, "r") as handle:
        if key is not None:
            obj = handle.get(key)
            if not isinstance(obj, h5py.Dataset):
                raise ValueError(
                    f"{os.path.basename(path)} has no dataset named {key!r}"
                )
            return np.asarray(obj[()])
        for preferred in PREFERRED_KEYS:
            obj = handle.get(preferred)
            if isinstance(obj, h5py.Dataset):
                return np.asarray(obj[()])
        found: list[np.ndarray] = []

        def _collect(_name, obj):
            if isinstance(obj, h5py.Dataset):
                found.append(np.asarray(obj[()]))
                return True  # stop at the first dataset
            return None

        handle.visititems(_collect)
        if not found:
            raise ValueError(f"{os.path.basename(path)} has no datasets")
        return found[0]


# --------------------------------------------------------------------------- #
# Saving: named arrays -> file on disk (inverse of load_array).
# --------------------------------------------------------------------------- #
def save_arrays(
    path: str, arrays: Mapping[str, np.ndarray], fmt: Optional[str] = None
) -> List[str]:
    """Write named ``arrays`` to disk as a container or a series of ``.npy`` files.

    ``fmt`` is ``"npz"``, ``"h5"``, or ``"npy"``. When ``None`` it is inferred
    from ``path``'s extension (``.npz`` -> npz, ``.h5``/``.hdf5`` -> h5); anything
    else defaults to ``"npy"``, in which case ``path`` is treated as a directory
    and one ``<name>.npy`` file is written per array. For ``npz``/``h5`` all the
    arrays go into the single file at ``path``. Returns the paths written.
    """
    fmt = (fmt or _infer_save_format(path)).lower()
    if fmt == "npz":
        target = path if path.lower().endswith(".npz") else path + ".npz"
        np.savez(target, **arrays)
        return [target]
    if fmt in ("h5", "hdf5"):
        h5py = _import_h5py()
        target = path if path.lower().endswith((".h5", ".hdf5")) else path + ".h5"
        with h5py.File(target, "w") as handle:
            for name, array in arrays.items():
                handle.create_dataset(name, data=np.asarray(array))
        return [target]
    if fmt == "npy":
        os.makedirs(path, exist_ok=True)
        written = []
        for name, array in arrays.items():
            dest = os.path.join(path, f"{name}.npy")
            np.save(dest, np.asarray(array))
            written.append(dest)
        return written
    raise ValueError(f"Unsupported save format: {fmt!r}")


def _infer_save_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        return "npz"
    if ext in (".h5", ".hdf5"):
        return "h5"
    return "npy"
