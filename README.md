# MARGARET
Modular Analysis and Reactor Graphics Application for Radiation and Energy Tallies

A PyQt6 application for loading and visualizing neutron flux data.

## Features

- Pick a **flux type** (1D scalar, 1D time-dependent scalar; easy to extend) and
  declare the input **dimension ordering** (e.g. `I x G` vs `G x I`) so flux loads
  correctly however it was stored (`I` = spatial cell, `G` = energy group,
  `T` = time step).
- Load flux from `.npy`/`.npz`, `.csv`/`.txt`, and `.h5`/`.hdf5`. Multi-array
  containers prompt you to pick which array to load.
- Load an **energy grid** to label groups; **generate or load** the spatial and
  time grids. Time-dependent flux is scrubbed with a slider (y-axis held constant
  across time).
- Embedded matplotlib canvas with pan/zoom/save.

## Download

Prebuilt executables for Windows, macOS, and Linux are attached to each
[GitHub Release](https://github.com/bwhewe-13/margaret/releases) (no Python
install needed). Releases are built automatically when a `v*` tag is pushed;
bump `version` in `pyproject.toml` to match when tagging.

- **Windows**: unzip and run `margaret.exe`. SmartScreen may warn because the
  exe is unsigned — click "More info" → "Run anyway".
- **macOS**: unzip and open `margaret.app`. The app is unsigned, so Gatekeeper
  blocks the first launch — right-click → Open, or run
  `xattr -dr com.apple.quarantine margaret.app`. Pick the `arm64` zip for
  Apple Silicon or `x64` for Intel.
- **Linux**: untar and run `./margaret`.

The single-file executables self-extract on launch, so the first start takes a
few seconds.

## Install

```bash
pip install -e .          # runtime deps (PyQt6, matplotlib, numpy, h5py)
pip install -e ".[dev]"   # plus pytest / pytest-qt
```

## Run

```bash
margaret              # installed console script (after pip install -e .)
python -m margaret    # run as a module
```

## Layout

```
src/margaret/
  constants.py        app metadata + registries (FLUX_TYPES, FILE_FILTER, ...)
  core/flux_model.py  FluxModel - flux + grids + reductions (no Qt, unit-tested)
  io/arrays.py        file -> numpy array loaders + container introspection
  gui/plot_canvas.py  matplotlib-in-Qt canvas (1D line)
  gui/array_picker.py dialog to choose an array from a multi-array file
  gui/start_page.py   main window: builds UI, owns model + canvas, wires signals
  app.py              entry point (main())
  __main__.py         python -m margaret
```

## Extending

- **New flux type**: add an entry to `FLUX_TYPES` in `constants.py` (its `axes`
  tuple drives the dimension-ordering options).
- **New file format**: handle its extension in `io/arrays.py::load_array` (and, for
  a multi-array container, in `list_arrays`).

## Test

```bash
pytest          # core model + loader tests run headless (no display)
```
