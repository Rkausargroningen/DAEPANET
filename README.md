# Water Network Modeler (Desktop)

**Version 2.0.0**

A Windows desktop application for building and analysing water distribution
networks, converted from an earlier Streamlit web app. Built with
**PySide6** (including Qt `QGraphicsScene`/`QGraphicsView` network visualization),
**NetworkX**, **NumPy** and **Matplotlib**, and grounded in a
research paper on the graph-theoretic and switched-DAE modelling of water
networks.

## Features

- Build a network interactively: add junctions, tanks and reservoirs; connect
  them with pipes and pumps.
- Directed network graph with pump edges shown separately.
- Symbolic **E, A, F, G** matrices of the underlying DAE model
  `E x' = A x + f + g(x)`, with LaTeX and CSV export.
- **Incidence matrices**: full `I(G)`, reduced `I^r(G)`, modified `I^m(G)`,
  and pipe pressure differences `ΔP = Iᵀ P`.
- **Checks tab**: numeric regularity test of the matrix pencil `(E, A)`, plus
  plain-language validation (every connected component must reach a supply
  node; pumps must not form a loop, including pumps in series between two
  supplies).
- Save/open projects as `.wnet` (JSON) files, with two bundled example
  networks.
- Packaged as a self-contained Windows executable (PyInstaller) — no Python
  installation required to run it.

## Entering parameters

Physical network parameters can be entered during node/pipe creation or modified in the Properties panel (stored for the planned comparison with EPANET):
- **Node Elevation**: elevation in metres ($m$).
- **Reservoir Head**: total hydraulic head of the water surface in metres ($m$, reservoirs only).
- **Pipe Length**: pipe length in metres ($m$).
- **Pipe Diameter**: internal diameter in metres ($m$).
- **Pipe Roughness**: absolute equivalent sand roughness ($\varepsilon$) in metres ($m$, stored for the planned comparison with EPANET; e.g. `3.59e-4`).
- **Pump Power**: pump power in kilowatts ($kW$, stored only).

Empty fields represent unset parameters (`None`); no default numbers are invented. The parameter status bar indicates if all parameters are complete or provides a detailed breakdown of missing fields.

## Moving nodes and bending pipes

Network elements can be positioned and shaped interactively on the Network canvas:
- **Drag nodes**: Click and drag any node using the mouse. All attached pipes, arrowheads, and labels follow the movement live, preserving distinct curvatures and bend ratios. Moving a node marks the project as modified, and positions are preserved when saving to `.wnet`.
- **Bend pipes**: Drag a pipe's circular handle (visible on hover or when selected) or drag its label to bend the pipe live. Endpoints stay neatly clipped at node boundaries while the curve, arrowhead, and label update continuously. Double-clicking the handle resets the bend to automatic curvature. Dragging the pipe body does nothing, while clicking the pipe body selects it.
- **Canvas panning and zoom**: Dragging on empty background pans the view without moving nodes or bending pipes. Use the mouse wheel to zoom smoothly around the cursor.
- **Selection sync**: Clicking a node or pipe in the graph selects its row in the corresponding table and displays its properties; selecting a table row highlights the element with a blue selection ring or line in the graph. Clicking the canvas background clears the selection.
- **Tooltips**: Hovering over any node or pipe reveals physical parameters and connection details in metric units.
- **Re-layout and Fit**: Click **Re-layout** to recalculate automatic spring layout positions and clear all custom pipe bends (marking the project modified), or click **Fit** to frame all elements with comfortable padding within the viewport.


## Project layout

```
core/       Pure calculation logic: network model, matrix construction,
            regularity checks, incidence matrices, save/load. No UI code.
ui/         PySide6 desktop interface (Network / Matrices / Incidence /
            Checks tabs). Imports only from core/.
tests/      Pytest suite (unit tests for core/, offscreen smoke tests for ui/).
legacy/     The original Streamlit web app, kept for reference. Not used by
            the desktop app.
tools/      Icon generation script.
assets/     App icon.
main.py     Desktop app entry point.
```

## Running from source

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python main.py
```

## Web app

The same model runs in the browser through Streamlit. `web/` is the interface.
`core/` is unchanged and does the calculations.

```powershell
.\.venv\Scripts\python -m streamlit run streamlit_app.py
```

That opens a local page. To give someone a link, push this repository to GitHub,
open [share.streamlit.io](https://share.streamlit.io), and create an app whose
main file is `streamlit_app.py`. Streamlit Community Cloud installs
`requirements.txt` and publishes a URL.

The comparison calls the EPANET toolkit. On Windows it uses `core/lib/epanet2.dll`
when that file is present. On the Linux host behind a public link it uses the
library shipped inside the `epyt` package. Pumps and tanks stay in the matrices
and the checks; the numeric comparison still requires reservoirs, valves, and a
complete set of physical parameters.

Run the tests with:

```powershell
.\.venv\Scripts\pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest -q
```

## Building the Windows executable

```powershell
.\.venv\Scripts\pip install -r requirements-dev.txt
.\build.ps1
```

This produces `dist/WaterNetworkModeler/WaterNetworkModeler.exe`. Prebuilt
binaries are not committed to this repository (they are large and
machine-specific) — see the project's Releases page instead, if published
there.

## Background

The matrix model follows the paper *"Water network as graph and its
solvability"*, which formulates a water distribution network as a directed
graph and studies the regularity of its switched differential-algebraic
equation model. `ASSUMPTIONS.md` documents every modelling choice made where
the paper was ambiguous (e.g. tank sign convention, treating tanks as supply
nodes for connectivity).

## Status

This is an early version. Valve/pump on-off switching, demand values, and
solving the DAE numerically are not yet implemented — see `ASSUMPTIONS.md`
and the "Not included yet" notes for details.

## License

License: not specified yet. All rights reserved by the author unless stated otherwise.
