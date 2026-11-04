# Changelog

All notable changes to the Water Network Modeler application will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-10-18

### Added
- **Valves & Switching Scenarios**: Support for valves at the start and end of any pipe, with configurable switching scenarios (presets for MATLAB specimens 1 to 5 or custom whole-second switching events).
- **Switched DAE Solver & Dirac Impulses**: Full implementation of the Theorem 8 switched DAE solver using Wong sequences and quasi-Weierstrass form, calculating continuous state trajectories and signed Dirac impulses $\delta(t_s)$ upon valve switching.
- **Embedded Real EPANET Engine**: Direct integration of the official EPANET 2.3.5 hydraulic engine (`epanet2.dll`) via ctypes for side-by-side simulation without external dependencies.
- **Comparison Tab**: Dedicated "Comparison with EPANET" tab with setup controls, multi-panel trajectory plots, point-by-point differences table, steady-state equilibrium comparison, Dirac impulses table, and simulation parameters breakdown.
- **Multi-Format Exports**: Export comparison plots to high-resolution PNG (200 DPI), complete simulation timeseries and impulses to CSV, and network configurations to standard EPANET `.inp` files.
- **Help Menu & Dialogs**: Help menu with About dialog (displaying app version, EPANET engine version, and license statement), Third-party notices viewer, and a 6-step "How to compare with EPANET" guide.

### Changed
- **Network Canvas Interactivity**: Interactive node dragging and pipe bending with live-updating geometry and curve handles.
- **Plot Area Layout**: Comparison plots dynamically allocate space, hiding the left network build panel and providing a "Hide setup" toggle to maximize plot viewing width.
- **Small Screen Adaptability**: Dynamic vertical layout and scrolling for small displays (down to 1024x600 and 1280x672).
- **Friction Formulations**: Flexible friction law options including constant $f = 0.02$ (matching MATLAB specimen), Swamee-Jain, and Darcy-Weisbach models.

### Fixed
- **Scientific Notation Input**: Scientific double spin-box for kinematic viscosity $\nu$, preserving high precision (`1.0219e-06`) across UI edits without truncation.
- **Theme & Contrast Polish**: High-contrast styling across both light and dark themes for tab headers, toolbar icons, and panel controls (WCAG AA compliant contrast ratios $\ge 4.5:1$).
- **Clean Project State**: Simulation runs alone no longer dirty the project modified state or trigger unsaved changes prompts.

---

## [1.1.0] - 2026-10-17

### Added
- **Physical Parameters**: Support for entering and editing physical network properties in SI metric units:
  - Node elevation ($m$) for all node types.
  - Reservoir hydraulic head ($m$, valid only for reservoirs).
  - Pipe length ($m$), internal diameter ($m$), and absolute sand roughness ($\varepsilon$, $m$) stored for the planned comparison with EPANET.
  - Pump power rating ($kW$, stored for reference).
- **Properties Editor & Missing-Parameter Indicator**:
  - Interactive Properties panel in the build tab to view and edit properties of the selected node or pipe.
  - Live status indicator showing when all parameters are complete or reporting the exact count and list of missing parameters.
- **File Format Version 2 (.wnet v2)**:
  - Format version 2 persists physical network parameters, node coordinates, and custom pipe edge bends.
  - Full backward compatibility: v1 files load seamlessly with empty physical parameters defaulting to `None`.
  - Resilience against forward extensions: unknown JSON keys are safely ignored.
- **Native Qt Network Graph View**:
  - Replaced Matplotlib canvas with native Qt `QGraphicsScene` / `QGraphicsView` architecture for smooth, responsive 60 FPS rendering.
  - Fixed-size cosmetic node-type legend pinned to the viewport top-right that does not scale with zoom.
- **Interactive Node Dragging**:
  - Nodes can be moved freely across the canvas with the mouse.
  - Connected pipes, arrowheads, and labels follow node movement live in real time.
  - Moving a node marks the project modified.
- **Bidirectional Selection Sync**:
  - Selecting an element in the node or pipe table highlights it in the graph with a blue selection ring or line.
  - Clicking an element in the graph selects its row in the corresponding table and displays its properties in the Properties editor.
  - Clicking empty canvas background clears selection.
- **Interactive Tooltips**:
  - Hovering over nodes or pipes displays connection topology and physical properties in metric units (or `"not set"` when unset).
- **Bendable Pipes (Moving Edges)**:
  - Small cosmetic handle (visible on hover or when pipe is selected) at the midpoint of each pipe curve.
  - Dragging the handle or label bends the pipe live with real-time curve, arrowhead tangent, and label updates.
  - Double-clicking the handle resets the bend to automatic curvature.
  - Bend ratios are preserved across node dragging and saved in `.wnet` v2 files.
- **Fit to View & Cropped Image Export**:
  - One-click "Fit" button frames all network components with comfortable margins (centered on node bounds).
  - "Save image..." exports clean, cropped high-resolution PNG images on a white background with legend and margins.

### Fixed
- **Default elevation is unset**: Adding a node without an explicit elevation leaves elevation unset (`None`) instead of defaulting to 50.0.
- **Empty Fields = Not Set**: Unset parameters strictly default to `None` with no invented placeholder numbers; clearing a field and pressing Apply sets it back to `None`.
- **Stale Positions Between Examples**: Loading a new example or project clears existing positions cleanly so node positions do not carry over between distinct network topologies.
- **Bend Ratio Consistency**: Curvature and midpoint displacement are calculated relative to the chord connecting clipped node-boundary endpoints, ensuring exact mathematical agreement between perpendicular drag distance and rendered curve displacement.

---

## [1.0.0] - 2026-10-15

### Added
- Initial desktop release converted from Streamlit web application.
- Interactive network builder (junctions, tanks, reservoirs, pipes, pumps).
- Symbolic DAE matrices ($E, A, F, G$) with LaTeX and CSV export.
- Full, reduced, and modified incidence matrices ($I(G)$, $I^r(G)$, $I^m(G)$) and pressure difference vector ($\Delta P$).
- Matrix pencil regularity check and topological rule validation.
- Standalone self-contained Windows executable built with PyInstaller.
