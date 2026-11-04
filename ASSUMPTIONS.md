# ASSUMPTIONS.md – Judgment Calls in core/

This document records every non-obvious decision made while building the
`core/` package for the water-distribution-network PySide6 migration (Step 1).

---

## 1. Tank Inflow Sign Convention

**Decision**: `TANK_INFLOW_SIGN = +1`.

A pipe whose **end** node is a tank contributes `+1` (positive) to the tank
row of the A matrix (inflow is positive). A pipe whose **start** node is a
tank contributes `−1` (outflow is negative). This exactly matches the junction
continuity convention used by the legacy code for junctions.

**Why not the legacy tank sign?**  
The legacy code only wrote `+1` when a tank was the *start* of a pipe and
wrote nothing when a tank was the *end*. This left the tank row all-zero for
networks where water only flows *into* the tank, making those networks appear
singular even when they are well-posed. The corrected sign is a **bug fix**.

**Rationale for +1 = inflow**  
From the mass-balance perspective the tank's state variable (pressure / water
level) is driven by net inflow. Matching the junction sign keeps the physical
interpretation consistent across node types.

**Flipping the sign**: if a future paper revision or regulatory standard
requires outflow-positive, change `TANK_INFLOW_SIGN = +1` to
`TANK_INFLOW_SIGN = −1` in `core/matrices.py`; no other code needs to change.

---

## 2. Tanks Counted as Supply Nodes (Validation)

**Decision**: tanks are treated as supply nodes in the connectivity check
(`validate_network`).

A tank can supply or receive water; it provides a non-trivial boundary
condition (known pressure trajectory). Therefore a component that contains
a tank but no reservoir is treated as *having a supply* and is **not** flagged
as supply-free.

---

## 3. Symbolic String Format (kept from legacy)

The following symbolic strings are kept exactly as in the legacy code:

| Matrix | Entry | String |
|--------|-------|--------|
| G | non-pumped pipe i (1-based) | `f"-C_aQ{i+1}|Q{i+1}|"` |
| G | pumped pipe | `"-S_puG_pu(Q_pu)"` |
| A | non-pumped Pl | `f"c{i+1}"` |
| A | non-pumped Pr | `f"-c{i+1}"` |
| A | pump diagonal | `f"1-s{sym_idx}"` |
| A | pump start pressure | `f"-s{sym_idx}"` |
| A | pump end pressure | `f"s{sym_idx}"` |
| A | pipe end-side flow | `f"1-s{i+1}"` |
| A | pipe end-side Pr | `f"s{i+1}"` |
| A | pipe end-side node | `f"-s{i+1}"` |
| F | reservoir i (0-based) | `f"-P_{chr(97+i)}"` i.e. `-P_a`, `-P_b`, … |

Symbol numbering: `c{i+1}` and `s{i+1}` index over **non-pumped** pipes
(1-based, in creation order); pump symbols continue at `s{N_np + i + 1}` where
`i` is 0-based within the pumped list. This is identical to the legacy
numbering when normal pipes come before pumps.

---

## 4. Column Ordering Bug Fix (ORDER BUG)

**Legacy assumption**: the first `non_pumped` pipes in the input list are the
non-pumped ones, and column indices are derived by parsing the integer suffix
of pipe/node IDs (e.g., `int("P3"[1:])` → 3).

**New approach**: pipes are partitioned into `non_pumped` and `pumped` lists
(preserving creation order within each group). All column indices are derived
from position in those lists via `pipe_flow_col` and `node_col` dictionaries.
No IDs are ever parsed. The resulting matrix is independent of the order in
which pipes were added to the network.

---

## 5. Numeric Pencil Evaluation (s=1, c=1)

**Substitution rules for `numeric_pencil`**:

- All switch variables `s_i` → 1 (all valves open, all pumps on).
- All conductance coefficients `c_i` → 1.
- Expressions like `1-s_i` → 0; `s_i` → 1; `-s_i` → -1; `c_i` → 1.

This represents the fully-activated network state (worst case for regularity
testing: if it is singular here it is structurally singular).

---

## 6. Regularity Check — Tolerance and Sampling

**Method**: evaluate `np.linalg.slogdet(λE − A)` at `_REGULARITY_N_SAMPLES = 8`
complex values of λ, drawn from a **fixed** RNG seed (`_REGULARITY_SEED = 42`).

**Tolerance**: `_REGULARITY_TOL = 1e-7`.  
A pencil is declared regular if `|det| > 1e-7` for at least one sample.

**Why `slogdet`?**  
`slogdet` returns `(sign, log|det|)`. It never overflows for large matrices,
unlike `np.linalg.det`. We check `sign != 0` (non-trivially-zero determinant)
AND `logabsdet > log(tol)`.

**Why multiple samples?**  
A single λ might accidentally coincide with an eigenvalue of a *regular* pencil
and give zero determinant. Using 8 samples from the complex plane (with
non-zero imaginary parts to avoid real eigenvalues) makes a false-negative
extremely unlikely.

**Limitation**: this is a heuristic test, not a formal algebraic rank check.
For production use, a Smith-McMillan form or Kronecker structure analysis
would be more rigorous. That is left for a later step.

---

## 7. Pump Cycle Detection — Cyclomatic Number Method

**Paper method**: merge all supply nodes (reservoirs + tanks) into a single
virtual super-node called `__supply__`, build an undirected multigraph of pump
edges only, and for each connected component compute:

```
cyclomatic_number = E − V + 1
```

where E = number of pump edges, V = number of vertices, and the "+1" assumes
the component is connected (C=1). A cyclomatic number > 0 means there is at
least one independent cycle in the pump sub-graph, which is a structural
problem for the switched-DAE formulation.

**Cross-check**: the rank of the pump-column block of the modified incidence
matrix is also computed. If `rank < number_of_pump_edges` but the cyclomatic
method did not flag a cycle, an additional warning is emitted.

---

## 8. Incidence Matrix Signs

Convention: −1 at the **start** (initial) node of each edge, +1 at the
**end** (terminal) node. Each column therefore sums to 0.

Reference row for the *reduced* incidence: first reservoir, else first tank,
else first node in creation order.

---

## 9. Node ID Auto-Generation

Prefixes: `J` (junction), `T` (tank), `R` (reservoir), `P` (pipe).
Counters track the **maximum issued number**, so manual IDs (supplied via
`id=` keyword) are respected and auto-generated IDs never collide with them
after `from_dict` round-trips.

---

## 10. Physical parameters and .wnet v2

- **SI Units**: All physical network quantities are expressed in standard SI units: pipe length ($m$), pipe diameter ($m$), pipe roughness ($m$), node elevation ($m$), and reservoir total head ($m$).
- **None = Not Set (No Invented Defaults)**: Unset parameters default to `None`. No arbitrary default numbers (e.g., standard pipe diameters or lengths) are ever invented or assumed.
- **Reservoir Head Semantics**: `head` represents the total hydraulic head of the water surface ($H_R$, in metres) and is valid only for reservoirs ($P_R = H_R - z$, where $z$ is elevation). Attempting to set `head` on junctions or tanks raises `ValueError`.
- **Legacy Node Elevations as Placeholders**: v1.0 files stored default elevations of `50.0 m` on all nodes. These numbers serve as structural placeholders rather than true surveyed data.
- **Roughness Interpretation**: Pipe roughness specifies absolute equivalent sand roughness ($\varepsilon$, in metres) stored for the planned comparison with EPANET, not dimensionless coefficients.
- **Backward Compatibility and Schema Evolution**: Format version 2 extends `.wnet` to persist physical parameters while preserving full backward compatibility with version 1 files. Unknown keys in node, pipe, or top-level objects are ignored during deserialization to maintain forward resilience.

---

## 11. UI Physical Parameter Entry and Properties Editor

- **Empty Field = Not Set (`None`)**: In the Add Node/Pipe forms and the Properties editor, an empty text field denotes an unset parameter (`None`). Clearing a value and pressing Apply unsets the parameter in the model.
- **Pump Toggle Not Editable After Creation**: A pipe cannot be switched between pumped and non-pumped in the Properties editor. The `has_pump` flag is fixed upon creation; if `has_pump` is true, its power rating can be adjusted in the Properties editor.
- **Tank Capacity Stored Only**: Tank `capacity` is editable in the Properties editor with a tooltip noting it is stored for reference and not yet incorporated into DAE matrix equations.
- **Input Fields Retained on Add**: Values in the Add Node and Add Pipe forms remain populated after clicking "Add" so users can rapidly add multiple network elements with matching properties without retyping.

---

## 12. Qt GraphicsScene Graph View

- **Native Qt Graphics Architecture**: Replaced Matplotlib canvas with native `QGraphicsScene` / `QGraphicsView` for network rendering, eliminating Matplotlib redraw latency, layout resets, and thread/context switching overhead while preserving Matplotlib in the project dependencies for other potential consumers.
- **Coordinate System Mapping**: Network models use Cartesian coordinates in `[-1.0, 1.0]` with $+Y$ pointing upward. The Qt scene uses computer graphics convention where $+Y$ points downward. Conversion is defined by $x_{scene} = x_{model} \times \text{SCALE}$ and $y_{scene} = -y_{model} \times \text{SCALE}$, with fixed $\text{SCALE} = 300.0$.
- **Edge Geometry & Curvature Separation**: Pipes stop cleanly at the node circular boundary rather than intersecting node centers. Parallel edges between identical endpoints are separated via quadratic Bezier curves with opposing lateral curvature offsets ($\pm 0.25$) to prevent visual overlap.
- **Fixed-Size Legend & Overlay Elements**: Node type legend is implemented with `QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations` and pinned to the viewport top-right so it remains readable and stationary under zoom and pan operations.
- **Zoom & Pan Stability**: Viewport transformations persist across property edits and network refreshes; automatic recentering is triggered only on project load or explicit user click on the "Fit" button.
- **Fit Quality & Margins**: Dynamic fit calculates usable viewport bounds excluding explicit margins (minimum 20 px padding from any viewport edge) and sets scene bounds tightly aligned with network contents to prevent center drift caused by untransformed UI overlays (such as the legend).

---

## 13. Interactive Node Dragging, Selection Sync, and Visual Styling

- **Dragging Marks Project Modified**: When a node is dragged with the mouse, `GraphView.positions_changed` is emitted once upon mouse release if and only if a real translation occurred. MainWindow connects this signal to mark the project modified (displaying window title asterisk and triggering the save prompt on close/new). Programmatic positioning (such as loading a file or example) does not mark the project modified.
- **Re-layout Discards Custom Positions**: Clicking "Re-layout" recalculates automatic spring layout coordinates for all nodes and marks the project modified.
- **Live Pipe Following & Curvature Preservation**: While dragging a node, attached pipes, arrowheads, and labels update live at 60 FPS using an adjacency map (`_node_pipes`). Parallel pipes maintain their opposing curvature offsets throughout dragging.
- **Live Internal Position Sync**: The internal `_pos` dictionary in model coordinates is kept in sync during dragging, ensuring that background refreshes (such as editing pipe parameters in Properties) never reset or discard dragged coordinates.
- **Selection Visual Distinction & Dual Styling**:
  - Blue selection ring / thicker stroke (`#0078D7`, 3.5–4.0 px) indicates the active element selected in the table or canvas.
  - Red outline (`#FF0000`, 3.0–3.5 px) indicates warnings highlighted by the Checks tab.
  - When an element is simultaneously selected and highlighted, dual concentric rings (outer blue ring at radius 24, inner red ring at radius 20) or stacked strokes ensure both states are independently distinguishable.
- **Empty Canvas Click Clears Selection**: Clicking empty canvas background clears both table selection and graph selection without modifying the network. Panning the view does not clear selection.
- **Tooltip Formatting & Metric Units**: Hover tooltips show physical quantities in standard metric units (`m` for length/elevation/head/diameter/roughness, `kW` for pump power) and explicitly display `"not set"` for `None` values. Reservoir total head is displayed only for reservoirs.

---

## 14. Pipe Bending (Moving Edges)

- **Moving an Edge = Bending It**: The client requested that "edges and nodes can be moved via cursor". Since nodes already move with all attached pipes following them, moving an edge has been interpreted as bending it (the client has not explicitly confirmed this interpretation). Dragging the pipe body does nothing (so panning on the background and node dragging remain unaffected), while dragging the curve midpoint handle or pipe label bends the pipe.
- **Bend Ratio Definition**: A bend is parameterized by a dimensionless bend ratio = perpendicular offset of the curve midpoint divided by the chord length, added to the automatic curvature used to separate parallel pipes. Bend ratios are clamped to `[-1.5, 1.5]`. Sign convention: positive = left of the travel direction (start $\to$ end) in screen coordinates ($+Y$ down).
- **Bend Ratio Survives Node Dragging**: When an attached node is moved, the pipe retains its bend ratio, so the physical displacement scales proportionally with the new chord length.
- **Optional Key in `.wnet` v2**: Saved files store non-zero bends in an optional top-level `"edge_bends": {pipe_id: float}` mapping. If no pipes are bent, the key is omitted. File version remains 2, and loaders ignore unknown keys, ensuring full backward compatibility.
- **Re-layout Clears Bends**: Clicking "Re-layout" recalculates the node layout and clears all custom edge bends back to automatic separation curvature. Deleting a pipe also removes its associated bend.
- **Single Modified Notification**: `edge_bends_changed` is emitted exactly once on mouse release after an actual drag has completed (not on clicks or programmatic setting). Loading a project or example resets bends silently and never marks the project modified.



---

## 15. v2.0 Design Decisions (not confirmed by client)

These decisions were made while implementing the v2.0 switched-DAE comparison feature. The client has not confirmed them; they are listed in delivery emails as open items.

- **Valves at both pipe ends**: The MATLAB reference places valves only at the start and end of pipe 2. The app generalises this: any pipe can have an optional valve at its start end (`valve_start`) and/or end end (`valve_end`). This is required to match the MATLAB reference for the specimen network.
- **Valve row uses pipe-local flow**: The valve row is `(1-s)*Q_pipe + s*(P_after - P_before) = 0` where `Q_pipe` is the flow of the pipe the valve belongs to. The MATLAB reference writes `Q1` in the V1 row, but V1 belongs to pipe 2 and `Q1 = Q2` by junction continuity for the specimen network. Both formulations are equivalent for the specimen; this app uses `Q_pipe` for generality.
- **Scale-aware regularity test**: Regularity is tested by normalising each row of the matrix pencil `(λE - A)` to unit L2-norm before computing the determinant / rank. This avoids false singularity results when physical parameters span many orders of magnitude (e.g. `b ≈ 1e-3` vs pressure heads ≈ 50).
- **No new dependencies**: The DAE solver, scenario engine, and matrix builder use only `numpy`. No `scipy`, `sympy`, or other third-party libraries are added in v2.0.
- **Default elevation is None (unset)**: Starting from v1.1.0, `add_node` without an explicit `elevation` argument produces `elevation=None` (not `50.0`). Old `.wnet` v1 files that contain `"elevation": 50.0` still load `50.0` correctly. This is a deliberate break from the v1.0.0 default.
- **Physical parameters stored, not yet used**: All physical parameters (elevation, head, length, diameter, roughness) are stored in `.wnet` files and editable in the UI, but the current symbolic matrix builder (`build_matrices`) ignores them. The planned v2.0 numeric DAE builder will use them. All documentation (README, CHANGELOG, ASSUMPTIONS, in-app text) explicitly says "stored for the planned comparison with EPANET".
- **PhysicsSettings dataclass**: `PhysicsSettings` lives in `core/physics.py` and carries `g`, `nu`, `friction_mode`, and `friction_factor`. Default mode is `"constant"` with `friction_factor = 0.02`.
- **Scenario and valve timing**: `Scenario` stores a list of `(switch_time, mode)` pairs. The initial mode is the mode at time 0. The client has not confirmed whether valve times are entered by the user or fixed; for the specimen they are hardcoded in `Scenario.matlab_specimen(k)`.

---

## 16. Friction law (default constant f = 0.02 as in the MATLAB file; Swamee-Jain/Darcy-Weisbach optional)

**Decision**: `PhysicsSettings.friction_mode` supports `"constant"`, `"swamee_jain"`, and `"darcy_weisbach"`.

1. **Constant friction factor** (default):
   - $f = \text{friction\_factor} = 0.02$.
   - Quadratic head loss term: $\bar{g}_i(Q) = -\frac{f}{2 D_i \text{Area}_i} Q |Q|$.

2. **Swamee-Jain**:
   - Turbulent formulation evaluated at every flow:
     $$f(Q) = \frac{0.25}{\left[ \log_{10}\left( \frac{\varepsilon}{3.7 D} + \frac{5.74}{\text{Re}^{0.9}} \right) \right]^2}, \quad \text{Re} = \frac{4 |Q|}{\pi D \nu}$$
   - At $Q = 0$, $\bar{g}_i(0) = 0.0$ continuously without division by zero.

3. **Darcy-Weisbach (EPANET rule)**:
   - Implemented from EPANET 2.2 source code (`hydcoeffs.c` lines 622–670):
     - Laminar ($\text{Re} \le 2000$): $f = 64 / \text{Re}$.
       The friction term is implemented directly as $\bar{g}_i(Q) = -\frac{32 \nu}{D_i^2} Q$, which is strictly linear in $Q$ and avoids $0/0$ indeterminacy at $Q = 0$.
     - Transition ($2000 < \text{Re} < 4000$): Dunlop cubic interpolation:
       $w = \frac{|Q|}{\nu D} = \frac{\text{Re} \pi}{4}$, $r = \frac{w}{500 \pi} = \frac{\text{Re}}{2000} \in (1, 2)$
       $y_2 = \frac{\varepsilon}{3.7 D} + \text{AB}$, where $\text{AB} = \frac{5.74}{4000^{0.9}}$
       $y_3 = A_9 \ln(y_2) = -2 \log_{10}(y_2)$
       $f_a = \frac{1}{y_3^2}$, $f_b = \left(2.0 + \frac{\text{AC}}{y_2 y_3}\right) f_a$ with $\text{AC} = \left(-\frac{3.6}{\ln 10}\right) \text{AB}$
       $x_1 = 7 f_a - f_b$
       $x_2 = 0.128 - 17 f_a + 2.5 f_b$
       $x_3 = -0.128 + 13 f_a - 2 f_b$
       $x_4 = 0.032 - 3 f_a + 0.5 f_b$
       $f = x_1 + r (x_2 + r (x_3 + r x_4))$.
     - Turbulent ($\text{Re} \ge 4000$): Swamee-Jain formula above.
   - Status: VERIFIED against EPANET 2.2 source code (`hydcoeffs.c`).

---

## 17. Initial state is the rest state (zero flow) of the first valve mode

**Decision**: The initial state $x(0)$ is the stationary rest state of the first operating mode of the scenario.

- In the MATLAB reference (`restState`), the physical water system is initially at rest on $(-\infty, 0)$ with zero flows ($v = 0$).
- The algebraic components follow directly from equations (10b) and (10c):
  $$w_1 = -f_{1w}, \quad \xi = M(T_{1w} w_1), \quad w_2 = -(f_{2w} + N_{2w} \bar{g}(\xi)), \quad x_0 = T_{1w} w_1 + T_{2w} w_2$$
- Consequently, $x(0)$ satisfies all algebraic constraints of the initial mode; there is no algebraic jump and no Dirac impulse at $t = 0$.

---

## 18. Impulses are reported, not plotted as bars; the plot shows an arrow (not confirmed by client)

**Decision**: Dirac impulses $\delta_{t_s}$ resulting from instantaneous valve closures are recorded as analytical impulse coefficients in `ImpulseRecord` with cross-check validation against formula (6) of the paper ($x[t_s] = -E^{\text{imp}} x(t_s^-)$).

---

## 19. EPANET engine and its license

**Decision**: The application runs the official real EPANET engine using Python's standard `ctypes` library directly on the pre-compiled C dynamic library `epanet2.dll` (Open Water Analytics / EPA EPANET 2.3.5, version code `20305`).

- **Provenance**: Extracted from `epyt` wheel package (version 1.0.9, path `epyt/libraries/win/epanet2.dll`), upstream C repository `https://github.com/OpenWaterAnalytics/EPANET`.
- **SHA-256**: `0FEC08B4BA0320A08E821E4E79ED380CE09AA926867A2B1D33E878B6159C02FA` (464,896 bytes, Windows x64).
- **License**: MIT License (OWA EPANET 2.2+) and Public Domain (US EPA EPANET 2.0). License text is MIT/Public Domain, fully compatible with commercial distribution and offline bundled executables.
- **Python dependencies**: 0 new pip packages added (no pandas, no scipy, no wntr; numpy-only on the Python side).
- **Executable packaging**: `epanet2.dll` adds only 454 KB to the PyInstaller distribution. It operates 100% offline at runtime without requiring Python or administrative rights.
- **Thread safety**: Every run uses EPANET's project handle API (`EN_createproject`, `EN_open`, `EN_openH`, `EN_closeH`, `EN_close`, `EN_deleteproject`) ensuring independent state for worker threads with zero leakage between runs.

---

## 20. EPANET works in whole seconds

**Decision**: Non-integer valve switch times are rejected with a clear `ScenarioError`.

- EPANET's internal clock and input file syntax specify event times in integer seconds (or `h:mm:ss`).
- Sub-second switching times cannot be represented natively in standard EPANET control rules. To prevent ambiguous rounding, non-integer event times raise `ScenarioError: EPANET requires switch times in whole seconds, got <t>s for valve <valve>`.

---

## 21. Constant friction is exported as minor losses

**Decision**: When `settings.friction_mode == "constant"`, each pipe is exported to EPANET as a 1 cm stub pipe ($L = 0.01$ m) carrying a minor loss coefficient $K = f \cdot L / D$.

- This reproduces exactly the MATLAB reference implementation (`writeInp2` in `switchedDAE3_vs_EPANET.m`).
- In this mode, pipe wall roughness is irrelevant because head loss is governed by the lumped minor loss $h_L = K \frac{v^2}{2g} = \frac{f L}{D} \frac{v^2}{2g}$. For the specimen network with $f = 0.02, D = 0.4$ m, this yields $K_1 = 0.02 \times 400 / 0.4 = 20.0$ and $K_2 = 0.02 \times 250 / 0.4 = 12.5$.
- For `darcy_weisbach` and `swamee_jain` modes, the real physical length ($L$), diameter ($D$), and roughness ($\varepsilon$) are exported with $K = 0$.

---

## 22. Valve elements are TCV with zero loss

**Decision**: Valves are represented in EPANET as Throttle Control Valves (`TCV`) with open setting `0` (or `OPEN`), minor loss `0`, and diameter matching the connected pipe diameter.

- When a valve is at the start of a pipe, an intermediate junction `<pipe>_L` is created at the elevation of the start node. The TCV link connects the start node junction to `<pipe>_L`, and the pipe link connects `<pipe>_L` to the downstream node.
- Similarly, a valve at the end of a pipe introduces intermediate junction `<pipe>_R` at the end node's elevation.
- This reproduces MATLAB's N5/N4/N6/N9 layout and provides exact correspondence with DAE state variables $P_L$ and $P_R$.

---

## 23. Valves are edited in the Comparison tab

*(Not confirmed by the client)*

**Decision**: Valve existence at pipe boundaries (`valve_start`, `valve_end`) is edited via checkboxes in the dedicated "Valves" table inside the "Comparison with EPANET" tab rather than on the primary Build panel.

- This keeps the general Network Builder clean and focused on standard pipe/node properties while centralizing comparison-specific switching controls.
- Toggling a valve immediately updates the underlying Network model, marks the project as modified, updates the graph visualization with valve symbols, and automatically purges references to any removed valves from the active scenario with an inline notification.

---

## 24. Scenario times are whole seconds

*(Not confirmed by the client)*

**Decision**: Switching event times and total simulation end times in the UI scenario editor are constrained to integer seconds via whole-second spin boxes.

- Sub-second switching events are rejected by the comparison engine because EPANET's rule-based controls operate on integer-second discrete time steps. Constraining the UI controls prevents user confusion and invalid input.

---

## 25. Presets use the first two valves

*(Not confirmed by the client)*

**Decision**: The built-in MATLAB specimen scenario presets (1 to 5) automatically bind to the first two valves found in the network in canonical pipe creation order (start valve before end valve).

- The UI explicitly displays the resolved binding (e.g. `V1 = P2:start, V2 = P2:end`).
- If the network contains fewer than two valves, presets are disabled with an informative label (`Need >= 2 valves for presets`), preventing misconfiguration.

---

## 26. A run uses a snapshot of the network

*(Not confirmed by the client)*

**Decision**: The comparison simulation is executed on a frozen deep copy (snapshot) of the network, scenario, and physics settings taken at the exact moment the "Run comparison" button is clicked.

- Edits made to the network or settings after a run do not mutate or invalidate the existing results in-place.
- Instead, a deterministic SHA-256 signature is evaluated against the current network state; if any discrepancy is detected, a prominent yellow banner alerts the user: `Network or settings changed since this result - run again`.

---

## 27. Valve symbol in the graph

*(Not confirmed by the client)*

**Decision**: Valves are rendered in the interactive network graph as small bow-tie symbols (two point-to-point opposing triangles with dark outline) positioned just inside the pipe boundary at the valve location (approx. 5% along the pipe curve).

- The symbols follow interactive node dragging, live pipe curvature bending, and automatic layout.
- The symbol has a fixed cosmetic pixel size (does not scale awkwardly with zoom), displays a descriptive tooltip (`Valve at start/end of <pipe>`), forwards clicks to select the parent pipe without disrupting hit-testing, is rendered in exported images ("Save image..."), and is included in the graph view legend.

## 28. Friction law choice in the UI

*(Not confirmed by the client)*

**Decision**: The Comparison tab exposes three friction formulations via a dropdown: "Constant friction factor" (default $f = 0.02$), "Swamee-Jain", and "Darcy-Weisbach as in EPANET".

- When non-constant friction is selected, the constant factor spin box is hidden, and the simulation uses the real pipe roughness, diameter, fluid viscosity, and Reynolds number.
- Advanced physical parameters (gravity $g$ and kinematic viscosity $\nu$) are grouped under a collapsed "Advanced Parameters" group with standard physical defaults ($g = 9.81456\text{ m/s}^2$, $\nu = 1.0219\times 10^{-6}\text{ m}^2/\text{s}$).

---

## 29. Plot reproduces the MATLAB figure; states selectable

*(Not confirmed by the client)*

**Decision**: The comparison plot visually reproduces MATLAB's `comparisonFigure` (`switchedDAE3_vs_EPANET.m` lines 614-690):
- Vertically stacked panels with shared time axis and boxed axes, $x$-limits $[-0.01 T, T]$, grid alpha $0.12$, plain numbers on y-axis without scientific offsets.
- DAE curves are drawn in solid `#0072B2` (linewidth 1.6) per mode interval (never drawing across switching discontinuities), with dotted jump segments at switch times.
- EPANET trajectories are drawn on top in `#D91A1A` (linewidth 0.9) with `drawstyle='steps-post'`.
- Light grey vertical lines (`#CCCCCC`) mark every switch event across all panels.
- State selection is configurable via a dropdown checklist with quick buttons ("MATLAB five", "All flows", "All"). On the MATLAB specimen preset, default selection matches MATLAB's 5 states in order (`Pipe_1_Flow`, `Pipe_2_Flow`, `Pipe_1_R`, `Pipe_2_L`, `Pipe_2_R`). Changing selected states updates the figure instantly without re-running the simulation.

---

## 30. Total head option

*(Not confirmed by the client)*

**Decision**: A radio toggle "Pressure head | Total head (EPANET Head)" allows users to switch between gauge pressure head ($m$) and total hydraulic head ($m = \text{pressure head} + \text{elevation}$).
- Flows ($Q$) are unaffected by the elevation shift.
- Pressure states are adjusted by adding their associated node elevation ($J1$ elevation $= 5.0\text{ m}$ in the specimen network).
- Y-axis labels dynamically update between `pressure head [m]` and `total head [m]`.

---

## 31. Dirac arrow height is 0.5 of the panel range (as in MATLAB)

*(Not confirmed by the client)*

**Decision**: In panels where a state exhibits a Dirac impulse ($|\text{coefficient}| > 10^{-9}$), an impulse arrow is drawn starting from the pre-switch DAE value with height $h = 0.5 \times (\text{y-range}) \times \text{sign}(\text{coefficient})$.
- The arrow has a solid line (width 2.2) and a filled triangle marker (size 7, up for positive, down for negative).
- An annotation box with white background is positioned at $t_s + 0.03 T, y = \text{base} + 0.62 h$, displaying signed coefficient and switch time: `DAE: Dirac ±XX.XX δ(t - t_s)\nEPANET: none`.
- Panel y-limits are symmetrically scaled by a factor of 1.35 to guarantee the arrow and text never clip outside the axes bounds.

---

## 32. Graph canvas is always light

*(Not confirmed by the client)*

**Decision**: The network graph canvas (both `QGraphicsScene` and `QGraphicsView` viewport background) is explicitly assigned a clean white background (`#FFFFFF`), and pipes have a minimum cosmetic pen width of 2.0 px.
- This ensures full visual contrast for pipes, direction arrows, labels, and valve symbols regardless of whether Windows is operating in Light or Dark theme.
- Contrast ratio between pipe strokes and canvas background meets or exceeds WCAG AA standards ($\ge 4.5$, measured $\approx 12.6$).

---

## 33. CSV/PNG/.inp exports

*(Not confirmed by the client)*

**Decision**: Three export buttons are provided directly underneath the comparison plots:
1. "Save figure...": exports high-resolution 200 DPI PNG (or PDF/SVG depending on chosen extension) with crisp white background.
2. "Export data (CSV)...": writes a comprehensive CSV containing time, DAE, EPANET, and difference columns for all states sampled at EPANET's report times, along with a secondary `<basename>_impulses.csv` detailing impulse times, states, and coefficients.
3. "Export EPANET input file (.inp)...": exports the exact INP text produced by `export_epanet` for the run snapshot, allowing independent validation and third-party EPANET simulation.
---

## 34. Minimum visible range of plot panels

*(Not confirmed by the client)*

**Decision**: After normal autoscaling, if the visible span of a PRESSURE or TOTAL HEAD panel is smaller than $\max(0.5\% \text{ of } \max|y|, 0.01\text{ m})$, it is widened symmetrically around the midpoint to that span. For FLOW panels, $1\%$ of the largest $|flow|$ over all displayed flow series (global scale) is used; if all flows are exactly zero, a fixed $\pm 10^{-3}\text{ m}^3/\text{s}$ window ($[-0.001, 0.001]$) is applied. Panels with Dirac arrows, jumps, or real transients remain unchanged. Tick labels format to at most 4 decimals with no scientific offset, omitting ticks beyond data limits.

---

## 35. Plots scroll instead of shrinking below 640x820

*(Not confirmed by the client)*

**Decision**: The comparison plot canvas is housed in a `QScrollArea` with a minimum figure size of $640 \times 820$ px (each panel at least 120 px high). Below this window size, scrollbars appear rather than squeezing the panels and overlapping text. If available canvas width is below 800 px, the legend moves above the first panel (horizontal, 2-column, centered) rather than the right side. A "Hide setup" toggle button on the left collapsible splitter allows users to collapse/expand the setup column, increasing plot area width.

---

## 36. Outdated banner is signature based

*(Not confirmed by the client)*

**Decision**: The warning banner "⚠️ Network or settings changed since this result - run again" relies strictly on comparing SHA-256 signatures of inputs (network snapshot, scenario, physics settings) against the signature stored in the simulation result (`ComparisonResult.is_outdated()`). No event counters or "dirty" flags are used. Signatures round floating-point parameters (e.g. gravitational acceleration $g$ and viscosity $\nu$) to prevent IEEE 754 precision mismatches. The banner dynamically recomputes after every edit, run, and preset change.

---

## 37. Editing panel hidden on the Comparison tab to give the plots room

*(Confirmed by the client)*

**Decision**: While the Comparison tab is active, the left editing panel (Add Node / Add Pipe / Nodes list) is automatically hidden to give the comparison plots maximum horizontal room. When switching back to any other tab (Network, Matrices, Checks, Incidence), the editing panel is automatically restored. No data is changed, and the state of the left panel (entered text, selected items) is strictly preserved.











