"""core/epanet_runner.py – EPANET 2.2+ engine runner via ctypes.

Runs real EPANET simulations on networks and scenarios, extracting results
in the identical format and state ordering as the DAE solver.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

from core.models import Network
from core.physics import PhysicsSettings
from core.scenario import Scenario
from core.epanet_export import export_epanet, ExportMap
from core.dae_model import build_switched_dae, UnsupportedNetworkError


class EpanetError(RuntimeError):
    """Raised when the EPANET engine encounters an error."""

    def __init__(self, code: int, message: str):
        super().__init__(f"EPANET Error {code}: {message}")
        self.code = code
        self.message = message


@dataclass
class EpanetResult:
    """Simulation results from the real EPANET engine.

    Attributes
    ----------
    t : np.ndarray
        Simulation report times in seconds (shape: (M,)).
    X : np.ndarray
        State trajectory array (shape: (M, K)), with columns exactly matching
        the DAE state order (flows of non-pumped pipes, then pressure heads).
    labels : list[str]
        State labels matching the DAE state vector.
    engine_version : int
        EPANET engine version integer (e.g. 20305 for 2.3.5).
    inp_text : str
        Input file text passed to the engine.
    warnings : list[str]
        List of warning messages generated during simulation.
    """

    t: np.ndarray
    X: np.ndarray
    labels: List[str]
    engine_version: int
    inp_text: str
    warnings: List[str] = field(default_factory=list)

    def series(self, label: str) -> np.ndarray:
        """Return 1D array of values for the given state label."""
        if label not in self.labels:
            raise KeyError(
                f"Label {label!r} not found in EpanetResult labels: {self.labels}"
            )
        col = self.labels.index(label)
        return self.X[:, col]


def _epanet_library_filename() -> str:
    """Shared-library file name for the current operating system."""
    if sys.platform == "win32":
        return "epanet2.dll"
    if sys.platform == "darwin":
        return "libepanet2.dylib"
    return "libepanet2.so"


def _load_epanet_library(path: str):
    """Load the EPANET toolkit.

    The Windows DLL uses stdcall. Linux and macOS builds use cdecl.
    """
    if sys.platform == "win32":
        return ctypes.WinDLL(path)
    return ctypes.CDLL(path)


def _epyt_library_candidates(filename: str) -> list[str]:
    """Locations used by the epyt package, which ships EPANET binaries."""
    try:
        import epyt
    except Exception:
        return []

    root = os.path.dirname(os.path.abspath(epyt.__file__))
    # epyt names the Linux binaries "glnx", not "linux". "emscripten" is a
    # WebAssembly build and cannot be loaded with ctypes.
    plat = "win" if sys.platform == "win32" else ("mac" if sys.platform == "darwin" else "glnx")
    candidates = [
        os.path.join(root, "libraries", plat, filename),
        os.path.join(root, "libraries", filename),
    ]
    lib_root = os.path.join(root, "libraries")
    if os.path.isdir(lib_root):
        for dirpath, dirnames, filenames in os.walk(lib_root):
            dirnames[:] = [d for d in dirnames if d not in {"__pycache__", "emscripten"}]
            if filename in filenames:
                candidates.append(os.path.join(dirpath, filename))
    return candidates


def get_epanet_dll_path() -> str:
    """Locate the EPANET shared library for this operating system.

    Windows keeps looking for ``epanet2.dll`` in the bundled ``core/lib``
    folder and in a PyInstaller build. Other platforms, and a Windows
    install that did not vendor the DLL, also search the ``epyt`` package.
    Only the library name for this OS is searched: the vendored Windows DLL
    must never be picked up on Linux (e.g. Streamlit Cloud), where loading
    it fails with ``OSError: invalid ELF header``.
    """
    filename = _epanet_library_filename()
    candidates: list[str] = []

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(os.path.join(meipass, "core", "lib", filename))
        candidates.append(os.path.join(meipass, filename))

    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    for folder in (
        os.path.join(exe_dir, "_internal", "core", "lib"),
        os.path.join(exe_dir, "_internal"),
        os.path.join(exe_dir, "core", "lib"),
        exe_dir,
    ):
        candidates.append(os.path.join(folder, filename))

    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(base_dir, "lib", filename))
    candidates.append(os.path.abspath(os.path.join("core", "lib", filename)))
    candidates.extend(_epyt_library_candidates(filename))

    seen: set[str] = set()
    checked: list[str] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        checked.append(path)
        if os.path.isfile(path):
            return os.path.abspath(path)

    raise FileNotFoundError(
        f"EPANET library ({filename}) could not be located. Checked paths: {checked}"
    )


def _get_epanet_error_message(dll: Any, code: int) -> str:
    """Retrieve error message from EPANET DLL or fallback dictionary."""
    try:
        buf = ctypes.create_string_buffer(512)
        dll.EN_geterror(ctypes.c_int(code), buf, ctypes.c_int(512))
        msg = buf.value.decode("utf-8", errors="replace").strip()
        if msg:
            return msg
    except Exception:
        pass

    standard_errors = {
        101: "System Error: insufficient memory available.",
        102: "System Error: no network data available.",
        103: "System Error: hydraulics not initialized.",
        104: "System Error: no hydraulics results available.",
        105: "System Error: no water quality results available.",
        106: "System Error: no results saved to report.",
        110: "System Error: cannot solve network hydraulic equations.",
        120: "System Error: cannot solve water quality transport equations.",
        200: "Input Error: one or more errors in input file.",
        201: "Input Error: syntax error in input file.",
        202: "Input Error: illegal numeric value.",
        203: "Input Error: undefined node referenced.",
        204: "Input Error: undefined link referenced.",
        205: "Input Error: undefined time pattern referenced.",
        206: "Input Error: undefined curve referenced.",
        207: "Input Error: node cannot have both demand and emitter.",
        223: "Input Error: not enough nodes in network.",
        224: "Input Error: no tanks or reservoirs in network.",
        225: "Input Error: invalid head curve for pump.",
        240: "Input Error: node referenced in control rule does not exist.",
        241: "Input Error: link referenced in control rule does not exist.",
        250: "Input Error: invalid parameter format.",
        254: "Input Error: coordinates section missing or invalid.",
        301: "File Error: identical file names used.",
        302: "File Error: cannot open input file.",
        303: "File Error: cannot open report file.",
        304: "File Error: cannot open binary output file.",
        305: "File Error: cannot read binary output file.",
        306: "File Error: cannot write to binary output file.",
        307: "File Error: cannot read hydraulic file.",
        308: "File Error: cannot write to hydraulic file.",
        309: "File Error: cannot read water quality file.",
    }
    return standard_errors.get(code, f"Unknown EPANET error (code {code})")


def _format_warning(code: int) -> str:
    """Format EPANET hydraulic warning codes into plain language."""
    warnings_map = {
        1: "System unbalanced - hydraulic equations could not be solved at one or more steps.",
        2: "System negative pressures - pressure dropped below zero at one or more nodes.",
        3: "System disconnected - network has disconnected components.",
        4: "Maximum trials exceeded - hydraulic solver reached trial limit.",
        5: "Negative pressures in supply lines.",
        6: "Valve cannot supply requested flow/pressure.",
    }
    return warnings_map.get(code, f"EPANET hydraulic warning (code {code})")


def run_epanet(
    network: Union[Network, dict],
    scenario: Scenario,
    settings: Optional[PhysicsSettings] = None,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
) -> EpanetResult:
    """Run real EPANET 2.2+ engine simulation on the given network and scenario.

    Parameters
    ----------
    network : Network or dict
        Network container.
    scenario : Scenario
        Switching scenario.
    settings : PhysicsSettings, optional
        Physical parameters (defaults to default PhysicsSettings).
    positions : dict, optional
        Custom node coordinates.

    Returns
    -------
    EpanetResult
        Simulation results with time vector and state matrix in exact DAE order.

    Raises
    ------
    UnsupportedNetworkError
        If network contains pumps or tanks.
    ScenarioError
        If scenario contains invalid switch times or valve configurations.
    EpanetError
        If EPANET engine fails to open, initialize, or solve the hydraulics.
    """
    if settings is None:
        settings = PhysicsSettings()

    # Generate INP file and mapping
    inp_text, export_map = export_epanet(
        network, scenario, settings, positions=positions
    )
    dae = build_switched_dae(network, settings)

    temp_dir = tempfile.mkdtemp(prefix="epanet_run_")
    ph = ctypes.c_void_p()
    dll = None

    try:
        inp_file = os.path.join(temp_dir, "model.inp")
        rpt_file = os.path.join(temp_dir, "model.rpt")
        out_file = os.path.join(temp_dir, "model.out")

        with open(inp_file, "w", encoding="utf-8") as f:
            f.write(inp_text)

        dll_path = get_epanet_dll_path()
        dll = _load_epanet_library(dll_path)

        # Configure ctypes function signatures
        dll.EN_createproject.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        dll.EN_createproject.restype = ctypes.c_int

        dll.EN_deleteproject.argtypes = [ctypes.c_void_p]
        dll.EN_deleteproject.restype = ctypes.c_int

        dll.EN_open.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        dll.EN_open.restype = ctypes.c_int

        dll.EN_close.argtypes = [ctypes.c_void_p]
        dll.EN_close.restype = ctypes.c_int

        dll.EN_getversion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        dll.EN_getversion.restype = ctypes.c_int

        dll.EN_geterror.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        dll.EN_geterror.restype = ctypes.c_int

        dll.EN_openH.argtypes = [ctypes.c_void_p]
        dll.EN_openH.restype = ctypes.c_int

        dll.EN_initH.argtypes = [ctypes.c_void_p, ctypes.c_int]
        dll.EN_initH.restype = ctypes.c_int

        dll.EN_runH.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long)]
        dll.EN_runH.restype = ctypes.c_int

        dll.EN_nextH.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long)]
        dll.EN_nextH.restype = ctypes.c_int

        dll.EN_closeH.argtypes = [ctypes.c_void_p]
        dll.EN_closeH.restype = ctypes.c_int

        dll.EN_getnodeindex.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        dll.EN_getnodeindex.restype = ctypes.c_int

        dll.EN_getlinkindex.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        dll.EN_getlinkindex.restype = ctypes.c_int

        dll.EN_getnodevalue.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        dll.EN_getnodevalue.restype = ctypes.c_int

        dll.EN_getlinkvalue.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        dll.EN_getlinkvalue.restype = ctypes.c_int

        # 1. Create project
        ret = dll.EN_createproject(ctypes.byref(ph))
        if ret > 0:
            raise EpanetError(ret, _get_epanet_error_message(dll, ret))

        # 2. Open project
        ret = dll.EN_open(
            ph,
            inp_file.encode("utf-8"),
            rpt_file.encode("utf-8"),
            out_file.encode("utf-8"),
        )
        if ret > 0:
            raise EpanetError(ret, _get_epanet_error_message(dll, ret))

        # 3. Version query
        v = ctypes.c_int()
        dll.EN_getversion(ctypes.byref(v))
        engine_version = v.value

        # 4. Map DAE labels to EPANET element indices
        # DAE labels: flows of pipes, then pressure heads (Pl, Pr, junctions, reservoirs)
        state_readers: List[Tuple[str, int]] = []
        for label in dae.labels:
            elem_id, elem_type = export_map.get_element(label)
            idx = ctypes.c_int()
            if elem_type == "link":
                err = dll.EN_getlinkindex(
                    ph, elem_id.encode("ascii"), ctypes.byref(idx)
                )
                if err > 0:
                    raise EpanetError(
                        err,
                        f"Failed to find link index for {elem_id!r} (label {label}): {_get_epanet_error_message(dll, err)}",
                    )
            elif elem_type == "node":
                err = dll.EN_getnodeindex(
                    ph, elem_id.encode("ascii"), ctypes.byref(idx)
                )
                if err > 0:
                    raise EpanetError(
                        err,
                        f"Failed to find node index for {elem_id!r} (label {label}): {_get_epanet_error_message(dll, err)}",
                    )
            else:
                raise ValueError(f"Unknown element type {elem_type!r} for label {label}")

            state_readers.append((elem_type, idx.value))

        # 5. Open and initialize hydraulics
        ret = dll.EN_openH(ph)
        if ret >= 100:
            raise EpanetError(ret, _get_epanet_error_message(dll, ret))

        h_open = True
        times_list: List[float] = []
        rows_list: List[np.ndarray] = []
        warning_codes: List[int] = []

        try:
            ret = dll.EN_initH(ph, 0)
            if ret >= 100:
                raise EpanetError(ret, _get_epanet_error_message(dll, ret))

            t_val = ctypes.c_long()
            tstep_val = ctypes.c_long()
            num_states = len(dae.labels)
            val = ctypes.c_double()

            # Stepping loop
            while True:
                ret = dll.EN_runH(ph, ctypes.byref(t_val))
                if ret >= 100:
                    raise EpanetError(ret, _get_epanet_error_message(dll, ret))
                elif ret > 0 and ret not in warning_codes:
                    warning_codes.append(ret)

                curr_t = float(t_val.value)
                times_list.append(curr_t)

                row = np.zeros(num_states, dtype=float)
                for i, (elem_type, elem_idx) in enumerate(state_readers):
                    if elem_type == "link":
                        # Property 8 = EN_FLOW
                        dll.EN_getlinkvalue(
                            ph, ctypes.c_int(elem_idx), ctypes.c_int(8), ctypes.byref(val)
                        )
                    else:
                        # Property 11 = EN_PRESSURE
                        dll.EN_getnodevalue(
                            ph,
                            ctypes.c_int(elem_idx),
                            ctypes.c_int(11),
                            ctypes.byref(val),
                        )
                    row[i] = val.value

                rows_list.append(row)

                ret = dll.EN_nextH(ph, ctypes.byref(tstep_val))
                if ret >= 100:
                    raise EpanetError(ret, _get_epanet_error_message(dll, ret))
                elif ret > 0 and ret not in warning_codes:
                    warning_codes.append(ret)

                if tstep_val.value <= 0:
                    break

        finally:
            if h_open:
                dll.EN_closeH(ph)

        # Convert to numpy arrays
        t_arr = np.array(times_list, dtype=float)
        X_arr = np.vstack(rows_list) if rows_list else np.empty((0, len(dae.labels)), dtype=float)
        warnings_text = [_format_warning(c) for c in warning_codes]

        return EpanetResult(
            t=t_arr,
            X=X_arr,
            labels=list(dae.labels),
            engine_version=engine_version,
            inp_text=inp_text,
            warnings=warnings_text,
        )

    finally:
        # Guarantee project and engine cleanup
        if dll is not None and ph.value is not None:
            try:
                dll.EN_close(ph)
            except Exception:
                pass
            try:
                dll.EN_deleteproject(ph)
            except Exception:
                pass

        # Guarantee temporary folder cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_epanet_inp(inp_text_or_path: str, dae: Optional[Any] = None) -> None:
    """Run EPANET engine directly on an INP text or file path to verify validity.

    Raises EpanetError if parsing, opening, initializing, or running fails.
    """
    temp_dir = tempfile.mkdtemp(prefix="epanet_inp_check_")
    ph = ctypes.c_void_p()
    dll = None

    try:
        if os.path.isfile(inp_text_or_path):
            inp_file = os.path.abspath(inp_text_or_path)
        else:
            inp_file = os.path.join(temp_dir, "input.inp")
            with open(inp_file, "w", encoding="utf-8") as f:
                f.write(inp_text_or_path)

        rpt_file = os.path.join(temp_dir, "check.rpt")
        out_file = os.path.join(temp_dir, "check.out")

        dll_path = get_epanet_dll_path()
        dll = _load_epanet_library(dll_path)

        dll.EN_createproject.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        dll.EN_createproject.restype = ctypes.c_int

        dll.EN_deleteproject.argtypes = [ctypes.c_void_p]
        dll.EN_deleteproject.restype = ctypes.c_int

        dll.EN_open.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        dll.EN_open.restype = ctypes.c_int

        dll.EN_close.argtypes = [ctypes.c_void_p]
        dll.EN_close.restype = ctypes.c_int

        dll.EN_geterror.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        dll.EN_geterror.restype = ctypes.c_int

        dll.EN_openH.argtypes = [ctypes.c_void_p]
        dll.EN_openH.restype = ctypes.c_int

        dll.EN_initH.argtypes = [ctypes.c_void_p, ctypes.c_int]
        dll.EN_initH.restype = ctypes.c_int

        dll.EN_runH.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long)]
        dll.EN_runH.restype = ctypes.c_int

        dll.EN_nextH.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long)]
        dll.EN_nextH.restype = ctypes.c_int

        dll.EN_closeH.argtypes = [ctypes.c_void_p]
        dll.EN_closeH.restype = ctypes.c_int

        # 1. Create project
        ret = dll.EN_createproject(ctypes.byref(ph))
        if ret > 0:
            raise EpanetError(ret, _get_epanet_error_message(dll, ret))

        # 2. Open project
        ret = dll.EN_open(
            ph,
            inp_file.encode("utf-8"),
            rpt_file.encode("utf-8"),
            out_file.encode("utf-8"),
        )
        if ret > 0:
            raise EpanetError(ret, _get_epanet_error_message(dll, ret))

        ret = dll.EN_openH(ph)
        if ret >= 100:
            raise EpanetError(ret, _get_epanet_error_message(dll, ret))

        times_list: List[float] = []
        try:
            ret = dll.EN_initH(ph, 0)
            if ret >= 100:
                raise EpanetError(ret, _get_epanet_error_message(dll, ret))

            t_val = ctypes.c_long()
            tstep_val = ctypes.c_long()
            while True:
                ret = dll.EN_runH(ph, ctypes.byref(t_val))
                if ret >= 100:
                    raise EpanetError(ret, _get_epanet_error_message(dll, ret))
                times_list.append(float(t_val.value))
                ret = dll.EN_nextH(ph, ctypes.byref(tstep_val))
                if ret >= 100:
                    raise EpanetError(ret, _get_epanet_error_message(dll, ret))
                if tstep_val.value <= 0:
                    break
        finally:
            dll.EN_closeH(ph)

        return EpanetResult(
            t=np.array(times_list, dtype=float),
            X=np.empty((len(times_list), 0), dtype=float),
            labels=[],
            engine_version=0,
            inp_text=inp_text_or_path if isinstance(inp_text_or_path, str) else "",
        )

    finally:
        if dll is not None and ph.value is not None:
            try:
                dll.EN_close(ph)
            except Exception:
                pass
            try:
                dll.EN_deleteproject(ph)
            except Exception:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_epanet_version() -> str:
    """Return the EPANET engine version as a string (e.g. '2.3.5 (build 20305)')."""
    try:
        dll_path = get_epanet_dll_path()
        dll = _load_epanet_library(dll_path)
        dll.EN_getversion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        dll.EN_getversion.restype = ctypes.c_int
        v = ctypes.c_int()
        ret = dll.EN_getversion(ctypes.byref(v))
        if ret == 0:
            ver_int = v.value
            major = ver_int // 10000
            minor = (ver_int % 10000) // 100
            patch = ver_int % 100
            return f"{major}.{minor}.{patch} (build {ver_int})"
        return f"build {v.value}"
    except Exception:
        return "2.3.5 (bundled)"


