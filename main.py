"""Entry point for Water Network Modeler desktop application."""

import os
import sys
import ctypes
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QIcon, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from app_info import APP_NAME, APP_ID, COMPANY_NAME, __version__, resource_path
from ui.main_window import MainWindow
from core.models import Network
from core.params import missing_parameters
from core.examples import example_paper_network, example_broken_network
from core.matrices import build_matrices
from core.io import save_project, load_project_full


def setup_exception_hook():
    """Install global exception hook to catch unhandled errors and display dialog."""
    def excepthook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_str = "".join(tb_lines)

        try:
            local_appdata = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
            log_dir = Path(local_appdata) / APP_ID
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "error.log"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now().isoformat()}] Unhandled Exception:\n")
                f.write(tb_str)
                f.write("=" * 60 + "\n")
            log_path_str = str(log_file)
        except Exception as e:
            log_path_str = f"<Unable to write log file: {e}>"

        sys.stderr.write(tb_str)

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Something went wrong")
        msg_box.setText("Something went wrong in Water Network Modeler.")
        msg_box.setInformativeText(
            f"An unexpected error occurred.\n\n"
            f"A detailed error log has been saved to:\n{log_path_str}\n\n"
            f"{exc_type.__name__}: {exc_value}"
        )
        msg_box.setDetailedText(tb_str)
        msg_box.exec()

    sys.excepthook = excepthook


def run_selftest(log_path: Optional[Path] = None) -> int:
    """Run offscreen selftest suite: load examples, verify matrices/checks, render tabs, test I/O."""
    import faulthandler
    try:
        faulthandler.enable()
        faulthandler.dump_traceback_later(90, exit=True)
    except Exception:
        pass

    # In selftest mode no modal dialog may ever open
    from PySide6.QtWidgets import QMessageBox, QFileDialog, QDialog
    QMessageBox.information = lambda *args, **kwargs: QMessageBox.Ok
    QMessageBox.warning = lambda *args, **kwargs: QMessageBox.Ok
    QMessageBox.critical = lambda *args, **kwargs: QMessageBox.Ok
    QMessageBox.question = lambda *args, **kwargs: QMessageBox.Discard
    QMessageBox.exec = lambda *args, **kwargs: QMessageBox.Ok
    QMessageBox.exec_ = lambda *args, **kwargs: QMessageBox.Ok
    QFileDialog.getSaveFileName = lambda *args, **kwargs: ("", "")
    QFileDialog.getOpenFileName = lambda *args, **kwargs: ("", "")
    QFileDialog.getExistingDirectory = lambda *args, **kwargs: ""

    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    if log_path is None:
        temp_dir = Path(os.environ.get("TEMP", os.environ.get("TMP", str(Path.home()))))
        log_path = temp_dir / "wnm_selftest.log"

    def write_log(msg: str):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {msg}"
        print(line)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    # Clear log file initially
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== WaterNetworkModeler {__version__} Selftest Log Started ===\n")
    except Exception as e:
        print(f"Failed to initialize log file {log_path}: {e}")

    try:
        write_log(f"Starting selftest for {APP_NAME} {__version__}...")
        write_log(f"Platform: {sys.platform}, Python: {sys.version.split()[0]}")
        write_log(f"QT_QPA_PLATFORM: {os.environ.get('QT_QPA_PLATFORM')}")

        # Ensure QApplication
        app = setup_application(argv=["--selftest"])

        # 1. Build a network with parameters, check missing_parameters, save v2 with positions and edge bend, reload and compare
        write_log("Testing physical parameters, missing_parameters, and .wnet v2 save/reload...")
        net = Network()
        net.add_node(type="reservoir", id="R1", elevation=25.0, head=60.0)
        net.add_node(type="junction", id="J1", elevation=15.0)
        net.add_pipe(start="R1", end="J1", id="P1", length=120.0, diameter=0.25, roughness=0.00015)

        missing = missing_parameters(net)
        assert len(missing) == 0, f"Expected 0 missing parameters, got: {missing}"

        incomplete = Network()
        incomplete.add_node(type="junction", id="J_inc", elevation=None)
        missing_inc = missing_parameters(incomplete)
        assert len(missing_inc) == 1 and "elevation" in missing_inc[0], f"Expected elevation missing warning, got {missing_inc}"

        temp_dir = Path(os.environ.get("TEMP", os.environ.get("TMP", str(Path.home()))))
        temp_v2_file = temp_dir / f"wnm_selftest_v2_{os.getpid()}.wnet"
        positions_v2 = {"R1": (-0.4, 0.2), "J1": (0.4, -0.2)}
        edge_bends_v2 = {"P1": 0.18}
        save_project(temp_v2_file, net, positions=positions_v2, edge_bends=edge_bends_v2)
        assert temp_v2_file.is_file(), f"v2 project file was not created: {temp_v2_file}"

        proj_data = load_project_full(temp_v2_file)
        assert len(proj_data.network.nodes) == 2, f"Expected 2 nodes, got {len(proj_data.network.nodes)}"
        assert len(proj_data.network.pipes) == 1, f"Expected 1 pipe, got {len(proj_data.network.pipes)}"
        r1_loaded = next(n for n in proj_data.network.nodes if n.id == "R1")
        assert r1_loaded.elevation == 25.0 and r1_loaded.head == 60.0
        j1_loaded = next(n for n in proj_data.network.nodes if n.id == "J1")
        assert j1_loaded.elevation == 15.0
        p1_loaded = next(p for p in proj_data.network.pipes if p.id == "P1")
        assert p1_loaded.length == 120.0 and p1_loaded.diameter == 0.25 and p1_loaded.roughness == 0.00015
        assert "R1" in proj_data.positions and "J1" in proj_data.positions
        assert abs(proj_data.edge_bends.get("P1", 0.0) - 0.18) < 1e-4

        try:
            temp_v2_file.unlink()
        except Exception:
            pass
        write_log("Physical parameters and .wnet v2 round-trip verified.")

        # 2. Load embedded v1-format JSON string and check fields are None
        write_log("Testing embedded v1-format project loading...")
        v1_json = """{
  "format": "water-network",
  "version": 1,
  "nodes": [
    {"id": "R1", "type": "reservoir"},
    {"id": "J1", "type": "junction"}
  ],
  "pipes": [
    {"id": "P1", "start": "R1", "end": "J1", "pump": false}
  ],
  "counters": {"junction": 1, "tank": 0, "reservoir": 1, "pipe": 1, "pump": 0},
  "positions": {"R1": [-0.5, 0.5], "J1": [0.5, -0.5]}
}"""
        temp_v1_file = temp_dir / f"wnm_selftest_v1_{os.getpid()}.wnet"
        temp_v1_file.write_text(v1_json, encoding="utf-8")
        v1_proj = load_project_full(temp_v1_file)
        v1_r1 = next(n for n in v1_proj.network.nodes if n.id == "R1")
        v1_j1 = next(n for n in v1_proj.network.nodes if n.id == "J1")
        v1_p1 = next(p for p in v1_proj.network.pipes if p.id == "P1")
        assert v1_r1.elevation is None, f"Expected R1 elevation None, got {v1_r1.elevation}"
        assert v1_r1.head is None, f"Expected R1 head None, got {v1_r1.head}"
        assert v1_j1.elevation is None, f"Expected J1 elevation None, got {v1_j1.elevation}"
        assert v1_p1.length is None, f"Expected P1 length None, got {v1_p1.length}"
        assert v1_p1.diameter is None, f"Expected P1 diameter None, got {v1_p1.diameter}"
        assert v1_p1.roughness is None, f"Expected P1 roughness None, got {v1_p1.roughness}"
        try:
            temp_v1_file.unlink()
        except Exception:
            pass
        write_log("v1-format project compatibility verified (fields are None).")

        # 3. Create real MainWindow, load paper example
        write_log("Instantiating MainWindow...")
        window = MainWindow()
        window.show()
        app.processEvents()
        write_log("MainWindow initialized and shown offscreen.")

        write_log("Loading paper example network...")
        window.network = example_paper_network()
        window.refresh()
        app.processEvents()

        num_nodes = len(window.network.nodes)
        num_pipes = len(window.network.pipes)
        write_log(f"Paper network loaded: {num_nodes} nodes, {num_pipes} pipes.")
        assert num_nodes == 6, f"Expected 6 nodes in paper network, got {num_nodes}"
        assert num_pipes == 8, f"Expected 8 pipes in paper network, got {num_pipes}"

        # 4. Check GraphView node and pipe item counts, render_to_image, simulate node move
        node_items = window.graph_view.node_items()
        pipe_items = window.graph_view.pipe_items()
        write_log(f"GraphView items: {len(node_items)} nodes, {len(pipe_items)} pipes.")
        assert len(node_items) == 6, f"Expected 6 node items in GraphView, got {len(node_items)}"
        assert len(pipe_items) == 8, f"Expected 8 pipe items in GraphView, got {len(pipe_items)}"

        write_log("Calling render_to_image on GraphView...")
        exported_img = window.graph_view.render_to_image()
        assert not exported_img.isNull(), "render_to_image() returned null image"
        assert exported_img.width() > 0 and exported_img.height() > 0, (
            f"render_to_image() returned empty dimensions: {exported_img.width()}x{exported_img.height()}"
        )
        write_log(f"render_to_image verified: non-empty image {exported_img.width()}x{exported_img.height()}.")

        write_log("Simulating node drag through public API...")
        moved_signaled = [False]
        window.graph_view.positions_changed.connect(lambda: moved_signaled.__setitem__(0, True))

        j1_item = next((it for it in node_items if it.node_id == "J1"), None)
        assert j1_item is not None, "Node item J1 not found in GraphView"
        old_positions = window.graph_view.get_positions()
        old_j1_pos = old_positions.get("J1")
        assert old_j1_pos is not None, "J1 not found in get_positions()"

        vp = window.graph_view.view.viewport()
        start_view = window.graph_view.view.mapFromScene(j1_item.pos())
        target_view = start_view + QPoint(50, 50)

        QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, start_view)
        app.processEvents()
        QTest.mouseMove(vp, target_view)
        app.processEvents()
        QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, target_view)
        app.processEvents()

        assert moved_signaled[0] is True, "positions_changed was not emitted after node drag"
        new_positions = window.graph_view.get_positions()
        new_j1_pos = new_positions.get("J1")
        assert new_j1_pos is not None
        old_x, old_y = float(old_j1_pos[0]), float(old_j1_pos[1])
        new_x, new_y = float(new_j1_pos[0]), float(new_j1_pos[1])
        assert (old_x != new_x) or (old_y != new_y), f"Node J1 position did not change: ({old_x}, {old_y}) vs ({new_x}, {new_y})"
        write_log(f"Node drag simulated: positions_changed emitted, J1 moved from ({old_x:.4f}, {old_y:.4f}) to ({new_x:.4f}, {new_y:.4f}).")

        # 2. Build matrices & checks for paper example
        write_log("Verifying matrices and checks on paper network...")
        mat = build_matrices(window.network.to_dict())
        assert mat is not None, "Matrices could not be built for paper network"
        assert mat["K_dimension"] == 28, f"Expected K=28, got {mat['K_dimension']}"
        banner_text = window.checks_view.banner.text()
        assert "Regular" in banner_text, f"Expected 'Regular' banner, got {banner_text}"
        write_log(f"Paper network verified: Regular banner confirmed, K={mat['K_dimension']}.")

        # 3. Render graph and every tab once for paper example
        write_log("Rendering graph and all right-panel tabs for paper network...")
        window.graph_view.draw_network(window.network)
        tab_count = window.right_tabs.count()
        for idx in range(tab_count):
            window.right_tabs.setCurrentIndex(idx)
            app.processEvents()
            tab_name = window.right_tabs.tabText(idx)
            write_log(f"  Rendered tab [{idx}]: {tab_name}")
        app.processEvents()

        # 4. Load broken example
        write_log("Loading broken example network...")
        window.network = example_broken_network()
        window.refresh()
        app.processEvents()

        broken_nodes = len(window.network.nodes)
        broken_pipes = len(window.network.pipes)
        write_log(f"Broken network loaded: {broken_nodes} nodes, {broken_pipes} pipes.")

        # 5. Build matrices & checks for broken example
        write_log("Verifying matrices and checks on broken network...")
        mat_broken = build_matrices(window.network.to_dict())
        assert mat_broken is not None, "Matrices could not be built for broken network"
        broken_banner = window.checks_view.banner.text()
        assert "NOT regular" in broken_banner, f"Expected 'NOT regular' banner, got {broken_banner}"
        warnings_count = window.checks_view.warnings_table.rowCount()
        assert warnings_count >= 2, f"Expected >= 2 warnings for broken network, got {warnings_count}"
        write_log(f"Broken network verified: 'NOT regular' banner and {warnings_count} warnings confirmed.")

        # 6. Render graph and every tab once for broken example
        write_log("Rendering graph and all right-panel tabs for broken network...")
        window.graph_view.draw_network(window.network)
        for idx in range(tab_count):
            window.right_tabs.setCurrentIndex(idx)
            app.processEvents()
            tab_name = window.right_tabs.tabText(idx)
            write_log(f"  Rendered tab [{idx}]: {tab_name}")
        app.processEvents()

        # 7. Save and reload a temporary .wnet
        write_log("Testing project save and reload to temporary .wnet...")
        window.network = example_paper_network()
        window.refresh()
        temp_dir = Path(os.environ.get("TEMP", os.environ.get("TMP", str(Path.home()))))
        temp_wnet = temp_dir / f"wnm_selftest_temp_{os.getpid()}.wnet"

        save_project(temp_wnet, window.network, window.graph_view.get_positions())
        assert temp_wnet.is_file(), f"Saved project file does not exist: {temp_wnet}"
        file_size = temp_wnet.stat().st_size
        write_log(f"Project saved to {temp_wnet} ({file_size} bytes).")

        # Clear and verify empty
        window.network.clear()
        window.refresh()
        app.processEvents()
        assert len(window.network.nodes) == 0, "Network should be empty after clear"
        write_log("Network cleared.")

        # Reload from temp file
        window.load_file(temp_wnet)
        app.processEvents()
        reloaded_nodes = len(window.network.nodes)
        reloaded_pipes = len(window.network.pipes)
        assert reloaded_nodes == 6, f"Expected 6 reloaded nodes, got {reloaded_nodes}"
        assert reloaded_pipes == 8, f"Expected 8 reloaded pipes, got {reloaded_pipes}"
        assert "Regular" in window.checks_view.banner.text(), "Reloaded network should be regular"
        write_log("Project successfully reloaded and verified.")

        # Cleanup temp file
        try:
            temp_wnet.unlink()
            write_log("Temporary .wnet file deleted.")
        except Exception as e:
            write_log(f"Warning: could not delete temp file: {e}")

        # 8. Real EPANET engine execution
        write_log("Testing real EPANET engine runner...")
        from core.epanet_runner import run_epanet, get_epanet_dll_path
        from core.scenario import Scenario, ValveId
        from core.physics import PhysicsSettings
        from core.dae_model import build_switched_dae, steady_state

        dll_path = get_epanet_dll_path()
        write_log(f"EPANET DLL located at: {dll_path}")

        # Build specimen network
        net_spec = Network()
        net_spec.add_node("reservoir", id="R1", elevation=20.0, head=50.0)
        net_spec.add_node("junction", id="J1", elevation=5.0)
        net_spec.add_node("reservoir", id="R2", elevation=0.0, head=46.0)
        net_spec.add_pipe("R1", "J1", id="P1", length=400.0, diameter=0.4, roughness=0.000359)
        net_spec.add_pipe("J1", "R2", id="P2", length=250.0, diameter=0.4, roughness=0.000359, valve_start=True, valve_end=True)

        v1 = ValveId("P2", "start")
        v2 = ValveId("P2", "end")
        sc2 = Scenario.matlab_specimen(2, v1, v2)
        sc2 = Scenario(title=sc2.title, initial=sc2.initial, events=sc2.events, t_end=20.0)
        settings = PhysicsSettings(friction_mode="constant", friction_factor=0.02)

        ep_res = run_epanet(net_spec, sc2, settings)
        q_ep = float(ep_res.series("Pipe_1_Flow")[-1])

        dae_spec = build_switched_dae(net_spec, settings)
        x_dae = steady_state(dae_spec, (1, 1))
        q_dae = float(x_dae[0])

        diff_q = abs(q_ep - q_dae)
        write_log(f"EPANET engine version: {ep_res.engine_version}")
        write_log(f"EPANET Q (t=20s): {q_ep:.6f} m3/s, DAE steady Q: {q_dae:.6f} m3/s, diff: {diff_q:.6e}")
        write_log("EPANET engine execution verified successfully.")

        # 9. Comparison tab verification
        write_log("Verifying Comparison with EPANET tab...")
        last_tab_idx = window.right_tabs.count() - 1
        last_tab_title = window.right_tabs.tabText(last_tab_idx)
        assert last_tab_title == "Comparison with EPANET", f"Expected last tab 'Comparison with EPANET', got {last_tab_title!r}"
        assert window.right_tabs.widget(last_tab_idx) is window.comparison_tab, "Last tab widget is not comparison_tab"

        window._on_load_specimen_example()
        app.processEvents()
        assert "Ready to compare" in window.comparison_tab.ready_label.text(), "Specimen should be ready to compare"
        assert window.comparison_tab.run_btn.isEnabled(), "Run button should be enabled for specimen"
        write_log("Comparison tab verified successfully.")

        # 10. Offscreen comparison figure check (> 10 kB)
        write_log("Testing comparison figure offscreen export (> 10 kB)...")
        from core.comparison import run_comparison
        from ui.comparison_plots import draw_comparison
        from matplotlib.figure import Figure

        res_comp = run_comparison(net_spec, sc2, settings)
        fig_test = Figure(figsize=(7, 9), facecolor="white")
        draw_comparison(fig_test, res_comp)

        temp_plot_png = temp_dir / f"wnm_selftest_plot_{os.getpid()}.png"
        fig_test.savefig(temp_plot_png, dpi=200, facecolor="white", bbox_inches="tight")
        assert temp_plot_png.is_file(), f"Comparison plot PNG was not created: {temp_plot_png}"
        plot_size = temp_plot_png.stat().st_size
        write_log(f"Comparison plot PNG created: {plot_size} bytes (target > 10000 bytes)")
        assert plot_size > 10000, f"Comparison plot PNG size {plot_size} bytes <= 10 kB"

        try:
            temp_plot_png.unlink()
        except Exception:
            pass
        write_log("Comparison plot figure export verified successfully.")

        window.close()
        app.processEvents()

        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass

        write_log(f"=== Selftest PASSED (code 0) === ({APP_NAME} {__version__})")
        return 0

    except Exception as e:
        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass
        tb = traceback.format_exc()
        write_log(f"FAIL: Selftest exception: {e}\n{tb}")
        write_log("=== Selftest FAILED (code 1) ===")
        return 1


def setup_application(app: Optional[QApplication] = None, argv: Optional[list] = None) -> QApplication:
    """Shared application configuration: Fusion style, metadata, icon, and attributes."""
    if app is None:
        app = QApplication.instance()
        if app is None:
            app = QApplication(argv if argv is not None else sys.argv)
    app.setStyle("Fusion")
    app.setOrganizationName(COMPANY_NAME)
    app.setApplicationName(APP_ID)

    icon_path = resource_path("assets/app.ico")
    if icon_path.exists():
        app_icon = QIcon(str(icon_path))
        app.setWindowIcon(app_icon)
    return app


def main():
    setup_exception_hook()

    if "--selftest" in sys.argv:
        if sys.platform == "win32":
            try:
                ctypes.windll.kernel32.AttachConsole(-1)
            except Exception:
                pass
        ret = run_selftest()
        sys.exit(ret)

    # Windows taskbar icon setup
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception as e:
            print(f"Warning: SetCurrentProcessExplicitAppUserModelID failed: {e}")

    app = setup_application()

    window = MainWindow()
    if app.windowIcon():
        window.setWindowIcon(app.windowIcon())
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

