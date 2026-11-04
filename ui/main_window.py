"""Main application window wiring build panel, views, menus, and file I/O."""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QKeySequence, QAction, QPalette
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget, QSplitter,
    QMessageBox, QStatusBar, QFileDialog, QScrollArea, QApplication,
    QDialog, QLabel, QDialogButtonBox, QTextEdit
)

from core.models import Network
from core.matrices import build_matrices
from core.io import save_project, load_project, load_project_full
from core.examples import (
    example_paper_network,
    example_broken_network,
    specimen_network,
    specimen_scenario,
    specimen_physics,
)

from ui.build_panel import BuildPanel
from ui.graph_view import GraphView
from ui.matrix_view import MatrixView
from ui.checks_view import ChecksView
from ui.incidence_view import IncidenceView
from ui.comparison_tab import ComparisonTab


from app_info import APP_NAME, __version__, resource_path

class MainWindow(QMainWindow):
    """Main desktop application window for Water Network Modeler."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.network = Network()
        self._current_filepath: Optional[Path] = None
        self._last_dir: Path = Path.home()

        self.setMinimumSize(1000, 600)
        self._init_ui()
        self._size_and_center_window()
        self._update_window_title()
        self.refresh()
        self.setWindowModified(False)

    def _size_and_center_window(self):
        screen = self.screen() or QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            w, h = int(geom.width() * 0.9), int(geom.height() * 0.9)
            self.resize(w, h)
            self.move(geom.x() + (geom.width() - w) // 2, geom.y() + (geom.height() - h) // 2)

    def _update_window_title(self):
        filename = self._current_filepath.name if self._current_filepath else "Untitled"
        self.setWindowTitle(f"Water Network Modeler {__version__} - {filename}[*]")

    def _init_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        self._add_action(file_menu, "&New", QKeySequence.New, self._on_new)
        self._add_action(file_menu, "&Open...", QKeySequence.Open, self._on_open)
        self._add_action(file_menu, "&Save", QKeySequence.Save, self._on_save)
        self._add_action(file_menu, "Save &As...", QKeySequence.SaveAs, self._on_save_as)
        file_menu.addSeparator()

        ex_menu = file_menu.addMenu("Load &Example")
        ex_menu.addAction("Paper sample network").triggered.connect(lambda: self._on_load_example(example_paper_network))
        ex_menu.addAction("Broken network (demo of warnings)").triggered.connect(lambda: self._on_load_example(example_broken_network))
        ex_menu.addAction("Specimen with valves (EPANET comparison)").triggered.connect(self._on_load_specimen_example)
        file_menu.addSeparator()

        file_menu.addAction("E&xit").triggered.connect(self.close)

        help_menu = menubar.addMenu("&Help")
        help_menu.addAction("&About").triggered.connect(self._show_about)
        help_menu.addAction("&Third-party notices").triggered.connect(self._show_third_party_notices)
        help_menu.addAction("&How to compare with EPANET").triggered.connect(self._show_how_to_compare)

        self.setStatusBar(QStatusBar(self))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        splitter = QSplitter(Qt.Horizontal)

        self.build_panel = BuildPanel()
        self.build_panel.node_added.connect(self._on_add_node)
        self.build_panel.pipe_added.connect(self._on_add_pipe)
        self.build_panel.node_deleted.connect(self._on_delete_node)
        self.build_panel.pipe_deleted.connect(self._on_delete_pipe)
        self.build_panel.node_updated.connect(self._on_update_node)
        self.build_panel.pipe_updated.connect(self._on_update_pipe)
        self.build_panel.cleared.connect(self._on_clear_all)

        self.build_scroll_area = QScrollArea()
        self.build_scroll_area.setWidget(self.build_panel)
        self.build_scroll_area.setWidgetResizable(True)
        self.build_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.build_scroll_area.setFixedWidth(330)
        splitter.addWidget(self.build_scroll_area)

        self.right_tabs = QTabWidget()
        self.right_tabs.currentChanged.connect(self._on_tab_changed)
        self.graph_view = GraphView()
        self.matrix_view = MatrixView()
        self.checks_view = ChecksView()
        self.incidence_view = IncidenceView()
        self.comparison_tab = ComparisonTab(self.network, self.graph_view.get_positions, self)

        self._syncing_selection = False

        self.checks_view.warning_double_clicked.connect(self._on_warning_double_clicked)
        self.graph_view.positions_changed.connect(self._on_positions_changed)
        self.graph_view.edge_bends_changed.connect(self._on_positions_changed)
        self.graph_view.node_clicked.connect(self._on_graph_node_clicked)
        self.graph_view.pipe_clicked.connect(self._on_graph_pipe_clicked)
        self.graph_view.background_clicked.connect(self._on_graph_background_clicked)
        self.graph_view.relayout_btn.clicked.connect(self._mark_edited)
        self.comparison_tab.network_modified.connect(self._mark_edited)

        self.build_panel.node_selected.connect(self._on_panel_node_selected)
        self.build_panel.pipe_selected.connect(self._on_panel_pipe_selected)
        self.build_panel.selection_cleared.connect(self._on_panel_selection_cleared)

        self.right_tabs.addTab(self.graph_view, "Network")
        self.right_tabs.addTab(self.matrix_view, "Matrices")
        self.right_tabs.addTab(self.checks_view, "Checks")
        self.right_tabs.addTab(self.incidence_view, "Incidence")
        self.right_tabs.addTab(self.comparison_tab, "Comparison with EPANET")

        splitter.addWidget(self.right_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)
        self._update_window_title()
        self.apply_tab_style()

    def _on_tab_changed(self, index: int):
        """Hide left editing panel while Comparison tab is active to give plots maximum space."""
        is_comp = (self.right_tabs.widget(index) == self.comparison_tab)
        if is_comp:
            self.build_scroll_area.hide()
        else:
            self.build_scroll_area.show()

    def apply_tab_style(self):
        """Ensure right_tabs and child tab bars have high contrast >= 4.5:1 in both light and dark themes."""
        pal = self.palette()
        is_dark = pal.color(QPalette.Window).value() < 128
        self.build_panel.update_theme_style(is_dark)
        if is_dark:
            ss = """
            QTabBar::tab {
                background: #2D2D2D;
                color: #DCDCDC;
                padding: 6px 14px;
                border: 1px solid #444444;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #1E1E1E;
                color: #FFFFFF;
                font-weight: bold;
                border-bottom: 2px solid #0078D7;
            }
            QTabBar::tab:!selected:hover {
                background: #383838;
                color: #FFFFFF;
            }
            """
        else:
            ss = """
            QTabBar::tab {
                background: #E4E6EA;
                color: #222222;
                padding: 6px 14px;
                border: 1px solid #C0C4CC;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #FFFFFF;
                color: #000000;
                font-weight: bold;
                border-bottom: 2px solid #0078D7;
            }
            QTabBar::tab:!selected:hover {
                background: #D8DCE2;
                color: #000000;
            }
            """
        self.right_tabs.setStyleSheet(ss)
        if hasattr(self, "comparison_tab") and hasattr(self.comparison_tab, "results_tabs"):
            self.comparison_tab.results_tabs.setStyleSheet(ss)
        if hasattr(self, "comparison_tab") and hasattr(self.comparison_tab, "update_theme_style"):
            self.comparison_tab.update_theme_style(is_dark)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.PaletteChange:
            self.apply_tab_style()

    def _add_action(self, menu, text, shortcut, slot):
        act = QAction(text, self)
        act.setShortcut(shortcut)
        act.triggered.connect(slot)
        menu.addAction(act)

    def _show_about(self):
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<b>{APP_NAME}</b><br>Version {__version__}<br><br>"
            "Based on the research paper on water network solvability."
        )

    def _maybe_save_changes(self) -> bool:
        if not self.isWindowModified():
            return True
        filename = self._current_filepath.name if self._current_filepath else "Untitled"
        res = QMessageBox.question(
            self, "Unsaved Changes", f"Do you want to save changes to {filename}?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
        )
        if res == QMessageBox.Save:
            return self._on_save()
        elif res == QMessageBox.Discard:
            return True
        return False

    def closeEvent(self, event):
        if hasattr(self, "comparison_tab"):
            self.comparison_tab.stop_worker()
        if self._maybe_save_changes():
            event.accept()
        else:
            event.ignore()

    def load_file(self, path: Path) -> None:
        path = Path(path)
        loaded = load_project_full(path)
        self.network = loaded.network
        self.comparison_tab.set_network(loaded.network)
        self.graph_view.set_positions(loaded.positions)
        self.graph_view.set_edge_bends(loaded.edge_bends)
        self.comparison_tab.set_scenario(loaded.scenario)
        self.comparison_tab.set_physics(loaded.physics)
        self._current_filepath = path
        self._last_dir = path.parent
        self.refresh()
        self.setWindowModified(False)
        self._update_window_title()

    def _on_new(self):
        if not self._maybe_save_changes():
            return
        self.network.clear()
        self.comparison_tab.set_network(self.network)
        self.graph_view.set_positions({})
        self.graph_view.set_edge_bends({})
        self.comparison_tab.reset()
        self._current_filepath = None
        self.refresh()
        self.setWindowModified(False)
        self._update_window_title()

    def _on_open(self):
        if not self._maybe_save_changes():
            return
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open Project File", str(self._last_dir), "Water Network Files (*.wnet)"
        )
        if path_str:
            try:
                self.load_file(Path(path_str))
            except Exception as e:
                self._handle_error("Failed to load project", e)

    def _on_save(self) -> bool:
        if self._current_filepath is None:
            return self._on_save_as()
        try:
            save_project(
                self._current_filepath,
                self.network,
                self.graph_view.get_positions(),
                self.graph_view.get_edge_bends(),
                scenario=self.comparison_tab.get_scenario(),
                physics=self.comparison_tab.get_physics(),
            )
            self.setWindowModified(False)
            self._update_window_title()
            return True
        except Exception as e:
            self._handle_error("Failed to save project", e)
            return False

    def _on_save_as(self) -> bool:
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", str(self._last_dir / "network.wnet"), "Water Network Files (*.wnet)"
        )
        if not path_str:
            return False
        path = Path(path_str)
        self._last_dir = path.parent
        self._current_filepath = path
        return self._on_save()

    def _on_load_example(self, example_factory):
        if not self._maybe_save_changes():
            return
        self.network = example_factory()
        pos = getattr(self.network, "default_positions", {})
        self.comparison_tab.set_network(self.network)
        self.graph_view.set_positions(dict(pos) if pos else {})
        self.graph_view.set_edge_bends({})
        self.comparison_tab.reset()
        self._current_filepath = None
        self.refresh()
        self.graph_view.fit_in_view()
        self.setWindowModified(False)
        self._update_window_title()

    def _on_load_specimen_example(self):
        if not self._maybe_save_changes():
            return
        net, pos = specimen_network()
        self.network = net
        self.comparison_tab.set_network(net)
        self.graph_view.set_positions(pos)
        self.graph_view.set_edge_bends({})
        sc1 = specimen_scenario(1)
        phys = specimen_physics()
        self.comparison_tab.set_scenario(sc1)
        self.comparison_tab.set_physics(phys)
        self._current_filepath = None
        self.refresh()
        self.graph_view.fit_in_view()
        self.setWindowModified(False)
        self._update_window_title()

    def _on_add_node(self, ntype: str, params: Optional[dict] = None):
        try:
            kwargs = dict(params) if params else {}
            self.network.add_node(ntype, **kwargs)
            self._mark_edited()
        except Exception as e:
            self._handle_error("Failed to add node", e)

    def _on_add_pipe(self, start: str, end: str, has_pump: bool, power: float, params: Optional[dict] = None):
        try:
            kwargs = dict(params) if params else {}
            self.network.add_pipe(start, end, has_pump=has_pump, pump_power=power, **kwargs)
            self._mark_edited()
        except Exception as e:
            self._handle_error("Failed to add pipe", e)

    def _on_update_node(self, node_id: str, fields: dict):
        try:
            self.network.update_node(node_id, **fields)
            self._mark_edited()
        except Exception as e:
            self._handle_error("Failed to update node", e)

    def _on_update_pipe(self, pipe_id: str, fields: dict):
        try:
            self.network.update_pipe(pipe_id, **fields)
            self._mark_edited()
        except Exception as e:
            self._handle_error("Failed to update pipe", e)

    def _on_delete_node(self, node_id: str):
        try:
            self.network.remove_node(node_id)
            remaining_pids = {p.id for p in self.network.pipes}
            for pid in list(self.graph_view._edge_bends.keys()):
                if pid not in remaining_pids:
                    self.graph_view._edge_bends.pop(pid, None)
            self._mark_edited()
        except Exception as e:
            self._handle_error("Failed to delete node", e)

    def _on_delete_pipe(self, pipe_id: str):
        try:
            self.network.remove_pipe(pipe_id)
            self.graph_view._edge_bends.pop(pipe_id, None)
            self._mark_edited()
        except Exception as e:
            self._handle_error("Failed to delete pipe", e)

    def _on_clear_all(self):
        try:
            self.network.clear()
            self.graph_view.set_positions({})
            self.comparison_tab.reset()
            self._mark_edited()
        except Exception as e:
            self._handle_error("Failed to clear network", e)

    def _on_positions_changed(self):
        self.setWindowModified(True)
        self._update_window_title()

    def _on_graph_node_clicked(self, node_id: str):
        if self._syncing_selection:
            return
        self._syncing_selection = True
        try:
            self.build_panel.select_node(node_id)
            self.graph_view.set_selection(node_id=node_id)
        finally:
            self._syncing_selection = False

    def _on_graph_pipe_clicked(self, pipe_id: str):
        if self._syncing_selection:
            return
        self._syncing_selection = True
        try:
            self.build_panel.select_pipe(pipe_id)
            self.graph_view.set_selection(pipe_id=pipe_id)
        finally:
            self._syncing_selection = False

    def _on_graph_background_clicked(self):
        if self._syncing_selection:
            return
        self._syncing_selection = True
        try:
            self.build_panel.clear_selection()
            self.graph_view.clear_selection()
        finally:
            self._syncing_selection = False

    def _on_panel_node_selected(self, node_id: str):
        if self._syncing_selection:
            return
        self._syncing_selection = True
        try:
            self.graph_view.set_selection(node_id=node_id)
        finally:
            self._syncing_selection = False

    def _on_panel_pipe_selected(self, pipe_id: str):
        if self._syncing_selection:
            return
        self._syncing_selection = True
        try:
            self.graph_view.set_selection(pipe_id=pipe_id)
        finally:
            self._syncing_selection = False

    def _on_panel_selection_cleared(self):
        if self._syncing_selection:
            return
        self._syncing_selection = True
        try:
            self.graph_view.clear_selection()
        finally:
            self._syncing_selection = False

    def _mark_edited(self):
        self.graph_view.clear_highlight()
        self.setWindowModified(True)
        self.refresh()

    def _on_warning_double_clicked(self, ids: list):
        self.right_tabs.setCurrentWidget(self.graph_view)
        self.graph_view.highlight(ids)

    def _handle_error(self, title: str, exc: Exception):
        msg = str(exc)
        self.statusBar().showMessage(f"Error: {msg}")
        QMessageBox.critical(self, title, f"{title}: {msg}")

    def refresh(self):
        try:
            self.build_panel.update_network(self.network)
            self.graph_view.draw_network(self.network)

            matrices_res = build_matrices(self.network.to_dict()) if self.network.pipes else None
            self.matrix_view.update_matrices(self.network, matrices_res)
            self.checks_view.update_checks(self.network)
            self.incidence_view.update_incidence(self.network)

            num_nodes, num_pipes = len(self.network.nodes), len(self.network.pipes)
            K = matrices_res["K_dimension"] if matrices_res else 0
            self.statusBar().showMessage(f"Nodes: {num_nodes}   Pipes: {num_pipes}   K = {K}")

            if hasattr(self, "comparison_tab"):
                self.comparison_tab.update_tab(self.network)
        except Exception as e:
            msg = str(e)
            self.statusBar().showMessage(f"Error during refresh: {msg}")
            QMessageBox.critical(self, "Refresh Error", f"An error occurred while refreshing the UI:\n{msg}")

    def _create_about_dialog(self) -> QDialog:
        from core.epanet_runner import get_epanet_version
        epanet_ver = get_epanet_version()
        dlg = QDialog(self)
        dlg.setWindowTitle("About Water Network Modeler")
        dlg.setMinimumWidth(440)
        layout = QVBoxLayout(dlg)

        title_label = QLabel(f"<h2>Water Network Modeler {__version__}</h2>")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        epanet_label = QLabel(f"<b>EPANET Engine Version:</b> {epanet_ver}")
        epanet_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(epanet_label)

        layout.addSpacing(10)
        license_text = (
            "License: not specified yet. All rights reserved by the author unless stated otherwise."
        )
        lic_label = QLabel(license_text)
        lic_label.setWordWrap(True)
        lic_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(lic_label)

        layout.addSpacing(15)
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok)
        btn_box.accepted.connect(dlg.accept)
        layout.addWidget(btn_box)

        return dlg

    def _show_about(self):
        dlg = self._create_about_dialog()
        dlg.exec()

    def _create_third_party_notices_dialog(self) -> QDialog:
        dlg = QDialog(self)
        dlg.setWindowTitle("Third-party notices")
        dlg.resize(700, 500)
        layout = QVBoxLayout(dlg)

        notices_path = resource_path("THIRD_PARTY_NOTICES.txt")
        if not notices_path.is_file():
            notices_path = Path(__file__).resolve().parent.parent / "THIRD_PARTY_NOTICES.txt"

        content = ""
        if notices_path.is_file():
            try:
                content = notices_path.read_text(encoding="utf-8")
            except Exception as e:
                content = f"Failed to read notices: {e}"
        else:
            content = "THIRD_PARTY_NOTICES.txt not found."

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(content)
        layout.addWidget(text_edit)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok)
        btn_box.accepted.connect(dlg.accept)
        layout.addWidget(btn_box)
        return dlg

    def _show_third_party_notices(self):
        dlg = self._create_third_party_notices_dialog()
        dlg.exec()

    def _create_how_to_compare_dialog(self) -> QDialog:
        dlg = QDialog(self)
        dlg.setWindowTitle("How to compare with EPANET")
        dlg.resize(650, 420)
        layout = QVBoxLayout(dlg)

        title_label = QLabel("<h3>How to compare with EPANET</h3>")
        layout.addWidget(title_label)

        steps_text = (
            "<ol>"
            "<li><b>Build the network and fill all parameters:</b> Ensure all junctions, reservoirs, and pipes have their physical parameters (elevation, head, length, diameter, roughness) set.</li>"
            "<li><b>Open the tab \"Comparison with EPANET\":</b> Switch to the Comparison tab in the right-hand panel.</li>"
            "<li><b>Tick the valves:</b> In the Valves table, select which pipes contain valves at their start or end.</li>"
            "<li><b>Choose or build the switching scenario:</b> Choose a preset specimen scenario (1 to 5) or specify initial valve states and switching times.</li>"
            "<li><b>Press Run:</b> Click the Run button to simulate both the switched DAE model and the EPANET engine.</li>"
            "<li><b>Read plots/tables and export:</b> Inspect the dynamic flows, pressure heads, equilibrium differences, and Dirac impulses in the plots and summary table. Use the Export buttons to save PNG plots, CSV data, or EPANET .inp files.</li>"
            "</ol>"
        )
        label = QLabel(steps_text)
        label.setWordWrap(True)
        layout.addWidget(label)

        layout.addStretch()
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok)
        btn_box.accepted.connect(dlg.accept)
        layout.addWidget(btn_box)
        return dlg

    def _show_how_to_compare(self):
        dlg = self._create_how_to_compare_dialog()
        dlg.exec()

