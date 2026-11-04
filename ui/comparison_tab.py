"""ui/comparison_tab.py – Tab for DAE vs EPANET comparison, valve setup, and scenario editing.

Provides:
- Live ready/problems box
- Valve configuration table per pipe
- Scenario builder with MATLAB specimen presets (1..5) and custom events
- Physics settings (Constant, Swamee-Jain, Darcy-Weisbach)
- Background execution via QThread
- Results tables: Parameters, Equilibrium differences, Dirac impulses
- Outdated results banner
- Empty 'Plots appear here' placeholder panel
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple, Any

from PySide6.QtCore import Qt, QThread, Signal, QObject, QSize, QEvent
from PySide6.QtGui import QColor, QFont, QKeySequence, QValidator, QPalette
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QSplitter,
    QScrollArea,
    QTabWidget,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QProgressBar,
    QFrame,
    QSizePolicy,
)

from core.models import Network, Pipe
from core.scenario import Scenario, SwitchEvent, ValveId
from core.physics import PhysicsSettings
from core.comparison import (
    problems,
    parameters_table,
    run_comparison,
    ComparisonResult,
    ComparisonError,
)
from core.plotdata import (
    default_states,
    series,
    difference_rows,
    max_abs_difference_after_last_switch,
)
from ui.comparison_plots import ComparisonPlots


class CompactComboBox(QComboBox):
    """QComboBox whose size hints do not expand to the longest item, preventing container blowout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
        if self.view() is not None:
            self.view().setTextElideMode(Qt.ElideRight)

    def minimumSizeHint(self):
        return QSize(80, 24)

    def sizeHint(self):
        return QSize(160, 24)



def _clear_table_cell_widgets(table: QTableWidget):
    """Cleanly remove and delete all cell widgets in the table."""
    for r in range(table.rowCount()):
        for c in range(table.columnCount()):
            w = table.cellWidget(r, c)
            if w is not None:
                table.removeCellWidget(r, c)
                w.deleteLater()
                w.setParent(None)


class CopyableTableWidget(QTableWidget):
    """QTableWidget with alternating row colors, Ctrl+C clipboard copy, and empty hint text."""

    def __init__(self, parent=None, placeholder_text: str = ""):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self._placeholder_text: str = placeholder_text

    def setPlaceholderText(self, text: str):
        self._placeholder_text = text
        self.viewport().update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.rowCount() == 0 and self._placeholder_text:
            from PySide6.QtGui import QPainter, QColor
            painter = QPainter(self.viewport())
            painter.setPen(QColor("#777777"))
            font = painter.font()
            font.setItalic(True)
            painter.setFont(font)
            rect = self.viewport().rect().adjusted(15, 30, -15, -30)
            painter.drawText(
                rect,
                Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,
                self._placeholder_text,
            )
            painter.end()

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            self.copy_selection_to_clipboard()
            event.accept()
        else:
            super().keyPressEvent(event)

    def copy_selection_to_clipboard(self):
        selected_ranges = self.selectedRanges()
        if not selected_ranges:
            r_min, r_max = 0, self.rowCount() - 1
            c_min, c_max = 0, self.columnCount() - 1
        else:
            r_min = min(rg.topRow() for rg in selected_ranges)
            r_max = max(rg.bottomRow() for rg in selected_ranges)
            c_min = min(rg.leftColumn() for rg in selected_ranges)
            c_max = max(rg.rightColumn() for rg in selected_ranges)

        if r_max < r_min or c_max < c_min:
            return

        headers = []
        for c in range(c_min, c_max + 1):
            h_item = self.horizontalHeaderItem(c)
            headers.append(h_item.text() if h_item else f"Col {c}")
        lines = ["\t".join(headers)]

        for r in range(r_min, r_max + 1):
            row_vals = []
            for c in range(c_min, c_max + 1):
                it = self.item(r, c)
                row_vals.append(it.text() if it else "")
            lines.append("\t".join(row_vals))

        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText("\n".join(lines))


class ScientificDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox displaying values in scientific notation (e.g. 1.0219e-06)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDecimals(10)

    def textFromValue(self, val: float) -> str:
        return f"{val:.4e}"

    def valueFromText(self, text: str) -> float:
        try:
            return float(text)
        except Exception:
            return 0.0

    def validate(self, text: str, pos: int):
        try:
            float(text)
            return (QValidator.Acceptable, text, pos)
        except Exception:
            return (QValidator.Intermediate, text, pos)


def format_table(table: QTableWidget):
    """Ensure columns are sized to contents, headers fully visible, no '...'."""
    table.resizeColumnsToContents()
    fm = table.fontMetrics()
    for c in range(table.columnCount()):
        h_item = table.horizontalHeaderItem(c)
        if h_item:
            needed_w = fm.horizontalAdvance(h_item.text()) + 28
            if table.columnWidth(c) < needed_w:
                table.setColumnWidth(c, needed_w)


class ComparisonWorker(QObject):
    """Background worker for running DAE and EPANET simulations."""

    finished = Signal(object)  # ComparisonResult
    error = Signal(str)

    def __init__(
        self,
        network: Network,
        scenario: Scenario,
        settings: PhysicsSettings,
        positions: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        super().__init__()
        self.network = copy.deepcopy(network)
        self.scenario = copy.deepcopy(scenario)
        self.settings = copy.deepcopy(settings)
        self.positions = dict(positions) if positions else None

    def run(self):
        try:
            res = run_comparison(
                self.network,
                self.scenario,
                self.settings,
                positions=self.positions,
            )
            self.finished.emit(res)
        except ComparisonError as ce:
            self.error.emit("\n".join(ce.problems))
        except Exception as e:
            self.error.emit(f"Comparison run failed: {e}")


class ComparisonTab(QWidget):
    """Comparison with EPANET tab."""

    network_modified = Signal()

    def __init__(self, network: Network, get_positions_cb=None, parent=None):
        super().__init__(parent)
        self.network = network
        self.get_positions_cb = get_positions_cb

        self._scenario: Optional[Scenario] = None
        self._physics: PhysicsSettings = PhysicsSettings()
        self._last_result: Optional[ComparisonResult] = None

        self._thread: Optional[QThread] = None
        self._worker: Optional[ComparisonWorker] = None
        self._updating_ui: bool = False

        self._init_ui()
        self.update_theme_style()
        self.refresh()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)

        self.splitter = QSplitter(Qt.Horizontal)
        splitter = self.splitter
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, False)
        self._saved_setup_width = 450
        splitter.splitterMoved.connect(self._on_splitter_moved)

        # -------------------------------------------------------------
        # LEFT COLUMN (Setup, in QScrollArea)
        # -------------------------------------------------------------
        self.left_scroll = QScrollArea()
        left_scroll = self.left_scroll
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(0)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # 1. Ready? box
        self.ready_group = QGroupBox("Ready?")
        self.ready_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        ready_layout = QVBoxLayout(self.ready_group)
        self.ready_label = QLabel()
        self.ready_label.setWordWrap(True)
        self.ready_label.setTextFormat(Qt.RichText)
        self.ready_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        ready_layout.addWidget(self.ready_label)
        left_layout.addWidget(self.ready_group)

        # 2. Valves table
        valves_group = QGroupBox("Valves")
        valves_layout = QVBoxLayout(valves_group)
        valve_desc = QLabel(
            "Valves at pipe boundaries switch the pipe between open (flow allowed) and closed (zero flow) states."
        )
        valve_desc.setWordWrap(True)
        valve_desc.setStyleSheet("color: #555555; font-style: italic;")
        valves_layout.addWidget(valve_desc)

        self.valves_table = QTableWidget()
        self.valves_table.setColumnCount(4)
        self.valves_table.setHorizontalHeaderLabels(["Pipe", "From -> To", "Valve at Start", "Valve at End"])
        self.valves_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.valves_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.valves_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.valves_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.valves_table.setMinimumHeight(110)
        self.valves_table.cellChanged.connect(self._on_valve_cell_changed)
        valves_layout.addWidget(self.valves_table)

        self.valve_notice_label = QLabel()
        self.valve_notice_label.setStyleSheet("color: #D9534F; font-size: 11px;")
        self.valve_notice_label.setWordWrap(True)
        self.valve_notice_label.setVisible(False)
        valves_layout.addWidget(self.valve_notice_label)

        left_layout.addWidget(valves_group)

        # 3. Scenario builder
        scen_group = QGroupBox("Scenario")
        scen_layout = QVBoxLayout(scen_group)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_combo = CompactComboBox()
        self.preset_combo.addItem("Custom")
        self.preset_combo.addItem("MATLAB specimen 1: valve 1 closed at t_s = 60 s (started from rest)")
        self.preset_combo.addItem("MATLAB specimen 2: no switching: relaxation from rest onto EPANET's steady state")
        self.preset_combo.addItem("MATLAB specimen 3: equilibrium first, then valve 1 closes at t_s = 300 s")
        self.preset_combo.addItem("MATLAB specimen 4: equilibrium first, then valve 2 closes at t_s = 300 s")
        self.preset_combo.addItem("MATLAB specimen 5: valve 1 opened at t_s = 60 s (start with valve 1 closed)")
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.preset_combo, 1)
        scen_layout.addLayout(preset_row)

        self.preset_mapping_label = QLabel("V1 = None, V2 = None")
        self.preset_mapping_label.setStyleSheet("color: #337AB7; font-weight: bold; font-size: 11px;")
        self.preset_mapping_label.setWordWrap(True)
        scen_layout.addWidget(self.preset_mapping_label)

        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Title:"))
        self.title_edit = QLineEdit("Custom Scenario")
        self.title_edit.textChanged.connect(self._on_scenario_edited)
        title_row.addWidget(self.title_edit, 1)
        scen_layout.addLayout(title_row)

        scen_layout.addWidget(QLabel("Initial valve states (t = 0):"))
        self.initial_table = QTableWidget()
        self.initial_table.setColumnCount(2)
        self.initial_table.setHorizontalHeaderLabels(["Valve", "Initial State"])
        self.initial_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.initial_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.initial_table.setMaximumHeight(100)
        scen_layout.addWidget(self.initial_table)

        scen_layout.addWidget(QLabel("Switching events:"))
        self.events_table = QTableWidget()
        self.events_table.setColumnCount(3)
        self.events_table.setHorizontalHeaderLabels(["Time (s)", "Valve", "New State"])
        self.events_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.events_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.events_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.events_table.setMinimumHeight(110)
        scen_layout.addWidget(self.events_table)

        ev_btn_row = QHBoxLayout()
        self.add_event_btn = QPushButton("Add Event")
        self.add_event_btn.clicked.connect(self._on_add_event)
        self.remove_event_btn = QPushButton("Remove Event")
        self.remove_event_btn.clicked.connect(self._on_remove_event)
        ev_btn_row.addWidget(self.add_event_btn)
        ev_btn_row.addWidget(self.remove_event_btn)
        ev_btn_row.addStretch()
        scen_layout.addLayout(ev_btn_row)

        end_time_row = QHBoxLayout()
        end_time_row.addWidget(QLabel("End time (s):"))
        self.end_time_spin = QSpinBox()
        self.end_time_spin.setRange(1, 86400)
        self.end_time_spin.setValue(120)
        self.end_time_spin.valueChanged.connect(self._on_scenario_edited)
        end_time_row.addWidget(self.end_time_spin)
        end_time_row.addStretch()
        scen_layout.addLayout(end_time_row)

        left_layout.addWidget(scen_group)

        # 4. Physics settings
        phys_group = QGroupBox("Physics")
        phys_layout = QVBoxLayout(phys_group)

        fric_row = QHBoxLayout()
        fric_row.addWidget(QLabel("Friction law:"))
        self.friction_combo = CompactComboBox()
        self.friction_combo.addItems([
            "Constant friction factor",
            "Swamee-Jain",
            "Darcy-Weisbach as in EPANET",
        ])
        self.friction_combo.currentIndexChanged.connect(self._on_friction_mode_changed)
        fric_row.addWidget(self.friction_combo, 1)
        phys_layout.addLayout(fric_row)

        self.friction_desc_label = QLabel("Fixed friction factor f across all flow velocities.")
        self.friction_desc_label.setStyleSheet("color: #666666; font-size: 11px; font-style: italic;")
        self.friction_desc_label.setWordWrap(True)
        phys_layout.addWidget(self.friction_desc_label)

        self.factor_row = QHBoxLayout()
        self.factor_label = QLabel("Friction factor f:")
        self.factor_spin = QDoubleSpinBox()
        self.factor_spin.setRange(0.0001, 1.0)
        self.factor_spin.setDecimals(4)
        self.factor_spin.setSingleStep(0.005)
        self.factor_spin.setValue(0.0200)
        self.factor_spin.valueChanged.connect(self._on_physics_edited)
        self.factor_row.addWidget(self.factor_label)
        self.factor_row.addWidget(self.factor_spin)
        self.factor_row.addStretch()
        phys_layout.addLayout(self.factor_row)

        # Advanced group (collapsible)
        self.adv_group = QGroupBox("Advanced Parameters")
        self.adv_group.setCheckable(True)
        self.adv_group.setChecked(False)
        adv_layout = QGridLayout(self.adv_group)

        adv_layout.addWidget(QLabel("Gravity g (m/s²):"), 0, 0)
        self.g_spin = QDoubleSpinBox()
        self.g_spin.setRange(1.0, 30.0)
        self.g_spin.setDecimals(6)
        self.g_spin.setValue(32.2 * 0.3048)
        self.g_spin.valueChanged.connect(self._on_physics_edited)
        adv_layout.addWidget(self.g_spin, 0, 1)

        adv_layout.addWidget(QLabel("Viscosity ν (m²/s):"), 1, 0)
        self.nu_spin = ScientificDoubleSpinBox()
        self.nu_spin.setRange(1e-8, 1e-3)
        self.nu_spin.setValue(1.0219e-06)
        self.nu_spin.valueChanged.connect(self._on_physics_edited)
        adv_layout.addWidget(self.nu_spin, 1, 1)

        phys_layout.addWidget(self.adv_group)
        left_layout.addWidget(phys_group)

        # 5. Run button & progress
        run_box = QVBoxLayout()
        self.run_btn = QPushButton("Run comparison")
        self.run_btn.setFixedHeight(34)
        self.run_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #0078D7; color: white; border-radius: 4px; }"
            "QPushButton:disabled { background-color: #CCCCCC; color: #666666; }"
        )
        self.run_btn.clicked.connect(self.run_comparison_action)
        run_box.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        run_box.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        run_box.addWidget(self.status_label)

        left_layout.addLayout(run_box)
        left_layout.addStretch()

        left_scroll.setWidget(left_widget)
        left_scroll.setMinimumWidth(440)
        left_scroll.setMaximumWidth(460)
        splitter.addWidget(left_scroll)

        # -------------------------------------------------------------
        # RIGHT RESULTS AREA (Splitter right child)
        # -------------------------------------------------------------
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)

        # 0. Hide / Show setup toggle button
        toggle_bar = QHBoxLayout()
        toggle_bar.setContentsMargins(0, 0, 0, 0)
        self.toggle_setup_btn = QPushButton("Hide setup")
        self.toggle_setup_btn.setToolTip("Collapse or expand the left setup column")
        self.toggle_setup_btn.clicked.connect(self._toggle_setup)
        toggle_bar.addWidget(self.toggle_setup_btn)
        toggle_bar.addStretch()
        right_layout.addLayout(toggle_bar)

        # 1. Outdated banner (full width, wraps text, never cut)
        self.outdated_banner = QFrame()
        self.outdated_banner.setStyleSheet(
            "background-color: #FFF3CD; border: 1px solid #FFEEBA; border-radius: 4px; padding: 6px;"
        )
        outdated_layout = QHBoxLayout(self.outdated_banner)
        outdated_layout.setContentsMargins(8, 4, 8, 4)
        self.outdated_label = QLabel("⚠️ Network or settings changed since this result - run again")
        self.outdated_label.setStyleSheet("color: #856404; font-weight: bold;")
        self.outdated_label.setWordWrap(True)
        outdated_layout.addWidget(self.outdated_label)
        self.outdated_banner.setVisible(False)
        right_layout.addWidget(self.outdated_banner)

        # 2. Results Tabs
        self.results_tabs = QTabWidget()

        # Tab 1: Plots
        self.plots_widget = ComparisonPlots()
        self.plots_widget.status_message.connect(self._on_plot_status_message)
        self.results_tabs.addTab(self.plots_widget, "Plots")

        # Tab 2: Differences
        diff_widget = QWidget()
        diff_layout = QVBoxLayout(diff_widget)
        diff_layout.setContentsMargins(6, 6, 6, 6)
        diff_layout.setSpacing(6)
        self.diff_table = CopyableTableWidget(
            placeholder_text="No results yet. Fill in the network, choose the valves and scenario on the left, then press Run."
        )
        self.diff_table.setColumnCount(5)
        self.diff_table.setHorizontalHeaderLabels([
            "Time (s)",
            "State Label",
            "DAE",
            "EPANET",
            "DAE - EPANET",
        ])
        diff_layout.addWidget(self.diff_table, 1)
        self.diff_summary_label = QLabel("No results yet. Run comparison to view point-by-point differences.")
        self.diff_summary_label.setStyleSheet("color: #666666; font-style: italic;")
        diff_layout.addWidget(self.diff_summary_label)
        self.results_tabs.addTab(diff_widget, "Differences")

        # Tab 3: Equilibrium
        eq_widget = QWidget()
        eq_layout = QVBoxLayout(eq_widget)
        eq_layout.setContentsMargins(6, 6, 6, 6)
        eq_layout.setSpacing(6)
        self.eq_table = CopyableTableWidget(
            placeholder_text="No results yet. Fill in the network, choose the valves and scenario on the left, then press Run."
        )
        self.eq_table.setColumnCount(8)
        self.eq_table.setHorizontalHeaderLabels([
            "Interval",
            "Mode",
            "State Label",
            "DAE End",
            "DAE Eq.",
            "EPANET",
            "Diff (End vs EP)",
            "Diff (Eq vs EP)",
        ])
        eq_layout.addWidget(self.eq_table, 1)
        self.eq_summary_label = QLabel("Run comparison to see equilibrium differences.")
        self.eq_summary_label.setStyleSheet("color: #666666; font-style: italic;")
        eq_layout.addWidget(self.eq_summary_label)
        self.results_tabs.addTab(eq_widget, "Equilibrium")

        # Tab 4: Dirac impulses
        imp_widget = QWidget()
        imp_layout = QVBoxLayout(imp_widget)
        imp_layout.setContentsMargins(6, 6, 6, 6)
        imp_layout.setSpacing(6)
        self.imp_table = CopyableTableWidget(
            placeholder_text="No results yet. Fill in the network, choose the valves and scenario on the left, then press Run."
        )
        self.imp_table.setColumnCount(4)
        self.imp_table.setHorizontalHeaderLabels([
            "Time (s)",
            "State Label",
            "Coefficient (m*s)",
            "EPANET",
        ])
        imp_layout.addWidget(self.imp_table, 1)
        self.imp_summary_label = QLabel("EPANET: none (EPANET has no impulses).")
        self.imp_summary_label.setStyleSheet("color: #666666; font-style: italic;")
        imp_layout.addWidget(self.imp_summary_label)
        self.results_tabs.addTab(imp_widget, "Dirac impulses")

        # Tab 5: Parameters used
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        params_layout.setContentsMargins(6, 6, 6, 6)
        params_layout.setSpacing(6)
        self.params_note = QLabel("No results yet. Parameters will be recorded when comparison is run.")
        self.params_note.setStyleSheet("color: #666666; font-style: italic;")
        params_layout.addWidget(self.params_note)
        self.params_table = CopyableTableWidget(
            placeholder_text="No results yet. Fill in the network, choose the valves and scenario on the left, then press Run."
        )
        self.params_table.setColumnCount(4)
        self.params_table.setHorizontalHeaderLabels(["Element", "Parameter", "Value", "Unit"])
        params_layout.addWidget(self.params_table, 1)
        self.results_tabs.addTab(params_widget, "Parameters used")

        right_layout.addWidget(self.results_tabs, 1)
        splitter.addWidget(right_container)

        splitter.setSizes([450, 2000])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

    def _toggle_setup(self):
        """Toggle left setup column between collapsed (0) and expanded."""
        sizes = self.splitter.sizes()
        if sizes[0] > 0:
            self._saved_setup_width = max(440, min(460, sizes[0]))
            self.left_scroll.setMinimumWidth(0)
            self.left_scroll.setMaximumWidth(0)
            self.splitter.setSizes([0, sizes[0] + sizes[1]])
            self.toggle_setup_btn.setText("Show setup")
        else:
            restore_w = getattr(self, "_saved_setup_width", 450)
            self.left_scroll.setMinimumWidth(440)
            self.left_scroll.setMaximumWidth(460)
            self.splitter.setSizes([restore_w, max(200, sizes[1] - restore_w)])
            self.toggle_setup_btn.setText("Hide setup")

    def _on_splitter_moved(self, pos: int, index: int):
        """Keep toggle button text in sync when user drags the splitter handle."""
        sizes = self.splitter.sizes()
        if sizes[0] == 0:
            self.toggle_setup_btn.setText("Show setup")
            self.left_scroll.setMinimumWidth(0)
            self.left_scroll.setMaximumWidth(0)
        else:
            self._saved_setup_width = max(440, min(460, sizes[0]))
            self.left_scroll.setMinimumWidth(440)
            self.left_scroll.setMaximumWidth(460)
            self.toggle_setup_btn.setText("Hide setup")

    # -----------------------------------------------------------------
    # Update / Refresh logic
    # -----------------------------------------------------------------

    def clear_results(self):
        """Reset every result widget (tables, captions, summary lines, plots, banner)."""
        self._last_result = None
        self.outdated_banner.setVisible(False)
        self.eq_table.setRowCount(0)
        self.imp_table.setRowCount(0)
        self.params_table.setRowCount(0)
        if hasattr(self, "diff_table"):
            self.diff_table.setRowCount(0)
        if hasattr(self, "diff_summary_label"):
            self.diff_summary_label.setText("No results yet. Run comparison to view point-by-point differences.")
            self.diff_summary_label.setStyleSheet("color: #666666; font-style: italic;")
        self.eq_summary_label.setText("Run comparison to see equilibrium differences.")
        self.eq_summary_label.setStyleSheet("color: #666666; font-style: italic;")
        self.imp_summary_label.setText("EPANET: none (EPANET has no impulses).")
        self.imp_summary_label.setStyleSheet("color: #666666; font-style: italic;")
        if hasattr(self, "params_note"):
            self.params_note.setText("No results yet. Parameters will be recorded when comparison is run.")
        if hasattr(self, "plots_widget"):
            self.plots_widget.clear()
        self.status_label.setText("Ready")

    def set_network(self, network: Network):
        """Update network reference, clear results and refresh."""
        self.network = network
        self.clear_results()
        self.refresh()

    def update_tab(self, network: Network):
        """Called by MainWindow on refresh()."""
        self.network = network
        self.refresh()

    def refresh(self):
        """Re-render UI components based on active network and scenario."""
        if self._updating_ui:
            return
        self._updating_ui = True
        try:
            valves = self._get_network_valves()

            # 1. Update valves table
            self._update_valves_table()

            # 2. Update scenario UI
            self._update_scenario_ui(valves)

            # 3. Check problems & update ready box
            current_scen = self.get_scenario()
            current_phys = self.get_physics()
            probs = problems(self.network, current_scen, current_phys)

            if probs:
                html = (
                    "<div style='color: #721C24; background-color: #F8D7DA; padding: 6px; border-radius: 4px; "
                    "word-wrap: break-word; overflow-wrap: break-word;'>"
                    "<b>Not ready to compare:</b>"
                    "<ul style='margin: 4px 0 0 16px; padding: 0; word-wrap: break-word;'>"
                )
                for p in probs:
                    html += f"<li style='margin-bottom: 2px; word-wrap: break-word;'>{p}</li>"
                html += "</ul></div>"
                self.ready_label.setText(html)
                self.run_btn.setEnabled(False)
            else:
                html = (
                    "<div style='color: #155724; background-color: #D4EDDA; padding: 6px; border-radius: 4px; "
                    "word-wrap: break-word; overflow-wrap: break-word;'>"
                    "<b>Ready to compare</b>"
                    "</div>"
                )
                self.ready_label.setText(html)
                self.run_btn.setEnabled(self._thread is None)

            # 4. Outdated banner check
            if self._last_result is not None:
                is_out = self._last_result.is_outdated(self.network, current_scen, current_phys)
                self.outdated_banner.setVisible(is_out)
            else:
                self.outdated_banner.setVisible(False)

        finally:
            self._updating_ui = False

    def _get_network_valves(self) -> List[ValveId]:
        """Return list of ValveId objects in the network in pipe order (start before end)."""
        valves: List[ValveId] = []
        for p in self.network.pipes:
            if p.valve_start:
                valves.append(ValveId(p.id, "start"))
            if p.valve_end:
                valves.append(ValveId(p.id, "end"))
        return valves

    def _update_valves_table(self):
        """Populate valves table with one row per pipe."""
        self.valves_table.blockSignals(True)
        try:
            pipes = list(self.network.pipes)
            self.valves_table.setRowCount(len(pipes))

            for row, p in enumerate(pipes):
                # 1. Pipe ID
                it_id = QTableWidgetItem(p.id)
                it_id.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.valves_table.setItem(row, 0, it_id)

                # 2. Endpoints
                it_nodes = QTableWidgetItem(f"{p.start} -> {p.end}")
                it_nodes.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.valves_table.setItem(row, 1, it_nodes)

                # 3. Start valve checkbox
                it_start = QTableWidgetItem()
                it_start.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                it_start.setCheckState(Qt.Checked if p.valve_start else Qt.Unchecked)
                if p.has_pump:
                    it_start.setFlags(Qt.NoItemFlags)
                    it_start.setToolTip("Valves cannot be placed on pumped pipes.")
                self.valves_table.setItem(row, 2, it_start)

                # 4. End valve checkbox
                it_end = QTableWidgetItem()
                it_end.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                it_end.setCheckState(Qt.Checked if p.valve_end else Qt.Unchecked)
                if p.has_pump:
                    it_end.setFlags(Qt.NoItemFlags)
                    it_end.setToolTip("Valves cannot be placed on pumped pipes.")
                self.valves_table.setItem(row, 3, it_end)
        finally:
            self.valves_table.blockSignals(False)

    def _on_valve_cell_changed(self, row: int, col: int):
        """Handle valve toggle from the table."""
        if self._updating_ui:
            return
        pipes = list(self.network.pipes)
        if row >= len(pipes) or col not in (2, 3):
            return

        p = pipes[row]
        it = self.valves_table.item(row, col)
        is_checked = (it.checkState() == Qt.Checked)

        valve_end_type = "start" if col == 2 else "end"
        valve_id = ValveId(p.id, valve_end_type)

        if col == 2:
            self.network.update_pipe(p.id, valve_start=is_checked)
        else:
            self.network.update_pipe(p.id, valve_end=is_checked)

        # If a valve was removed, purge from scenario
        if not is_checked and self._scenario:
            removed = False
            if valve_id in self._scenario.initial:
                del self._scenario.initial[valve_id]
                removed = True
            new_events = [e for e in self._scenario.events if e.valve != valve_id]
            if len(new_events) != len(self._scenario.events):
                self._scenario.events = new_events
                removed = True
            if removed:
                self.valve_notice_label.setText(f"Removed deleted valve '{valve_id}' from scenario.")
                self.valve_notice_label.setVisible(True)

        self.network_modified.emit()
        self.refresh()

    def _update_scenario_ui(self, valves: List[ValveId]):
        """Synchronize scenario editor with active valves and scenario state."""
        has_two_valves = len(valves) >= 2
        self.preset_combo.setEnabled(has_two_valves)

        if has_two_valves:
            v1, v2 = valves[0], valves[1]
            self.preset_mapping_label.setText(f"V1 = {v1}, V2 = {v2}")
        else:
            self.preset_mapping_label.setText("V1 = None, V2 = None (Need ≥2 valves for presets)")

        # Ensure initial states exist for all network valves if we have a scenario
        if self._scenario is None:
            if valves:
                self._scenario = Scenario(
                    title="Default Scenario",
                    initial={v: 1 for v in valves},
                    events=[],
                    t_end=120.0,
                )

        if self._scenario is not None:
            # Sync initial dict to ensure only valid valves exist
            valve_set = set(valves)
            for v in valves:
                if v not in self._scenario.initial:
                    self._scenario.initial[v] = 1
            # Remove valves that no longer exist
            for v in list(self._scenario.initial.keys()):
                if v not in valve_set:
                    del self._scenario.initial[v]
            self._scenario.events = [e for e in self._scenario.events if e.valve in valve_set]

            self.title_edit.setText(self._scenario.title)
            self.end_time_spin.setValue(int(round(self._scenario.t_end)))

            # Populate initial states table
            _clear_table_cell_widgets(self.initial_table)
            self.initial_table.setRowCount(len(valves))
            for r, v in enumerate(valves):
                it_v = QTableWidgetItem(str(v))
                it_v.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.initial_table.setItem(r, 0, it_v)
                self.initial_table.removeCellWidget(r, 0)

                cb = QComboBox()
                cb.addItems(["Open", "Closed"])
                st = self._scenario.initial.get(v, 1)
                cb.setCurrentIndex(0 if st == 1 else 1)
                cb.currentIndexChanged.connect(self._on_initial_state_combo_changed)
                self.initial_table.setCellWidget(r, 1, cb)

            # Populate events table
            self._rebuild_events_table(valves)
        else:
            _clear_table_cell_widgets(self.initial_table)
            self.initial_table.setRowCount(0)
            _clear_table_cell_widgets(self.events_table)
            self.events_table.setRowCount(0)

    def _rebuild_events_table(self, valves: List[ValveId]):
        """Populate switching events table."""
        _clear_table_cell_widgets(self.events_table)
        if not self._scenario:
            self.events_table.setRowCount(0)
            return

        self.events_table.setRowCount(len(self._scenario.events))
        valve_strs = [str(v) for v in valves]

        for r, ev in enumerate(self._scenario.events):
            # Time spin box
            sp = QSpinBox()
            sp.setRange(0, int(self.end_time_spin.value()))
            sp.setValue(int(round(ev.time)))
            sp.valueChanged.connect(self._on_events_table_edited)
            self.events_table.setCellWidget(r, 0, sp)

            # Valve combo
            v_cb = QComboBox()
            v_cb.addItems(valve_strs)
            ev_v_str = str(ev.valve)
            if ev_v_str in valve_strs:
                v_cb.setCurrentText(ev_v_str)
            v_cb.currentIndexChanged.connect(self._on_events_table_edited)
            self.events_table.setCellWidget(r, 1, v_cb)

            # State combo
            st_cb = QComboBox()
            st_cb.addItems(["Open", "Closed"])
            st_cb.setCurrentIndex(0 if ev.state == 1 else 1)
            st_cb.currentIndexChanged.connect(self._on_events_table_edited)
            self.events_table.setCellWidget(r, 2, st_cb)

    def _on_initial_state_combo_changed(self):
        """Update scenario initial states when user selects Open/Closed."""
        if self._updating_ui or not self._scenario:
            return
        for r in range(self.initial_table.rowCount()):
            it_v = self.initial_table.item(r, 0)
            cb = self.initial_table.cellWidget(r, 1)
            if it_v and isinstance(cb, QComboBox):
                vid = ValveId.from_str(it_v.text())
                self._scenario.initial[vid] = 1 if cb.currentIndex() == 0 else 0
        self.preset_combo.setCurrentIndex(0)  # Custom
        self.network_modified.emit()
        self.refresh()

    def _on_events_table_edited(self):
        """Update scenario events when table spin/combo changes."""
        if self._updating_ui or not self._scenario:
            return
        new_events: List[SwitchEvent] = []
        for r in range(self.events_table.rowCount()):
            sp = self.events_table.cellWidget(r, 0)
            v_cb = self.events_table.cellWidget(r, 1)
            st_cb = self.events_table.cellWidget(r, 2)
            if isinstance(sp, QSpinBox) and isinstance(v_cb, QComboBox) and isinstance(st_cb, QComboBox):
                if v_cb.currentText():
                    vid = ValveId.from_str(v_cb.currentText())
                    st = 1 if st_cb.currentIndex() == 0 else 0
                    new_events.append(SwitchEvent(time=float(sp.value()), valve=vid, state=st))
        self._scenario.events = new_events
        self.preset_combo.setCurrentIndex(0)  # Custom
        self.network_modified.emit()
        self.refresh()

    def _on_add_event(self):
        """Add a new default switching event."""
        valves = self._get_network_valves()
        if not valves:
            return
        if self._scenario is None:
            self._scenario = Scenario(title="Custom Scenario", initial={v: 1 for v in valves}, events=[])
        t_sw = min(30.0, max(1.0, float(self.end_time_spin.value()) / 2.0))
        self._scenario.events.append(SwitchEvent(time=t_sw, valve=valves[0], state=0))
        self.preset_combo.setCurrentIndex(0)
        self.network_modified.emit()
        self.refresh()

    def _on_remove_event(self):
        """Remove selected or last event."""
        if not self._scenario or not self._scenario.events:
            return
        sel = self.events_table.currentRow()
        if 0 <= sel < len(self._scenario.events):
            self._scenario.events.pop(sel)
        else:
            self._scenario.events.pop()
        self.preset_combo.setCurrentIndex(0)
        self.network_modified.emit()
        self.refresh()

    def _on_preset_changed(self, idx: int):
        """Apply MATLAB specimen preset 1..5."""
        if self._updating_ui:
            return
        if idx == 0:
            return  # Custom
        valves = self._get_network_valves()
        if len(valves) < 2:
            return
        v1, v2 = valves[0], valves[1]
        try:
            self._scenario = Scenario.matlab_specimen(idx, v1, v2)
            self.network_modified.emit()
            self.refresh()
        except Exception as e:
            self.status_label.setText(f"Preset error: {e}")

    def _on_scenario_edited(self):
        """Handle title or end time edits."""
        if self._updating_ui or not self._scenario:
            return
        self._scenario.title = self.title_edit.text()
        self._scenario.t_end = float(self.end_time_spin.value())
        self.network_modified.emit()
        self.refresh()

    # -----------------------------------------------------------------
    # Physics settings
    # -----------------------------------------------------------------

    def _on_friction_mode_changed(self, idx: int):
        if idx == 0:
            mode = "constant"
            desc = "Constant Darcy friction factor f across all flow velocities."
            self.factor_label.setVisible(True)
            self.factor_spin.setVisible(True)
        elif idx == 1:
            mode = "swamee_jain"
            desc = "Explicit turbulent approximation based on relative roughness and Reynolds number."
            self.factor_label.setVisible(False)
            self.factor_spin.setVisible(False)
        else:
            mode = "darcy_weisbach"
            desc = "EPANET hydraulic formulation using laminar, transition, and Swamee-Jain regimes."
            self.factor_label.setVisible(False)
            self.factor_spin.setVisible(False)

        self.friction_desc_label.setText(desc)
        self._physics.friction_mode = mode
        self.network_modified.emit()
        self.refresh()

    def _on_physics_edited(self):
        if self._updating_ui:
            return
        self._physics.friction_factor = float(self.factor_spin.value())
        self._physics.g = float(self.g_spin.value())
        self._physics.nu = float(self.nu_spin.value())
        self.network_modified.emit()
        self.refresh()

    def get_physics(self) -> PhysicsSettings:
        """Return active PhysicsSettings."""
        return copy.deepcopy(self._physics)

    def set_physics(self, phys: Optional[PhysicsSettings]):
        """Load PhysicsSettings into the UI."""
        if phys is None:
            phys = PhysicsSettings()
        self._physics = copy.deepcopy(phys)
        self._updating_ui = True
        try:
            if phys.friction_mode == "constant":
                self.friction_combo.setCurrentIndex(0)
                self.factor_label.setVisible(True)
                self.factor_spin.setVisible(True)
            elif phys.friction_mode == "swamee_jain":
                self.friction_combo.setCurrentIndex(1)
                self.factor_label.setVisible(False)
                self.factor_spin.setVisible(False)
            else:
                self.friction_combo.setCurrentIndex(2)
                self.factor_label.setVisible(False)
                self.factor_spin.setVisible(False)

            self.factor_spin.setValue(float(phys.friction_factor))
            self.g_spin.setValue(float(phys.g))
            self.nu_spin.setValue(float(phys.nu))
        finally:
            self._updating_ui = False
        self.refresh()

    def get_scenario(self) -> Optional[Scenario]:
        """Return active Scenario object."""
        if self._scenario is None:
            return None
        return copy.deepcopy(self._scenario)

    def set_scenario(self, scen: Optional[Scenario]):
        """Load Scenario into the UI."""
        self._scenario = copy.deepcopy(scen) if scen else None
        idx = 0
        if self._scenario:
            valves = self._get_network_valves()
            if len(valves) >= 2:
                for p_idx in range(1, 6):
                    try:
                        spec = Scenario.matlab_specimen(p_idx, valves[0], valves[1])
                        if spec.title == self._scenario.title:
                            idx = p_idx
                            break
                    except Exception:
                        pass
        self._updating_ui = True
        try:
            self.preset_combo.setCurrentIndex(idx)
        finally:
            self._updating_ui = False
        self.refresh()

    def reset(self):
        """Clear results and reset scenario and physics."""
        self._scenario = None
        self._physics = PhysicsSettings()
        self.clear_results()
        self.valve_notice_label.setVisible(False)
        self.refresh()

    # -----------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        """True while comparison worker is executing in background."""
        return getattr(self, "_is_running", False)

    def run_comparison_action(self):
        """Trigger asynchronous comparison run in background thread."""
        scen = self.get_scenario()
        phys = self.get_physics()
        probs = problems(self.network, scen, phys)
        if probs:
            self.status_label.setStyleSheet("color: red;")
            self.status_label.setText("Cannot run: " + "; ".join(probs))
            return

        # Prepare UI
        self._is_running = True
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setStyleSheet("color: black;")
        self.status_label.setText("Running DAE and EPANET comparison...")

        positions = self.get_positions_cb() if self.get_positions_cb else None

        # Start worker thread
        self._thread = QThread()
        self._worker = ComparisonWorker(self.network, scen, phys, positions)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_run_finished)
        self._worker.error.connect(self._on_run_error)

        self._thread.start()

    def _on_run_finished(self, result: ComparisonResult):
        """Handle successful run completion."""
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        self._is_running = False
        self._last_result = result
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)

        t_dae = result.timings.get("dae", 0.0)
        t_ep = result.timings.get("epanet", 0.0)
        self.status_label.setStyleSheet("color: green;")
        self.status_label.setText(f"Comparison completed successfully (DAE: {t_dae:.3f}s, EPANET: {t_ep:.3f}s).")

        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        self._populate_results(result)
        QApplication.processEvents()

        self.results_tabs.setCurrentWidget(self.plots_widget)
        QApplication.processEvents()

        self.refresh()
        QApplication.processEvents()

    def _on_run_error(self, err_msg: str):
        """Handle error from comparison run."""
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        self._is_running = False
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)

        self.status_label.setStyleSheet("color: red;")
        self.status_label.setText(f"Comparison failed: {err_msg}")
        self.refresh()

    def _on_plot_status_message(self, msg: str):
        """Show plot status update in status label."""
        self.status_label.setStyleSheet("color: #0055A5;")
        self.status_label.setText(msg)

    def _populate_results(self, res: ComparisonResult):
        """Fill all results tables from ComparisonResult."""
        from PySide6.QtWidgets import QApplication

        # 1. Plots Widget
        self.plots_widget.set_result(res)
        QApplication.processEvents()

        tables = [self.diff_table, self.params_table, self.eq_table, self.imp_table]
        for tbl in tables:
            tbl.setUpdatesEnabled(False)
            tbl.blockSignals(True)

        try:
            # 2. Differences Table
            d_rows = difference_rows(res)
            self.diff_table.setRowCount(len(d_rows))
            for r, (t_val, s_lbl, dae_val, ep_val, diff_val) in enumerate(d_rows):
                it_t = QTableWidgetItem(f"{t_val:.6g}")
                it_t.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.diff_table.setItem(r, 0, it_t)

                self.diff_table.setItem(r, 1, QTableWidgetItem(s_lbl))

                it_dae = QTableWidgetItem(f"{dae_val:.6g}")
                it_dae.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.diff_table.setItem(r, 2, it_dae)

                it_ep = QTableWidgetItem(f"{ep_val:.6g}")
                it_ep.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.diff_table.setItem(r, 3, it_ep)

                it_diff = QTableWidgetItem(f"{diff_val:.6g}")
                it_diff.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.diff_table.setItem(r, 4, it_diff)

            max_d = max_abs_difference_after_last_switch(res)
            if res.scenario.events:
                self.diff_summary_label.setText(
                    f"After the last switch the largest difference is {max_d:.2e} (they coincide)."
                )
            else:
                self.diff_summary_label.setText(f"Largest difference across run: {max_d:.2e}")

            # 3. Parameters Table
            self.params_table.setRowCount(len(res.params))
            for r, (el, p_name, val, unit) in enumerate(res.params):
                self.params_table.setItem(r, 0, QTableWidgetItem(el))
                self.params_table.setItem(r, 1, QTableWidgetItem(p_name))
                self.params_table.setItem(r, 2, QTableWidgetItem(val))
                self.params_table.setItem(r, 3, QTableWidgetItem(unit))

            # 4. Equilibrium Comparison Table
            self.eq_table.setRowCount(len(res.equilibrium))
            intervals_summary: Dict[str, Dict[str, float]] = {}

            for r, row in enumerate(res.equilibrium):
                inv = row["interval"]
                if inv not in intervals_summary:
                    intervals_summary[inv] = {
                        "max_flow_end": 0.0,
                        "max_head_end": 0.0,
                        "max_flow_eq": 0.0,
                        "max_head_eq": 0.0,
                    }

                self.eq_table.setItem(r, 0, QTableWidgetItem(row["interval"]))
                self.eq_table.setItem(r, 1, QTableWidgetItem(row["mode"]))
                self.eq_table.setItem(r, 2, QTableWidgetItem(row["label"]))

                it_dae_end = QTableWidgetItem(f"{row.get('dae_end', row.get('dae', 0.0)):.6g}")
                it_dae_end.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.eq_table.setItem(r, 3, it_dae_end)

                it_dae_eq = QTableWidgetItem(f"{row.get('dae_eq', 0.0):.6g}")
                it_dae_eq.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.eq_table.setItem(r, 4, it_dae_eq)

                it_ep = QTableWidgetItem(f"{row['epanet']:.6g}")
                it_ep.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.eq_table.setItem(r, 5, it_ep)

                diff_end_val = row.get("diff_end", row.get("diff", 0.0))
                it_diff_end = QTableWidgetItem(f"{diff_end_val:.6g}")
                it_diff_end.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.eq_table.setItem(r, 6, it_diff_end)

                diff_eq_val = row.get("diff_eq", 0.0)
                it_diff_eq = QTableWidgetItem(f"{diff_eq_val:.6g}")
                it_diff_eq.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.eq_table.setItem(r, 7, it_diff_eq)

                lbl = row["label"].lower()
                if "flow" in lbl:
                    intervals_summary[inv]["max_flow_end"] = max(intervals_summary[inv]["max_flow_end"], diff_end_val)
                    intervals_summary[inv]["max_flow_eq"] = max(intervals_summary[inv]["max_flow_eq"], diff_eq_val)
                else:
                    intervals_summary[inv]["max_head_end"] = max(intervals_summary[inv]["max_head_end"], diff_end_val)
                    intervals_summary[inv]["max_head_eq"] = max(intervals_summary[inv]["max_head_eq"], diff_eq_val)

            summary_parts = []
            for inv, s in intervals_summary.items():
                inv_short = inv.split()[0] if " " in inv else inv
                summary_parts.append(
                    f"{inv_short}: End diff (Flow: {s['max_flow_end']:.2e}, Head: {s['max_head_end']:.2e}) | "
                    f"Eq diff (Flow: {s['max_flow_eq']:.2e}, Head: {s['max_head_eq']:.2e})"
                )
            self.eq_summary_label.setText("Max differences per interval: " + " ; ".join(summary_parts))

            # 5. Dirac Impulses Table
            self.imp_table.setRowCount(len(res.impulses))
            for r, imp in enumerate(res.impulses):
                it_t = QTableWidgetItem(f"{imp['time']:.6g}")
                it_t.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.imp_table.setItem(r, 0, it_t)

                lbl_display = f"{imp['state']} ({imp.get('state_index', '')})" if imp.get('state_index') else imp['state']
                self.imp_table.setItem(r, 1, QTableWidgetItem(lbl_display))

                it_c = QTableWidgetItem(f"{imp['coefficient']:+.6g}")
                it_c.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.imp_table.setItem(r, 2, it_c)

                it_ep = QTableWidgetItem("EPANET: none")
                self.imp_table.setItem(r, 3, it_ep)

            if not res.impulses:
                self.imp_summary_label.setText("No Dirac impulses occurred during switching. (EPANET: none)")
            else:
                self.imp_summary_label.setText(f"{len(res.impulses)} Dirac impulse entries at switching instants. (EPANET: none)")

            # Format columns of all results tables to fit content without clipping headers while updates disabled
            for tbl in tables:
                format_table(tbl)

        finally:
            for tbl in tables:
                tbl.blockSignals(False)
                tbl.setUpdatesEnabled(True)

    def stop_worker(self):
        """Wait for worker thread to finish if running (called during window close)."""
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)

    def update_theme_style(self, is_dark: Optional[bool] = None):
        """Ensure valves table checkboxes have >= 3:1 contrast in dark and light themes."""
        if is_dark is None:
            is_dark = self.palette().color(QPalette.Window).value() < 128
        border_color = "#A8A8A8" if is_dark else "#5A5A5A"
        bg_color = "#252526" if is_dark else "#FFFFFF"
        ss = f"""
        QTableWidget::indicator {{
            width: 14px;
            height: 14px;
            border: 2px solid {border_color};
            border-radius: 2px;
            background-color: {bg_color};
        }}
        QTableWidget::indicator:checked {{
            border: 2px solid {border_color};
            background-color: #007ACC;
        }}
        """
        self.valves_table.setStyleSheet(ss)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.PaletteChange:
            self.update_theme_style()
