"""Widget for displaying solvability checks, warnings, explanation banner, and graph stats."""

import re
import networkx as nx
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox
)

from core.matrices import build_matrices
from core.checks import numeric_pencil, is_regular, validate_network, incidence_matrices


class ChecksView(QWidget):
    """View displaying network regularity status, graph warnings, stats, and explanatory help."""

    warning_double_clicked = Signal(list)  # list of node/pipe IDs to highlight

    def __init__(self, parent=None):
        super().__init__(parent)
        self.warnings_data = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # 1. Status Banner
        self.banner = QLabel("Add pipes to run the checks")
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setStyleSheet(
            "background-color: #6C757D; color: white; font-weight: bold; font-size: 15px; padding: 12px; border-radius: 6px;"
        )
        layout.addWidget(self.banner)

        # 2. Network Stats & Rank Line
        self.stats_label = QLabel("Connected components: 0  |  Modified matrix rank: 0 / 0 rows")
        self.stats_label.setStyleSheet("font-size: 12px; color: #333; margin-top: 4px; margin-bottom: 8px;")
        layout.addWidget(self.stats_label)

        # 3. Warnings Table
        warnings_group = QGroupBox("Validation Warnings (Double-click a warning to highlight in graph)")
        w_layout = QVBoxLayout(warnings_group)

        self.warnings_table = QTableWidget(0, 2)
        self.warnings_table.setHorizontalHeaderLabels(["Type", "Description"])
        self.warnings_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.warnings_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.warnings_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.warnings_table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        w_layout.addWidget(self.warnings_table)

        layout.addWidget(warnings_group)

        # 4. Collapsible Explanation Box
        self.help_group = QGroupBox("What do these checks mean?")
        self.help_group.setCheckable(True)
        self.help_group.setChecked(True)
        h_layout = QVBoxLayout(self.help_group)

        help_text = (
            "<b>Solvability Conditions & Assumptions:</b><br>"
            "• <b>(C-II) Supply Requirement:</b> Every group of connected nodes needs at least one supply (Reservoir or Tank).<br>"
            "• <b>(C-I) Pump Loop Constraint:</b> Pumps must not form a closed loop or run in series directly between two supplies.<br>"
            "• <b>Numerical Test Assumptions:</b> Solvability is evaluated assuming all valves are fully open, all pumps are operational, and friction coefficients c<sub>i</sub> = 1."
        )
        self.help_label = QLabel(help_text)
        self.help_label.setWordWrap(True)
        self.help_label.setStyleSheet("font-size: 11px; color: #444;")
        h_layout.addWidget(self.help_label)

        self.help_group.toggled.connect(self.help_label.setVisible)
        layout.addWidget(self.help_group)

    def update_checks(self, network):
        """Re-run network validation and update banner, stats, and warnings table."""
        if not network.nodes or not network.pipes:
            self.banner.setText("Add pipes to run the checks")
            self.banner.setStyleSheet(
                "background-color: #6C757D; color: white; font-weight: bold; font-size: 15px; padding: 12px; border-radius: 6px;"
            )
            self.stats_label.setText("Connected components: 0  |  Modified matrix rank: 0 / 0 rows")
            self.warnings_table.setRowCount(0)
            self.warnings_data = []
            return

        # 1. Compute regularity
        mats = build_matrices(network.to_dict())
        E_num, A_num = numeric_pencil(mats)
        regular = is_regular(E_num, A_num)

        if regular:
            self.banner.setText("Regular - the system is solvable (under the assumptions below)")
            self.banner.setStyleSheet(
                "background-color: #28A745; color: white; font-weight: bold; font-size: 15px; padding: 12px; border-radius: 6px;"
            )
        else:
            self.banner.setText("NOT regular")
            self.banner.setStyleSheet(
                "background-color: #DC3545; color: white; font-weight: bold; font-size: 15px; padding: 12px; border-radius: 6px;"
            )

        # 2. Compute Graph & Matrix Stats
        G = nx.Graph()
        for n in network.nodes:
            G.add_node(n.id)
        for p in network.pipes:
            G.add_edge(p.start, p.end)

        num_cc = nx.number_connected_components(G)
        inc_res = incidence_matrices(network)
        mod_mat = inc_res["modified"]
        rank_mod = int(np.linalg.matrix_rank(mod_mat)) if mod_mat.size > 0 else 0
        num_junc_rows = mod_mat.shape[0]

        self.stats_label.setText(
            f"Connected components: {num_cc}  |  Modified incidence matrix rank: {rank_mod} / {num_junc_rows} rows"
        )

        # 3. Warnings Table
        warnings = validate_network(network)
        self.warnings_data = warnings
        self.warnings_table.setRowCount(len(warnings))

        for r, warn in enumerate(warnings):
            tag_item = QTableWidgetItem("⚠️ Warning")
            tag_item.setTextAlignment(Qt.AlignCenter)
            tag_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            desc_item = QTableWidgetItem(warn)
            desc_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            self.warnings_table.setItem(r, 0, tag_item)
            self.warnings_table.setItem(r, 1, desc_item)

    def _on_cell_double_clicked(self, row: int, col: int):
        if row < 0 or row >= len(self.warnings_data):
            return
        warn_text = self.warnings_data[row]
        ids = re.findall(r'\b[JTRP]\d+\b', warn_text)
        if ids:
            self.warning_double_clicked.emit(ids)
