"""Widget for displaying full, reduced, and modified incidence matrices and pressure difference equations."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget,
    QTableWidgetItem, QPushButton, QLabel, QStackedWidget
)

from core.checks import incidence_matrices
from ui.table_utils import copy_matrix_to_latex, export_matrix_to_csv, style_table_item


class IncidenceTabWidget(QWidget):
    """Single tab for an incidence matrix with descriptive header and export buttons."""

    def __init__(self, tab_title: str, parent=None):
        super().__init__(parent)
        self.tab_title = tab_title
        self.row_labels = []
        self.col_labels = []
        self.data = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        self.info_label = QLabel("Sign convention: -1 = start node, +1 = end node")
        self.info_label.setStyleSheet("font-size: 11px; color: #555; margin-bottom: 2px;")
        layout.addWidget(self.info_label)

        self.table = QTableWidget()
        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.copy_latex_btn = QPushButton("Copy as LaTeX")
        self.copy_latex_btn.clicked.connect(self._copy_latex)
        self.export_csv_btn = QPushButton("Export CSV...")
        self.export_csv_btn.clicked.connect(self._export_csv)

        btn_layout.addStretch()
        btn_layout.addWidget(self.copy_latex_btn)
        btn_layout.addWidget(self.export_csv_btn)
        layout.addLayout(btn_layout)

    def set_data(self, data, row_labels, col_labels, info_text=None):
        self.data = [list(r) if hasattr(r, "__iter__") and not isinstance(r, (str, bytes)) else [r] for r in data]
        self.row_labels = row_labels
        self.col_labels = col_labels

        if info_text:
            self.info_label.setText(f"Sign convention: -1 = start node, +1 = end node | {info_text}")
        else:
            self.info_label.setText("Sign convention: -1 = start node, +1 = end node")

        rows = len(self.data)
        cols = len(self.data[0]) if rows > 0 else 0

        self.table.setRowCount(rows)
        self.table.setColumnCount(cols)
        self.table.setHorizontalHeaderLabels(col_labels[:cols])
        self.table.setVerticalHeaderLabels(row_labels[:rows])

        for r in range(rows):
            for c in range(cols):
                val_str = str(self.data[r][c])
                item = QTableWidgetItem(val_str)
                style_table_item(item, val_str)
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()

    def _copy_latex(self):
        copy_matrix_to_latex(self.data)

    def _export_csv(self):
        export_matrix_to_csv(self, self.tab_title, self.row_labels, self.col_labels, self.data)


class IncidenceView(QWidget):
    """Main view for network incidence matrices with sub-tabs Full I(G), Reduced I^r(G), Modified I^m(G), Pressure diffs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.stack = QStackedWidget()

        # Placeholder page
        placeholder = QWidget()
        ph_layout = QVBoxLayout(placeholder)
        ph_label = QLabel("Add nodes and pipes to view incidence matrices")
        ph_label.setAlignment(Qt.AlignCenter)
        ph_label.setStyleSheet("font-size: 14px; color: gray;")
        ph_layout.addWidget(ph_label)
        self.stack.addWidget(placeholder)

        # Content page
        content = QWidget()
        c_layout = QVBoxLayout(content)
        c_layout.setContentsMargins(0, 0, 0, 0)

        self.sub_tabs = QTabWidget()
        self.tab_full = IncidenceTabWidget("Full Incidence Matrix")
        self.tab_reduced = IncidenceTabWidget("Reduced Incidence Matrix")
        self.tab_modified = IncidenceTabWidget("Modified Incidence Matrix")
        self.tab_pressures = IncidenceTabWidget("Pressure Differences")

        self.sub_tabs.addTab(self.tab_full, "Full I(G)")
        self.sub_tabs.addTab(self.tab_reduced, "Reduced I^r(G)")
        self.sub_tabs.addTab(self.tab_modified, "Modified I^m(G)")
        self.sub_tabs.addTab(self.tab_pressures, "Pressure differences")

        c_layout.addWidget(self.sub_tabs)
        self.stack.addWidget(content)

        layout.addWidget(self.stack)
        self.stack.setCurrentIndex(0)

    def update_incidence(self, network):
        """Re-calculate and update all incidence tabs."""
        if not network.nodes or not network.pipes:
            self.stack.setCurrentIndex(0)
            return

        self.stack.setCurrentIndex(1)

        res = incidence_matrices(network)
        full = res["incidence"].tolist()
        reduced = res["reduced"].tolist()
        modified = res["modified"].tolist()

        node_labels = res["node_labels"]
        edge_labels = res["edge_labels"]
        ref_node = res["reference_node"]

        # 1. Full
        self.tab_full.set_data(full, node_labels, edge_labels)

        # 2. Reduced
        red_rows = [n for n in node_labels if n != ref_node]
        self.tab_reduced.set_data(
            reduced, red_rows, edge_labels, info_text=f"Reference node removed: {ref_node}"
        )

        # 3. Modified
        mod_rows = [n.id for n in network.nodes if n.type == "junction"]
        non_juncs = [n.id for n in network.nodes if n.type in ("reservoir", "tank")]
        removed_str = ", ".join(non_juncs) if non_juncs else "None"
        self.tab_modified.set_data(
            modified, mod_rows, edge_labels, info_text=f"Reservoirs/Tanks removed: {removed_str}"
        )

        # 4. Pressure Differences
        press_data = []
        pipe_ids = []
        for p in network.pipes:
            pipe_ids.append(p.id)
            press_data.append([f"dP_{p.id} = P_{p.end} - P_{p.start}"])

        self.tab_pressures.set_data(
            press_data, pipe_ids, ["Expression"], info_text="dP = I^T * P (end pressure minus start pressure)"
        )
