"""Widget for displaying symbolic network matrices, vector equations, patterns, LaTeX & CSV exports."""

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget,
    QTableWidgetItem, QPushButton, QLabel, QStackedWidget
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ui.table_utils import copy_matrix_to_latex, export_matrix_to_csv, style_table_item


class MatrixTabWidget(QWidget):
    """Sub-tab widget containing a QTableWidget and LaTeX/CSV export buttons."""

    def __init__(self, matrix_name: str, parent=None):
        super().__init__(parent)
        self.matrix_name = matrix_name
        self.row_labels = []
        self.col_labels = []
        self.data = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

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

    def set_data(self, data, labels, col_labels=None):
        self.data = data
        self.row_labels = labels
        self.col_labels = col_labels if col_labels is not None else labels
        is_2d = isinstance(data, list) and len(data) > 0 and isinstance(data[0], list)

        if is_2d:
            rows = len(data)
            cols = len(data[0])
            self.table.setRowCount(rows)
            self.table.setColumnCount(cols)
            self.table.setHorizontalHeaderLabels(self.col_labels[:cols])
            self.table.setVerticalHeaderLabels(self.row_labels[:rows])

            for r in range(rows):
                for c in range(cols):
                    val = str(data[r][c])
                    item = QTableWidgetItem(val)
                    style_table_item(item, val)
                    self.table.setItem(r, c, item)
            self.table.resizeColumnsToContents()
        else:
            rows = len(data)
            self.table.setRowCount(rows)
            self.table.setColumnCount(1)
            self.table.setHorizontalHeaderLabels([self.matrix_name])
            self.table.setVerticalHeaderLabels(self.row_labels[:rows])

            for r in range(rows):
                val = str(data[r])
                item = QTableWidgetItem(val)
                style_table_item(item, val)
                self.table.setItem(r, 0, item)
            self.table.resizeColumnsToContents()

    def _copy_latex(self):
        copy_matrix_to_latex(self.data)

    def _export_csv(self):
        export_matrix_to_csv(self, self.matrix_name, self.row_labels, self.col_labels, self.data)


class PatternsTabWidget(QWidget):
    """Sub-tab showing non-zero heatmaps for E and A matrices."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

    def set_patterns(self, E_matrix, A_matrix):
        self.figure.clear()
        if not E_matrix or not A_matrix:
            self.canvas.draw()
            return

        E_bin = np.array([[1 if str(v) != '0' else 0 for v in row] for row in E_matrix])
        A_bin = np.array([[1 if str(v) != '0' else 0 for v in row] for row in A_matrix])

        ax1 = self.figure.add_subplot(121)
        ax1.imshow(E_bin, cmap='binary', interpolation='nearest')
        ax1.set_title("E Non-zero Pattern")
        ax1.set_xlabel("Columns")
        ax1.set_ylabel("Rows")

        ax2 = self.figure.add_subplot(122)
        ax2.imshow(A_bin, cmap='binary', interpolation='nearest')
        ax2.set_title("A Non-zero Pattern")
        ax2.set_xlabel("Columns")

        self.figure.tight_layout()
        self.canvas.draw()


class MatrixView(QWidget):
    """Main view for network matrices with summary line and E, A, F, G, Patterns sub-tabs."""

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
        ph_label = QLabel("Add at least one pipe to see matrices")
        ph_label.setAlignment(Qt.AlignCenter)
        ph_label.setStyleSheet("font-size: 14px; color: gray;")
        ph_layout.addWidget(ph_label)
        self.stack.addWidget(placeholder)

        # Main content page
        content = QWidget()
        c_layout = QVBoxLayout(content)
        c_layout.setContentsMargins(0, 0, 0, 0)

        self.summary_label = QLabel("K = 0   |   pipes 0, pumps 0, junctions 0, reservoirs 0, tanks 0")
        self.summary_label.setStyleSheet("font-weight: bold; font-size: 13px; margin-bottom: 5px;")
        c_layout.addWidget(self.summary_label)

        self.sub_tabs = QTabWidget()
        self.tab_E = MatrixTabWidget("E")
        self.tab_A = MatrixTabWidget("A")
        self.tab_F = MatrixTabWidget("F")
        self.tab_G = MatrixTabWidget("G")
        self.tab_patterns = PatternsTabWidget()

        self.sub_tabs.addTab(self.tab_E, "E")
        self.sub_tabs.addTab(self.tab_A, "A")
        self.sub_tabs.addTab(self.tab_F, "F")
        self.sub_tabs.addTab(self.tab_G, "G")
        self.sub_tabs.addTab(self.tab_patterns, "Patterns")

        c_layout.addWidget(self.sub_tabs)
        self.stack.addWidget(content)

        layout.addWidget(self.stack)
        self.stack.setCurrentIndex(0)

    def update_matrices(self, network, matrices_result):
        """Update summary and all matrix sub-tabs."""
        if not network.pipes or matrices_result is None:
            self.stack.setCurrentIndex(0)
            return

        self.stack.setCurrentIndex(1)

        num_pipes = len(network.pipes)
        num_pumps = sum(1 for p in network.pipes if p.has_pump)
        num_juncs = sum(1 for n in network.nodes if n.type == "junction")
        num_res = sum(1 for n in network.nodes if n.type == "reservoir")
        num_tanks = sum(1 for n in network.nodes if n.type == "tank")
        K = matrices_result["K_dimension"]

        self.summary_label.setText(
            f"K = {K}   |   pipes {num_pipes}, pumps {num_pumps}, junctions {num_juncs}, reservoirs {num_res}, tanks {num_tanks}"
        )

        labels = matrices_result["labels"]
        self.tab_E.set_data(matrices_result["E_matrix"], labels)
        self.tab_A.set_data(matrices_result["A_matrix"], labels)
        self.tab_F.set_data(matrices_result["F_matrix"], labels)
        self.tab_G.set_data(matrices_result["G_matrix"], labels)
        self.tab_patterns.set_patterns(matrices_result["E_matrix"], matrices_result["A_matrix"])
