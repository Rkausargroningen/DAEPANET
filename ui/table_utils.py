"""Shared table utilities for matrix and incidence rendering, LaTeX copying, and CSV exporting."""

import csv
import re
from typing import List, Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import QWidget, QTableWidget, QTableWidgetItem, QFileDialog


def format_latex_symbol(cell_str: Any) -> str:
    """Format symbolic matrix strings into LaTeX math syntax."""
    s = str(cell_str)
    s = re.sub(r'([sc])(\d+)', r'\1_{\2}', s)
    s = s.replace('Q_pu', r'Q_{\text{pu}}')
    s = s.replace('G_pu', r'G_{\text{pu}}')
    s = s.replace('S_pu', r'S_{\text{pu}}')
    s = s.replace('C_a', r'C_{\text{a}}')
    s = s.replace('Pl', r'P_{\text{l}}')
    s = s.replace('Pr', r'P_{\text{r}}')
    return s


def matrix_to_latex(matrix_data: Any) -> str:
    """Convert a 2D matrix or 1D vector to a LaTeX bmatrix string."""
    rows = []
    if not matrix_data:
        return r"\begin{bmatrix}\end{bmatrix}"

    for r in matrix_data:
        if isinstance(r, (list, tuple)):
            row_str = " & ".join(format_latex_symbol(cell) for cell in r)
        else:
            row_str = format_latex_symbol(r)
        rows.append(row_str)
    body = " \\\\\n  ".join(rows)
    return f"\\begin{{bmatrix}}\n  {body}\n\\end{{bmatrix}}"


def copy_matrix_to_latex(data: Any) -> None:
    """Copy matrix/vector data to system clipboard as LaTeX bmatrix."""
    latex_str = matrix_to_latex(data)
    QGuiApplication.clipboard().setText(latex_str)


def export_matrix_to_csv(
    parent: QWidget,
    matrix_name: str,
    row_labels: List[str],
    col_labels: List[str],
    data: Any
) -> None:
    """Export matrix or table data to CSV file with UTF-8 BOM encoding."""
    default_name = f"{matrix_name.replace(' ', '_').lower()}.csv"
    path, _ = QFileDialog.getSaveFileName(
        parent, f"Export {matrix_name} to CSV", default_name, "CSV Files (*.csv)"
    )
    if not path:
        return

    is_2d = isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (list, tuple))

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if is_2d:
            writer.writerow([""] + list(col_labels))
            for r, row in enumerate(data):
                label = row_labels[r] if r < len(row_labels) else f"Row {r+1}"
                writer.writerow([label] + [str(x) for x in row])
        else:
            writer.writerow(["Label", matrix_name])
            for r, val in enumerate(data):
                label = row_labels[r] if r < len(row_labels) else f"Row {r+1}"
                writer.writerow([label, str(val)])


def style_table_item(item: QTableWidgetItem, val_str: str) -> None:
    """Apply standard visual styling (colors, alignment) to table cells."""
    item.setTextAlignment(Qt.AlignCenter)
    if val_str in ("0", "0.0", ""):
        item.setForeground(QColor("#A0A0A0"))
    elif val_str in ("-1", "-1.0"):
        item.setBackground(QColor("#E6F0FA"))  # Soft Blue for start node / -1
        item.setForeground(QColor("#004085"))
    elif val_str in ("1", "1.0", "+1"):
        item.setBackground(QColor("#FCE8E6"))  # Soft Red/Salmon for end node / +1
        item.setForeground(QColor("#721C24"))
    else:
        item.setBackground(QColor("#EBF5FB"))  # Soft Highlight for non-zero symbolic
        item.setForeground(QColor("#000000"))
