"""Plain-text helpers for the web app. No Qt imports."""

from __future__ import annotations

import csv
import io
import math
import re
from typing import Any


def format_optional(value: Any) -> str:
    """Render a stored number for an input, or an empty string when it is unset."""
    if value is None:
        return ""
    return f"{float(value):.6g}"


def parse_optional_number(
    text: str,
    label: str,
    *,
    minimum: float | None = None,
    exclusive_min: float | None = None,
) -> float | None:
    """Parse an optional number. A blank field means the value is unset."""
    raw = (text or "").strip()
    if raw == "":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number, got {raw!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}.")
    if exclusive_min is not None and value <= exclusive_min:
        raise ValueError(f"{label} must be > {exclusive_min}.")
    return value


def format_latex_symbol(cell: Any) -> str:
    """Format one symbolic matrix entry as LaTeX."""
    s = str(cell)
    s = re.sub(r"([sc])(\d+)", r"\1_{\2}", s)
    s = s.replace("Q_pu", r"Q_{\text{pu}}")
    s = s.replace("G_pu", r"G_{\text{pu}}")
    s = s.replace("S_pu", r"S_{\text{pu}}")
    s = s.replace("C_a", r"C_{\text{a}}")
    s = s.replace("Pl", r"P_{\text{l}}")
    s = s.replace("Pr", r"P_{\text{r}}")
    return s


def matrix_to_latex(matrix_data: Any) -> str:
    """Convert a matrix or a column vector to a LaTeX bmatrix."""
    if not matrix_data:
        return r"\begin{bmatrix}\end{bmatrix}"
    rows = []
    for row in matrix_data:
        if isinstance(row, (list, tuple)):
            body = " & ".join(format_latex_symbol(cell) for cell in row)
        else:
            body = format_latex_symbol(row)
        rows.append(body)
    return "\\begin{bmatrix}\n  " + " \\\\\n  ".join(rows) + "\n\\end{bmatrix}"


def matrix_to_csv(row_labels: list[str], col_labels: list[str], data: Any) -> bytes:
    """CSV bytes with a UTF-8 BOM so Excel keeps symbols intact."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    is_2d = bool(data) and isinstance(data[0], (list, tuple))
    if is_2d:
        writer.writerow([""] + list(col_labels))
        for index, row in enumerate(data):
            label = row_labels[index] if index < len(row_labels) else f"Row {index + 1}"
            writer.writerow([label] + [str(cell) for cell in row])
    else:
        writer.writerow(["Label", col_labels[0] if col_labels else "Value"])
        for index, value in enumerate(data):
            label = row_labels[index] if index < len(row_labels) else f"Row {index + 1}"
            writer.writerow([label, str(value)])
    return buffer.getvalue().encode("utf-8-sig")
