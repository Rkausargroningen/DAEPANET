"""ui/comparison_plots.py – Matplotlib-based comparison plot widget reproducing MATLAB comparisonFigure.

Includes:
- draw_comparison(fig, result, options)
- ComparisonPlots widget with navigation toolbar and export buttons
- Options bar: States dropdown checklist, Pressure vs Total head, Show Dirac arrows
- Export actions: Save figure (PNG/PDF/SVG), Export data (CSV), Export EPANET input (.inp)
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtGui import QColor, QFont, QAction, QKeySequence, QPalette, QPixmap, QIcon
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QMenu,
    QRadioButton,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QScrollArea,
    QFrame,
)

QWIDGETSIZE_MAX = 16777215

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, FuncFormatter

ICON_MAP = {
    "Home": "home",
    "Back": "back",
    "Forward": "forward",
    "Pan": "move",
    "Zoom": "zoom_to_rect",
    "Subplots": "subplots",
    "Customize": "qt4_editor_options",
    "Save": "filesave",
}

from core.comparison import ComparisonResult
from core.plotdata import (
    default_states,
    series,
    decimate_series,
    impulse_arrows,
    engine_label,
)


def apply_panel_span_rules(
    current_ymin: float,
    current_ymax: float,
    panel_max_abs_y: float,
    is_flow: bool,
    global_max_abs_flow: float,
) -> Tuple[float, float]:
    """Calculate widened (ymin, ymax) limits according to Step 1a rules."""
    span = current_ymax - current_ymin
    ymid = (current_ymin + current_ymax) / 2.0
    if is_flow:
        if global_max_abs_flow == 0.0:
            return (-0.001, 0.001)
        else:
            target_span = 0.01 * global_max_abs_flow
            if span < target_span:
                return (ymid - target_span / 2.0, ymid + target_span / 2.0)
    else:
        target_span = max(0.005 * panel_max_abs_y, 0.01)
        if span < target_span:
            return (ymid - target_span / 2.0, ymid + target_span / 2.0)
    return (current_ymin, current_ymax)


_apply_panel_span_rules = apply_panel_span_rules


def make_tick_formatter(get_limits_fn):
    """Return a FuncFormatter that prints at most 4 decimals, no scientific offset,
    and returns empty string for ticks outside the axis limits."""
    def _format(val, pos=None):
        lims = get_limits_fn()
        if lims is not None:
            if val < lims[0] - 1e-6 or val > lims[1] + 1e-6:
                return ""
        if abs(val) < 1e-12:
            return "0"
        s = f"{val:.4f}".rstrip("0").rstrip(".")
        if s == "-0" or s == "":
            return "0"
        return s

    return FuncFormatter(_format)


from matplotlib.font_manager import FontProperties

_FONT_PROP_CACHE: Dict[float, FontProperties] = {}
_TEXT_WIDTH_CACHE: Dict[Tuple[str, float, float], float] = {}


def _cached_text_width(renderer, text: str, size: float) -> float:
    key = (text, size, renderer.dpi)
    val = _TEXT_WIDTH_CACHE.get(key)
    if val is None:
        fp = _FONT_PROP_CACHE.get(size)
        if fp is None:
            fp = FontProperties(size=size)
            _FONT_PROP_CACHE[size] = fp
        val = renderer.get_text_width_height_descent(text, fp, False)[0]
        _TEXT_WIDTH_CACHE[key] = val
    return val


def draw_comparison(
    fig: Figure,
    result: Optional[ComparisonResult],
    options: Optional[Dict[str, Any]] = None,
):
    """Draw stacked comparison figure faithfully reproducing MATLAB's comparisonFigure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The target figure to draw on.
    result : ComparisonResult, optional
        Simulation comparison result.
    options : dict, optional
        Display options:
        - 'selected_states': list of state labels to show
        - 'head_mode': 'pressure' or 'total'
        - 'show_dirac': bool (default True)
        - 'title': optional scenario title
    """
    opts = options or {}
    selected = opts.get("selected_states")
    if not selected and result is not None:
        selected = default_states(result)
    elif not selected:
        selected = []
    head_mode = opts.get("head_mode", "pressure")
    show_dirac = opts.get("show_dirac", True)

    n_panels = len(selected)

    if result is None:
        fig.clf()
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")
        hint = "No results yet. Fill in the network, choose the valves and scenario on the left, then press Run."
        fig_w = fig.bbox.width if hasattr(fig, "bbox") and fig.bbox.width > 0 else 600
        max_chars = max(30, int((fig_w * 0.72) / 8.5))
        import textwrap
        wrapped_hint = textwrap.fill(hint, width=max_chars)
        ax.text(
            0.5,
            0.5,
            wrapped_hint,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            color="#666666",
            fontsize=11,
            multialignment="center",
        )
        ax.axis("off")
        fig.canvas.draw_idle()
        return

    if not hasattr(fig.canvas, "get_renderer"):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig)

    if n_panels == 0:
        fig.clf()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No states selected.", ha="center", va="center", color="#888888")
        ax.axis("off")
        fig.canvas.draw_idle()
        return

    # Palette matching MATLAB comparisonFigure
    # DAE: (0, 0.447, 0.698) = #0072B2
    # EPANET: (0.85, 0.10, 0.10) = #D91A1A
    # Grid: alpha 0.12, switch line: #CCCCCC
    cD = (0.0, 0.447, 0.698)
    cE = (0.85, 0.10, 0.10)
    cSwitch = "#CCCCCC"

    T = float(result.scenario.t_end)
    xlims = (-0.01 * T, T)

    # Check if network is the specimen network
    is_specimen = (
        len(result.snapshot.pipes) == 2
        and all(s in result.dae.labels for s in ["Pipe_1_Flow", "Pipe_2_Flow", "Pipe_1_R", "Pipe_2_L", "Pipe_2_R"])
    )

    specimen_labels_full = {
        "Pipe_1_Flow": "x1 = Q1 [m³/s]",
        "Pipe_2_Flow": "x2 = Q2 [m³/s]",
        "Pipe_1_R": f"x5 pipe-1 end {'total head ' if head_mode == 'total' else ''}[m]",
        "Pipe_2_L": f"x4 d/s of V1 {'total head ' if head_mode == 'total' else ''}[m]",
        "Pipe_2_R": f"x6 pipe-2 end {'total head ' if head_mode == 'total' else ''}[m]",
    }
    specimen_labels_short = {
        "Pipe_1_Flow": "x1 [m³/s]",
        "Pipe_2_Flow": "x2 [m³/s]",
        "Pipe_1_R": "x5 [m]",
        "Pipe_2_L": "x4 [m]",
        "Pipe_2_R": "x6 [m]",
    }

    def get_labels_pair(lbl_name: str) -> Tuple[str, str]:
        if is_specimen and lbl_name in specimen_labels_full:
            return specimen_labels_full[lbl_name], specimen_labels_short[lbl_name]
        if "Flow" in lbl_name:
            return f"{lbl_name}  Q [m³/s]", f"{lbl_name} [m³/s]"
        return (
            f"{lbl_name}  {'total head [m]' if head_mode == 'total' else 'pressure head [m]'}",
            f"{lbl_name} [m]",
        )

    # Precompute Dirac impulses
    all_imp_arrows = impulse_arrows(result, selected) if show_dirac else {lbl: [] for lbl in selected}

    # Step 1a: compute global maximum |flow| over all displayed flow series
    global_max_flow = 0.0
    for lbl in selected:
        if "Flow" in lbl:
            d_segs, _, ep_data = series(result, lbl, kind=head_mode)
            for _, yk in d_segs:
                if len(yk) > 0:
                    global_max_flow = max(global_max_flow, float(np.max(np.abs(yk))))
            t_ep, y_ep = ep_data
            if len(y_ep) > 0:
                global_max_flow = max(global_max_flow, float(np.max(np.abs(y_ep))))

    if len(fig.axes) == n_panels:
        axes = np.array(fig.axes)
        for ax in axes:
            ax.clear()
    else:
        fig.clf()
        axes = fig.subplots(nrows=n_panels, ncols=1, sharex=True, squeeze=False)[:, 0]

    for r, lbl in enumerate(selected):
        ax = axes[r]
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.12, color="gray", linestyle="-")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#333333")

        ax.set_xlim(xlims)

        # Step 1a: Tick formatter with at most 4 decimals, no offset, pruning outside ticks
        ax.yaxis.set_major_formatter(make_tick_formatter(ax.get_ylim))

        dae_segs, jumps, ep_data = series(result, lbl, kind=head_mode)

        # 1. DAE segments (one per mode, solid line, decimated to <= max_pts for drawing only)
        max_pts = opts.get("max_points", 4000)
        for tk, yk in dae_segs:
            t_plot, y_plot = decimate_series(tk, yk, max_points=max_pts)
            ax.plot(t_plot, y_plot, "-", color=cD, linewidth=1.6)

        # 2. Dotted jump line and vertical switch line
        for ts, y_before, y_after in jumps:
            ax.plot([ts, ts], [y_before, y_after], ":", color=cD, linewidth=1.5)
            ax.axvline(x=ts, color=cSwitch, linestyle="-", linewidth=1.0, zorder=1)

        # In scenarios with switches where jump is 0 (or no flow jump), still draw switch line
        if result.scenario.events and not jumps:
            for ev in result.scenario.events:
                ax.axvline(x=ev.time, color=cSwitch, linestyle="-", linewidth=1.0, zorder=1)

        # 3. EPANET step plot on top
        t_ep, y_ep = ep_data
        ax.plot(t_ep, y_ep, "-", color=cE, linewidth=0.9, drawstyle="steps-post", zorder=4)

        # 4. Y-axis label (initial setting)
        ylab_init, _ = get_labels_pair(lbl)
        ax.set_ylabel(ylab_init, fontsize=9)

        # 5. Dirac arrows
        yl = ax.get_ylim()
        arrows = all_imp_arrows.get(lbl, [])
        if arrows:
            for ts, coeff, base in arrows:
                yrange = yl[1] - yl[0]
                if abs(yrange) < 1e-9:
                    yrange = 1.0
                h = 0.5 * yrange * (1.0 if coeff > 0 else -1.0)
                marker = "^" if coeff > 0 else "v"

                # Solid arrow stem
                ax.plot([ts, ts], [base, base + h], "-", color=cD, linewidth=2.2, zorder=6)
                # Triangle marker
                ax.plot(
                    ts,
                    base + h,
                    marker=marker,
                    color=cD,
                    markerfacecolor=cD,
                    markersize=7,
                    zorder=7,
                )
                # Text label
                sign_str = f"{coeff:+.2f}"
                txt = f"DAE: Dirac {sign_str} δ(t - {ts:g})\nEPANET: none"
                if ts >= 0.5 * T:
                    x_txt = ts - 0.03 * T
                    ha_txt = "right"
                else:
                    x_txt = ts + 0.03 * T
                    ha_txt = "left"
                ax.text(
                    x_txt,
                    base + 0.62 * h,
                    txt,
                    color=cD,
                    fontsize=8.5,
                    fontweight="bold",
                    horizontalalignment=ha_txt,
                    verticalalignment="center",
                    bbox=dict(
                        boxstyle="square,pad=0.25",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.95,
                    ),
                    zorder=8,
                )
                # Extend y limits
                new_ymin = min(yl[0], base + 1.35 * min(0.0, h))
                new_ymax = max(yl[1], base + 1.35 * max(0.0, h))
                ax.set_ylim(new_ymin, new_ymax)
        else:
            ax.set_ylim(yl)

        # Step 1a: Minimum visible range rule
        ymin, ymax = ax.get_ylim()
        is_flow = ("Flow" in lbl)
        max_abs_y = max(abs(ymin), abs(ymax))
        new_ymin, new_ymax = apply_panel_span_rules(ymin, ymax, max_abs_y, is_flow, global_max_flow)
        ax.set_ylim(new_ymin, new_ymax)

        # Top panel: title and legend
        if r == 0:
            if opts.get("title"):
                title_text = opts.get("title")
            elif result is not None and result.scenario and result.scenario.title:
                s_title = result.scenario.title
                if fig.bbox.width < 1200 or len(s_title) > 25:
                    title_text = f"Switched DAE vs EPANET\n{s_title}"
                else:
                    title_text = f"Switched DAE vs EPANET: {s_title}"
            else:
                title_text = "Switched DAE vs EPANET"
            while "  " in title_text:
                title_text = title_text.replace("  ", " ")

            line_dae = Line2D([0], [0], color=cD, linewidth=1.6)
            line_ep = Line2D([0], [0], color=cE, linewidth=0.9)
            leg_labels = ["switched DAE", engine_label(result)]

            fig_w = fig.bbox.width
            engine = fig.get_layout_engine()

            if fig_w < 800:
                if engine and hasattr(engine, "set"):
                    engine.set(rect=[0.0, 0.0, 1.0, 1.0])
                ax.set_title(title_text, fontsize=9.5, wrap=True)
                panel_h_est = max(30.0, (fig.bbox.height - 60.0) / max(1, n_panels))
                if panel_h_est < 130.0:
                    leg_y = 1.50 if "\n" in title_text else 1.32
                else:
                    leg_y = 1.35 if "\n" in title_text else 1.22
                ax.legend(
                    [line_dae, line_ep],
                    leg_labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, leg_y),
                    ncol=2,
                    frameon=False,
                    fontsize=8,
                )
            else:
                renderer = fig.canvas.get_renderer()
                w_max = max(
                    _cached_text_width(renderer, lbl, 8.5)
                    for lbl in leg_labels
                )
                leg_w_px = w_max + 75.0
                margin_px = leg_w_px + 28.0
                right_rect = max(0.60, min(0.92, (fig_w - margin_px) / fig_w))
                if engine and hasattr(engine, "set"):
                    engine.set(rect=[0.0, 0.0, right_rect, 1.0])

                ax.set_title(title_text, fontsize=10, pad=8, wrap=True)
                ax.legend(
                    [line_dae, line_ep],
                    leg_labels,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=False,
                    fontsize=8.5,
                )

        # Bottom panel: x axis label and formatter
        if r == n_panels - 1:
            ax.xaxis.set_major_formatter(make_tick_formatter(ax.get_xlim))
            ax.set_xlabel("t [s]", fontsize=10)

    # Adaptive Y-label sizing and short form
    renderer = fig.canvas.get_renderer()
    panel_est_h = max(30.0, (fig.bbox.height - 60.0) / max(1, n_panels))
    shortened_any = False
    for r, lbl in enumerate(selected):
        ax = axes[r]
        full_txt, short_txt = get_labels_pair(lbl)
        ax_h = panel_est_h

        chosen_txt = full_txt
        chosen_fs = 9.0

        w9 = _cached_text_width(renderer, full_txt, 9.0)
        w8 = _cached_text_width(renderer, full_txt, 8.0)
        w7 = _cached_text_width(renderer, full_txt, 7.0)

        if w9 <= 0.82 * ax_h:
            chosen_txt, chosen_fs = full_txt, 9.0
        elif w8 <= 0.82 * ax_h:
            chosen_txt, chosen_fs = full_txt, 8.0
        elif w7 <= 0.82 * ax_h:
            chosen_txt, chosen_fs = full_txt, 7.0
        else:
            shortened_any = True
            sw9 = _cached_text_width(renderer, short_txt, 9.0)
            sw8 = _cached_text_width(renderer, short_txt, 8.0)
            if sw9 <= 0.85 * ax_h:
                chosen_txt, chosen_fs = short_txt, 9.0
            elif sw8 <= 0.85 * ax_h:
                chosen_txt, chosen_fs = short_txt, 8.0
            else:
                chosen_txt, chosen_fs = short_txt, 7.0

        ax.set_ylabel(chosen_txt, fontsize=chosen_fs)

    fig.align_ylabels()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Verify no two adjacent y-labels overlap
    for r in range(n_panels - 1):
        bb_top = axes[r].yaxis.label.get_window_extent(renderer)
        bb_bot = axes[r + 1].yaxis.label.get_window_extent(renderer)
        gap = bb_top.y0 - bb_bot.y1
        if gap < 6.0:
            shortened_any = True
            _, short_top = get_labels_pair(selected[r])
            _, short_bot = get_labels_pair(selected[r + 1])
            axes[r].set_ylabel(short_top, fontsize=7.0)
            axes[r + 1].set_ylabel(short_bot, fontsize=7.0)

    if shortened_any and is_specimen:
        fig.text(
            0.5,
            0.008,
            "States: x1: Pipe 1 Flow, x2: Pipe 2 Flow, x4: Pipe 2 (d/s of V1), x5: Pipe 1 (end), x6: Pipe 2 (end)",
            ha="center",
            fontsize=7.5,
            color="#555555",
        )

    fig.align_ylabels()
    fig.canvas.draw_idle()


class ComparisonPlots(QWidget):
    """Widget containing comparison plot canvas, options bar, and export buttons."""

    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: Optional[ComparisonResult] = None
        self._selected_states: List[str] = []
        self._head_mode: str = "pressure"
        self._show_dirac: bool = True

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # -------------------------------------------------------------
        # TOP OPTIONS BAR & TOOLBAR (responsive combined row)
        # -------------------------------------------------------------
        self.top_container = QWidget(self)
        self.top_box = QVBoxLayout(self.top_container)
        self.top_box.setContentsMargins(0, 0, 0, 0)
        self.top_box.setSpacing(4)

        self.states_btn = QToolButton()
        self.states_btn.setText("States...")
        self.states_btn.setPopupMode(QToolButton.InstantPopup)
        self.states_menu = QMenu(self)
        self.states_btn.setMenu(self.states_menu)

        self.display_lbl = QLabel("Display:")
        self.radio_group = QButtonGroup(self)
        self.radio_pressure = QRadioButton("Pressure head")
        self.radio_pressure.setChecked(True)
        self.radio_total = QRadioButton("Total head (EPANET)")
        self.radio_group.addButton(self.radio_pressure, 0)
        self.radio_group.addButton(self.radio_total, 1)
        self.radio_group.buttonClicked.connect(self._on_head_mode_changed)
        self.radio_pressure.setMinimumWidth(self.radio_pressure.fontMetrics().horizontalAdvance(self.radio_pressure.text()) + 26)
        self.radio_total.setMinimumWidth(self.radio_total.fontMetrics().horizontalAdvance(self.radio_total.text()) + 26)

        self.dirac_check = QCheckBox("Show Dirac arrows")
        self.dirac_check.setChecked(True)
        self.dirac_check.toggled.connect(self._on_dirac_toggled)
        self.dirac_check.setMinimumWidth(self.dirac_check.fontMetrics().horizontalAdvance(self.dirac_check.text()) + 26)

        # -------------------------------------------------------------
        # MATPLOTLIB CANVAS & TOOLBAR (In QScrollArea)
        # -------------------------------------------------------------
        self.fig = Figure(figsize=(7, 9), facecolor="white", layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.canvas.setMinimumWidth(560)
        self.scroll_area.setWidget(self.canvas)

        self.update_toolbar_style()

        layout.addWidget(self.top_container)
        layout.addWidget(self.scroll_area, 1)
        self._relayout_top_bar(self.width())

        # -------------------------------------------------------------
        # BOTTOM EXPORT BUTTONS BAR
        # -------------------------------------------------------------
        export_bar = QHBoxLayout()
        export_bar.setSpacing(10)

        self.save_fig_btn = QPushButton("Save figure...")
        self.save_fig_btn.setToolTip("Save high-resolution plot (PNG 200 dpi, PDF, SVG)")
        self.save_fig_btn.clicked.connect(self.save_figure_action)
        export_bar.addWidget(self.save_fig_btn)

        self.export_csv_btn = QPushButton("Export data (CSV)...")
        self.export_csv_btn.setToolTip("Export comparison time series and Dirac impulses as CSV")
        self.export_csv_btn.clicked.connect(self.export_csv_action)
        export_bar.addWidget(self.export_csv_btn)

        self.export_inp_btn = QPushButton("Export EPANET input file (.inp)...")
        self.export_inp_btn.setToolTip("Export the exact EPANET input file used in this simulation")
        self.export_inp_btn.clicked.connect(self.export_inp_action)
        export_bar.addWidget(self.export_inp_btn)

        export_bar.addStretch()
        layout.addLayout(export_bar)

        self.clear()

    def update_toolbar_style(self):
        """Ensure toolbar and buttons have high contrast in both dark and light themes."""
        pal = self.palette()
        is_dark = pal.color(QPalette.Window).value() < 128
        if is_dark:
            bg = "#2D2D2D"
            border = "#444444"
            fg = QColor("#F0F0F0")
        else:
            bg = "#F8F9FA"
            border = "#DDDDDD"
            fg = QColor("#222222")

        self.toolbar.setStyleSheet(f"""
            QToolBar {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 2px;
            }}
            QToolButton {{
                background-color: {bg};
                border: none;
                padding: 3px;
                border-radius: 3px;
            }}
            QToolButton:hover {{
                background-color: {'#3D3D3D' if is_dark else '#E2E6EA'};
            }}
        """)
        import matplotlib.cbook as cbook
        for btn in self.toolbar.findChildren(QToolButton):
            txt = btn.text()
            base_name = ICON_MAP.get(txt)
            if not base_name:
                continue
            try:
                path_reg = cbook._get_data_path("images", f"{base_name}.png")
                path_large = path_reg.with_name(path_reg.name.replace(".png", "_large.png"))
                fn = str(path_large if path_large.exists() else path_reg)
                pm = QPixmap(fn)
                mask = pm.createMaskFromColor(QColor("black"), Qt.MaskMode.MaskOutColor)
                pm.fill(fg)
                pm.setMask(mask)
                btn.setIcon(QIcon(pm))
            except Exception:
                pass

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.PaletteChange:
            self.update_toolbar_style()

    def _relayout_top_bar(self, width: int):
        """Put options and toolbar on ONE row when width >= 900 px, wrap when narrower."""
        while self.top_box.count():
            item = self.top_box.takeAt(0)
            if item.layout():
                sub = item.layout()
                while sub.count():
                    sub.takeAt(0)

        if width >= 900:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            row.addWidget(self.states_btn)
            row.addWidget(self.display_lbl)
            row.addWidget(self.radio_pressure)
            row.addWidget(self.radio_total)
            row.addWidget(self.dirac_check)
            row.addStretch()
            row.addWidget(self.toolbar)
            self.top_box.addLayout(row)
        elif width >= 650:
            row1 = QHBoxLayout()
            row1.setContentsMargins(0, 0, 0, 0)
            row1.setSpacing(8)
            row1.addWidget(self.states_btn)
            row1.addWidget(self.display_lbl)
            row1.addWidget(self.radio_pressure)
            row1.addWidget(self.radio_total)
            row1.addWidget(self.dirac_check)
            row1.addStretch()

            row2 = QHBoxLayout()
            row2.setContentsMargins(0, 0, 0, 0)
            row2.setSpacing(8)
            row2.addWidget(self.toolbar)
            row2.addStretch()

            self.top_box.addLayout(row1)
            self.top_box.addLayout(row2)
        else:
            row1 = QHBoxLayout()
            row1.setContentsMargins(0, 0, 0, 0)
            row1.setSpacing(8)
            row1.addWidget(self.states_btn)
            row1.addWidget(self.display_lbl)
            row1.addWidget(self.radio_pressure)
            row1.addStretch()

            row2 = QHBoxLayout()
            row2.setContentsMargins(0, 0, 0, 0)
            row2.setSpacing(8)
            row2.addWidget(self.radio_total)
            row2.addWidget(self.dirac_check)
            row2.addStretch()

            row3 = QHBoxLayout()
            row3.setContentsMargins(0, 0, 0, 0)
            row3.setSpacing(8)
            row3.addWidget(self.toolbar)
            row3.addStretch()

            self.top_box.addLayout(row1)
            self.top_box.addLayout(row2)
            self.top_box.addLayout(row3)

    def _update_canvas_height(self):
        """Ensure all panels are shown without scrollbar if visible area is tall enough for 110 px per panel."""
        n = max(1, len(self._selected_states) if self._selected_states else 5)
        needed_h = n * 110
        visible_h = self.scroll_area.viewport().height()
        if visible_h >= needed_h:
            self.canvas.setMinimumHeight(0)
            self.canvas.setMaximumHeight(QWIDGETSIZE_MAX)
            self.canvas.resize(self.scroll_area.viewport().width(), visible_h)
            self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        else:
            self.canvas.setMinimumHeight(needed_h)
            self.canvas.setMaximumHeight(QWIDGETSIZE_MAX)
            self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout_top_bar(self.width())
        self._update_canvas_height()

    # -----------------------------------------------------------------
    # State & Redraw logic
    # -----------------------------------------------------------------

    def set_result(self, result: Optional[ComparisonResult]):
        """Store result and redraw comparison plot."""
        self._result = result
        has_res = result is not None
        self.top_container.setEnabled(has_res)
        self.toolbar.setEnabled(has_res)
        self.save_fig_btn.setEnabled(has_res)
        self.export_csv_btn.setEnabled(has_res)
        self.export_inp_btn.setEnabled(has_res)

        if has_res:
            self._selected_states = default_states(result)
            self._build_states_menu()
        else:
            self._selected_states = []
            self.states_menu.clear()

        self.redraw()

    def clear(self):
        """Clear the plot canvas and reset state."""
        self._result = None
        self._selected_states = []
        self.states_menu.clear()
        self.top_container.setEnabled(False)
        self.toolbar.setEnabled(False)
        self.save_fig_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)
        self.export_inp_btn.setEnabled(False)
        draw_comparison(self.fig, None)

    def redraw(self):
        """Redraw plot using stored result and current options (no re-simulation)."""
        self._update_canvas_height()
        opts = {
            "selected_states": list(self._selected_states),
            "head_mode": self._head_mode,
            "show_dirac": self._show_dirac,
        }
        draw_comparison(self.fig, self._result, opts)

    def _build_states_menu(self):
        """Populate the States dropdown checklist with actions."""
        self.states_menu.clear()
        if self._result is None:
            return

        all_states = list(self._result.dae.labels)

        # Quick actions
        act_matlab = self.states_menu.addAction("MATLAB five")
        act_matlab.triggered.connect(self._select_matlab_five)

        act_flows = self.states_menu.addAction("All flows")
        act_flows.triggered.connect(self._select_all_flows)

        act_all = self.states_menu.addAction("All states")
        act_all.triggered.connect(self._select_all_states)

        self.states_menu.addSeparator()

        self._state_actions: Dict[str, QAction] = {}
        for s in all_states:
            act = QAction(s, self.states_menu)
            act.setCheckable(True)
            act.setChecked(s in self._selected_states)
            act.toggled.connect(lambda chk, state=s: self._on_state_toggled(state, chk))
            self.states_menu.addAction(act)
            self._state_actions[s] = act

    def _on_state_toggled(self, state: str, checked: bool):
        if checked and state not in self._selected_states:
            self._selected_states.append(state)
        elif not checked and state in self._selected_states:
            self._selected_states.remove(state)
        self.redraw()

    def _select_matlab_five(self):
        if self._result is None:
            return
        m5 = ["Pipe_1_Flow", "Pipe_2_Flow", "Pipe_1_R", "Pipe_2_L", "Pipe_2_R"]
        self._selected_states = [s for s in m5 if s in self._result.dae.labels]
        self._update_menu_checkboxes()
        self.redraw()

    def _select_all_flows(self):
        if self._result is None:
            return
        self._selected_states = [s for s in self._result.dae.labels if "Flow" in s]
        self._update_menu_checkboxes()
        self.redraw()

    def _select_all_states(self):
        if self._result is None:
            return
        self._selected_states = list(self._result.dae.labels)
        self._update_menu_checkboxes()
        self.redraw()

    def _update_menu_checkboxes(self):
        if hasattr(self, "_state_actions"):
            for s, act in self._state_actions.items():
                act.blockSignals(True)
                act.setChecked(s in self._selected_states)
                act.blockSignals(False)

    def _on_head_mode_changed(self):
        self._head_mode = "total" if self.radio_total.isChecked() else "pressure"
        self.redraw()

    def _on_dirac_toggled(self, checked: bool):
        self._show_dirac = checked
        self.redraw()

    # -----------------------------------------------------------------
    # Export Actions
    # -----------------------------------------------------------------

    def save_figure_action(self):
        """Export high-resolution comparison plot to PNG, PDF, or SVG."""
        if self._result is None:
            QMessageBox.information(self, "No Results", "Run a comparison first to export the plot.")
            return

        default_name = f"comparison_{self._default_filename_slug()}.png"
        path_str, sel_filter = QFileDialog.getSaveFileName(
            self,
            "Save Comparison Figure",
            default_name,
            "PNG Images (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*.*)",
        )
        if not path_str:
            return

        try:
            path = Path(path_str)
            dpi = 200 if path.suffix.lower() == ".png" else "figure"
            self.fig.savefig(
                path,
                dpi=dpi,
                facecolor="white",
                edgecolor="none",
                bbox_inches="tight",
            )
            msg = f"Saved figure to {path.name}"
            self.status_message.emit(msg)
        except Exception as e:
            QMessageBox.critical(self, "Save Figure Error", f"Failed to save figure: {e}")

    def export_csv_action(self):
        """Export time-series comparison data and Dirac impulses as CSV."""
        if self._result is None:
            QMessageBox.information(self, "No Results", "Run a comparison first to export data.")
            return

        default_name = f"comparison_{self._default_filename_slug()}.csv"
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export Comparison Data (CSV)",
            default_name,
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not path_str:
            return

        try:
            main_path = Path(path_str)
            self._write_csv_data(main_path)

            # Secondary impulses file: <name>_impulses.csv
            stem = main_path.stem
            imp_path = main_path.parent / f"{stem}_impulses.csv"
            self._write_csv_impulses(imp_path)

            msg = f"Exported comparison CSV and impulses to {main_path.name}"
            self.status_message.emit(msg)
        except Exception as e:
            QMessageBox.critical(self, "Export CSV Error", f"Failed to export CSV: {e}")

    def _write_csv_data(self, path: Path):
        """Write time, DAE, EPANET, diff columns sampled at EPANET report times."""
        res = self._result
        assert res is not None

        from core.plotdata import _strictly_increasing_dae_time_series
        tall, Xall = _strictly_increasing_dae_time_series(res)
        te = res.epanet.t

        states = list(res.dae.labels)
        header = ["time"]
        for s in states:
            header.extend([f"{s}_DAE", f"{s}_EPANET", f"{s}_diff"])

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for k, t_val in enumerate(te):
                row = [f"{t_val:.6f}"]
                for col_idx, s in enumerate(states):
                    dae_val = float(np.interp(t_val, tall, Xall[:, col_idx]))
                    ep_val = float(res.epanet.series(s)[k])
                    diff_val = dae_val - ep_val
                    row.extend([f"{dae_val:.6f}", f"{ep_val:.6f}", f"{diff_val:.6f}"])
                writer.writerow(row)

    def _write_csv_impulses(self, path: Path):
        """Write impulse records to CSV."""
        res = self._result
        assert res is not None

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "state", "coefficient (m*s)"])
            for imp in res.dae.impulses:
                for idx, lbl in enumerate(res.dae.labels):
                    coeff = float(imp.coefficients[idx])
                    if abs(coeff) > 1e-9:
                        writer.writerow([f"{imp.time:.6f}", lbl, f"{coeff:.6f}"])

    def export_inp_action(self):
        """Export the exact EPANET input file used in this run."""
        if self._result is None:
            QMessageBox.information(self, "No Results", "Run a comparison first to export the .inp file.")
            return

        default_name = f"comparison_{self._default_filename_slug()}.inp"
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export EPANET Input File",
            default_name,
            "EPANET Input Files (*.inp);;All Files (*.*)",
        )
        if not path_str:
            return

        try:
            inp_text = getattr(self._result.epanet, "inp_text", None)
            if not inp_text:
                from core.epanet_export import export_epanet
                inp_text = export_epanet(
                    self._result.snapshot,
                    self._result.scenario,
                    self._result.settings,
                )

            with open(path_str, "w", encoding="utf-8") as f:
                f.write(inp_text)

            msg = f"Exported EPANET input file to {Path(path_str).name}"
            self.status_message.emit(msg)
        except Exception as e:
            QMessageBox.critical(self, "Export INP Error", f"Failed to export EPANET .inp file: {e}")

    def _default_filename_slug(self) -> str:
        if self._result and self._result.scenario and self._result.scenario.title:
            slug = self._result.scenario.title.lower().replace(" ", "_")
            # Keep alphanumeric and underscores
            cleaned = "".join(c for c in slug if c.isalnum() or c == "_")
            if "specimen_1" in cleaned or "valve_1_closed" in cleaned:
                return "scenario1"
            return cleaned[:25] or "run"
        return "run"
