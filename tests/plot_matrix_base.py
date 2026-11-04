"""tests/plot_matrix_base.py – Shared test runner for plot layout validation."""

from typing import Dict, Any
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from ui.comparison_plots import draw_comparison


def run_matrix_subset(res, sc_name: str, dpr: float, mode: str) -> int:
    panel_heights = [100, 120, 145, 200]
    widths = [560, 700, 900, 1100, 1400]

    fig = Figure(layout="constrained")
    FigureCanvasAgg(fig)

    tested = 0
    for w_px in widths:
        for h_panel in panel_heights:
            fig.set_dpi(100.0 * dpr)
            fig.set_size_inches(w_px / 100.0, (5 * h_panel) / 100.0)

            opts: Dict[str, Any] = {
                "selected_states": list(res.dae.labels[:5]),
                "head_mode": mode,
                "show_dirac": True,
                "max_points": 50,
            }
            draw_comparison(fig, res, opts)
            renderer = fig.canvas.get_renderer()
            assert_plot_layout(fig, renderer, sc_name, mode, dpr, w_px, h_panel)
            tested += 1

    return tested


def assert_plot_layout(fig, renderer, sc_name: str, mode: str, dpr: float, w: int, h: int):
    fig_bbox = fig.get_window_extent(renderer)
    W, H = fig_bbox.width, fig_bbox.height

    # 1. Y-labels: no two adjacent y-labels overlap and inside figure
    axes = [ax for ax in fig.axes if ax.get_ylabel()]
    for i in range(len(axes) - 1):
        bb_top = axes[i].yaxis.label.get_window_extent(renderer)
        bb_bot = axes[i + 1].yaxis.label.get_window_extent(renderer)
        assert bb_top.y0 >= bb_bot.y1 - 1.0, (
            f"[{sc_name} {mode} dpr={dpr} w={w} h={h}] Y-label overlap: "
            f"top '{axes[i].get_ylabel()}' (y0={bb_top.y0:.1f}) "
            f"bot '{axes[i+1].get_ylabel()}' (y1={bb_bot.y1:.1f})"
        )
        assert bb_top.x0 >= -1.0 and bb_top.x1 <= W + 1.0
        assert bb_top.y0 >= -1.0 and bb_top.y1 <= H + 1.0

    if axes:
        bb_last = axes[-1].yaxis.label.get_window_extent(renderer)
        assert bb_last.x0 >= -1.0 and bb_last.x1 <= W + 1.0
        assert bb_last.y0 >= -1.0 and bb_last.y1 <= H + 1.0

    # 2. Legend: inside figure and does not overlap title
    leg = fig.axes[0].get_legend() if fig.axes else None
    if leg:
        leg_bb = leg.get_window_extent(renderer)
        assert leg_bb.x1 <= W + 1.0, f"[{sc_name} {mode} dpr={dpr} w={w} h={h}] Legend clipped right: x1={leg_bb.x1:.1f} > W={W:.1f}"
        assert leg_bb.x0 >= -1.0, f"[{sc_name} {mode} dpr={dpr} w={w} h={h}] Legend clipped left: x0={leg_bb.x0:.1f} < 0"
        assert leg_bb.y1 <= H + 1.0, f"[{sc_name} {mode} dpr={dpr} w={w} h={h}] Legend clipped top: y1={leg_bb.y1:.1f} > H={H:.1f}"
        assert leg_bb.y0 >= -1.0, f"[{sc_name} {mode} dpr={dpr} w={w} h={h}] Legend clipped bottom: y0={leg_bb.y0:.1f} < 0"

        title_bb = fig.axes[0].title.get_window_extent(renderer)
        if title_bb.width > 0 and title_bb.height > 0:
            overlap_x = max(0.0, min(leg_bb.x1, title_bb.x1) - max(leg_bb.x0, title_bb.x0))
            overlap_y = max(0.0, min(leg_bb.y1, title_bb.y1) - max(leg_bb.y0, title_bb.y0))
            assert overlap_x * overlap_y == 0.0 or overlap_x < 2.0 or overlap_y < 2.0, (
                f"[{sc_name} {mode} dpr={dpr} w={w} h={h}] Legend overlaps title: leg={leg_bb}, title={title_bb}"
            )

    # 3. Titles and Dirac labels inside figure
    for ax in fig.axes:
        title_bb = ax.title.get_window_extent(renderer)
        if title_bb.width > 0 and title_bb.height > 0:
            assert title_bb.x0 >= -1.0 and title_bb.x1 <= W + 1.0
            assert title_bb.y0 >= -1.0 and title_bb.y1 <= H + 1.0
        for txt in ax.texts:
            txt_bb = txt.get_window_extent(renderer)
            assert txt_bb.x0 >= -1.0 and txt_bb.x1 <= W + 1.0
            assert txt_bb.y0 >= -1.0 and txt_bb.y1 <= H + 1.0
