"""tests/test_plot_empty_state.py – Test empty-state hint text wrapping and margins."""

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from ui.comparison_plots import draw_comparison


def test_empty_state_wraps_within_80_percent():
    panel_heights = [100, 120, 145, 200]
    widths = [560, 700, 900, 1100, 1400]

    for w in widths:
        for h in panel_heights:
            fig = Figure(layout="constrained")
            FigureCanvasAgg(fig)
            fig.set_size_inches(w / 100.0, (5 * h) / 100.0)
            draw_comparison(fig, None)
            renderer = fig.canvas.get_renderer()
            fig_bb = fig.get_window_extent(renderer)

            assert len(fig.axes) == 1
            ax = fig.axes[0]
            assert len(ax.texts) == 1
            txt = ax.texts[0]
            txt_bb = txt.get_window_extent(renderer)

            # Assert wraps at most 80% of canvas width
            assert txt_bb.width <= 0.80 * fig_bb.width + 1.0, (
                f"Width {txt_bb.width:.1f} > 0.80 * {fig_bb.width:.1f} at w={w}, h={h}"
            )
            # Assert text does not touch or exceed right edge
            assert txt_bb.x1 <= fig_bb.width - 5.0, (
                f"Text x1={txt_bb.x1:.1f} too close to right edge {fig_bb.width:.1f} at w={w}, h={h}"
            )
            assert txt_bb.x0 >= 5.0, (
                f"Text x0={txt_bb.x0:.1f} too close to left edge at w={w}, h={h}"
            )
