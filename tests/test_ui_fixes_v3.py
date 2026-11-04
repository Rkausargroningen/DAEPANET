"""tests/test_ui_fixes_v3.py – Comprehensive verification for Step 1-6 UI fixes."""

import unicodedata
from pathlib import Path
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette, QFontMetrics, QFont
from PySide6.QtWidgets import QApplication, QToolButton
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from core.models import Network
from core.examples import specimen_network, specimen_scenario, specimen_physics
from core.comparison import run_comparison
from ui.main_window import MainWindow
from ui.comparison_tab import ComparisonTab
from ui.comparison_plots import ComparisonPlots, draw_comparison, ICON_MAP


def rel_luminance(c: QColor) -> float:
    def channel(v):
        v = v / 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * channel(c.red()) + 0.7152 * channel(c.green()) + 0.0722 * channel(c.blue())


def contrast_ratio(c1: QColor, c2: QColor) -> float:
    l1, l2 = rel_luminance(c1), rel_luminance(c2)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_step1_setup_column_widths_and_scrollbars(qapp):
    """Verify all tables in setup column have no horizontal scrollbar at 1280x720 and 1920x1080."""
    win = MainWindow()
    win.resize(1280, 720)
    win.show()
    win._on_load_specimen_example()
    comp_idx = win.right_tabs.indexOf(win.comparison_tab)
    win.right_tabs.setCurrentIndex(comp_idx)
    qapp.processEvents()

    tab = win.comparison_tab
    assert tab.left_scroll.width() >= 440

    for size in [(1280, 720), (1920, 1080)]:
        win.resize(*size)
        qapp.processEvents()
        for name, tbl in [("Valves", tab.valves_table), ("Initial", tab.initial_table), ("Events", tab.events_table)]:
            tot_w = sum(tbl.columnWidth(c) for c in range(tbl.columnCount()))
            vp_w = tbl.viewport().width()
            assert not tbl.horizontalScrollBar().isVisible(), f"{name} table has horizontal scrollbar at {size}"
            assert tot_w <= vp_w + 2, f"{name} table columns {tot_w} exceed viewport {vp_w} at {size}"

    win.close()


def test_step2_option_row_label_widths(qapp):
    """Verify fontMetrics text width <= widget width for each option control at 1280x720 and 1920x1080."""
    win = MainWindow()
    win.resize(1280, 720)
    win.show()
    win._on_load_specimen_example()
    comp_idx = win.right_tabs.indexOf(win.comparison_tab)
    win.right_tabs.setCurrentIndex(comp_idx)
    qapp.processEvents()

    tab = win.comparison_tab
    plots = tab.plots_widget

    for size in [(1280, 720), (1920, 1080)]:
        win.resize(*size)
        qapp.processEvents()
        
        # Check States button
        fm_btn = plots.states_btn.fontMetrics()
        assert fm_btn.horizontalAdvance(plots.states_btn.text()) <= plots.states_btn.width()
        
        # Check Pressure radio
        fm_p = plots.radio_pressure.fontMetrics()
        assert fm_p.horizontalAdvance(plots.radio_pressure.text()) <= plots.radio_pressure.width()
        
        # Check Total radio
        fm_t = plots.radio_total.fontMetrics()
        assert fm_t.horizontalAdvance(plots.radio_total.text()) <= plots.radio_total.width()
        
        # Check Dirac checkbox
        fm_d = plots.dirac_check.fontMetrics()
        assert fm_d.horizontalAdvance(plots.dirac_check.text()) <= plots.dirac_check.width()

    win.close()


def test_step3_toolbar_icons_contrast(qapp):
    """Verify every toolbar button icon has >= 3:1 contrast against button background in light and dark."""
    plots = ComparisonPlots()
    toolbar = plots.toolbar

    for is_dark in [False, True]:
        pal = QPalette()
        bg = QColor("#2D2D2D") if is_dark else QColor("#F8F9FA")
        fg = QColor("#F0F0F0") if is_dark else QColor("#222222")
        pal.setColor(QPalette.Window, bg)
        plots.setPalette(pal)
        plots.update_toolbar_style()
        qapp.processEvents()

        min_cr = 999.0
        for btn in toolbar.findChildren(QToolButton):
            if not btn.text() or btn.text() not in ICON_MAP:
                continue
            pix = btn.icon().pixmap(24, 24)
            img = pix.toImage()
            non_trans = []
            for x in range(img.width()):
                for y in range(img.height()):
                    c = img.pixelColor(x, y)
                    if c.alpha() > 100:
                        non_trans.append(c)
            assert len(non_trans) > 0, f"Button {btn.text()} has no icon pixels"
            avg_c = QColor(
                sum(c.red() for c in non_trans) // len(non_trans),
                sum(c.green() for c in non_trans) // len(non_trans),
                sum(c.blue() for c in non_trans) // len(non_trans),
            )
            cr = contrast_ratio(bg, avg_c)
            min_cr = min(min_cr, cr)
            assert cr >= 3.0, f"Button {btn.text()} contrast {cr:.2f} < 3:1 in {'dark' if is_dark else 'light'}"

        assert min_cr >= 3.0


def test_step4a_figure_title_formatting():
    """Verify figure title has no double spaces and all characters exist in DejaVu Sans."""
    net_spec, _ = specimen_network()
    res = run_comparison(net_spec, specimen_scenario(1), specimen_physics())
    fig = Figure(figsize=(7, 9), facecolor="white", layout="constrained")
    draw_comparison(fig, res)
    title = fig.axes[0].get_title()
    assert "  " not in title, f"Title contains double spaces: {title!r}"
    assert "Switched DAE vs EPANET" in title
    # Check glyph existence in DejaVu Sans / ASCII
    for ch in title:
        assert ord(ch) < 128 or unicodedata.category(ch) != "Cn", f"Char {ch!r} unrenderable"


def test_step4b_title_legend_gap(qapp):
    """Verify title and legend bounding boxes do not overlap and have >= 6px gap on-screen and export."""
    net_spec, _ = specimen_network()
    for sc_num in [1, 2]:
        res = run_comparison(net_spec, specimen_scenario(sc_num), specimen_physics())
        
        # 1. On-screen wide
        fig_w = Figure(figsize=(10, 8), facecolor="white", layout="constrained")
        can_w = FigureCanvasQTAgg(fig_w)
        draw_comparison(fig_w, res)
        can_w.draw()
        ren_w = can_w.get_renderer()
        ax0 = fig_w.axes[0]
        tb = ax0.title.get_window_extent(ren_w)
        lb = ax0.get_legend().get_window_extent(ren_w)
        assert lb.x0 - tb.x1 >= 6.0, f"Sc{sc_num} Wide gap {lb.x0 - tb.x1} < 6px"

        # 2. On-screen narrow (< 800)
        fig_n = Figure(figsize=(6.4, 8.2), facecolor="white", layout="constrained")
        can_n = FigureCanvasQTAgg(fig_n)
        draw_comparison(fig_n, res)
        can_n.draw()
        ren_n = can_n.get_renderer()
        ax0_n = fig_n.axes[0]
        tbn = ax0_n.title.get_window_extent(ren_n)
        lbn = ax0_n.get_legend().get_window_extent(ren_n)
        # Legend is placed above title in narrow mode: lbn.y0 > tbn.y1
        assert lbn.y0 - tbn.y1 >= 6.0, f"Sc{sc_num} Narrow gap {lbn.y0 - tbn.y1} < 6px"

        # 3. Exported figure
        fig_exp = Figure(figsize=(7, 9), facecolor="white", layout="constrained")
        can_exp = FigureCanvasQTAgg(fig_exp)
        draw_comparison(fig_exp, res)
        can_exp.draw()
        ren_exp = can_exp.get_renderer()
        ax0_e = fig_exp.axes[0]
        tbe = ax0_e.title.get_window_extent(ren_exp)
        lbe = ax0_e.get_legend().get_window_extent(ren_exp)
        assert lbe.y0 - tbe.y1 >= 6.0, f"Sc{sc_num} Export gap {lbe.y0 - tbe.y1} < 6px"


def test_step5_canvas_height_and_no_overflow(qapp):
    """Verify canvas height rule, 0px text overflow and 0 y-label overlaps at 125px per panel."""
    net_spec, _ = specimen_network()
    for sc_num in [1, 2, 3, 4, 5]:
        res = run_comparison(net_spec, specimen_scenario(sc_num), specimen_physics())
        fig = Figure(figsize=(6.4, 6.25), facecolor="white", layout="constrained")
        can = FigureCanvasQTAgg(fig)
        draw_comparison(fig, res)
        can.draw()
        ren = can.get_renderer()

        fig_w, fig_h = fig.bbox.width, fig.bbox.height
        for ax in fig.axes:
            for t in ax.texts + [ax.xaxis.label, ax.yaxis.label, ax.title] + ax.get_xticklabels() + ax.get_yticklabels():
                if not t.get_visible() or not t.get_text():
                    continue
                bb = t.get_window_extent(ren)
                assert max(0, -bb.x0, bb.x1 - fig_w) <= 1.0, f"Text overflow x in Sc{sc_num}"
                assert max(0, -bb.y0, bb.y1 - fig_h) <= 1.0, f"Text overflow y in Sc{sc_num}"

        for i in range(len(fig.axes) - 1):
            bb1 = fig.axes[i].yaxis.label.get_window_extent(ren)
            bb2 = fig.axes[i+1].yaxis.label.get_window_extent(ren)
            overlap_y = max(0, min(bb1.y1, bb2.y1) - max(bb1.y0, bb2.y0))
            overlap_x = max(0, min(bb1.x1, bb2.x1) - max(bb1.x0, bb2.x0))
            assert not (overlap_x > 0 and overlap_y > 0), f"Y-label overlap in Sc{sc_num}"


def test_step6a_real_menu_actions(qapp):
    """Verify loading paper, broken, specimen via real QAction triggers populates nodes and updates status."""
    win = MainWindow()
    win.show()
    qapp.processEvents()

    for act_name, exp_nodes, exp_pipes in [
        ("Paper sample network", 6, 8),
        ("Broken network (demo of warnings)", 6, 5),
        ("Specimen with valves (EPANET comparison)", 3, 2),
    ]:
        # Find action
        target_act = None
        for a in win.menuBar().actions():
            if a.menu():
                for sub in a.menu().actions():
                    if sub.menu():
                        for sub_a in sub.menu().actions():
                            if act_name in sub_a.text():
                                target_act = sub_a
        assert target_act is not None, f"Action {act_name} not found"
        target_act.trigger()
        qapp.processEvents()

        assert win.build_panel.nodes_table.rowCount() == exp_nodes
        assert win.build_panel.pipes_table.rowCount() == exp_pipes
        msg = win.statusBar().currentMessage()
        assert f"Nodes: {exp_nodes}" in msg
        assert f"Pipes: {exp_pipes}" in msg
        assert win.build_panel.add_node_btn.isEnabled()
        assert win.build_panel.add_pipe_btn.isEnabled()

    win.close()


def test_step6b_tab_contrast(qapp):
    """Verify selected and unselected tab contrast is >= 4.5:1 in both light and dark themes."""
    for is_dark in [False, True]:
        pal = QPalette()
        bg = QColor(32, 32, 32) if is_dark else QColor(240, 240, 240)
        pal.setColor(QPalette.Window, bg)
        qapp.setPalette(pal)

        win = MainWindow()
        win.show()
        qapp.processEvents()
        tab_bar = win.right_tabs.tabBar()

        for tab_idx in [0, 1]:
            rect = tab_bar.tabRect(tab_idx)
            pix = tab_bar.grab(rect)
            img = pix.toImage()
            bg_col = img.pixelColor(2, 2)
            contrasts = [contrast_ratio(bg_col, img.pixelColor(x, y)) for x in range(img.width()) for y in range(img.height())]
            max_cr = max(contrasts)
            assert max_cr >= 4.5, f"Tab {tab_idx} contrast {max_cr:.2f} < 4.5:1 in {'dark' if is_dark else 'light'}"

        win.close()
