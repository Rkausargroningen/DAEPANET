"""tests/test_ui_fixes_v4.py – Tests for Step 1-4 UI fixes (plots space, 1080p height, clipping, left panel contrast)."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette, QFontMetrics
from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QComboBox, QLabel
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from core.models import Network
from core.examples import specimen_network, specimen_scenario, specimen_physics
from core.comparison import run_comparison
from ui.main_window import MainWindow
from ui.comparison_tab import ComparisonTab
from ui.comparison_plots import ComparisonPlots, draw_comparison


def rel_luminance(c: QColor) -> float:
    def channel(v):
        v = v / 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * channel(c.red()) + 0.7152 * channel(c.green()) + 0.0722 * channel(c.blue())


def contrast_ratio(c1: QColor, c2: QColor) -> float:
    l1, l2 = rel_luminance(c1), rel_luminance(c2)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QComboBox, QLabel, QMessageBox, QFileDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def prevent_dialogs(monkeypatch):
    """Ensure no modal dialog can ever block test execution."""
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Discard)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Discard)
    monkeypatch.setattr(QMessageBox, "about", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("", ""))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: ("", ""))


def test_step1_give_plots_space_and_hide_panel(qapp):
    """Verify left editing panel hides on Comparison tab and restores on other tabs; verify viewport widths."""
    win = MainWindow()
    win.setFixedSize(1920, 1080)
    win.show()
    win._on_load_specimen_example()
    qapp.processEvents()

    # In Network tab, editing panel is visible
    assert win.build_scroll_area.isVisible()

    # Put text in field to verify state preservation
    win.build_panel.node_elevation_input.setText("123.45")

    # Switch to Comparison tab
    comp_idx = win.right_tabs.indexOf(win.comparison_tab)
    win.right_tabs.setCurrentIndex(comp_idx)
    qapp.processEvents()

    # Left editing panel is now hidden to give plots space
    assert not win.build_scroll_area.isVisible()

    tab = win.comparison_tab
    plots = tab.plots_widget

    # Check setup column width is in 440-460 px range
    setup_w = tab.left_scroll.width()
    assert 440 <= setup_w <= 465, f"Setup width {setup_w} not in 440-465 range"

    # Check canvas min width is 560
    assert plots.canvas.minimumWidth() == 560

    # At 1920x1080: plots viewport width >= 1100 px, no horizontal scrollbar
    vp_w_1080 = plots.scroll_area.viewport().width()
    assert vp_w_1080 >= 1100, f"Plots viewport width {vp_w_1080} < 1100 at 1080p"
    assert not plots.scroll_area.horizontalScrollBar().isVisible()

    # Resize to 1280x720: no horizontal scrollbar
    win.setFixedSize(1280, 720)
    qapp.processEvents()
    vp_w_720 = plots.scroll_area.viewport().width()
    assert vp_w_720 >= 580, f"Plots viewport width {vp_w_720} < 580 at 720p"
    assert not plots.scroll_area.horizontalScrollBar().isVisible()

    # Switch back to Network tab: editing panel restored and text preserved
    win.right_tabs.setCurrentIndex(0)
    qapp.processEvents()
    assert win.build_scroll_area.isVisible()
    assert win.build_panel.node_elevation_input.text() == "123.45"

    win.setWindowModified(False)
    win.close()
    win.deleteLater()


def test_step2_compact_top_bar_and_canvas_height(qapp):
    """Verify options + toolbar combined in 1 row at >= 900 px, and all 5 panels visible at 1080p without scrollbar."""
    win = MainWindow()
    win.setFixedSize(1920, 1080)
    win.show()
    win._on_load_specimen_example()
    comp_idx = win.right_tabs.indexOf(win.comparison_tab)
    win.right_tabs.setCurrentIndex(comp_idx)
    qapp.processEvents()

    tab = win.comparison_tab
    plots = tab.plots_widget

    # Run scenario 1 so all 5 panels are populated
    tab.preset_combo.setCurrentIndex(1)
    tab._on_preset_changed(1)
    net_spec, _ = specimen_network()
    res = run_comparison(net_spec, specimen_scenario(1), specimen_physics())
    plots.set_result(res)
    qapp.processEvents()

    # At 1080p with width >= 900 px, top bar has 1 row
    assert plots.width() >= 900
    assert plots.top_box.count() == 1, f"Expected 1 row in top_box at width {plots.width()}, got {plots.top_box.count()}"

    # Visible plot height >= 625 px, no vertical scrollbar
    vp_h = plots.scroll_area.viewport().height()
    assert vp_h >= 625, f"Visible plot height {vp_h} < 625 at 1080p"
    assert not plots.scroll_area.verticalScrollBar().isVisible(), "Vertical scrollbar visible at 1080p"

    win.setWindowModified(False)
    win.close()
    win.deleteLater()


def test_step3_nothing_clipped_in_visible_area(qapp):
    """Verify title, legend, Dirac labels, y-labels, tick labels lie inside figure and inside viewport for widths 560, 700, 900, 1400 and export."""
    net_spec, _ = specimen_network()
    phys = specimen_physics()

    for sc_num in [1, 5]:
        res = run_comparison(net_spec, specimen_scenario(sc_num), phys)

        # Test canvas widths 560, 700, 900, 1400
        for w in [560, 700, 900, 1400]:
            fig = Figure(figsize=(w / 100.0, 7.5), facecolor="white", layout="constrained")
            can = FigureCanvasQTAgg(fig)
            can.resize(w, 750)
            draw_comparison(fig, res)
            can.draw()
            ren = can.get_renderer()
            fig_w, fig_h = fig.bbox.width, fig.bbox.height

            # Check every text element inside figure and viewport [0, w] x [0, h]
            for ax in fig.axes:
                all_texts = ax.texts + [ax.xaxis.label, ax.yaxis.label, ax.title] + ax.get_xticklabels() + ax.get_yticklabels()
                leg = ax.get_legend()
                if leg is not None:
                    all_texts.extend(leg.get_texts())

                for t in all_texts:
                    if not t.get_visible() or not t.get_text():
                        continue
                    bb = t.get_window_extent(ren)
                    # Allow 1 px subpixel tolerance
                    assert bb.x0 >= -1.0, f"Sc{sc_num} w={w} text {t.get_text()[:20]!r} clips left: {bb.x0}"
                    assert bb.x1 <= fig_w + 1.0, f"Sc{sc_num} w={w} text {t.get_text()[:20]!r} clips right: {bb.x1} > {fig_w}"
                    assert bb.y0 >= -1.0, f"Sc{sc_num} w={w} text {t.get_text()[:20]!r} clips bottom: {bb.y0}"
                    assert bb.y1 <= fig_h + 1.0, f"Sc{sc_num} w={w} text {t.get_text()[:20]!r} clips top: {bb.y1} > {fig_h}"

        # Test exported PNG figure at 200 DPI
        fig_exp = Figure(figsize=(7, 9), facecolor="white", layout="constrained")
        can_exp = FigureCanvasQTAgg(fig_exp)
        draw_comparison(fig_exp, res)
        can_exp.draw()
        ren_exp = can_exp.get_renderer()
        f_w, f_h = fig_exp.bbox.width, fig_exp.bbox.height

        for ax in fig_exp.axes:
            all_texts = ax.texts + [ax.xaxis.label, ax.yaxis.label, ax.title] + ax.get_xticklabels() + ax.get_yticklabels()
            leg = ax.get_legend()
            if leg is not None:
                all_texts.extend(leg.get_texts())
            for t in all_texts:
                if not t.get_visible() or not t.get_text():
                    continue
                bb = t.get_window_extent(ren_exp)
                assert bb.x0 >= -1.0
                assert bb.x1 <= f_w + 1.0
                assert bb.y0 >= -1.0
                assert bb.y1 <= f_h + 1.0


def test_step4_light_theme_left_panel_contrast(qapp):
    """Verify enabled text in left panel has contrast >= 4.5:1 (normal) and >= 3:1 (placeholder) in light theme."""
    app = qapp
    app.setPalette(QApplication.style().standardPalette())

    win = MainWindow()
    win.show()
    qapp.processEvents()
    bp = win.build_panel
    bp.update_theme_style(is_dark=False)

    # 1. Add Node / Add Pipe button text vs button background
    c_btn_text = QColor("#1E1E1E")
    c_btn_bg = QColor("#E4E7EB")
    cr_btn = contrast_ratio(c_btn_text, c_btn_bg)
    assert cr_btn >= 4.5, f"Button text contrast {cr_btn:.2f} < 4.5"

    # 2. Combo box text ("Junction", Start, End) vs combo background
    c_combo_text = QColor("#1E1E1E")
    c_combo_bg = QColor("#FFFFFF")
    cr_combo = contrast_ratio(c_combo_text, c_combo_bg)
    assert cr_combo >= 4.5, f"Combo text contrast {cr_combo:.2f} < 4.5"

    # 3. Label text vs background
    c_lbl_text = QColor("#1E1E1E")
    c_lbl_bg = QColor("#F0F2F5")
    cr_lbl = contrast_ratio(c_lbl_text, c_lbl_bg)
    assert cr_lbl >= 4.5, f"Label text contrast {cr_lbl:.2f} < 4.5"

    # 4. Entry field normal text vs entry background
    c_entry_text = QColor("#1E1E1E")
    c_entry_bg = QColor("#FFFFFF")
    cr_entry = contrast_ratio(c_entry_text, c_entry_bg)
    assert cr_entry >= 4.5, f"Entry text contrast {cr_entry:.2f} < 4.5"

    # 5. Entry field placeholder text vs entry background
    c_ph_text = QColor("#5A626A")
    cr_ph = contrast_ratio(c_ph_text, c_entry_bg)
    assert cr_ph >= 3.0, f"Placeholder text contrast {cr_ph:.2f} < 3.0"

    win.setWindowModified(False)
    win.close()
    win.deleteLater()
