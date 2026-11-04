"""Tests for Step 1 bug fixes:
1a: No overlapping combo boxes in ComparisonTab initial valve states
1b: clear_results() resets all result widgets when switching networks
1c: Graph tab contrast WCAG >= 4.5 in dark and light themes
1d: Broken network example readable layout (min node distance >= 0.4)
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import math
import pytest
from PySide6.QtCore import Qt, QPoint, QRect
from PySide6.QtGui import QPalette, QColor, QImage
from PySide6.QtWidgets import QApplication, QComboBox

from core.models import Network
from core.examples import (
    example_paper_network,
    example_broken_network,
    example_broken_network_positions,
    specimen_network,
    specimen_scenario,
    specimen_physics,
)
from core.comparison import run_comparison
from ui.main_window import MainWindow
from ui.comparison_tab import ComparisonTab
from ui.graph_view import GraphView, PipeItem, NodeItem, model_to_scene


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# 1a: Overlapping combo box fix
# ---------------------------------------------------------------------------

def test_1a_no_overlapping_combo_boxes(qapp):
    net, pos = specimen_network()
    tab = ComparisonTab(net)
    tab.resize(1000, 700)
    tab.show()
    qapp.processEvents()

    def check_combos():
        qapp.processEvents()
        # 1. Valve-name cells must have no cell widget
        for r in range(tab.initial_table.rowCount()):
            assert tab.initial_table.cellWidget(r, 0) is None, f"Row {r} col 0 has unexpected cell widget"

        # 2. Collect all visible QComboBox widgets of the tab
        visible_combos = [c for c in tab.findChildren(QComboBox) if c.isVisible()]
        rects = []
        for c in visible_combos:
            top_left = c.mapTo(tab, QPoint(0, 0))
            rect = QRect(top_left, c.size())
            rects.append((c, rect))

        # 3. Assert that no two intersect
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                c1, r1 = rects[i]
                c2, r2 = rects[j]
                intersect = r1.intersected(r2)
                assert intersect.isEmpty(), (
                    f"Combos intersect: {c1} (items: {[c1.itemText(k) for k in range(c1.count())]}) "
                    f"at {r1} and {c2} at {r2}"
                )

    # Initial state
    check_combos()

    # Switch presets
    for p_idx in [1, 2, 3, 4, 5, 0, 1]:
        tab.preset_combo.setCurrentIndex(p_idx)
        check_combos()

    # Toggle valves
    tab.valves_table.item(0, 2).setCheckState(Qt.Checked) # P1 start valve
    tab._on_valve_cell_changed(0, 2)
    check_combos()

    tab.valves_table.item(0, 2).setCheckState(Qt.Unchecked)
    tab._on_valve_cell_changed(0, 2)
    check_combos()


# ---------------------------------------------------------------------------
# 1b: Stale text & clear_results
# ---------------------------------------------------------------------------

def test_1b_clear_results_on_network_change(qapp):
    win = MainWindow()
    win.resize(1280, 720)
    win.show()
    qapp.processEvents()

    # 1. Load specimen example and run comparison
    win._on_load_specimen_example()
    qapp.processEvents()

    tab = win.comparison_tab
    # Run synchronously for test
    scen = tab.get_scenario()
    phys = tab.get_physics()
    res = run_comparison(tab.network, scen, phys, positions=win.graph_view.get_positions())
    tab._on_run_finished(res)
    qapp.processEvents()

    # Check results are populated
    assert tab.eq_table.rowCount() > 0
    assert tab.params_table.rowCount() > 0
    assert "Dirac impulse entries" in tab.imp_summary_label.text()
    assert tab._last_result is not None

    # 2. Load paper example
    win._on_load_example(example_paper_network)
    qapp.processEvents()

    # 3. Assert all result widgets are empty / reset
    assert tab._last_result is None
    assert tab.eq_table.rowCount() == 0
    assert tab.imp_table.rowCount() == 0
    assert tab.params_table.rowCount() == 0
    assert "Run comparison" in tab.eq_summary_label.text()
    assert tab.imp_summary_label.text() == "EPANET: none (EPANET has no impulses)."
    assert not tab.outdated_banner.isVisible()


# ---------------------------------------------------------------------------
# 1c: Graph tab contrast WCAG >= 4.5 in dark and light themes
# ---------------------------------------------------------------------------

def _relative_luminance(c: QColor) -> float:
    def channel(v: int) -> float:
        s = v / 255.0
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4
    return 0.2126 * channel(c.red()) + 0.7152 * channel(c.green()) + 0.0722 * channel(c.blue())


def _contrast_ratio(c1: QColor, c2: QColor) -> float:
    l1 = _relative_luminance(c1)
    l2 = _relative_luminance(c2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def test_1c_graph_contrast_dark_and_light(qapp):
    net, pos = specimen_network()
    view = GraphView()
    view.set_positions(pos)
    view.draw_network(net)
    view.resize(800, 600)
    view.show()
    view.fit_in_view()
    qapp.processEvents()

    for theme_name, bg_color in [("dark", QColor(30, 30, 30)), ("light", QColor(240, 240, 240))]:
        palette = QPalette()
        palette.setColor(QPalette.Window, bg_color)
        palette.setColor(QPalette.Base, bg_color)
        view.setPalette(palette)
        view.view.viewport().setPalette(palette)
        qapp.processEvents()

        # Render viewport to image
        img = view.view.viewport().grab().toImage()
        dpr = img.devicePixelRatio()

        # Canvas background should be white (#FFFFFF)
        p1_item = next(it for it in view.scene.items() if isinstance(it, PipeItem) and it.pipe_id == "P1")
        assert p1_item is not None

        # Sample at 35% along the pipe curve (away from the middle label box)
        pt_scene = p1_item.path.pointAtPercent(0.35)
        vp_pt = view.view.mapFromScene(pt_scene)

        px = int(vp_pt.x() * dpr)
        py = int(vp_pt.y() * dpr)

        pipe_pixel = img.pixelColor(px, py)
        bg_pixel = img.pixelColor(px, max(0, py - int(25 * dpr)))

        cr = _contrast_ratio(pipe_pixel, bg_pixel)
        assert cr >= 4.5, (
            f"Contrast ratio in {theme_name} theme is {cr:.2f} < 4.5 "
            f"(pipe: {pipe_pixel.name()}, bg: {bg_pixel.name()})"
        )


# ---------------------------------------------------------------------------
# 1d: Broken network layout
# ---------------------------------------------------------------------------

def test_1d_broken_network_layout():
    pos = example_broken_network_positions()
    assert len(pos) == 6
    nodes = list(pos.keys())
    min_dist = float("inf")
    closest_pair = None

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            n1, n2 = nodes[i], nodes[j]
            d = math.hypot(pos[n1][0] - pos[n2][0], pos[n1][1] - pos[n2][1])
            if d < min_dist:
                min_dist = d
                closest_pair = (n1, n2)

    # Minimum distance between any two node centres in model units >= 0.4 (actual is 0.5)
    assert min_dist >= 0.4, f"Min distance {min_dist:.4f} between {closest_pair} is < 0.4"
