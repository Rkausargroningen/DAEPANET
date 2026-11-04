"""Tests for v2 polish:
Step 1: Plot readability (1a min span & tick formatting, 1b text inside bbox & no y-label overlap, 1c small window & setup toggle)
Step 2: Outdated banner signature-based reactivity
Step 3: Graph view legend clearance, export layout, and broken example distance ratio
"""

import math
import os
import pytest
from PySide6.QtCore import Qt, QPoint, QRectF
from PySide6.QtWidgets import QApplication

from core.models import Network
from core.examples import (
    specimen_network,
    specimen_scenario,
    specimen_physics,
    example_paper_network,
    example_broken_network,
    example_broken_network_positions,
)
from core.comparison import run_comparison, ComparisonResult
from ui.comparison_plots import ComparisonPlots, draw_comparison, apply_panel_span_rules
from ui.comparison_tab import ComparisonTab
from ui.graph_view import GraphView, NodeItem, PipeItem
from ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# Step 1a: Minimum visible range and tick formatting
# ---------------------------------------------------------------------------

def test_1a_synthetic_rule():
    """Unit test of the span rule on synthetic data."""
    # 1. Tiny pressure span
    # y = [42.0, 42.000002], max|y| = 42.000002.
    # min_span = max(0.005 * 42, 0.01) = max(0.21, 0.01) = 0.21.
    ymin, ymax = apply_panel_span_rules(
        current_ymin=42.0,
        current_ymax=42.000002,
        panel_max_abs_y=42.000002,
        is_flow=False,
        global_max_abs_flow=0.0,
    )
    span = ymax - ymin
    assert abs(span - 0.21) < 1e-6
    mid = 42.000001
    assert abs((ymin + ymax) / 2.0 - mid) < 1e-6

    # 2. Zero-flow panel
    ymin, ymax = apply_panel_span_rules(
        current_ymin=0.0,
        current_ymax=0.0,
        panel_max_abs_y=0.0,
        is_flow=True,
        global_max_abs_flow=0.0,
    )
    assert ymin == -1e-3
    assert ymax == 1e-3
    assert abs((ymax - ymin) - 2e-3) < 1e-9

    # 3. Flow panel with tiny variation, global_max_abs_flow = 0.5
    # min_span = 0.01 * 0.5 = 0.005
    ymin, ymax = apply_panel_span_rules(
        current_ymin=0.1,
        current_ymax=0.10001,
        panel_max_abs_y=0.10001,
        is_flow=True,
        global_max_abs_flow=0.5,
    )
    assert abs((ymax - ymin) - 0.005) < 1e-6

    # 4. Normal large transient span: unchanged
    ymin, ymax = apply_panel_span_rules(
        current_ymin=10.0,
        current_ymax=50.0,
        panel_max_abs_y=50.0,
        is_flow=False,
        global_max_abs_flow=0.0,
    )
    assert ymin == 10.0
    assert ymax == 50.0


def test_1a_scenarios_spans_and_limits(qapp):
    """Verify scenarios 2 and 5 have span >= 0.01 m and sc1 limits are unchanged."""
    net, _ = specimen_network()
    physics = specimen_physics()

    res1 = run_comparison(net, specimen_scenario(1), physics)
    res2 = run_comparison(net, specimen_scenario(2), physics)
    res5 = run_comparison(net, specimen_scenario(5), physics)

    plots2 = ComparisonPlots()
    plots2.set_result(res2)
    for ax in plots2.fig.axes:
        label = ax.get_ylabel()
        ymin, ymax = ax.get_ylim()
        span = ymax - ymin
        if any(v in label for v in ["x5", "x4", "x6", "Pipe_1_R", "Pipe_2_L", "Pipe_2_R"]):
            assert span >= 0.01, f"Scenario 2 panel {label} span {span} < 0.01 m"
        for tick in ax.yaxis.get_major_ticks():
            txt = tick.label1.get_text()
            if txt and "." in txt:
                decimals = len(txt.split(".")[1])
                assert decimals <= 4, f"Tick label '{txt}' has {decimals} > 4 decimals"

    plots5 = ComparisonPlots()
    plots5.set_result(res5)
    for ax in plots5.fig.axes:
        label = ax.get_ylabel()
        ymin, ymax = ax.get_ylim()
        span = ymax - ymin
        if any(v in label for v in ["x5", "x4", "x6", "Pipe_1_R", "Pipe_2_L", "Pipe_2_R"]):
            assert span >= 0.01, f"Scenario 5 panel {label} span {span} < 0.01 m"
        for tick in ax.yaxis.get_major_ticks():
            txt = tick.label1.get_text()
            if txt and "." in txt:
                decimals = len(txt.split(".")[1])
                assert decimals <= 4, f"Tick label '{txt}' has {decimals} > 4 decimals"

    plots1 = ComparisonPlots()
    plots1.set_result(res1)
    for ax in plots1.fig.axes:
        label = ax.get_ylabel()
        ymin, ymax = ax.get_ylim()
        span = ymax - ymin
        if "Q1" in label or "Q2" in label or "Pipe_1_Flow" in label or "Pipe_2_Flow" in label:
            assert span > 0.1
        elif any(v in label for v in ["x5", "x4", "Pipe_1_R", "Pipe_2_L"]):
            assert span > 1.0


# ---------------------------------------------------------------------------
# Step 1b: No clipped or overlapping text
# ---------------------------------------------------------------------------

def test_1b_no_text_overflow_and_no_ylabel_overlap(qapp):
    """Render scenarios 1-5 at 1900x900, 1100x800, 640x820 and export size; check bbox and y-label overlap."""
    net, _ = specimen_network()
    physics = specimen_physics()

    canvas_sizes = [
        (1900, 900),
        (1100, 800),
        (640, 820),
        (1600, 1000),  # export size
    ]

    for k in [1, 2, 3, 4, 5]:
        res = run_comparison(net, specimen_scenario(k), physics)
        plots = ComparisonPlots()
        plots.set_result(res)

        for w, h in canvas_sizes:
            plots.canvas.resize(w, h)
            plots.fig.set_size_inches(w / 100.0, h / 100.0)
            plots.canvas.draw()
            renderer = plots.canvas.get_renderer()
            fig_bb = plots.fig.get_window_extent(renderer)

            ylabels = []
            for ax in plots.fig.axes:
                yl = ax.yaxis.get_label()
                if yl.get_text():
                    ylabels.append(yl)

                for t in [yl, ax.xaxis.get_label(), ax.title]:
                    if t.get_text():
                        bb = t.get_window_extent(renderer)
                        assert bb.x0 >= fig_bb.x0 - 1.0
                        assert bb.x1 <= fig_bb.x1 + 1.0
                        assert bb.y0 >= fig_bb.y0 - 1.0
                        assert bb.y1 <= fig_bb.y1 + 1.0

                for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
                    for lbl in [tick.label1, tick.label2]:
                        if lbl.get_text():
                            bb = lbl.get_window_extent(renderer)
                            assert bb.x0 >= fig_bb.x0 - 1.0
                            assert bb.x1 <= fig_bb.x1 + 1.0
                            assert bb.y0 >= fig_bb.y0 - 1.0
                            assert bb.y1 <= fig_bb.y1 + 1.0

                for txt in ax.texts:
                    if txt.get_text():
                        bb = txt.get_window_extent(renderer)
                        assert bb.x0 >= fig_bb.x0 - 1.0
                        assert bb.x1 <= fig_bb.x1 + 1.0
                        assert bb.y0 >= fig_bb.y0 - 1.0
                        assert bb.y1 <= fig_bb.y1 + 1.0

            if plots.fig.axes:
                leg = plots.fig.axes[0].get_legend()
                if leg:
                    for t in leg.get_texts():
                        bb = t.get_window_extent(renderer)
                        assert bb.x0 >= fig_bb.x0 - 1.0
                        assert bb.x1 <= fig_bb.x1 + 1.0

            for i in range(len(ylabels)):
                bb_i = ylabels[i].get_window_extent(renderer)
                for j in range(i + 1, len(ylabels)):
                    bb_j = ylabels[j].get_window_extent(renderer)
                    overlaps = bb_i.overlaps(bb_j)
                    assert not overlaps, f"Y labels overlap: '{ylabels[i].get_text()}' and '{ylabels[j].get_text()}'"


import time


def wait_for_comparison(tab, timeout_s=60.0):
    """Wait for background ComparisonWorker to finish."""
    t0 = time.time()
    for _ in range(50):
        QApplication.processEvents()
        time.sleep(0.005)
        if tab.is_running:
            break
    while tab.is_running:
        QApplication.processEvents()
        time.sleep(0.005)
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Comparison run timed out after {timeout_s} seconds")
    QApplication.processEvents()


# ---------------------------------------------------------------------------
# Step 1c: Small windows & toggle setup
# ---------------------------------------------------------------------------

def test_1c_small_windows_and_toggle_setup(qapp):
    """Test setup toggle increases canvas width and scroll area keeps labels visible."""
    net, _ = specimen_network()
    tab = ComparisonTab(net)
    tab.resize(1280, 720)
    tab.show()
    qapp.processEvents()

    assert tab.toggle_setup_btn.text() == "Hide setup"
    assert tab.plots_widget.canvas.minimumSize().width() >= 560
    assert tab.plots_widget.canvas.minimumSize().height() >= 550

    # Toggle at 1280x720
    w_before = tab.plots_widget.canvas.width()
    setup_w = tab.splitter.sizes()[0]
    tab.toggle_setup_btn.click()
    qapp.processEvents()
    assert tab.toggle_setup_btn.text() == "Show setup"
    assert tab.splitter.sizes()[0] == 0
    w_after = tab.plots_widget.canvas.width()
    assert w_after >= w_before + (setup_w - 40)

    # Toggle back
    tab.toggle_setup_btn.click()
    qapp.processEvents()
    assert tab.toggle_setup_btn.text() == "Hide setup"
    assert tab.splitter.sizes()[0] > 0

    # Toggle at 1920x1080
    tab.resize(1920, 1080)
    qapp.processEvents()
    w_before_1080 = tab.plots_widget.canvas.width()
    setup_w_1080 = tab.splitter.sizes()[0]
    tab.toggle_setup_btn.click()
    qapp.processEvents()
    w_after_1080 = tab.plots_widget.canvas.width()
    assert w_after_1080 >= w_before_1080 + (setup_w_1080 - 40)

    # Test legend position at narrow width (< 800 px)
    res = run_comparison(net, specimen_scenario(1), specimen_physics())
    plots_narrow = ComparisonPlots()
    plots_narrow.fig.set_size_inches(6.4, 8.2)
    plots_narrow.set_result(res)
    plots_narrow.canvas.draw()
    leg = plots_narrow.fig.axes[0].get_legend()
    assert leg is not None
    renderer = plots_narrow.canvas.get_renderer()
    bbox = leg.get_window_extent(renderer)
    first_ax_bb = plots_narrow.fig.axes[0].get_window_extent(renderer)
    assert bbox.y0 >= first_ax_bb.y1 - 10


# ---------------------------------------------------------------------------
# Step 2: "Outdated" banner signature-based tests
# ---------------------------------------------------------------------------

def test_step2_outdated_banner_behavior(qapp):
    """Test outdated banner visibility across all required actions."""
    net, _ = specimen_network()
    tab = ComparisonTab(net)
    tab.resize(1280, 720)
    tab.show()
    qapp.processEvents()

    # 1. Run presets 1..5 through UI: banner hidden after each run
    for p_idx in [1, 2, 3, 4, 5]:
        tab.preset_combo.setCurrentIndex(p_idx)
        qapp.processEvents()
        tab.run_btn.click()
        wait_for_comparison(tab)
        assert not tab.outdated_banner.isVisible(), f"Banner visible after preset {p_idx} run"

        # 2. Select SAME preset again: banner remains hidden
        tab.preset_combo.setCurrentIndex(p_idx)
        qapp.processEvents()
        assert not tab.outdated_banner.isVisible(), f"Banner visible after re-selecting preset {p_idx}"

    # 3. Change one number (pipe length)
    orig_len = tab.network.pipes[0].length
    tab.network.pipes[0].length += 50.0
    tab.refresh()
    qapp.processEvents()
    assert tab.outdated_banner.isVisible(), "Banner not visible after editing pipe length"

    # Change it back: hidden again
    tab.network.pipes[0].length = orig_len
    tab.refresh()
    qapp.processEvents()
    assert not tab.outdated_banner.isVisible(), "Banner still visible after reverting pipe length"

    # Change friction factor
    orig_f = tab.factor_spin.value()
    tab.factor_spin.setValue(orig_f + 0.005)
    tab._on_physics_edited()
    qapp.processEvents()
    assert tab.outdated_banner.isVisible(), "Banner not visible after editing friction factor"

    tab.factor_spin.setValue(orig_f)
    tab._on_physics_edited()
    qapp.processEvents()
    assert not tab.outdated_banner.isVisible(), "Banner still visible after reverting friction factor"

    # 4. Toggle valve off and on: hidden again
    tab.valves_table.item(0, 2).setCheckState(Qt.Checked)
    tab._on_valve_cell_changed(0, 2)
    qapp.processEvents()
    assert tab.outdated_banner.isVisible(), "Banner not visible after toggling valve"

    tab.valves_table.item(0, 2).setCheckState(Qt.Unchecked)
    tab._on_valve_cell_changed(0, 2)
    qapp.processEvents()
    assert not tab.outdated_banner.isVisible(), "Banner still visible after reverting valve"

    # 5. Switch preset to another without running: banner visible
    tab.preset_combo.setCurrentIndex(3)
    qapp.processEvents()
    assert tab.outdated_banner.isVisible(), "Banner not visible after switching preset without running"

    # 6. Load another network: results and banner cleared
    net2 = example_paper_network()
    tab.clear_results()
    tab.set_network(net2)
    qapp.processEvents()
    assert tab._last_result is None
    assert not tab.outdated_banner.isVisible()


# ---------------------------------------------------------------------------
# Step 3: Graph Tab legend clearance, export, and broken layout
# ---------------------------------------------------------------------------

def test_3a_graph_legend_clearance_and_export(qapp):
    """Test legend clearance > 0 px at 1280x720 and 1920x1080 for all 3 networks, and no export overlap."""
    examples = [
        ("specimen", specimen_network()[0]),
        ("paper", example_paper_network()),
        ("broken", example_broken_network()),
    ]
    sizes = [(1280, 720), (1920, 1080)]

    for name, net in examples:
        for w, h in sizes:
            gv = GraphView()
            gv.resize(w, h)
            gv.show()
            gv.draw_network(net)
            gv.fit_in_view()
            qapp.processEvents()

            legend_item = gv.legend_item
            l_scene = legend_item.mapToScene(legend_item.boundingRect()).boundingRect()
            l_vp = gv.view.mapFromScene(l_scene).boundingRect()

            nodes = [it for it in gv.scene.items() if isinstance(it, NodeItem)]
            pipes = [it for it in gv.scene.items() if isinstance(it, PipeItem)]

            vp_rect = QRectF(0, 0, gv.view.viewport().width(), gv.view.viewport().height())

            for n in nodes:
                n_vp = gv.view.mapFromScene(n.sceneBoundingRect()).boundingRect()
                assert vp_rect.contains(n_vp), f"{name} node {n.node_id} outside viewport at {w}x{h}"
                assert not n_vp.intersects(l_vp), f"{name} node {n.node_id} intersects legend at {w}x{h}"
                clearance = l_vp.left() - n_vp.right()
                assert clearance >= 0.0, f"{name} node {n.node_id} clearance {clearance} < 0"

            for p in pipes:
                label_scene = p.mapToScene(p._label_box).boundingRect()
                label_vp = gv.view.mapFromScene(label_scene).boundingRect()
                assert not label_vp.intersects(l_vp), f"{name} pipe label {p.pipe_id} intersects legend at {w}x{h}"

            # Export image check
            img = gv.render_to_image()
            assert not img.isNull()
            export_legend_rect = gv._last_export_legend_rect
            for n in nodes:
                assert not export_legend_rect.intersects(n.sceneBoundingRect()), f"{name} export node overlap"
            for p in pipes:
                p_box = p.path.boundingRect().united(p.arrow_poly.boundingRect()).united(p._label_box)
                assert not export_legend_rect.intersects(p_box), f"{name} export pipe overlap"


def test_3b_broken_example_distance_ratio():
    """Verify broken example layout distance ratio <= 6 and min distance >= 0.5."""
    pos = example_broken_network_positions()
    nodes = list(pos.keys())
    dists = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            n1, n2 = nodes[i], nodes[j]
            d = math.hypot(pos[n1][0] - pos[n2][0], pos[n1][1] - pos[n2][1])
            dists.append(d)

    min_d = min(dists)
    max_d = max(dists)
    ratio = max_d / min_d

    assert min_d >= 0.5, f"Min distance {min_d} < 0.5"
    assert ratio <= 6.0, f"Distance ratio {ratio} > 6.0"
