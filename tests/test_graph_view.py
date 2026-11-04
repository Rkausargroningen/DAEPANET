"""Tests for Qt QGraphicsScene/QGraphicsView based GraphView widget."""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import math

import pytest
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from core.models import Network
from core.examples import example_paper_network, example_broken_network
from ui.graph_view import (
    GraphView, NodeItem, PipeItem, model_to_scene, scene_to_model, SCALE,
    measure_pipe_geometry
)
from ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def _patch_dialogs(monkeypatch):
    from PySide6.QtWidgets import QMessageBox, QFileDialog
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Discard)



def test_coordinate_helpers_and_v1_positions(qapp):
    # 1. Round-trip conversion
    test_points = [(0.0, 0.0), (1.0, 1.0), (-0.5, 0.75), (0.123, -0.987)]
    for mx, my in test_points:
        sx, sy = model_to_scene(mx, my)
        assert sx == pytest.approx(mx * SCALE)
        assert sy == pytest.approx(-my * SCALE)

        rx, ry = scene_to_model(sx, sy)
        assert rx == pytest.approx(mx)
        assert ry == pytest.approx(my)

    # 2. set_positions then get_positions round-trip
    view = GraphView()
    pos_dict = {"R1": (0.2, 0.8), "J1": (-0.4, -0.6)}
    net = Network()
    net.add_node("reservoir", id="R1")
    net.add_node("junction", id="J1")
    net.add_pipe("R1", "J1", id="P1")

    view.set_positions(pos_dict)
    view.draw_network(net)
    retrieved = view.get_positions()
    assert retrieved["R1"] == (0.2, 0.8)
    assert retrieved["J1"] == (-0.4, -0.6)

    # 3. Hand-written v1 position dict (y pointing UP) produces scene positions (y negated)
    r1_item = next(it for it in view.scene.items() if isinstance(it, NodeItem) and it.node_id == "R1")
    j1_item = next(it for it in view.scene.items() if isinstance(it, NodeItem) and it.node_id == "J1")

    assert r1_item.pos().x() == pytest.approx(0.2 * SCALE)
    assert r1_item.pos().y() == pytest.approx(-0.8 * SCALE)  # negated Y

    assert j1_item.pos().x() == pytest.approx(-0.4 * SCALE)
    assert j1_item.pos().y() == pytest.approx(0.6 * SCALE)  # negated Y (-(-0.6)*300 = +180)


def test_draw_example_networks(qapp):
    view = GraphView()

    # 1. Paper example: 6 nodes, 8 pipes
    net_paper = example_paper_network()
    view.draw_network(net_paper)

    node_items = [it for it in view.scene.items() if isinstance(it, NodeItem)]
    pipe_items = [it for it in view.scene.items() if isinstance(it, PipeItem)]

    assert len(node_items) == 6
    assert len(pipe_items) == 8

    # 2. Broken example: 6 nodes, 5 pipes
    net_broken = example_broken_network()
    view.draw_network(net_broken)

    node_items = [it for it in view.scene.items() if isinstance(it, NodeItem)]
    pipe_items = [it for it in view.scene.items() if isinstance(it, PipeItem)]

    assert len(node_items) == 6
    assert len(pipe_items) == 5


def test_highlight_and_clear_highlight(qapp):
    view = GraphView()
    net = example_paper_network()
    view.draw_network(net)

    # Highlight R1 and P1
    view.highlight(["R1", "P1"])

    r1_item = next(it for it in view.scene.items() if isinstance(it, NodeItem) and it.node_id == "R1")
    j1_item = next(it for it in view.scene.items() if isinstance(it, NodeItem) and it.node_id == "J1")
    p1_item = next(it for it in view.scene.items() if isinstance(it, PipeItem) and it.pipe_id == "P1")
    p2_item = next(it for it in view.scene.items() if isinstance(it, PipeItem) and it.pipe_id == "P2")

    assert r1_item._highlighted is True
    assert j1_item._highlighted is False
    assert p1_item._highlighted is True
    assert p2_item._highlighted is False

    # Clear highlight
    view.clear_highlight()
    assert r1_item._highlighted is False
    assert p1_item._highlighted is False


def test_parallel_pipes_produce_different_paths(qapp):
    net = Network()
    net.add_node("junction", id="J1")
    net.add_node("junction", id="J2")
    # Two pipes connecting J1 and J2
    net.add_pipe("J1", "J2", id="P1")
    net.add_pipe("J1", "J2", id="P2")

    view = GraphView()
    view.draw_network(net)

    pipe_items = [it for it in view.scene.items() if isinstance(it, PipeItem)]
    assert len(pipe_items) == 2

    p1 = next(p for p in pipe_items if p.pipe_id == "P1")
    p2 = next(p for p in pipe_items if p.pipe_id == "P2")

    # The curvatures and control points must differ
    assert p1.curvature != p2.curvature
    assert p1.label_pos != p2.label_pos


def test_empty_network_shows_hint_and_zero_nodes(qapp):
    view = GraphView()
    net = Network()
    view.draw_network(net)

    node_items = [it for it in view.scene.items() if isinstance(it, NodeItem)]
    pipe_items = [it for it in view.scene.items() if isinstance(it, PipeItem)]

    assert len(node_items) == 0
    assert len(pipe_items) == 0
    assert view.hint_item is not None
    assert "Add nodes and pipes to view network graph" in view.hint_item.text()
    assert view.legend_item.isVisible() is False


def test_relayout_and_position_preservation(qapp):
    net = Network()
    net.add_node("junction", id="J1")
    net.add_node("junction", id="J2")
    net.add_pipe("J1", "J2", id="P1")

    view = GraphView()
    view.draw_network(net)

    pos_initial = view.get_positions()
    assert "J1" in pos_initial and "J2" in pos_initial
    j1_pos_before = pos_initial["J1"]
    j2_pos_before = pos_initial["J2"]

    # Add a third node J3: existing positions for J1 and J2 must be preserved
    net.add_node("junction", id="J3")
    net.add_pipe("J2", "J3", id="P2")
    view.draw_network(net)

    pos_after_add = view.get_positions()
    assert len(pos_after_add) == 3
    assert tuple(pos_after_add["J1"]) == tuple(j1_pos_before)
    assert tuple(pos_after_add["J2"]) == tuple(j2_pos_before)

    # Re-layout button clicked: all nodes get positions
    view.relayout_btn.click()
    pos_after_relayout = view.get_positions()
    assert len(pos_after_relayout) == 3
    assert "J1" in pos_after_relayout and "J2" in pos_after_relayout and "J3" in pos_after_relayout


def test_zoom_level_survives_refresh_after_edit(qapp):
    window = MainWindow()
    window.show()
    window.network = example_paper_network()
    window.refresh()

    view = window.graph_view.view
    initial_scale_x = view.transform().m11()

    # User zooms in by factor of 2
    view.scale(2.0, 2.0)
    zoomed_scale_x = view.transform().m11()
    assert pytest.approx(zoomed_scale_x, rel=1e-3) == initial_scale_x * 2.0

    # User edits a pipe parameter (triggers window.refresh())
    window._on_update_pipe("P1", {"length": 550.0})

    # Scale transform must not have reset to initial_scale_x
    scale_after_edit = view.transform().m11()
    assert pytest.approx(scale_after_edit, rel=1e-3) == zoomed_scale_x
    assert window.network.pipes[0].length == 550.0

    window.setWindowModified(False)
    window.close()


def test_fit_on_first_draw_and_deferred_tab_switch(qapp):
    window = MainWindow()
    window.resize(1280, 800)
    # Start on Matrices tab (index 1) so GraphView is hidden
    window.right_tabs.setCurrentIndex(1)
    window.show()
    qapp.processEvents()

    # Load example paper network while Matrices tab is active
    window._on_load_example(example_paper_network)
    assert window.graph_view._pending_fit is True

    # Switch to Network tab (index 0)
    window.right_tabs.setCurrentIndex(0)
    qapp.processEvents()

    # showEvent/resizeEvent should have applied fit and cleared _pending_fit
    assert window.graph_view._pending_fit is False

    # Check all nodes are inside viewport rect
    vp_rect = window.graph_view.view.viewport().rect()
    for item in window.graph_view.scene.items():
        if isinstance(item, NodeItem):
            vp_poly = window.graph_view.view.mapFromScene(item.sceneBoundingRect())
            r = vp_poly.boundingRect()
            assert r.left() >= 0
            assert r.right() <= vp_rect.width()
            assert r.top() >= 0
            assert r.bottom() <= vp_rect.height()

    # Record scale
    scale_before = window.graph_view.view.transform().m11()

    # Window resize: zoom should NOT be reset
    window.resize(1400, 900)
    qapp.processEvents()
    scale_after_resize = window.graph_view.view.transform().m11()
    assert pytest.approx(scale_after_resize, rel=1e-3) == scale_before

    # Clicking Fit should refit
    window.graph_view.fit_btn.click()
    scale_after_fit = window.graph_view.view.transform().m11()
    assert scale_after_fit != scale_after_resize

    window.setWindowModified(False)
    window.close()


def test_cropped_image_export_and_legend_non_overlap(qapp, tmp_path):
    view = GraphView()
    net = example_paper_network()
    view.draw_network(net)

    # 1. Render to image
    img = view.render_to_image(scale=2.0, margin=24.0)
    assert not img.isNull()
    w, h = img.width(), img.height()

    # 2. Check pixel size is cropped (~1240 x 1264 rather than fixed 1200x900)
    assert 1000 <= w <= 1600
    assert 1000 <= h <= 1600

    # 3. Non-white pixel fraction
    non_white = 0
    step = 4
    for y in range(0, h, step):
        for x in range(0, w, step):
            c = img.pixelColor(x, y)
            if c.red() < 250 or c.green() < 250 or c.blue() < 250:
                non_white += 1
    sampled_total = ((w + step - 1) // step) * ((h + step - 1) // step)
    fraction = non_white / sampled_total
    assert fraction > 0.02

    # 4. Legend present and does not overlap any node
    legend_rect = view.legend_item.sceneBoundingRect()
    node_items = [it for it in view.scene.items() if isinstance(it, NodeItem)]
    assert len(node_items) == 6
    for node in node_items:
        node_rect = node.sceneBoundingRect()
        assert not legend_rect.intersects(node_rect)

    # 5. export_image file saving
    out_file = tmp_path / "test_export.png"
    ok = view.export_image(out_file)
    assert ok is True
    assert out_file.exists() and out_file.stat().st_size > 0


def test_regression_stale_positions_on_example_and_new(qapp):
    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    qapp.processEvents()

    # 1. Load paper example (positions exist)
    window._on_load_example(example_paper_network)
    paper_pos = window.graph_view.get_positions()
    assert len(paper_pos) == 6
    assert "J1" in paper_pos

    # 2. Load broken example through the exact same code path the menu uses
    window._on_load_example(example_broken_network)
    broken_pos = window.graph_view.get_positions()
    assert len(broken_pos) == 6

    # Assert no coordinate of paper network was kept for broken network
    for nid, coord in paper_pos.items():
        if nid in broken_pos:
            assert tuple(coord) != tuple(broken_pos[nid])

    # Assert all nodes of broken network are inside viewport after fit
    vp_rect = window.graph_view.view.viewport().rect()
    for item in window.graph_view.node_items():
        vp_poly = window.graph_view.view.mapFromScene(item.sceneBoundingRect())
        r = vp_poly.boundingRect()
        assert r.left() >= 0
        assert r.right() <= vp_rect.width()
        assert r.top() >= 0
        assert r.bottom() <= vp_rect.height()

    # 3. File > New followed by adding nodes
    window._on_new()
    assert len(window.graph_view.get_positions()) == 0

    window._on_add_node("reservoir")
    window._on_add_node("junction")
    window.graph_view.fit_in_view()
    qapp.processEvents()
    new_pos = window.graph_view.get_positions()
    assert len(new_pos) == 2
    for item in window.graph_view.node_items():
        vp_poly = window.graph_view.view.mapFromScene(item.sceneBoundingRect())
        r = vp_poly.boundingRect()
        assert r.left() >= 0
        assert r.right() <= vp_rect.width()
        assert r.top() >= 0
        assert r.bottom() <= vp_rect.height()

    window.setWindowModified(False)
    window.close()


def test_fit_quality_all_scenarios(qapp):
    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    qapp.processEvents()

    scenarios = [
        ("Direct paper", lambda: window._on_load_example(example_paper_network)),
        ("Direct broken", lambda: window._on_load_example(example_broken_network)),
        ("3-node network", lambda: (
            window._on_new(),
            window._on_add_node("reservoir"),
            window._on_add_node("junction"),
            window._on_add_node("junction"),
            window._on_add_pipe("R1", "J1", False, 0.0),
            window._on_add_pipe("J1", "J2", False, 0.0),
        )),
        ("Tab switch scenario", lambda: (
            window.right_tabs.setCurrentIndex(1),
            window._on_load_example(example_paper_network),
            window.right_tabs.setCurrentIndex(0),
        )),
    ]

    for name, action in scenarios:
        action()
        qapp.processEvents()
        window.graph_view.fit_in_view()
        qapp.processEvents()

        view = window.graph_view.view
        vp_rect = view.viewport().rect()
        vw, vh = vp_rect.width(), vp_rect.height()
        vp_cx, vp_cy = vw / 2.0, vh / 2.0

        node_items = window.graph_view.node_items()
        assert len(node_items) > 0, f"No nodes in scenario {name}"

        # Check node margins >= 20px
        min_x = min(view.mapFromScene(n.sceneBoundingRect()).boundingRect().left() for n in node_items)
        max_x = max(view.mapFromScene(n.sceneBoundingRect()).boundingRect().right() for n in node_items)
        min_y = min(view.mapFromScene(n.sceneBoundingRect()).boundingRect().top() for n in node_items)
        max_y = max(view.mapFromScene(n.sceneBoundingRect()).boundingRect().bottom() for n in node_items)

        margin_left = min_x
        margin_right = vw - max_x
        margin_top = min_y
        margin_bottom = vh - max_y

        min_margin = min(margin_left, margin_right, margin_top, margin_bottom)
        assert min_margin >= 20.0, f"Scenario {name} min_margin {min_margin:.1f} < 20px"

        # Content center within 10px of viewport center
        content_cx = (min_x + max_x) / 2.0
        content_cy = (min_y + max_y) / 2.0
        diff_x = abs(content_cx - vp_cx)
        diff_y = abs(content_cy - vp_cy)
        assert diff_x <= 10.0, f"Scenario {name} center diff_x {diff_x:.1f} > 10px"
        assert diff_y <= 10.0, f"Scenario {name} center diff_y {diff_y:.1f} > 10px"

    window.setWindowModified(False)
    window.close()


def test_drag_node_live_pipe_update_and_positions(qapp):
    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    window._on_load_example(example_paper_network)
    qapp.processEvents()

    gv = window.graph_view
    view = gv.view
    vp = view.viewport()

    j1_item = next(n for n in gv.node_items() if n.node_id == "J1")
    other_items = [n for n in gv.node_items() if n.node_id != "J1"]
    other_initial_positions = {n.node_id: QPointF(n.pos()) for n in other_items}

    initial_pos_dict = gv.get_positions()
    init_j1_model = initial_pos_dict["J1"]
    init_j1_scene = QPointF(j1_item.pos())

    # Attached pipes
    attached_pipes = [p for p in gv.pipe_items() if p.start_node_id == "J1" or p.end_node_id == "J1"]
    assert len(attached_pipes) >= 2

    # Track signals
    pos_changed_count = 0
    def _on_pos_changed():
        nonlocal pos_changed_count
        pos_changed_count += 1
    gv.positions_changed.connect(_on_pos_changed)

    # Map scene pos to viewport coordinates
    start_pt = view.mapFromScene(init_j1_scene)
    # Drag by +60px right, -45px up in viewport coordinates
    drag_dx_px, drag_dy_px = 60, -45
    end_pt = start_pt + QPoint(drag_dx_px, drag_dy_px)

    # Perform real QTest mouse drag
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, start_pt)
    qapp.processEvents()

    # Move halfway
    mid_pt = start_pt + QPoint(drag_dx_px // 2, drag_dy_px // 2)
    QTest.mouseMove(vp, mid_pt)
    qapp.processEvents()

    # Check live update: pipe endpoints should have moved
    for p in attached_pipes:
        if p.start_node_id == "J1":
            assert p.start_pos != init_j1_scene
        if p.end_node_id == "J1":
            assert p.end_pos != init_j1_scene

    # Finish drag to end_pt
    QTest.mouseMove(vp, end_pt)
    qapp.processEvents()
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, end_pt)
    qapp.processEvents()

    # Verify positions_changed emitted exactly once
    assert pos_changed_count == 1
    assert window.isWindowModified() is True

    # Check model space position change
    final_pos_dict = gv.get_positions()
    final_j1_model = final_pos_dict["J1"]
    final_j1_scene = j1_item.pos()

    # Scene shift
    actual_dx_scene = final_j1_scene.x() - init_j1_scene.x()
    actual_dy_scene = final_j1_scene.y() - init_j1_scene.y()

    # Model shift: dx = sx / SCALE, dy = -sy / SCALE
    expected_model_dx = actual_dx_scene / SCALE
    expected_model_dy = -actual_dy_scene / SCALE

    assert final_j1_model[0] == pytest.approx(init_j1_model[0] + expected_model_dx, rel=1e-3)
    assert final_j1_model[1] == pytest.approx(init_j1_model[1] + expected_model_dy, rel=1e-3)

    # Check other nodes did NOT move
    for n in other_items:
        assert n.pos() == other_initial_positions[n.node_id]

    window.setWindowModified(False)
    window.close()


def test_click_node_emits_node_clicked_without_positions_changed(qapp):
    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    window._on_load_example(example_paper_network)
    qapp.processEvents()

    gv = window.graph_view
    view = gv.view
    vp = view.viewport()

    j1_item = next(n for n in gv.node_items() if n.node_id == "J1")
    click_pt = view.mapFromScene(j1_item.scenePos())

    clicked_nodes = []
    gv.node_clicked.connect(clicked_nodes.append)
    pos_changed_count = 0
    gv.positions_changed.connect(lambda: pos_changed_count.__iadd__(1))

    # Plain click without moving
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, click_pt)
    qapp.processEvents()
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, click_pt)
    qapp.processEvents()

    assert clicked_nodes == ["J1"]
    assert pos_changed_count == 0

    window.setWindowModified(False)
    window.close()


def test_drag_empty_background_pans_view(qapp):
    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    window._on_load_example(example_paper_network)
    qapp.processEvents()

    gv = window.graph_view
    view = gv.view
    vp = view.viewport()

    initial_h = view.horizontalScrollBar().value()
    initial_v = view.verticalScrollBar().value()
    initial_node_positions = {n.node_id: QPointF(n.pos()) for n in gv.node_items()}

    # Zoom in first so scrollbars can move
    view.scale(2.0, 2.0)
    qapp.processEvents()

    # Pick empty background spot (e.g. top-left 30, 30)
    start_pt = QPoint(30, 30)
    end_pt = QPoint(150, 150)

    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, start_pt)
    qapp.processEvents()
    QTest.mouseMove(vp, end_pt)
    qapp.processEvents()
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, end_pt)
    qapp.processEvents()

    # No node moved in scene
    for n in gv.node_items():
        assert n.pos() == initial_node_positions[n.node_id]

    window.setWindowModified(False)
    window.close()


def test_after_drag_edit_param_preserves_position_and_zoom(qapp):
    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    window._on_load_example(example_paper_network)
    qapp.processEvents()

    gv = window.graph_view
    view = gv.view
    vp = view.viewport()

    j1_item = next(n for n in gv.node_items() if n.node_id == "J1")
    start_pt = view.mapFromScene(j1_item.scenePos())
    end_pt = start_pt + QPoint(50, 50)

    # Drag J1
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, start_pt)
    QTest.mouseMove(vp, end_pt)
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, end_pt)
    qapp.processEvents()

    dragged_pos = gv.get_positions()["J1"]
    view.scale(1.5, 1.5)
    scale_before_edit = view.transform().m11()

    # Edit pipe parameter
    window._on_update_pipe("P1", {"length": 777.0})
    qapp.processEvents()

    # Dragged position is retained
    pos_after_edit = gv.get_positions()["J1"]
    assert pos_after_edit == pytest.approx(dragged_pos)

    # Zoom scale is retained
    scale_after_edit = view.transform().m11()
    assert pytest.approx(scale_after_edit, rel=1e-3) == scale_before_edit

    window.setWindowModified(False)
    window.close()


def test_save_and_reopen_restores_dragged_positions(qapp, tmp_path):
    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    window._on_load_example(example_paper_network)
    qapp.processEvents()

    gv = window.graph_view
    view = gv.view
    vp = view.viewport()

    j1_item = next(n for n in gv.node_items() if n.node_id == "J1")
    start_pt = view.mapFromScene(j1_item.scenePos())
    end_pt = start_pt + QPoint(60, -40)

    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, start_pt)
    QTest.mouseMove(vp, end_pt)
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, end_pt)
    qapp.processEvents()

    expected_dragged_pos = gv.get_positions()["J1"]

    # Save to temp file
    save_file = tmp_path / "dragged_test.wnet"
    window._current_filepath = save_file
    saved_ok = window._on_save()
    assert saved_ok is True

    window.close()

    # Open in fresh MainWindow
    fresh_window = MainWindow()
    fresh_window.resize(1280, 800)
    fresh_window.show()
    fresh_window.load_file(save_file)
    qapp.processEvents()

    restored_pos = fresh_window.graph_view.get_positions()["J1"]
    assert restored_pos[0] == pytest.approx(expected_dragged_pos[0], rel=1e-4)
    assert restored_pos[1] == pytest.approx(expected_dragged_pos[1], rel=1e-4)
    assert fresh_window.isWindowModified() is False

    fresh_window.close()


def test_selection_sync_and_visual_styling(qapp):
    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    window._on_load_example(example_paper_network)
    qapp.processEvents()

    gv = window.graph_view
    bp = window.build_panel
    vp = gv.view.viewport()

    # 1. Click node J1 in graph: selects row in table, shows properties, sets blue selection ring
    j1_item = next(n for n in gv.node_items() if n.node_id == "J1")
    click_pt = gv.view.mapFromScene(j1_item.scenePos())
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, click_pt)
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, click_pt)
    qapp.processEvents()

    assert bp._selected_node_id == "J1"
    assert bp.props_stack.currentWidget() == bp.props_node_widget
    assert bp.prop_node_id_lbl.text() == "J1"
    assert j1_item._selected is True

    # 2. Select pipe row in table: sets blue selection line in graph
    bp.select_pipe("P1")
    qapp.processEvents()

    p1_item = next(p for p in gv.pipe_items() if p.pipe_id == "P1")
    assert j1_item._selected is False
    assert p1_item._selected is True
    assert bp.props_stack.currentWidget() == bp.props_pipe_widget

    # 3. Checks tab highlight works independently
    gv.highlight(["P1"])
    assert p1_item._selected is True
    assert p1_item._highlighted is True

    # 4. Click empty background: clears selection
    bg_pt = QPoint(30, 30)
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, bg_pt)
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, bg_pt)
    qapp.processEvents()

    assert bp._selected_node_id is None
    assert bp._selected_pipe_id is None
    assert p1_item._selected is False
    assert p1_item._highlighted is True  # Highlight still intact!

    window.setWindowModified(False)
    window.close()


def test_parallel_pipes_after_drag(qapp):
    net = Network()
    net.add_node("junction", id="J1")
    net.add_node("junction", id="J2")
    net.add_pipe("J1", "J2", id="P1")
    net.add_pipe("J1", "J2", id="P2")

    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    window.network = net
    window.refresh()
    qapp.processEvents()

    gv = window.graph_view
    j1_item = next(n for n in gv.node_items() if n.node_id == "J1")

    # Drag J1
    start_pt = gv.view.mapFromScene(j1_item.scenePos())
    end_pt = start_pt + QPoint(40, -50)
    QTest.mousePress(gv.view.viewport(), Qt.LeftButton, Qt.NoModifier, start_pt)
    QTest.mouseMove(gv.view.viewport(), end_pt)
    QTest.mouseRelease(gv.view.viewport(), Qt.LeftButton, Qt.NoModifier, end_pt)
    qapp.processEvents()

    pipes = gv.pipe_items()
    p1 = next(p for p in pipes if p.pipe_id == "P1")
    p2 = next(p for p in pipes if p.pipe_id == "P2")

    # Curvatures and paths remain distinct
    assert p1.curvature != p2.curvature
    assert p1.path != p2.path

    window.setWindowModified(False)
    window.close()


def test_tooltips_content(qapp):
    net = Network()
    net.add_node("reservoir", id="R1", elevation=25.0, head=60.0)
    net.add_node("junction", id="J1", elevation=None)  # unset elevation
    net.add_pipe("R1", "J1", id="P1", length=500.0, diameter=0.3, roughness=0.0002, has_pump=True, pump_power=45.0)
    net.add_pipe("J1", "R1", id="P2")  # unset params

    gv = GraphView()
    gv.draw_network(net)

    r1_item = next(n for n in gv.node_items() if n.node_id == "R1")
    j1_item = next(n for n in gv.node_items() if n.node_id == "J1")
    p1_item = next(p for p in gv.pipe_items() if p.pipe_id == "P1")
    p2_item = next(p for p in gv.pipe_items() if p.pipe_id == "P2")

    r1_tip = r1_item.toolTip()
    assert "ID: R1" in r1_tip
    assert "Reservoir" in r1_tip
    assert "Elevation: 25.0 m" in r1_tip or "Elevation: 25 m" in r1_tip
    assert "Head: 60.0 m" in r1_tip or "Head: 60 m" in r1_tip

    j1_tip = j1_item.toolTip()
    assert "ID: J1" in j1_tip
    assert "Junction" in j1_tip
    assert "Elevation: not set" in j1_tip
    assert "Head" not in j1_tip  # Head only for reservoirs

    p1_tip = p1_item.toolTip()
    assert "ID: P1" in p1_tip
    assert "R1 -> J1" in p1_tip
    assert "500" in p1_tip and "m" in p1_tip
    assert "0.3" in p1_tip and "m" in p1_tip
    assert "0.0002" in p1_tip and "m" in p1_tip
    assert "45" in p1_tip and "kW" in p1_tip

    p2_tip = p2_item.toolTip()
    assert "ID: P2" in p2_tip
    assert "J1 -> R1" in p2_tip
    assert "Length: not set" in p2_tip
    assert "Diameter: not set" in p2_tip
    assert "Roughness: not set" in p2_tip


def test_pipe_bend_interaction_path_and_endpoints(qapp):
    """Test that dragging the handle bends the pipe, midpoint moves about ratio*chord,

    endpoints stay clipped at node boundaries, label follows midpoint, and sign convention holds.
    """
    net = Network()
    net.add_node("junction", id="J1")
    net.add_node("junction", id="J2")
    net.add_pipe("J1", "J2", id="P1")

    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    window.network = net
    # Position J1 at (0, 0) and J2 at (1, 0) so travel direction is +x (East)
    window.graph_view.set_positions({"J1": (0.0, 0.0), "J2": (1.0, 0.0)})
    window.refresh()
    qapp.processEvents()
    window.graph_view.fit_in_view()
    qapp.processEvents()

    gv = window.graph_view
    p1 = next(p for p in gv.pipe_items() if p.pipe_id == "P1")
    p1.setSelected(True)
    qapp.processEvents()

    chord = math.hypot(p1.end_pos.x() - p1.start_pos.x(), p1.end_pos.y() - p1.start_pos.y())
    mx = (p1.start_pos.x() + p1.end_pos.x()) / 2.0
    my = (p1.start_pos.y() + p1.end_pos.y()) / 2.0

    # Drag handle upward (screen -y, left of East travel direction)
    handle_vp = gv.view.mapFromScene(p1.handle_item.scenePos())
    target_vp = handle_vp + QPoint(0, -50)

    QTest.mousePress(gv.view.viewport(), Qt.LeftButton, Qt.NoModifier, handle_vp)
    qapp.processEvents()
    QTest.mouseMove(gv.view.viewport(), target_vp)
    qapp.processEvents()
    QTest.mouseRelease(gv.view.viewport(), Qt.LeftButton, Qt.NoModifier, target_vp)
    qapp.processEvents()

    ratio = gv.get_edge_bends().get("P1")
    assert ratio is not None
    assert ratio > 0.0, "Upward drag for East travel must be positive bend ratio"

    # Midpoint displacement equals ratio * chord within 0.5 px
    m = measure_pipe_geometry(p1)
    disp = m["signed_displacement"]
    chord = m["chord_length"]
    assert abs(disp - ratio * chord) <= 0.5

    # Label stays at curve midpoint
    assert p1.label_pos == p1.curve_midpoint

    # Both endpoints stay on node boundaries (distance to node center == NODE_RADIUS = 20.0)
    start_pt = m["start_pt"]
    end_pt = m["end_pt"]
    d_start = math.hypot(start_pt.x() - p1.start_pos.x(), start_pt.y() - p1.start_pos.y())
    d_end = math.hypot(end_pt.x() - p1.end_pos.x(), end_pt.y() - p1.end_pos.y())
    assert abs(d_start - 20.0) < 1e-3
    assert abs(d_end - 20.0) < 1e-3

    window.setWindowModified(False)
    window.close()


def test_pipe_bend_ratio_survives_node_drag(qapp):
    """Test that bend ratio is preserved when a node is dragged and displacement scales with chord."""
    net = Network()
    net.add_node("junction", id="J1")
    net.add_node("junction", id="J2")
    net.add_pipe("J1", "J2", id="P1")

    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    window.network = net
    window.graph_view.set_positions({"J1": (0.0, 0.0), "J2": (1.0, 0.0)})
    window.graph_view.set_edge_bends({"P1": 0.25})
    window.refresh()
    qapp.processEvents()
    window.graph_view.fit_in_view()
    qapp.processEvents()

    gv = window.graph_view
    p1 = next(p for p in gv.pipe_items() if p.pipe_id == "P1")
    assert p1.bend_ratio == 0.25
    m_init = measure_pipe_geometry(p1)
    initial_chord = m_init["chord_length"]
    initial_disp = m_init["signed_displacement"]

    # Drag J2 to change chord length
    j2 = next(n for n in gv.node_items() if n.node_id == "J2")
    j2_vp = gv.view.mapFromScene(j2.scenePos())
    target_vp = j2_vp + QPoint(80, -40)

    QTest.mousePress(gv.view.viewport(), Qt.LeftButton, Qt.NoModifier, j2_vp)
    qapp.processEvents()
    QTest.mouseMove(gv.view.viewport(), target_vp)
    qapp.processEvents()
    QTest.mouseRelease(gv.view.viewport(), Qt.LeftButton, Qt.NoModifier, target_vp)
    qapp.processEvents()

    # Bend ratio is strictly preserved
    assert p1.bend_ratio == 0.25
    assert gv.get_edge_bends().get("P1") == 0.25

    m_new = measure_pipe_geometry(p1)
    new_chord = m_new["chord_length"]
    new_disp = m_new["signed_displacement"]

    # Chord length really changed by more than 1 px
    assert abs(new_chord - initial_chord) > 1.0, f"Chord length must change: was {initial_chord}, now {new_chord}"
    # Displacement equals ratio * new chord within 0.5 px
    assert abs(new_disp - p1.bend_ratio * new_chord) <= 0.5

    window.setWindowModified(False)
    window.close()


def test_pipe_bend_double_click_resets(qapp):
    """Test that double clicking the handle resets bend to automatic curvature."""
    net = Network()
    net.add_node("junction", id="J1")
    net.add_node("junction", id="J2")
    net.add_pipe("J1", "J2", id="P1")

    gv = GraphView()
    gv.set_positions({"J1": (0.0, 0.0), "J2": (1.0, 0.0)})
    gv.set_edge_bends({"P1": 0.4})
    gv.draw_network(net)
    gv.resize(800, 600)
    gv.show()
    qapp.processEvents()
    gv.fit_in_view()
    qapp.processEvents()

    p1 = gv.pipe_items()[0]
    p1.setSelected(True)
    qapp.processEvents()
    assert p1.handle_item.isVisible()

    signals = []
    gv.edge_bends_changed.connect(lambda: signals.append(1))

    handle_vp = gv.view.mapFromScene(p1.handle_item.scenePos())
    QTest.mouseDClick(gv.view.viewport(), Qt.LeftButton, Qt.NoModifier, handle_vp)
    qapp.processEvents()

    assert len(signals) == 1
    assert p1.bend_ratio == 0.0
    assert gv.get_edge_bends() == {}
    gv.close()


def test_edge_bends_changed_signal_and_click_behavior(qapp):
    """Test edge_bends_changed fires exactly once per drag, 0 on click, and pipe body click selects."""
    net = Network()
    net.add_node("junction", id="J1")
    net.add_node("junction", id="J2")
    net.add_pipe("J1", "J2", id="P1")

    gv = GraphView()
    gv.set_positions({"J1": (0.0, 0.0), "J2": (1.0, 0.0)})
    gv.draw_network(net)
    gv.resize(800, 600)
    gv.show()
    qapp.processEvents()
    gv.fit_in_view()
    qapp.processEvents()

    p1 = gv.pipe_items()[0]
    p1.setSelected(True)
    qapp.processEvents()

    bend_signals = []
    pipe_clicks = []
    gv.edge_bends_changed.connect(lambda: bend_signals.append(1))
    gv.pipe_clicked.connect(lambda pid: pipe_clicks.append(pid))

    # 1. Plain click on handle without move
    handle_vp = gv.view.mapFromScene(p1.handle_item.scenePos())
    QTest.mousePress(gv.view.viewport(), Qt.LeftButton, Qt.NoModifier, handle_vp)
    qapp.processEvents()
    QTest.mouseRelease(gv.view.viewport(), Qt.LeftButton, Qt.NoModifier, handle_vp)
    qapp.processEvents()
    assert len(bend_signals) == 0, "Plain click must not emit edge_bends_changed"

    # 2. Drag handle
    target_vp = handle_vp + QPoint(0, -40)
    QTest.mousePress(gv.view.viewport(), Qt.LeftButton, Qt.NoModifier, handle_vp)
    qapp.processEvents()
    QTest.mouseMove(gv.view.viewport(), target_vp)
    qapp.processEvents()
    QTest.mouseRelease(gv.view.viewport(), Qt.LeftButton, Qt.NoModifier, target_vp)
    qapp.processEvents()
    assert len(bend_signals) == 1, "Drag must emit edge_bends_changed exactly once"

    # 3. Click pipe body
    body_scene = p1.path.pointAtPercent(0.3)
    body_vp = gv.view.mapFromScene(body_scene)
    QTest.mousePress(gv.view.viewport(), Qt.LeftButton, Qt.NoModifier, body_vp)
    qapp.processEvents()
    QTest.mouseRelease(gv.view.viewport(), Qt.LeftButton, Qt.NoModifier, body_vp)
    qapp.processEvents()
    assert pipe_clicks == ["P1"]
    assert len(bend_signals) == 1  # No extra bend signal

    # 4. Drag pipe body does nothing to bends
    QTest.mousePress(gv.view.viewport(), Qt.LeftButton, Qt.NoModifier, body_vp)
    qapp.processEvents()
    QTest.mouseMove(gv.view.viewport(), body_vp + QPoint(30, 30))
    qapp.processEvents()
    QTest.mouseRelease(gv.view.viewport(), Qt.LeftButton, Qt.NoModifier, body_vp + QPoint(30, 30))
    qapp.processEvents()
    assert len(bend_signals) == 1
    gv.close()


def test_pipe_bend_modified_flag_and_save_reopen(qapp, tmp_path):
    """Test MainWindow modified flag behavior and save/reopen round-trip."""
    from core.examples import example_paper_network
    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    window._on_load_example(example_paper_network)
    qapp.processEvents()
    assert not window.isWindowModified()

    gv = window.graph_view
    p1 = next(p for p in gv.pipe_items() if p.pipe_id == "P1")
    p1.setSelected(True)
    qapp.processEvents()

    handle_vp = gv.view.mapFromScene(p1.handle_item.scenePos())
    target_vp = handle_vp + QPoint(0, -35)

    QTest.mousePress(gv.view.viewport(), Qt.LeftButton, Qt.NoModifier, handle_vp)
    qapp.processEvents()
    QTest.mouseMove(gv.view.viewport(), target_vp)
    qapp.processEvents()
    QTest.mouseRelease(gv.view.viewport(), Qt.LeftButton, Qt.NoModifier, target_vp)
    qapp.processEvents()

    assert window.isWindowModified()
    bends = gv.get_edge_bends()
    assert "P1" in bends
    bend_val = bends["P1"]

    save_path = tmp_path / "saved_bend.wnet"
    window._current_filepath = save_path
    assert window._on_save()
    assert not window.isWindowModified()

    # Reopen in fresh window
    w2 = MainWindow()
    w2.resize(1280, 800)
    w2.show()
    w2.load_file(save_path)
    qapp.processEvents()

    assert not w2.isWindowModified(), "Loading file must not leave window modified"
    restored = w2.graph_view.get_edge_bends()
    assert "P1" in restored
    assert abs(restored["P1"] - bend_val) < 1e-4

    # Open file without bends
    no_bends_path = tmp_path / "no_bends.wnet"
    from core.io import save_project
    save_project(no_bends_path, window.network, window.graph_view.get_positions(), edge_bends=None)

    w3 = MainWindow()
    w3.resize(1280, 800)
    w3.show()
    w3.load_file(no_bends_path)
    qapp.processEvents()
    assert not w3.isWindowModified()
    assert w3.graph_view.get_edge_bends() == {}

    window.close()
    w2.close()
    w3.close()


def test_parallel_pipes_bend_distinct_paths(qapp):
    """Test that bending one of two parallel pipes keeps their paths and curvatures distinct."""
    net = Network()
    net.add_node("junction", id="J1")
    net.add_node("junction", id="J2")
    net.add_pipe("J1", "J2", id="P1")
    net.add_pipe("J1", "J2", id="P2")

    gv = GraphView()
    gv.set_positions({"J1": (0.0, 0.0), "J2": (1.0, 0.0)})
    gv.draw_network(net)

    p1 = next(p for p in gv.pipe_items() if p.pipe_id == "P1")
    p2 = next(p for p in gv.pipe_items() if p.pipe_id == "P2")

    assert p1.curvature != p2.curvature
    assert p1.path != p2.path

    # Bend P1
    gv.set_edge_bends({"P1": 0.5})
    assert p1.bend_ratio == 0.5
    assert p2.bend_ratio == 0.0
    assert p1.curvature != p2.curvature
    assert p1.path != p2.path


def test_deleting_pipe_and_relayout_clears_bends(qapp):
    """Test that deleting a pipe removes its bend, and re-layout clears all bends."""
    from core.examples import example_paper_network
    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    window._on_load_example(example_paper_network)
    qapp.processEvents()

    window.graph_view.set_edge_bends({"P1": 0.3, "P2": -0.2})
    assert "P1" in window.graph_view.get_edge_bends()
    assert "P2" in window.graph_view.get_edge_bends()

    # Delete P1
    window._on_delete_pipe("P1")
    qapp.processEvents()
    assert "P1" not in window.graph_view.get_edge_bends()
    assert "P2" in window.graph_view.get_edge_bends()

    # Re-layout clears all bends
    window.graph_view.relayout_btn.click()
    qapp.processEvents()
    assert window.graph_view.get_edge_bends() == {}

    window.setWindowModified(False)
    window.close()


def test_bend_handle_visibility_hover_and_selection(qapp):
    """Test that the bend handle is visible only on hover or selection, and hidden otherwise."""
    net = Network()
    net.add_node("junction", id="J1")
    net.add_node("junction", id="J2")
    net.add_pipe("J1", "J2", id="P1")

    gv = GraphView()
    gv.set_positions({"J1": (0.0, 0.0), "J2": (1.0, 0.0)})
    gv.draw_network(net)
    gv.resize(800, 600)
    gv.show()
    qapp.processEvents()
    gv.fit_in_view()
    qapp.processEvents()

    p1 = gv.pipe_items()[0]
    handle = p1.handle_item

    # Start mouse outside
    QTest.mouseMove(gv.view.viewport(), QPoint(10, 10))
    qapp.processEvents()
    assert not handle.isVisible()

    # Hover
    body_scene = p1.path.pointAtPercent(0.3)
    body_vp = gv.view.mapFromScene(body_scene)
    QTest.mouseMove(gv.view.viewport(), body_vp)
    qapp.processEvents()
    assert handle.isVisible()

    # Hover out
    QTest.mouseMove(gv.view.viewport(), QPoint(10, 10))
    qapp.processEvents()
    assert not handle.isVisible()

    # Select
    p1.setSelected(True)
    qapp.processEvents()
    assert handle.isVisible()

    # Deselect
    p1.setSelected(False)
    qapp.processEvents()
    assert not handle.isVisible()

    gv.close()


