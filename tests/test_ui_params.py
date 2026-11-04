"""Tests for UI parameter entry, properties editing, and status indicators."""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import json
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox
from ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_ui_add_node_and_pipe_with_parameters(qapp):
    window = MainWindow()
    panel = window.build_panel

    # 1. Add Reservoir with Elevation=30.0 and Head=65.0
    panel.node_type_combo.setCurrentText("Reservoir")
    assert panel.node_head_input.isEnabled() is True

    panel.node_elevation_input.setText("30.0")
    panel.node_head_input.setText("65.0")
    panel.add_node_btn.click()

    assert len(window.network.nodes) == 1
    r1 = window.network.nodes[0]
    assert r1.id == "R1"
    assert r1.type == "reservoir"
    assert r1.elevation == 30.0
    assert r1.head == 65.0

    # 2. Add Junction with Elevation=15.0
    panel.node_type_combo.setCurrentText("Junction")
    assert panel.node_head_input.isEnabled() is False
    panel.node_elevation_input.setText("15.0")
    panel.add_node_btn.click()

    assert len(window.network.nodes) == 2
    j1 = window.network.nodes[1]
    assert j1.id == "J1"
    assert j1.type == "junction"
    assert j1.elevation == 15.0
    assert j1.head is None

    # 2b. Add Junction with empty elevation ("not set"): elevation must be None
    panel.node_elevation_input.setText("")
    panel.add_node_btn.click()
    assert len(window.network.nodes) == 3
    j2 = window.network.nodes[2]
    assert j2.elevation is None


    # 3. Add Pipe with Length=350.0, Diameter=0.4, Roughness=3.59e-4
    panel.start_combo.setCurrentText("R1")
    panel.end_combo.setCurrentText("J1")
    panel.pipe_length_input.setText("350.0")
    panel.pipe_diameter_input.setText("0.4")
    panel.pipe_roughness_input.setText("3.59e-4")
    panel.add_pipe_btn.click()

    assert len(window.network.pipes) == 1
    p1 = window.network.pipes[0]
    assert p1.id == "P1"
    assert p1.length == 350.0
    assert p1.diameter == 0.4
    assert pytest.approx(p1.roughness, rel=1e-5) == 3.59e-4

    # Verify input fields retain their values so user can add several similar items
    assert panel.pipe_length_input.text() == "350.0"
    assert panel.pipe_diameter_input.text() == "0.4"
    assert panel.pipe_roughness_input.text() == "3.59e-4"


def test_ui_properties_editor_edit_node_and_pipe(qapp):
    window = MainWindow()
    panel = window.build_panel

    window._on_add_node("reservoir", {"elevation": 20.0, "head": 50.0})
    window._on_add_node("junction", {"elevation": 10.0})
    window._on_add_pipe("R1", "J1", False, 0.0, {"length": 100.0, "diameter": 0.3, "roughness": 0.001})

    # Select node R1
    panel.nodes_table.selectRow(0)
    assert panel.props_stack.currentWidget() == panel.props_node_widget
    assert panel.prop_node_id_lbl.text() == "R1"
    assert panel.prop_node_elev_edit.text() == "20"
    assert panel.prop_node_head_edit.text() == "50"

    # Edit elevation and head, then click Apply
    panel.prop_node_elev_edit.setText("25.5")
    panel.prop_node_head_edit.setText("75.0")
    panel.apply_node_btn.click()

    r1 = window.network.nodes[0]
    assert r1.elevation == 25.5
    assert r1.head == 75.0

    # Select pipe P1
    panel.pipes_table.selectRow(0)
    assert panel.props_stack.currentWidget() == panel.props_pipe_widget
    assert panel.prop_pipe_id_lbl.text() == "P1"
    assert panel.prop_pipe_length_edit.text() == "100"

    # Edit pipe properties and click Apply
    panel.prop_pipe_length_edit.setText("250.0")
    panel.prop_pipe_diam_edit.setText("0.5")
    panel.prop_pipe_rough_edit.setText("0.0002")
    panel.apply_pipe_btn.click()

    p1 = window.network.pipes[0]
    assert p1.length == 250.0
    assert p1.diameter == 0.5
    assert p1.roughness == 0.0002

    # Test Revert button
    panel.prop_pipe_length_edit.setText("999.0")
    panel.revert_pipe_btn.click()
    assert panel.prop_pipe_length_edit.text() == "250"


def test_ui_invalid_input_shows_error_and_keeps_network(qapp, monkeypatch):
    window = MainWindow()
    panel = window.build_panel

    window._on_add_node("junction", {"elevation": 10.0})
    window._on_add_node("junction", {"elevation": 10.0})
    window._on_add_pipe("J1", "J2", False, 0.0, {"length": 100.0, "diameter": 0.3, "roughness": 0.001})

    critical_called = []

    def mock_critical(parent, title, msg, *args, **kwargs):
        critical_called.append((title, msg))

    monkeypatch.setattr(QMessageBox, "critical", mock_critical)

    panel.pipes_table.selectRow(0)

    # 1. Negative length
    panel.prop_pipe_length_edit.setText("-5.0")
    panel.apply_pipe_btn.click()
    assert len(critical_called) == 1
    assert "length must be > 0" in critical_called[-1][1]
    assert window.network.pipes[0].length == 100.0  # unchanged

    # 2. Text in length
    panel.prop_pipe_length_edit.setText("invalid_text")
    panel.apply_pipe_btn.click()
    assert len(critical_called) == 2
    assert "must be a number" in critical_called[-1][1]
    assert window.network.pipes[0].length == 100.0

    # 3. NaN in diameter
    panel.prop_pipe_length_edit.setText("100.0")
    panel.prop_pipe_diam_edit.setText("nan")
    panel.apply_pipe_btn.click()
    assert len(critical_called) == 3
    assert "must be a finite number" in critical_called[-1][1]
    assert window.network.pipes[0].diameter == 0.3


def test_ui_missing_parameters_indicator(qapp, monkeypatch):
    window = MainWindow()
    window.show()
    panel = window.build_panel

    # 1. Empty network: no indicator text
    assert panel.params_status_label.text() == ""
    assert panel.params_details_btn.isHidden() is True

    # 2. Incomplete network (v1-style additions without physical parameters)
    window._on_add_node("reservoir")
    window._on_add_node("junction")
    window._on_add_pipe("R1", "J1", False, 0.0)

    assert "missing" in panel.params_status_label.text()
    assert panel.params_details_btn.isHidden() is False

    # Test Details dialog pop-up
    details_opened = []

    def mock_info(parent, title, msg, *args, **kwargs):
        details_opened.append((title, msg))

    monkeypatch.setattr(QMessageBox, "information", mock_info)
    panel.params_details_btn.click()
    assert len(details_opened) == 1
    assert "Pipe P1: length is not set" in details_opened[0][1]

    # 3. Complete all parameters
    window.network.update_node("R1", head=50.0, elevation=10.0)
    window.network.update_node("J1", elevation=5.0)
    window.network.update_pipe("P1", length=100.0, diameter=0.3, roughness=0.0001)
    window.refresh()

    assert panel.params_status_label.text() == "Parameters: complete"
    assert panel.params_details_btn.isHidden() is True



def test_ui_save_and_reopen_survives_all_parameters(qapp, tmp_path):
    window = MainWindow()
    window._on_add_node("reservoir", {"elevation": 22.0, "head": 55.0})
    window._on_add_node("junction", {"elevation": 11.0})
    window._on_add_pipe(
        "R1", "J1", True, 20.0,
        {"length": 230.0, "diameter": 0.35, "roughness": 0.00012}
    )

    save_path = tmp_path / "params_survive.wnet"
    window._current_filepath = save_path
    assert window._on_save() is True

    # Clear and verify empty
    window.network.clear()
    window.refresh()
    assert len(window.network.nodes) == 0

    # Reload and check parameters
    window.load_file(save_path)
    assert len(window.network.nodes) == 2
    r1 = next(n for n in window.network.nodes if n.id == "R1")
    j1 = next(n for n in window.network.nodes if n.id == "J1")
    p1 = window.network.pipes[0]

    assert r1.elevation == 22.0
    assert r1.head == 55.0
    assert j1.elevation == 11.0
    assert p1.length == 230.0
    assert p1.diameter == 0.35
    assert p1.roughness == 0.00012


def test_ui_opening_v1_wnet_shows_missing_and_empty_fields(qapp, tmp_path):
    v1_json = {
        "format": "water-network",
        "version": 1,
        "nodes": [
            {"id": "R1", "type": "reservoir", "capacity": 1000.0, "elevation": 50.0},
            {"id": "J1", "type": "junction", "capacity": 1000.0, "elevation": 50.0}
        ],
        "pipes": [
            {"id": "P1", "start": "R1", "end": "J1", "has_pump": False, "pump_power": 0.0}
        ],
        "counters": {"R": 1, "J": 1, "P": 1, "T": 0},
        "positions": {}
    }
    v1_path = tmp_path / "legacy_test.wnet"
    v1_path.write_text(json.dumps(v1_json), encoding="utf-8")

    window = MainWindow()
    window.load_file(v1_path)
    panel = window.build_panel

    # Check missing parameters indicator
    assert "missing" in panel.params_status_label.text()

    # Select pipe P1: physical fields should be empty
    panel.pipes_table.selectRow(0)
    assert panel.prop_pipe_length_edit.text() == ""
    assert panel.prop_pipe_diam_edit.text() == ""
    assert panel.prop_pipe_rough_edit.text() == ""

    # Select reservoir R1: head should be empty
    panel.nodes_table.selectRow(0)
    assert panel.prop_node_head_edit.text() == ""
    assert panel.prop_node_elev_edit.text() == "50"  # v1 placeholder


def test_ui_selection_survives_refresh(qapp):
    window = MainWindow()
    panel = window.build_panel

    window._on_add_node("reservoir", {"elevation": 20.0, "head": 50.0})
    window._on_add_node("junction", {"elevation": 10.0})
    window._on_add_pipe("R1", "J1", False, 0.0, {"length": 100.0, "diameter": 0.3, "roughness": 0.001})

    # Select node J1
    panel.nodes_table.selectRow(1)
    assert panel._selected_node_id == "J1"
    assert panel.props_stack.currentWidget() == panel.props_node_widget

    window.refresh()
    assert panel._selected_node_id == "J1"
    assert panel.props_stack.currentWidget() == panel.props_node_widget
    assert panel.prop_node_id_lbl.text() == "J1"

    # Select pipe P1
    panel.pipes_table.selectRow(0)
    assert panel._selected_pipe_id == "P1"
    assert panel.props_stack.currentWidget() == panel.props_pipe_widget

    window.refresh()
    assert panel._selected_pipe_id == "P1"
    assert panel.props_stack.currentWidget() == panel.props_pipe_widget
    assert panel.prop_pipe_id_lbl.text() == "P1"
