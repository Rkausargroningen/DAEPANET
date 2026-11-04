"""UI smoke test using Qt offscreen platform."""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
from core.examples import example_paper_network, example_broken_network


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_ui_smoke_flow(qapp):
    window = MainWindow()

    # 1. Add 3 nodes (R1, J1, J2)
    window._on_add_node("reservoir")
    window._on_add_node("junction")
    window._on_add_node("junction")

    # Assert node table has 3 rows
    assert window.build_panel.nodes_table.rowCount() == 3

    # 2. Add 2 pipes (P1: R1->J1, P2: J1->J2 with pump)
    window._on_add_pipe("R1", "J1", False, 0.0)
    window._on_add_pipe("J1", "J2", True, 15.0)

    # Assert pipe table has 2 rows
    assert window.build_panel.pipes_table.rowCount() == 2

    # 3. Assert matrix A table is KxK matching K_dimension
    a_table = window.matrix_view.tab_A.table
    rows = a_table.rowCount()
    cols = a_table.columnCount()
    assert rows > 0
    assert rows == cols

    expected_k = len(window.network.pipes) + 2 * 1 + sum(1 for n in window.network.nodes if n.type == "junction") + sum(1 for n in window.network.nodes if n.type in ("reservoir", "tank"))
    assert rows == expected_k

    # 4. Refresh graph without exception
    window.graph_view.draw_network(window.network)

    # 5. Delete a node (J2) and verify deletion updates tables & matrices
    window._on_delete_node("J2")

    assert window.build_panel.nodes_table.rowCount() == 2
    assert window.build_panel.pipes_table.rowCount() == 1  # P2 attached to J2 removed
    assert window.matrix_view.tab_A.table.rowCount() < rows


def test_ui_smoke_nodes_without_pipes(qapp):
    """0a: GraphView draws nodes as soon as 1+ nodes exist, even with 0 pipes."""
    window = MainWindow()
    window._on_add_node("reservoir")
    window._on_add_node("junction")

    assert window.build_panel.nodes_table.rowCount() == 2
    assert window.build_panel.pipes_table.rowCount() == 0

    window.graph_view.draw_network(window.network)

    # Verify node labels are drawn on scene and placeholder text is absent
    node_labels = [n.node_id for n in window.graph_view.node_items()]
    assert "R1" in node_labels
    assert "J1" in node_labels
    assert not any("Add nodes" in n.node_id for n in window.graph_view.node_items())


def test_ui_smoke_examples_checks_and_incidence(qapp, tmp_path):
    """Test loading broken example, paper example, incidence tables, and file save/reopen."""
    window = MainWindow()

    # 1. Load broken network example
    window.network = example_broken_network()
    window.refresh()

    # Assert Checks tab shows "NOT regular" (red banner) and >= 2 warnings
    assert "NOT regular" in window.checks_view.banner.text()
    assert window.checks_view.warnings_table.rowCount() >= 2

    # 2. Load paper network example
    window.network = example_paper_network()
    window.refresh()

    # Assert Checks tab shows "Regular" (green banner)
    assert "Regular" in window.checks_view.banner.text()

    # Assert Incidence table shapes (6 nodes x 8 pipes)
    inc_tab = window.incidence_view.tab_full.table
    assert inc_tab.rowCount() == 6
    assert inc_tab.columnCount() == 8

    # 3. Save to temp .wnet and reopen it
    save_file = tmp_path / "smoke_save.wnet"
    window._current_filepath = save_file
    saved = window._on_save()
    assert saved is True
    assert save_file.is_file()

    # Clear window state
    window.network.clear()
    window.refresh()
    assert window.build_panel.nodes_table.rowCount() == 0

    # Load file back
    window.load_file(save_file)
    assert window.build_panel.nodes_table.rowCount() == 6
    assert window.build_panel.pipes_table.rowCount() == 8
    assert "Regular" in window.checks_view.banner.text()
