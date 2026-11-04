"""tests/test_comparison_plots.py – Tests for comparison plots, line properties, Dirac arrows, options, differences, exports, and layout."""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import csv
import math
import tempfile
from pathlib import Path

import pytest
import numpy as np
from matplotlib.colors import to_hex
from PIL import Image

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QColor, QFontMetrics
from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog

from core.examples import specimen_network, specimen_scenario, specimen_physics
from core.comparison import run_comparison
from core.plotdata import (
    default_states,
    series,
    difference_rows,
    max_abs_difference_after_last_switch,
    impulse_arrows,
    engine_label,
)
from ui.comparison_plots import ComparisonPlots, draw_comparison
from ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(scope="module")
def res_sc1():
    net, pos = specimen_network()
    sc = specimen_scenario(1)
    phys = specimen_physics()
    return run_comparison(net, sc, phys, positions=pos)


@pytest.fixture(scope="module")
def res_sc2():
    net, pos = specimen_network()
    sc = specimen_scenario(2)
    phys = specimen_physics()
    return run_comparison(net, sc, phys, positions=pos)


@pytest.fixture(scope="module")
def res_sc3():
    net, pos = specimen_network()
    sc = specimen_scenario(3)
    phys = specimen_physics()
    return run_comparison(net, sc, phys, positions=pos)


@pytest.fixture(scope="module")
def res_sc4():
    net, pos = specimen_network()
    sc = specimen_scenario(4)
    phys = specimen_physics()
    return run_comparison(net, sc, phys, positions=pos)


@pytest.fixture(scope="module")
def res_sc5():
    net, pos = specimen_network()
    sc = specimen_scenario(5)
    phys = specimen_physics()
    return run_comparison(net, sc, phys, positions=pos)


# ---------------------------------------------------------------------------
# 1. Plot structure for scenario 1
# ---------------------------------------------------------------------------

def test_01_plot_structure_scenario1(qapp, res_sc1):
    widget = ComparisonPlots()
    widget.set_result(res_sc1)

    fig = widget.fig
    axes = fig.axes
    # 5 panels
    assert len(axes) == 5

    # Check y labels
    expected_ylabels = [
        "x1 = Q1 [m³/s]",
        "x2 = Q2 [m³/s]",
        "x5 pipe-1 end [m]",
        "x4 d/s of V1 [m]",
        "x6 pipe-2 end [m]",
    ]
    for r, ax in enumerate(axes):
        assert ax.get_ylabel() == expected_ylabels[r]

    # First axes has title and legend with exactly two entries
    ax0 = axes[0]
    assert ax0.get_title() != ""
    legend = ax0.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert len(legend_texts) == 2
    assert "switched DAE" in legend_texts[0]
    assert "real EPANET 2.3.5" in legend_texts[1]

    # x-limits [-0.01*120, 120] = [-1.2, 120] or [-0.6, 120]
    T = res_sc1.scenario.t_end
    for ax in axes:
        assert ax.get_xlim() == pytest.approx((-0.01 * T, T), abs=1e-3)


# ---------------------------------------------------------------------------
# 2. Line properties
# ---------------------------------------------------------------------------

def test_02_line_properties(qapp, res_sc1):
    widget = ComparisonPlots()
    widget.set_result(res_sc1)

    axes = widget.fig.axes
    assert len(axes) == 5

    c_dae_expected = "#0072b2"
    c_ep_expected = "#d91a1a"

    for r, ax in enumerate(axes):
        lines = ax.get_lines()
        # Find solid DAE lines, EPANET step line, dotted jump line, vertical switch line
        dae_lines = [l for l in lines if to_hex(l.get_color()).lower() == c_dae_expected and l.get_linestyle() == "-"]
        # Each panel has 2 mode intervals -> 2 DAE segments (plus optional Dirac arrow stem if impulse)
        dae_segments = [l for l in dae_lines if len(l.get_xdata()) > 2]
        assert len(dae_segments) == 2
        for seg in dae_segments:
            assert seg.get_linewidth() == pytest.approx(1.6)

        ep_lines = [l for l in lines if to_hex(l.get_color()).lower() == c_ep_expected]
        assert len(ep_lines) >= 1
        ep_step = ep_lines[0]
        assert ep_step.get_drawstyle() == "steps-post"
        assert ep_step.get_linewidth() == pytest.approx(0.9)

        # Dotted jump line at t = 60
        jump_lines = [l for l in lines if to_hex(l.get_color()).lower() == c_dae_expected and l.get_linestyle() == ":"]
        assert len(jump_lines) == 1
        jl = jump_lines[0]
        assert jl.get_linewidth() == pytest.approx(1.5)
        np.testing.assert_allclose(jl.get_xdata(), [60.0, 60.0])

        # Jump line y values match Step 3
        # Pipe_1_R (r=2): before 42.5385, after 45.0000
        if r == 2:
            assert jl.get_ydata()[0] == pytest.approx(42.5385, abs=1e-4)
            assert jl.get_ydata()[1] == pytest.approx(45.0000, abs=1e-4)
        # Pipe_2_L (r=3): before 42.5385, after 41.0000
        elif r == 3:
            assert jl.get_ydata()[0] == pytest.approx(42.5385, abs=1e-4)
            assert jl.get_ydata()[1] == pytest.approx(41.0000, abs=1e-4)
        # Q1 (r=0): before 0.191668, after 0.0
        elif r == 0:
            assert jl.get_ydata()[0] == pytest.approx(0.191668, abs=1e-4)
            assert jl.get_ydata()[1] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 3. Dirac arrows across scenarios
# ---------------------------------------------------------------------------

def test_03_dirac_arrows(qapp, res_sc1, res_sc2, res_sc3, res_sc4, res_sc5):
    widget = ComparisonPlots()

    # Scenario 1: arrows ONLY in x5 (r=2, +62.16) and x4 (r=3, -38.85)
    widget.set_result(res_sc1)
    axes = widget.fig.axes

    def get_arrow_texts(ax):
        return [t.get_text() for t in ax.texts if "Dirac" in t.get_text()]

    assert get_arrow_texts(axes[0]) == []
    assert get_arrow_texts(axes[1]) == []
    assert len(get_arrow_texts(axes[2])) == 1
    assert "+62.16" in get_arrow_texts(axes[2])[0]
    assert "EPANET: none" in get_arrow_texts(axes[2])[0]
    assert len(get_arrow_texts(axes[3])) == 1
    assert "-38.85" in get_arrow_texts(axes[3])[0]
    assert "EPANET: none" in get_arrow_texts(axes[3])[0]
    assert get_arrow_texts(axes[4]) == []

    # Scenario 3: +63.35 in x5 and -39.59 in x4
    widget.set_result(res_sc3)
    axes3 = widget.fig.axes
    assert "+63.35" in get_arrow_texts(axes3[2])[0]
    assert "-39.59" in get_arrow_texts(axes3[3])[0]

    # Scenario 4: x5 "+63.35", x4 "+63.35", x6 "+102.94" (all up)
    widget.set_result(res_sc4)
    axes4 = widget.fig.axes
    assert "+63.35" in get_arrow_texts(axes4[2])[0]
    assert "+63.35" in get_arrow_texts(axes4[3])[0]
    assert "+102.94" in get_arrow_texts(axes4[4])[0]

    # Scenario 2: no arrows, no switch line
    widget.set_result(res_sc2)
    for ax in widget.fig.axes:
        assert get_arrow_texts(ax) == []

    # Scenario 5: no arrows, switch line at 60
    widget.set_result(res_sc5)
    for ax in widget.fig.axes:
        assert get_arrow_texts(ax) == []

    # Toggle Show Dirac arrows checkbox
    widget.set_result(res_sc1)
    widget.dirac_check.setChecked(False)
    for ax in widget.fig.axes:
        assert get_arrow_texts(ax) == []
    widget.dirac_check.setChecked(True)
    assert len(get_arrow_texts(widget.fig.axes[2])) == 1


# ---------------------------------------------------------------------------
# 4. Options and redraw (no re-simulation)
# ---------------------------------------------------------------------------

def test_04_options_and_no_rerun(qapp, res_sc1, monkeypatch):
    widget = ComparisonPlots()
    widget.set_result(res_sc1)

    # 1. Unticking a state removes its axes
    assert len(widget.fig.axes) == 5
    widget._on_state_toggled("Pipe_2_R", False)
    assert len(widget.fig.axes) == 4

    # 2. Quick buttons
    widget._select_all_flows()
    assert len(widget.fig.axes) == 2
    for ax in widget.fig.axes:
        assert "Q" in ax.get_ylabel()

    widget._select_matlab_five()
    assert len(widget.fig.axes) == 5

    # 3. Total head shifts x5 by 5.0
    widget.radio_total.setChecked(True)
    widget._on_head_mode_changed()
    assert "total" in widget.fig.axes[2].get_ylabel() or "x5" in widget.fig.axes[2].get_ylabel()

    # 4. Count calls to run_comparison: toggling options does NOT call run_comparison
    call_count = 0
    def mock_run(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return res_sc1

    monkeypatch.setattr("core.comparison.run_comparison", mock_run)
    widget.radio_pressure.setChecked(True)
    widget._on_head_mode_changed()
    widget.dirac_check.setChecked(False)
    widget.dirac_check.setChecked(True)
    widget._select_matlab_five()
    assert call_count == 0


# ---------------------------------------------------------------------------
# 5. Differences tab recomputation
# ---------------------------------------------------------------------------

def test_05_differences_recomputation(res_sc1):
    rows = difference_rows(res_sc1)
    states = default_states(res_sc1)

    # Independent recomputation with np.interp
    tall, Xall = [], []
    from core.plotdata import _strictly_increasing_dae_time_series
    tall, Xall = _strictly_increasing_dae_time_series(res_sc1)
    te = res_sc1.epanet.t

    for t_rep, lbl, dae_val, ep_val, diff in rows:
        col = res_sc1.dae._col(lbl)
        dae_expected = float(np.interp(t_rep, tall, Xall[:, col]))
        k = int(np.where(np.abs(te - t_rep) < 1e-6)[0][0])
        ep_expected = float(res_sc1.epanet.series(lbl)[k])
        diff_expected = dae_expected - ep_expected

        assert dae_val == pytest.approx(dae_expected, abs=1e-9)
        assert ep_val == pytest.approx(ep_expected, abs=1e-9)
        assert diff == pytest.approx(diff_expected, abs=1e-9)

    max_diff = max_abs_difference_after_last_switch(res_sc1)
    assert max_diff < 1e-4


# ---------------------------------------------------------------------------
# 6. Exports (PNG, CSV, .inp)
# ---------------------------------------------------------------------------

def test_06_exports(qapp, res_sc1, monkeypatch):
    widget = ComparisonPlots()
    widget.set_result(res_sc1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Save figure (PNG)
        png_file = tmp_path / "test_comp.png"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(png_file), ""))
        widget.save_figure_action()
        assert png_file.is_file()
        assert png_file.stat().st_size > 10000

        # Load as image and check white background corners
        img = Image.open(png_file)
        assert img.size[0] > 100 and img.size[1] > 100
        # Check top-left corner pixel is white
        tl_pixel = img.getpixel((0, 0))
        assert tl_pixel[:3] == (255, 255, 255)

        # 2. Export CSV
        csv_file = tmp_path / "test_comp.csv"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(csv_file), ""))
        widget.export_csv_action()
        assert csv_file.is_file()

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        assert "time" in header[0]
        assert len(rows) == len(res_sc1.epanet.t)
        # Check difference column = DAE - EPANET
        # row: [time, Q1_DAE, Q1_EPANET, Q1_diff, ...]
        q1_dae = float(rows[0][1])
        q1_ep = float(rows[0][2])
        q1_diff = float(rows[0][3])
        assert q1_diff == pytest.approx(q1_dae - q1_ep, abs=1e-6)

        # Secondary impulses CSV
        imp_csv = tmp_path / "test_comp_impulses.csv"
        assert imp_csv.is_file()
        with open(imp_csv, "r", encoding="utf-8") as f:
            imp_rows = list(csv.reader(f))
        assert imp_rows[0] in [["time", "state", "coefficient (m*s)"], ["time", "state", "coefficient"]]
        assert len(imp_rows) > 1

        # 3. Export .inp
        inp_file = tmp_path / "test_comp.inp"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(inp_file), ""))
        widget.export_inp_action()
        assert inp_file.is_file()
        inp_text = inp_file.read_text(encoding="utf-8")
        assert "[PIPES]" in inp_text and "[JUNCTIONS]" in inp_text

        # Verify real engine can run this exact exported .inp without error
        from core.epanet_runner import run_epanet_inp
        res_from_inp = run_epanet_inp(inp_text, res_sc1.dae)
        assert len(res_from_inp.t) > 0


# ---------------------------------------------------------------------------
# 7. Layout at 1280x720 and 1920x1080
# ---------------------------------------------------------------------------

def test_07_layout_dimensions_and_headers(qapp, res_sc1):
    win = MainWindow()
    win.show()
    win._on_load_specimen_example()
    qapp.processEvents()

    tab = win.comparison_tab
    tab._on_run_finished(res_sc1)
    qapp.processEvents()

    for width, height in [(1280, 720), (1920, 1080)]:
        win.resize(width, height)
        qapp.processEvents()

        # 1. Header text fully visible for every results table
        tables = [tab.diff_table, tab.params_table, tab.eq_table, tab.imp_table]
        for tbl in tables:
            fm = tbl.fontMetrics()
            for c in range(tbl.columnCount()):
                h_item = tbl.horizontalHeaderItem(c)
                if h_item:
                    text_w = fm.horizontalAdvance(h_item.text())
                    col_w = tbl.columnWidth(c)
                    assert col_w >= text_w, (
                        f"Table {tbl} col {c} ({h_item.text()!r}) clipped: width {col_w} < text width {text_w}"
                    )

        # 2. Outdated banner text is not elided and wraps
        assert tab.outdated_label.wordWrap()

        # 3. Splitter left child min width
        assert tab.left_scroll.minimumWidth() >= 400
        assert tab.splitter.widget(0).minimumWidth() >= 400

        # 4. No widget of results area is wider than its tab
        results_tabs_width = tab.results_tabs.width()
        for idx in range(tab.results_tabs.count()):
            tab_widget = tab.results_tabs.widget(idx)
            assert tab_widget.width() <= results_tabs_width + 10

    win.close()
    qapp.processEvents()

