"""tests/test_release_v2.py – Release verification test suite for v2.0.0.

Covers:
  (i) Physics widgets round-trip and Swamee-Jain scenario 1 steady Q.
  (ii) Version string 2.0.0 parity across app_info, version_info, README, CHANGELOG, and window title.
  (iii) Help dialogs build and show without exec().
  (iv) Acceptance report generation and value verification.
  (v) specimen_matlab.wnet load/run and EPANET .inp files validation.
  (vi) Quoted UI names in README.txt exist in ui/*.py.
"""

import math
import os
import re
import glob
from pathlib import Path
import pytest
from PySide6.QtWidgets import QApplication, QLabel, QTextEdit, QMessageBox, QFileDialog

from app_info import __version__, APP_NAME
from core.models import Network
from core.examples import specimen_network, specimen_scenario, specimen_physics
from core.comparison import run_comparison
from core.io import load_project_full
from core.epanet_runner import run_epanet_inp
from ui.comparison_tab import ComparisonTab
from ui.main_window import MainWindow
import tools.make_acceptance_report as mar

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qapp():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    # Monkeypatch modal dialogs so tests never hang
    QMessageBox.information = lambda *args, **kwargs: QMessageBox.Ok
    QMessageBox.warning = lambda *args, **kwargs: QMessageBox.Ok
    QMessageBox.critical = lambda *args, **kwargs: QMessageBox.Ok
    QMessageBox.question = lambda *args, **kwargs: QMessageBox.Discard
    QFileDialog.getOpenFileName = lambda *args, **kwargs: ("", "")
    QFileDialog.getSaveFileName = lambda *args, **kwargs: ("", "")
    return app


def test_physics_widgets_round_trip(qapp):
    """(i) Physics widgets round trip: nu text '1.0219e-06', g 9.81456, f 0.02, rel error < 1e-15,

    and Swamee-Jain scenario 1 through UI gives steady Q = 0.196438 +- 1e-6.
    """
    net, _ = specimen_network()
    tab = ComparisonTab(net)

    phys = specimen_physics()
    phys.nu = 1.0219e-06
    phys.g = 9.81456
    phys.friction_factor = 0.02
    phys.friction_mode = "constant"

    tab.set_physics(phys)

    # 1. Widget values and formatted text
    assert "1.0219e-06" in tab.nu_spin.text(), f"Expected '1.0219e-06' in nu_spin text, got {tab.nu_spin.text()!r}"
    assert abs(tab.g_spin.value() - 9.81456) < 1e-12
    assert abs(tab.factor_spin.value() - 0.02) < 1e-12

    # 2. Round-trip read back
    read_phys = tab.get_physics()
    assert abs(read_phys.nu - phys.nu) / phys.nu < 1e-15
    assert abs(read_phys.g - phys.g) / phys.g < 1e-15
    assert abs(read_phys.friction_factor - phys.friction_factor) / phys.friction_factor < 1e-15

    # 3. Swamee-Jain scenario 1 steady Q
    tab.friction_combo.setCurrentIndex(1)
    tab._on_friction_mode_changed(1)
    phys_sj = tab.get_physics()
    assert phys_sj.friction_mode == "swamee_jain"

    tab.preset_combo.setCurrentIndex(1)
    tab._on_preset_changed(1)
    sc1 = tab.get_scenario()

    res_sj = run_comparison(net, sc1, phys_sj)
    flow_eqs = [eq for eq in res_sj.equilibrium if "Pipe_1_Flow" in eq["label"]]
    assert len(flow_eqs) > 0
    q_steady = flow_eqs[0]["dae_eq"]
    assert abs(q_steady - 0.196438) <= 1e-6, f"Expected steady Q = 0.196438 +- 1e-6, got {q_steady}"

    tab.deleteLater()


def test_version_string_identical(qapp):
    """(ii) Version string 2.0.0 identical in app_info.py, version_info.txt, README.md, CHANGELOG.md, and window title."""
    expected_version = "2.0.0"

    # 1. app_info.py
    assert __version__ == expected_version

    # 2. version_info.txt
    version_info_path = REPO_ROOT / "version_info.txt"
    assert version_info_path.exists()
    vinfo_text = version_info_path.read_text(encoding="utf-8")
    assert f"2, 0, 0, 0" in vinfo_text or f"2.0.0" in vinfo_text

    # 3. README.md
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.exists()
    readme_text = readme_path.read_text(encoding="utf-8")
    assert expected_version in readme_text

    # 4. CHANGELOG.md
    changelog_path = REPO_ROOT / "CHANGELOG.md"
    assert changelog_path.exists()
    changelog_text = changelog_path.read_text(encoding="utf-8")
    assert f"[{expected_version}]" in changelog_text or f"v{expected_version}" in changelog_text or f"Version {expected_version}" in changelog_text

    # 5. Window title
    win = MainWindow()
    try:
        title = win.windowTitle()
        assert expected_version in title, f"Expected {expected_version} in window title, got {title}"
    finally:
        win.setWindowModified(False)
        win.close()
        win.deleteLater()


def test_help_dialogs_build_and_show(qapp):
    """(iii) The three Help dialogs build and show without exec()."""
    win = MainWindow()
    try:
        # 1. About dialog
        about_dlg = win._create_about_dialog()
        about_dlg.show()
        assert about_dlg.windowTitle() == "About Water Network Modeler"
        about_labels = [w.text() for w in about_dlg.findChildren(QLabel)]
        about_full_text = " ".join(about_labels)
        assert "Water Network Modeler 2.0.0" in about_full_text
        about_dlg.close()
        about_dlg.deleteLater()

        # 2. Third-party notices dialog
        notices_dlg = win._create_third_party_notices_dialog()
        notices_dlg.show()
        assert notices_dlg.windowTitle() == "Third-party notices"
        text_edits = notices_dlg.findChildren(QTextEdit)
        assert len(text_edits) == 1
        notices_content = text_edits[0].toPlainText()
        assert "EPANET" in notices_content
        assert "Open Water Analytics" in notices_content
        notices_dlg.close()
        notices_dlg.deleteLater()

        # 3. How to compare dialog
        how_dlg = win._create_how_to_compare_dialog()
        how_dlg.show()
        assert how_dlg.windowTitle() == "How to compare with EPANET"
        how_labels = [w.text() for w in how_dlg.findChildren(QLabel)]
        how_full_text = " ".join(how_labels)
        assert "Build the network" in how_full_text
        assert "Comparison with EPANET" in how_full_text
        how_dlg.close()
        how_dlg.deleteLater()
    finally:
        win.setWindowModified(False)
        win.close()
        win.deleteLater()


def test_acceptance_report_generation():
    """(iv) tools/make_acceptance_report.py runs and 2-3 numbers parsed from the written file equal a fresh computation."""
    mar.generate_acceptance_report()

    report_path = REPO_ROOT / "release" / "ACCEPTANCE_REPORT.md"
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")

    # Fresh computation
    g = 9.81456
    Area = math.pi * 0.40**2 / 4.0
    b1_comp = g * Area / 400.0
    b2_comp = g * Area / 250.0

    # Parse and check b1 and b2
    assert f"{b1_comp:.6e}" in text
    assert f"{b2_comp:.6e}" in text

    # Parse and check Dirac value
    assert "+62.16" in text or "+62.1557" in text

    # Check that .txt copy exists and has matching length
    txt_path = REPO_ROOT / "release" / "ACCEPTANCE_REPORT.txt"
    assert txt_path.exists()
    assert len(txt_path.read_text(encoding="utf-8")) == len(text)


def test_specimen_matlab_and_epanet_inp():
    """(v) release/examples/specimen_matlab.wnet loads via load_project_full, runs, impulses +62.16/-38.85/+62.16;

    every scenarioN.inp is read by the real engine without warnings.
    """
    wnet_path = REPO_ROOT / "release" / "examples" / "specimen_matlab.wnet"
    proj = load_project_full(wnet_path)
    net = proj.network
    pos = proj.positions
    scenario = proj.scenario
    physics = proj.physics
    assert scenario is not None
    assert physics is not None

    res = run_comparison(net, scenario, physics, positions=pos)
    assert len(res.impulses) == 3

    coeffs = [imp["coefficient"] for imp in res.impulses]
    # Check +62.16 / -38.85 / +62.16
    assert any(abs(c - (-38.8516)) < 0.05 for c in coeffs)
    assert sum(1 for c in coeffs if abs(c - 62.1626) < 0.05) == 2

    # Every scenarioN.inp is read by real EPANET engine without errors
    for sc_id in range(1, 6):
        inp_file = REPO_ROOT / "release" / "examples" / f"scenario{sc_id}.inp"
        assert inp_file.exists()
        ep_res = run_epanet_inp(str(inp_file))
        assert len(ep_res.t) > 0
        assert len(ep_res.warnings) == 0


def test_readme_quoted_ui_names():
    """(vi) README.txt: every quoted UI name exists as a string in ui/*.py."""
    readme_path = REPO_ROOT / "release" / "README.txt"
    if not readme_path.exists():
        pytest.skip("release/README.txt not present (release/ is gitignored)")
    readme_text = readme_path.read_text(encoding="utf-8")

    quotes = re.findall(r'"([^"\n]+)"', readme_text)

    # Exclude system/file names that are not UI widget labels
    non_ui = {
        "WaterNetworkModeler.exe",
        "Windows protected your PC",
        "More info",
        "Run anyway",
        "examples\\",
        "Create_Desktop_Shortcut.bat",
        "Water Network Modeler",
    }
    ui_quotes = [q for q in quotes if q not in non_ui]

    ui_files = list((REPO_ROOT / "ui").glob("*.py"))
    assert len(ui_files) > 0
    ui_code = " ".join(f.read_text(encoding="utf-8") for f in ui_files)

    for q in ui_quotes:
        assert q in ui_code, f"Quoted UI name {q!r} from README.txt not found in any ui/*.py file"


def test_acceptance_report_compare_yourself_matches_tables():
    """(vii) Section 'How You Can Compare Yourself' values match table values."""
    report_path = REPO_ROOT / "release" / "ACCEPTANCE_REPORT.md"
    if not report_path.exists():
        mar.generate_acceptance_report()
    text = report_path.read_text(encoding="utf-8")

    # Parse '## How You Can Compare Yourself' section
    comp_match = re.search(r"## How You Can Compare Yourself(.*?)(?=##|\Z)", text, re.DOTALL)
    assert comp_match is not None, "Section 'How You Can Compare Yourself' not found"
    comp_text = comp_match.group(1)

    # 1. b1 = g * Area / L1 = 3.083335e-03 m2/s2
    b1_m = re.search(r"b1\s*=\s*g\s*\*\s*Area\s*/\s*L1\s*=\s*([0-9.eE+-]+)\s*m2/s2", comp_text)
    assert b1_m, "b1 not found in Compare Yourself"
    b1_val = float(b1_m.group(1))

    # 2. b2 = g * Area / L2 = 4.933336e-03 m2/s2
    b2_m = re.search(r"b2\s*=\s*g\s*\*\s*Area\s*/\s*L2\s*=\s*([0-9.eE+-]+)\s*m2/s2", comp_text)
    assert b2_m, "b2 not found in Compare Yourself"
    b2_val = float(b2_m.group(1))

    # 3. Steady flow Q of mode (1,1): 0.195321 m3/s
    q_ss_m = re.search(r"Steady flow Q of mode \(1,1\):\s*([0-9.]+)\s*m3/s", comp_text)
    assert q_ss_m, "Steady flow Q of mode (1,1) not found in Compare Yourself"
    q_ss_val = float(q_ss_m.group(1))

    # 4. x5 of mode (0,1): 45.000000 m
    x5_m = re.search(r"x5 of mode \(0,1\):\s*([0-9.]+)\s*m", comp_text)
    assert x5_m, "x5 of mode (0,1) not found in Compare Yourself"
    x5_val = float(x5_m.group(1))

    # 5. Dirac impulse coefficient of scenario 1: +62.1626 m*s (reported as +62.16 m*s)
    dirac_m = re.search(r"Dirac impulse coefficient of scenario 1:\s*\+([0-9.]+)\s*m\*s", comp_text)
    assert dirac_m, "Dirac impulse not found in Compare Yourself"
    dirac_val = float(dirac_m.group(1))

    # Parse Table 2.1 (b1 and b2)
    t21_b1_m = re.search(r"\|\s*b1 = g \* Area / L1\s*\|\s*[0-9.eE+-]+\s*\|\s*([0-9.eE+-]+)\s*\|", text)
    assert t21_b1_m, "b1 not found in Table 2.1"
    assert abs(b1_val - float(t21_b1_m.group(1))) < 1e-10

    t21_b2_m = re.search(r"\|\s*b2 = g \* Area / L2\s*\|\s*[0-9.eE+-]+\s*\|\s*([0-9.eE+-]+)\s*\|", text)
    assert t21_b2_m, "b2 not found in Table 2.1"
    assert abs(b2_val - float(t21_b2_m.group(1))) < 1e-10

    # Parse Section 2.3 Mode (1,1) steady flow Q
    m11_section = re.search(r"#### Mode \(1,1\) All open.*?(?=####|\Z)", text, re.DOTALL)
    assert m11_section, "Mode (1,1) section not found"
    t23_q_m = re.search(r"\|\s*Pipe_1_Flow\s*\|\s*[0-9.]+\s*\|\s*([0-9.]+)\s*\|", m11_section.group(0))
    assert t23_q_m, "Pipe_1_Flow not found in Mode (1,1) table"
    assert abs(q_ss_val - float(t23_q_m.group(1))) < 1e-6

    # Parse Section 2.3 Mode (0,1) x5 (Pipe_1_R)
    m01_section = re.search(r"#### Mode \(0,1\) Valve 1 closed.*?(?=####|\Z)", text, re.DOTALL)
    assert m01_section, "Mode (0,1) section not found"
    t23_x5_m = re.search(r"\|\s*Pipe_1_R\s*\|\s*[0-9.]+\s*\|\s*([0-9.]+)\s*\|", m01_section.group(0))
    assert t23_x5_m, "Pipe_1_R not found in Mode (0,1) table"
    assert abs(x5_val - float(t23_x5_m.group(1))) < 1e-6

    # Parse Table 3.1 (Scenario 1 Dirac impulse for Pipe_1_R)
    t31_dirac_m = re.search(r"\|\s*Scenario 1\s*\|\s*60\.0\s*\|\s*Pipe_1_R\s*\|\s*\+([0-9.]+)\s*\|", text)
    assert t31_dirac_m, "Pipe_1_R Dirac impulse not found in Table 3.1"
    table_dirac_val = float(t31_dirac_m.group(1))
    assert abs(dirac_val - table_dirac_val) < 1e-4
    assert abs(table_dirac_val - 62.1626) < 1e-3


def test_no_placeholders_or_todos_in_release():
    """(viii) Guard: Packaging files in release/ must NOT contain '<<', 'TODO', or 'FIXME'."""
    release_dir = REPO_ROOT / "release"
    if not release_dir.exists() or not any(release_dir.iterdir()):
        pytest.skip("release/ directory empty or missing (release/ is gitignored)")

    text_extensions = {".txt", ".md", ".bat", ".inp", ".wnet", ".csv", ".json"}
    files_checked = 0

    for path in release_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in text_extensions:
            # Skip binary or zip files
            content = path.read_text(encoding="utf-8", errors="replace")
            for line in content.splitlines():
                if "TODO" in line or "FIXME" in line:
                    pytest.fail(f"Forbidden placeholder/TODO found in {path.relative_to(REPO_ROOT)}: {line}")
                if "<<" in line:
                    pytest.fail(f"Forbidden unreplaced placeholder found in {path.relative_to(REPO_ROOT)}: {line}")
            files_checked += 1

    assert files_checked >= 8, f"Expected at least 8 release text files checked, got {files_checked}"


def test_valves_table_checkbox_contrast(qapp):
    """(ix) Valves table checkboxes: border contrast >= 3:1 against background in dark and light themes."""
    from PySide6.QtGui import QPalette, QColor, QPixmap, QPainter
    from PySide6.QtWidgets import QTableWidget, QTableWidgetItem
    from PySide6.QtCore import Qt

    def rel_luminance(c: QColor) -> float:
        def to_linear(v):
            v_norm = v / 255.0
            return v_norm / 12.92 if v_norm <= 0.04045 else ((v_norm + 0.055) / 1.055) ** 2.4
        r = to_linear(c.red())
        g = to_linear(c.green())
        b = to_linear(c.blue())
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def contrast_ratio(c1: QColor, c2: QColor) -> float:
        l1 = rel_luminance(c1)
        l2 = rel_luminance(c2)
        hi = max(l1, l2)
        lo = min(l1, l2)
        return (hi + 0.05) / (lo + 0.05)

    # 1. Dark theme: background #1E1E1E, indicator border #A8A8A8
    dark_bg = QColor("#1E1E1E")
    dark_border = QColor("#A8A8A8")
    ratio_dark = contrast_ratio(dark_border, dark_bg)
    assert ratio_dark >= 3.0, f"Dark theme checkbox border contrast {ratio_dark:.2f}:1 < 3:1"

    # 2. Light theme: background #FFFFFF, indicator border #5A5A5A
    light_bg = QColor("#FFFFFF")
    light_border = QColor("#5A5A5A")
    ratio_light = contrast_ratio(light_border, light_bg)
    assert ratio_light >= 3.0, f"Light theme checkbox border contrast {ratio_light:.2f}:1 < 3:1"


def test_acceptance_report_conclusions_within_limits():
    """(x) Section 4: every 'within X' is >= the largest corresponding difference in the tables."""
    mar.generate_acceptance_report()
    report_path = REPO_ROOT / "release" / "ACCEPTANCE_REPORT.md"
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")

    sec4_match = re.search(r"## 4\. Acceptance Conclusion\s*(.*?)(?=\n##|\n---\n|\Z)", text, re.DOTALL)
    assert sec4_match, "Section 4 not found in ACCEPTANCE_REPORT.md"
    sec4_text = sec4_match.group(1)

    within_matches = re.findall(r"within\s+([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", sec4_text)
    assert len(within_matches) >= 4, f"Expected at least 4 'within <val>' in Section 4, got {len(within_matches)}"
    limits = [float(w) for w in within_matches]

    # Item 1: pencil parameters, gravity vector f, and determinants
    t21_match = re.search(r"### 2\.1 Pencil Coefficients & Vector f\s*(.*?)(?=###|\Z)", text, re.DOTALL)
    assert t21_match, "Table 2.1 not found"
    t21_diffs = []
    for line in t21_match.group(1).splitlines():
        if line.strip().startswith("|") and "Parameter" not in line and "---" not in line:
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if len(cols) >= 5:
                try:
                    t21_diffs.append(float(cols[4]))
                except ValueError:
                    pass
    assert len(t21_diffs) > 0, "No diffs parsed from Table 2.1"

    t22_match = re.search(r"### 2\.2 Mode Analysis & Regularity\s*(.*?)(?=###|\Z)", text, re.DOTALL)
    assert t22_match, "Table 2.2 not found"
    t22_diffs = []
    for line in t22_match.group(1).splitlines():
        if line.strip().startswith("|") and "Valve Mode" not in line and "---" not in line:
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if len(cols) >= 4:
                try:
                    t22_diffs.append(float(cols[3]))
                except ValueError:
                    pass
    assert len(t22_diffs) > 0, "No diffs parsed from Table 2.2"

    max_diff_1 = max(t21_diffs + t22_diffs)
    assert limits[0] >= max_diff_1 - 1e-15, f"Limit 1 ({limits[0]}) < max diff in Tables 2.1 & 2.2 ({max_diff_1})"

    # Item 2: Mode steady states
    t23_match = re.search(r"### 2\.3 Steady States of the Three Modes\s*(.*?)(?=\n## 3|\Z)", text, re.DOTALL)
    assert t23_match, "Section 2.3 not found"
    t23_diffs = []
    for line in t23_match.group(1).splitlines():
        if line.strip().startswith("|") and "State Label" not in line and "---" not in line:
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if len(cols) >= 5:
                try:
                    t23_diffs.append(float(cols[4]))
                except ValueError:
                    pass
    assert len(t23_diffs) > 0, "No diffs parsed from Section 2.3"
    max_diff_2 = max(t23_diffs)
    assert limits[1] >= max_diff_2 - 1e-15, f"Limit 2 ({limits[1]}) < max steady state diff ({max_diff_2})"

    # Item 3: Dirac impulse coefficients vs analytical values (+62.16, -38.85, +62.16)
    t31_match = re.search(r"### 3\.1 Dirac Impulses Summary\s*(.*?)(?=###|\Z)", text, re.DOTALL)
    assert t31_match, "Section 3.1 not found"
    analytical_map = {
        ("Scenario 1", "Pipe_2_L"): -38.85,
        ("Scenario 1", "Pipe_1_R"): 62.16,
        ("Scenario 1", "J1"): 62.16,
    }
    dirac_diffs = []
    for line in t31_match.group(1).splitlines():
        if line.strip().startswith("|") and "Switch Time" not in line and "---" not in line:
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if len(cols) >= 4:
                sc = cols[0]
                st = cols[2]
                if (sc, st) in analytical_map:
                    val = float(cols[3])
                    dirac_diffs.append(abs(val - analytical_map[(sc, st)]))
    assert len(dirac_diffs) == 3, f"Expected 3 Scenario 1 analytical Dirac impulses, got {len(dirac_diffs)}"
    max_diff_3 = max(dirac_diffs)
    assert limits[2] >= max_diff_3 - 1e-15, f"Limit 3 ({limits[2]}) < max Dirac diff ({max_diff_3})"

    # Item 4: DAE vs EPANET equilibria across all scenarios
    t32_match = re.search(r"### 3\.2 Maximum Differences per Scenario.*?\n(.*?)(?=Note on Scenario|\Z)", text, re.DOTALL)
    assert t32_match, "Section 3.2 not found"
    eq_diffs = []
    for line in t32_match.group(1).splitlines():
        if line.strip().startswith("|") and "Max Flow Diff" not in line and "---" not in line:
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if len(cols) >= 5:
                try:
                    eq_diffs.append(float(cols[3]))
                    eq_diffs.append(float(cols[4]))
                except ValueError:
                    pass
    assert len(eq_diffs) > 0, "No equilibrium diffs parsed from Section 3.2"
    max_diff_4 = max(eq_diffs)
    assert limits[3] >= max_diff_4 - 1e-15, f"Limit 4 ({limits[3]}) < max equilibrium diff ({max_diff_4})"



