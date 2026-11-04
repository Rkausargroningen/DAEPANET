"""In-process selftest pytest."""
import sys
from pathlib import Path


def test_in_process_selftest(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from main import run_selftest

    log_file = tmp_path / "wnm_selftest.log"
    exit_code = run_selftest(log_path=log_file)

    assert log_file.is_file(), "Selftest log file was not created"
    content = log_file.read_text(encoding="utf-8")

    assert exit_code == 0, (
        "run_selftest() returned %s (expected 0). Log:\n%s" % (exit_code, content[-2000:])
    )
    assert "=== Selftest PASSED (code 0) ===" in content
    assert "Physical parameters and .wnet v2 round-trip verified" in content
    assert "v1-format project compatibility verified" in content
    assert "render_to_image verified" in content
    assert "Node drag simulated: positions_changed emitted" in content
    assert "Paper network verified" in content
    assert "Broken network verified" in content
    assert "Project successfully reloaded and verified" in content
    assert "EPANET engine execution verified successfully" in content
