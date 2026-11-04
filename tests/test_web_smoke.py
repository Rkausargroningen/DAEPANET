"""Smoke test: the Streamlit app loads the example networks without a browser."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "web" / "app.py"


def _values(items) -> list[str]:
    return [str(getattr(item, "value", item)) for item in items]


def test_web_app_examples_render():
    app = AppTest.from_file(str(APP), default_timeout=90)
    app.run(timeout=90)
    assert not app.exception, app.exception

    app.button(key="load_paper").click()
    app.session_state["main_tabs"] = "Checks"
    app.run(timeout=90)
    assert not app.exception, app.exception
    assert any("Regular" in text for text in _values(app.success))

    app.button(key="load_broken").click()
    app.session_state["main_tabs"] = "Checks"
    app.run(timeout=90)
    assert not app.exception, app.exception
    assert any("NOT regular" in text for text in _values(app.error))

    app.button(key="load_specimen").click()
    app.session_state["main_tabs"] = "Comparison with EPANET"
    app.run(timeout=90)
    assert not app.exception, app.exception
    assert any("Ready to compare" in text for text in _values(app.success))
