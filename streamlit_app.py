"""Entry point Streamlit Community Cloud looks for at the repository root."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "web" / "app.py"), run_name="__main__")
