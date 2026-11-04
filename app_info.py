"""Application metadata and resource resolution helper."""

import sys
from pathlib import Path

APP_NAME = "Water Network Modeler"
APP_ID = "WaterNetworkModeler"
COMPANY_NAME = "Abdul Haseeb"
__version__ = "2.0.0"


def resource_path(relative_path: str | Path) -> Path:
    """Get absolute path to a resource.

    Works both during development and when bundled by PyInstaller
    (where sys._MEIPASS holds the temporary extraction directory).
    """
    if hasattr(sys, "_MEIPASS"):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parent

    return (base_path / relative_path).resolve()
