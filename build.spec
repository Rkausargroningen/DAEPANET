# -*- mode: python ; coding: utf-8 -*-
# build.spec -- PyInstaller spec for Water Network Modeler

import sys
from pathlib import Path

block_cipher = None

sys.path.insert(0, str(SPECPATH))

# ---------------------------------------------------------------------------
# Hidden imports needed when frozen
# ---------------------------------------------------------------------------
hidden_imports = [
    'matplotlib.backends.backend_qtagg',
    'matplotlib.backends.backend_qt5agg',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'PySide6.QtSvg',
    'PySide6.QtOpenGL',
    'PySide6.QtTest',
    'networkx.algorithms.components',
    'networkx.algorithms.cycles',
    'numpy.core._multiarray_umath',
]

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------
datas = [
    ('assets', 'assets'),
    ('THIRD_PARTY_NOTICES.txt', '.'),
]

# ---------------------------------------------------------------------------
# Exclusions
# ---------------------------------------------------------------------------
excludes = [
    'tkinter',
    'PyQt5',
    'PyQt6',
    'IPython',
    'pytest',
    'pandas',
]

binaries = [
    ('core/lib/epanet2.dll', 'core/lib'),
    ('core/lib/epanet2.dll', '.'),
]

a = Analysis(
    ['main.py'],
    pathex=[SPECPATH],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WaterNetworkModeler',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets\\app.ico',
    version='version_info.txt',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='WaterNetworkModeler',
)
