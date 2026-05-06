# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PySide6.QtWidgets import QApplication

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('config.yaml', '.'),
        ('driver', 'driver'),
    ],
    hiddenimports=[
        'mvsdk',
        'defect_detection',
        'camera',
        'anomalib',
        'anomalib.models',
        'anomalib.data',
        'anomalib.engine',
        'anomalib.models.padim',
        'anomalib.data.folder',
        'torch',
        'torchvision',
        'torchvision.transforms',
        'albumentations',
        'cv2',
        'yaml',
        'numpy',
        'sklearn',
        'sklearn.utils',
        'sklearn.utils._weight_vector',
        'scipy',
        'scipy.spatial',
        'scipy.spatial.distance',
        'lightning',
        'lightning.pytorch',
        'lightning.pytorch.cli',
        'jsonargparse',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'IPython',
        'jupyter',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_lapack',
        'scipy._csparsetools',
        'scipy._dsparsetools',
        'scipy._bsparsetools',
        'scipy._ssparsetools',
    ],
    noarchive=False,
    optimize=0,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DefectDetection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
    uac_admin=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    a.scripts,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='DefectDetection',
)