# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PySide6.QtWidgets import QApplication
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

_data_files = (
    collect_data_files('lightning_fabric')
    + collect_data_files('lightning')
    + collect_data_files('jsonargparse')
)

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('config.yaml', '.'),
        ('driver', 'driver'),
    ] + _data_files,
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
        'scipy.linalg',
        'scipy.linalg.cython_lapack',
        'scipy.linalg.cython_blas',
        'scipy.linalg._cythonized_array_utils',
        'lightning',
        'lightning.pytorch',
        'lightning.pytorch.cli',
        'lightning_fabric',
        'jsonargparse',
        'torchmetrics',
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'PIL.ImageFont',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'IPython',
        'jupyter',
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
    console=True,
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