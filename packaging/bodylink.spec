from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


PROJECT_ROOT = Path(SPECPATH).resolve().parent
SITE_PACKAGES = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
BUILD_DIR = PROJECT_ROOT / "build"

datas = [
    (str(PROJECT_ROOT / "LICENSE"), "."),
    (str(PROJECT_ROOT / "README.md"), "."),
    (str(PROJECT_ROOT / "CHANGELOG.md"), "."),
    (str(PROJECT_ROOT / "ROADMAP.md"), "."),
    (str(PROJECT_ROOT / "THIRD_PARTY_NOTICES.md"), "."),
    (str(PROJECT_ROOT / "VERSION"), "."),
]
datas += collect_data_files("mediapipe")

license_patterns = {
    "mediapipe": "mediapipe-*.dist-info/licenses/*",
    "numpy": "numpy-*.dist-info/LICENSE*",
    "nvidia-cublas": "nvidia_cublas-*.dist-info/licenses/*",
    "nvidia-cuda-runtime": "nvidia_cuda_runtime-*.dist-info/licenses/*",
    "nvidia-cudnn": "nvidia_cudnn_cu13-*.dist-info/licenses/*",
    "nvidia-cufft": "nvidia_cufft-*.dist-info/licenses/*",
    "onnxruntime": "onnxruntime_gpu-*.dist-info/LICENSE*",
    "opencv": "opencv_contrib_python-*.dist-info/LICENSE*",
    "openvr": "openvr-*.dist-info/licenses/*",
    "pygrabber": "pygrabber-*.dist-info/LICENSE*",
    "pyside6": "pyside6-*.dist-info/licenses/*",
    "rtmlib": "rtmlib-*.dist-info/licenses/*",
}
for component, pattern in license_patterns.items():
    for license_path in SITE_PACKAGES.glob(pattern):
        if license_path.is_file():
            datas.append((str(license_path), f"licenses/{component}"))

required_cuda_dlls = (
    "cublasLt64_13.dll",
    "cublas64_13.dll",
    "cufft64_12.dll",
    "cudart64_13.dll",
)
required_cudnn_dlls = (
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn64_9.dll",
    "cudnn_engines_tensor_ir64_9.dll",
)

binaries = [
    (str(SITE_PACKAGES / "openvr" / "libopenvr_api_64.dll"), "openvr"),
]
cuda_directory = SITE_PACKAGES / "nvidia" / "cu13" / "bin" / "x86_64"
for name in required_cuda_dlls:
    binaries.append((str(cuda_directory / name), "nvidia/cu13/bin/x86_64"))
cudnn_directory = SITE_PACKAGES / "nvidia" / "cudnn" / "bin"
for name in required_cudnn_dlls:
    binaries.append((str(cudnn_directory / name), "nvidia/cudnn/bin"))

hiddenimports = sorted(
    set(
        collect_submodules("rtmlib")
        + collect_submodules("comtypes.gen")
        + [
            "openvr",
            "pygrabber.dshow_graph",
            "mediapipe.tasks.python.vision",
            "onnxruntime.capi._pybind_state",
        ]
    )
)

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(PROJECT_ROOT / "packaging" / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "jupyter",
        "openvino",
        "pandas",
        "scipy",
        "tkinter",
        "torch",
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BodyLink",
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
    icon=str(BUILD_DIR / "bodylink.ico"),
    version=str(PROJECT_ROOT / "packaging" / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BodyLink",
)
