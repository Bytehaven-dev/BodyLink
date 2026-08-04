from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs


binaries = [
    entry
    for entry in collect_dynamic_libs("onnxruntime")
    if Path(entry[0]).name.lower() != "onnxruntime_providers_tensorrt.dll"
]
