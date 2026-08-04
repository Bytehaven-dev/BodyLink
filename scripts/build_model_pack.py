from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


MODEL_FILES = (
    "rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7.onnx",
    "yolox-m-humanart.onnx",
    "face_landmarker.task",
)
ZIP_TIMESTAMP = (2026, 8, 5, 0, 0, 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def build_model_pack(models_directory: Path, output: Path, version: str) -> None:
    files = [models_directory / name for name in MODEL_FILES]
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing models: " + ", ".join(missing))

    manifest = {
        "format": 1,
        "bodylink_version": version,
        "models": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            manifest_bytes = json.dumps(
                manifest,
                ensure_ascii=True,
                indent=2,
            ).encode("utf-8")
            archive.writestr(_zip_info("models/manifest.json"), manifest_bytes)
            for path in files:
                with path.open("rb") as source, archive.open(
                    _zip_info(f"models/{path.name}"),
                    "w",
                    force_zip64=True,
                ) as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_model_pack(
        args.models.resolve(),
        args.output.resolve(),
        args.version,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
