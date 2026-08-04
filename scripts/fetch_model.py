from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bodylink import __version__


RTMW3D_MODEL_URL = (
    "https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/"
    "rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx"
)
RTMW3D_MODEL_BYTES = 369_330_857
RTMW3D_MODEL_SHA256 = "4a289c0e99d47eb595e99679d9d4a2d1def1b4241f9adcbafba44b9ff585ebcd"

DETECTOR_ARCHIVE_URL = (
    "https://huggingface.co/Tau-J/RTMPose/resolve/main/rtmposev1/onnx_sdk/"
    "yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
)
DETECTOR_ARCHIVE_BYTES = 94_223_081
DETECTOR_ARCHIVE_SHA256 = "a000224fd8ba283202bc62d4a5fcdfe353adb9f468777dbac1ea2ada2093adde"

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MIN_FACE_MODEL_BYTES = 3_000_000
MIN_DETECTOR_MODEL_BYTES = 50_000_000
FACE_MODEL_BYTES = 3_758_596
FACE_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
DETECTOR_MODEL_BYTES = 101_400_344
DETECTOR_MODEL_SHA256 = "3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23"


def models_directory() -> Path:
    return Path(__file__).resolve().parents[1] / "models"


def model_path() -> Path:
    return models_directory() / "rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7.onnx"


def detector_model_path() -> Path:
    return models_directory() / "yolox-m-humanart.onnx"


def face_model_path() -> Path:
    return models_directory() / "face_landmarker.task"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ready(
    path: Path,
    *,
    minimum_bytes: int = 1,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> bool:
    if not path.exists() or path.stat().st_size < minimum_bytes:
        return False
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        return False
    return expected_sha256 is None or _sha256(path) == expected_sha256


def _download(
    url: str,
    target: Path,
    label: str,
    *,
    minimum_bytes: int = 1,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> Path:
    if _ready(
        target,
        minimum_bytes=minimum_bytes,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    ):
        print(f"{label} ready: {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if expected_bytes is not None and partial.exists() and partial.stat().st_size > expected_bytes:
        partial.unlink()
    last_percent = -1
    last_error: Exception | None = None
    for attempt in range(8):
        offset = partial.stat().st_size if partial.exists() else 0
        if expected_bytes is not None and offset == expected_bytes:
            break
        headers = {"User-Agent": f"BodyLink/{__version__}"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = int(getattr(response, "status", response.getcode()))
                if offset and status != 206:
                    offset = 0
                    partial.unlink(missing_ok=True)
                reported_length = int(response.headers.get("Content-Length") or 0)
                total = expected_bytes or (offset + reported_length)
                mode = "ab" if offset else "wb"
                with partial.open(mode) as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        offset += len(chunk)
                        if total:
                            percent = min(100, int(offset * 100 / total))
                            if percent != last_percent:
                                print(
                                    f"\rDownloading {label}: {percent:3d}%",
                                    end="",
                                    flush=True,
                                )
                                last_percent = percent
            if expected_bytes is None or partial.stat().st_size == expected_bytes:
                break
            last_error = RuntimeError(
                f"connection ended at {partial.stat().st_size} of {expected_bytes} bytes"
            )
        except Exception as exc:
            last_error = exc
        if attempt < 7:
            current = partial.stat().st_size if partial.exists() else 0
            print(f"\n{label} connection interrupted at {current} bytes; resuming...")
            time.sleep(min(8, 2 ** attempt))
    else:
        raise RuntimeError(f"{label} download failed after retries: {last_error}")

    if last_percent >= 0:
        print()
    downloaded = partial.stat().st_size if partial.exists() else 0
    if downloaded < minimum_bytes:
        raise RuntimeError(f"{label} download is unexpectedly small")
    if expected_bytes is not None and downloaded != expected_bytes:
        raise RuntimeError(
            f"{label} size mismatch: expected {expected_bytes}, got {downloaded}"
        )
    if expected_sha256 is not None and _sha256(partial) != expected_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{label} SHA256 mismatch")
    os.replace(partial, target)
    print(f"{label} ready: {target}")
    return target


def download_model(destination: Path | None = None) -> Path:
    return _download(
        RTMW3D_MODEL_URL,
        destination or model_path(),
        "RTMW3D model",
        expected_bytes=RTMW3D_MODEL_BYTES,
        expected_sha256=RTMW3D_MODEL_SHA256,
    )


def download_detector_model(destination: Path | None = None) -> Path:
    target = destination or detector_model_path()
    if _ready(
        target,
        minimum_bytes=MIN_DETECTOR_MODEL_BYTES,
        expected_bytes=DETECTOR_MODEL_BYTES,
        expected_sha256=DETECTOR_MODEL_SHA256,
    ):
        print(f"Person detector ready: {target}")
        return target

    archive = models_directory() / "yolox-m-humanart.zip"
    _download(
        DETECTOR_ARCHIVE_URL,
        archive,
        "person detector archive",
        expected_bytes=DETECTOR_ARCHIVE_BYTES,
        expected_sha256=DETECTOR_ARCHIVE_SHA256,
    )
    partial = target.with_suffix(target.suffix + ".part")
    partial.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(archive) as package:
            candidates = [
                name for name in package.namelist() if name.lower().endswith("end2end.onnx")
            ]
            if len(candidates) != 1:
                raise RuntimeError("person detector archive has unexpected contents")
            with package.open(candidates[0]) as source, partial.open("wb") as out:
                shutil.copyfileobj(source, out, length=1024 * 1024)
        if not _ready(
            partial,
            minimum_bytes=MIN_DETECTOR_MODEL_BYTES,
            expected_bytes=DETECTOR_MODEL_BYTES,
            expected_sha256=DETECTOR_MODEL_SHA256,
        ):
            raise RuntimeError("extracted person detector failed verification")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
        archive.unlink(missing_ok=True)
    print(f"Person detector ready: {target}")
    return target


def download_face_model(destination: Path | None = None) -> Path:
    return _download(
        FACE_MODEL_URL,
        destination or face_model_path(),
        "MediaPipe face model",
        minimum_bytes=MIN_FACE_MODEL_BYTES,
        expected_bytes=FACE_MODEL_BYTES,
        expected_sha256=FACE_MODEL_SHA256,
    )


if __name__ == "__main__":
    try:
        download_model()
        download_detector_model()
        download_face_model()
    except Exception as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
