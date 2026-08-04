# BodyLink Roadmap

## Optional TensorRT acceleration

BodyLink 0.3.0 uses ONNX Runtime CUDA for both YOLOX-M and RTMW3D-X. CUDA remains the default backend because it works with the bundled runtime and does not require a machine-specific engine build.

A future release may add TensorRT as an optional FP16 acceleration mode with these constraints:

- Use `TensorrtExecutionProvider` first and `CUDAExecutionProvider` as the explicit fallback.
- Keep the CUDA mode available and selectable; TensorRT must never be a mandatory dependency.
- Build and cache engines per model hash, GPU architecture, TensorRT version, and precision mode.
- Show the potentially long first-run engine build in the UI and allow cancellation.
- Invalidate incompatible caches after model, driver, CUDA, or TensorRT changes.
- Report the actual provider used by both detector and pose sessions instead of claiming TensorRT when a graph falls back to CUDA.

Acceptance criteria:

- YOLOX-M and RTMW3D-X both initialize on a supported NVIDIA GPU without silent CPU fallback.
- FP16 keypoint output stays within an agreed tolerance of the CUDA baseline on a recorded validation set.
- End-to-end tracking latency and sustained FPS improve on at least the target RTX 50-series test machine.
- Calibration, OSC output, face tracking, camera formats, and SteamVR/Pico alignment pass the existing regression suite.
- The installer remains below GitHub's per-asset limit, or TensorRT is shipped as a separate optional runtime package.

TensorRT engines are hardware- and version-sensitive. They will not be prebuilt into the public installer.
