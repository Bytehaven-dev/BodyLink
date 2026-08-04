# 更新日志

[English](CHANGELOG.md) | 简体中文

## 0.3.0 - 2026-08-05

### 新功能

- 新增基于 ONNX Runtime CUDA 的 RTMW3D-X 133 点全身追踪和 YOLOX-M 人体检测。
- 新增稳定的 3 点和完整 8 点 VRChat OSC 追踪器模式。
- 新增可选的 SteamVR/Pico 头部与双手对齐，以及摄像头辅助肘部 IK。
- 新增可选的 MediaPipe 面部追踪，支持 VRChat 原生眼动和 VRCFaceTracking v2 表情参数。
- 新增 DirectShow 真实摄像头名称扫描，以及 MJPEG、YUY2 和自动采集格式选择。
- 新增 Windows 安装器：捆绑 GPU 运行库，独立下载并校验模型包，同时支持安装器同目录离线模型包。
- 将 MediaPipe 固定为 0.10.33，避免较新 Windows wheel 中的 Clearcut 联网日志和退出延迟。

### 文档

- 补充摄像头机位、校准、Pico 4 Ultra 联动、GPU 使用方式和当前追踪限制说明。
- 将可选 TensorRT FP16 加速记录为后续后端，并保留 CUDA 回退方案。
