# BodyLink

BodyLink 是一个面向 Windows 和 VRChat 的本地摄像头全身追踪工具。它使用 RTMW3D-X 从普通 RGB 摄像头估计 133 个三维全身关键点，通过 ONNX Runtime CUDA 在 NVIDIA GPU 上推理。完成身高、地面和朝向校准后，软件通过 VRChat 官方 OSC Tracker 地址发送米制位置与旋转数据。

默认使用腰部和双脚三个追踪点。VRChat 官方文档明确指出，摄像头这类有遮挡和深度误差的追踪源通常在少量追踪点下更稳定；八点模式包含胸、双膝和双肘，适合光线与机位稳定后再尝试。

## 安装

要求：Windows 10/11、Python 3.12、普通摄像头、PC 版 VRChat，以及支持 CUDA 13 的 NVIDIA 显卡驱动。当前开发机为 RTX 5070 Ti、驱动 610.74。Pico 辅助模式还需要 PICO Connect 和 SteamVR；没有头显时仍可使用纯摄像头模式。

1. 双击 `install.bat`。它会创建独立的 `.venv`，安装 ONNX Runtime CUDA、CUDA/cuDNN 运行库与固定版本依赖，并下载 RTMW3D-X、YOLOX 人体检测和 MediaPipe 面捕模型。首次安装需要下载约 1.8 GB，解压后占用会更大。
2. 双击 `start.bat` 启动 BodyLink。

安装结束前会分别创建人体检测与姿态模型的 CUDA 会话；只要任一模型没有实际使用 `CUDAExecutionProvider`，安装就会失败，不会静默回退到 CPU。所有视频帧都在本机处理，不会上传。程序只向设置的 UDP 地址发送追踪器坐标。

## 使用

1. 将摄像头放在身体正前方，确保肩部、手臂和双脚全部入镜。头部可以在画面外。保持均匀照明，并让身体与背景有明显区分。
2. 在 VRChat 动作菜单中打开 `OSC > Enabled`。VRChat 默认接收端口是 `9000`。
3. 在 BodyLink 中填写真实身高，选择摄像头，点击“开启摄像头”。
4. 识别状态变为“肩部至双脚已识别”后点击“校准身体”。倒计时期间站直、面向摄像头，双脚自然分开，不要移动。
5. 点击“发送到 VRChat”。
6. 打开 VRChat 快捷菜单中的 Tracking & IK，执行 Calibrate FBT；向前看并按正常的双手柄触发键流程完成 VRChat 自身校准。

更改身高或摄像头水平视场角后必须重新校准。常见网络摄像头的水平视场角约为 55 到 78 度；数值不准确会主要影响前后移动的距离估计。

摄像头列表读取 Windows DirectShow 设备名，并在名称后保留 OpenCV 索引用于排障，例如 `UHD 4K AF Camera [0]` 和 `OBS Virtual Camera [2]`。

“高级 > 采集格式”可以选择 MJPEG、YUY2 或自动协商。MJPEG 使用较少 USB 带宽，通常更适合 720p/1080p 高帧率，但会增加少量 CPU 解码；YUY2 不压缩，解码开销低但占用更多 USB 带宽。当前测试的 `UHD 4K AF Camera` 在 MJPEG 下最高为 30 FPS，在 YUY2 下 720p 最高 15 FPS、1080p 最高 5 FPS，不支持 60 FPS。

## 模式

| 模式 | OSC Tracker | 适用情况 |
| --- | --- | --- |
| 稳定 3 点 | 腰、左脚、右脚 | 默认；单摄像头、日常使用 |
| 全身 8 点 | 加入胸、双膝、双肘 | 全身无遮挡、光线稳定、较高帧率 |

软件会对低置信度点短暂保持 280 ms，随后停止发送该点；单帧异常位移会被拒绝。OSC 使用 UDP，没有“已连接”握手，界面中的数据包计数表示本机已经发出数据，不代表 VRChat 已完成 FBT 校准。

## Pico / SteamVR 辅助

“高级 > 使用 SteamVR / Pico 头手辅助”默认开启。使用 Pico 4 Ultra 时，先通过 PICO Connect 进入 PC VR 并启动 SteamVR，确认 SteamVR 已识别头显和左右手柄，再启动 BodyLink。BodyLink 不会自行启动 SteamVR。

辅助校准时，头部仍然可以在摄像头画面外，但双手柄以及对应的手腕应保持在画面内，双手自然放在身体两侧。BodyLink 会将同一时刻的摄像头手腕位置与 SteamVR 手柄位置配对，用于计算摄像头空间到 SteamVR 站立空间的水平旋转和平移。连接或对齐不可用时，校准会自动完成为纯摄像头模式，并在界面中显示回退原因。

这里只运行一套辅助 3D IK 约束：Pico 的左右手 6DoF 是手部终点，摄像头提供肩部、肘部和身体骨架，BodyLink 只据此修正双肘位置。它不会再生成一套摄像头手部 Tracker，也不会发送手部 OSC。Pico 继续直接控制头部和双手；BodyLink 只发送腰、胸、脚、膝、肘等额外身体 Tracker，最后的 Avatar IK 由 VRChat 完成。

发送时，Pico 头显位姿还会通过 VRChat 官方 `/tracking/trackers/head/*` 地址作为 OSC Tracking Space 的连续对齐参考。该地址只移动和校正额外身体 Tracker 的坐标空间，不会接管 Avatar 头部，也不是第二套头部 IK。

## 可选面捕

“面捕”页中的 MediaPipe 面捕默认关闭。启用后，它复用全身追踪的同一摄像头，不会再次打开设备；RTMW3D 身体姿态仍按摄像头帧率优先处理，面捕默认以 20 FPS 运行，也可选择 15 或 30 FPS。

- 默认勾选“使用 VRChat 原生眼动”，眼球方向和眨眼发送到 `/tracking/eye/*`。
- 眉毛、脸颊、下巴和嘴部表情发送为 VRCFaceTracking Unified Expressions 的 `/avatar/parameters/v2/*` 参数。
- 取消原生眼动后，眼球方向与眼睑也改用 `v2/Eye*` Avatar 参数。
- 停止发送、关闭面捕、连续 0.5 秒丢失人脸或退出程序时，BodyLink 会发送一次中性复位，防止表情停留在最后一帧。

嘴部和眉毛动画要求当前 Avatar 已包含兼容 VRCFT v2 的 Expression Parameters 与 Animator；BodyLink 不会自动修改 Avatar。MediaPipe 官方 52 个 blendshape 不包含舌头追踪，因此本模式不发送 `TongueOut`。

## GPU 与模型

全身追踪由两个 ONNX 模型组成：YOLOX-M 负责找人，RTMW3D-X 负责输出 133 个二维/三维关键点。人体检测默认每 7 帧运行一次，姿态模型每帧运行。二者都固定使用 ONNX Runtime 的 `CUDAExecutionProvider`；界面的“GPU 推理”显示两阶段本帧合计耗时。

摄像头读取、图像预处理、后处理、滤波、预览绘制和 OSC 发送仍在 CPU 上执行。可选 MediaPipe 面捕也仍使用 CPU，因为官方 Windows wheel 没有启用 MediaPipe GPU delegate。关闭面捕时不会运行面部模型。

RTMW3D 与 rtmlib 使用 Apache-2.0 许可，ONNX Runtime 使用 MIT 许可；不需要 NVIDIA Maxine SDK、NGC API Key 或 NVIDIA AI Enterprise。安装器从公开模型仓库下载固定文件并校验 SHA-256。摄像头可采集 720p 或 1080p，但 RTMW3D 的单人裁剪输入固定为 384 x 288；更高采集分辨率主要改善远距离人体检测和裁剪质量，不会把模型本身变成 1080p 推理。

## 机位与限制

- 单个 RGB 摄像头无法看穿身体。转身、腿部互相遮挡、宽松纯色衣服和暗光都会降低精度。
- 摄像头无法达到 Lighthouse 或 IMU 追踪器的绝对位置精度，尤其是前后深度和脚部旋转。
- RTMW3D 输出关键点位置而不是关节四元数；BodyLink 会从骨架方向推导腰、脚、膝和肘的朝向。
- 程序使用 VRChat 的实验性 OSC Tracker 接口，不安装或模拟 SteamVR 驱动，也不是 VRChat 官方支持的软件。
- 头显和控制器仍由 PICO Connect、SteamVR 和 VRChat 原生追踪。BodyLink 只提供额外身体追踪点。
- 当前 Pico 接入读取 SteamVR 暴露的 HMD 与左右 Controller 6DoF；尚未在真实 Pico 4 Ultra 上验证，裸手关节追踪是否会被 PICO Connect 映射为 Controller 取决于其版本和设置。

## 故障排查

- `无法打开摄像头`：关闭会议软件、浏览器摄像头页面或其他占用摄像头的程序，再重新扫描。
- `ONNX Runtime CUDA 不可用`：更新 NVIDIA 驱动后重新运行 `install.bat`；不要同时安装 CPU 版 `onnxruntime`。
- `pip check` 报告 `rtmlib` 缺少 `onnxruntime` 或 `opencv-python`：这是发行包名称造成的元数据告警。BodyLink 实际使用 `onnxruntime-gpu` 和 `opencv-contrib-python`；不要为消除告警安装 CPU 版 `onnxruntime`，安装器末尾的 CUDA 会话检查才是运行时验收依据。
- 姿态置信度低：增加正面照明、换成有对比度的背景、退后让双脚完整入镜。
- 脚穿地：确认 BodyLink 与 VRChat 中的身高正确，站直后重新执行两边的校准。
- 前后移动比例不对：查摄像头规格中的水平 FOV，并在高级设置中修改后重新校准。
- VRChat 没有反应：确认 `OSC > Enabled`、目标为 `127.0.0.1:9000`，并在开始发送后重新进入 Calibrate FBT。
- Pico 辅助一直显示等待：先在 PICO Connect 中进入 PC VR，再启动 SteamVR，并确认 SteamVR 状态窗口能看到头显和左右手柄；BodyLink 不会主动拉起 SteamVR。
- Pico 辅助校准回退到摄像头：让两个手柄和两个手腕同时入镜，双手自然分开并保持不动后重新校准。

## 开发与验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\check_runtime.py
.\.venv\Scripts\python.exe scripts\check_camera.py
.\.venv\Scripts\python.exe main.py
```

协议依据：

- [VRChat OSC Trackers](https://docs.vrchat.com/docs/osc-trackers)
- [VRChat OSC Overview](https://docs.vrchat.com/docs/osc-overview)
- [VRChat Full-Body Tracking](https://docs.vrchat.com/docs/full-body-tracking)
- [OpenVR API Documentation](https://github.com/ValveSoftware/openvr/wiki/API-Documentation)
- [OpenMMLab RTMPose3D](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d)
- [rtmlib](https://github.com/Tau-J/rtmlib)
- [ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
