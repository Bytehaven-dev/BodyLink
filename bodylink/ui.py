from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QCloseEvent,
    QFont,
    QFontDatabase,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QApplication,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from bodylink import __version__
from bodylink.config import AppConfig, save_config
from bodylink.osc_sender import validate_target
from bodylink.rtmw3d import missing_model_paths
from bodylink.vision_worker import CameraDevice, CameraProbeThread, TrackingWorker


CAPTURE_FORMAT_HINTS = {
    "mjpg": "低 USB 带宽，适合 720p/1080p 高帧率；有少量 CPU 解码开销。",
    "yuy2": "无压缩、解码开销低；USB 带宽高，720p 常见上限为 15 FPS。",
    "auto": "由驱动自动协商；兼容性优先，但可能回退到低帧率 YUY2。",
}


APP_STYLESHEET = """
* {
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 14px;
    letter-spacing: 0px;
}
QMainWindow, QWidget#Root {
    background: #0d1014;
    color: #f2f5f4;
}
QLabel { color: #f2f5f4; background: transparent; }
QLabel#Brand {
    font-size: 24px;
    font-weight: 700;
}
QLabel#Version {
    color: #8f99a2;
    font-size: 12px;
    font-weight: 600;
}
QLabel#BrandMark {
    background: #38d39f;
    color: #07120e;
    border-radius: 5px;
    font-size: 16px;
    font-weight: 800;
    min-width: 34px;
    min-height: 34px;
    max-width: 34px;
    max-height: 34px;
}
QLabel#Subtle, QLabel[muted="true"] { color: #8f99a2; }
QLabel#SectionTitle {
    font-size: 16px;
    font-weight: 650;
}
QLabel#PoseBadge {
    border: 1px solid #353c45;
    border-radius: 5px;
    color: #a7b0b7;
    padding: 6px 10px;
    font-weight: 600;
}
QLabel#PoseBadge[state="ready"] {
    color: #65ddaF;
    background: #12271f;
    border-color: #285b48;
}
QLabel#PoseBadge[state="partial"] {
    color: #f2c66d;
    background: #2b2415;
    border-color: #64532a;
}
QLabel#PoseBadge[state="lost"] {
    color: #ef8581;
    background: #2b191a;
    border-color: #633235;
}
QLabel#PoseBadge[state="error"] {
    color: #ef8581;
    background: #2b191a;
    border-color: #633235;
}
QLabel#PoseBadge[state="loading"] {
    color: #f2c66d;
    background: #2b2415;
    border-color: #64532a;
}
QLabel#PoseBadge[state="disabled"] {
    color: #8f99a2;
    background: #15191f;
    border-color: #353c45;
}
QFrame#PreviewFrame {
    background: #080a0d;
    border: 1px solid #2a3038;
    border-radius: 6px;
}
QFrame#ToolPanel {
    background: #15191f;
    border: 1px solid #2a3038;
    border-radius: 6px;
}
QFrame#MetricTile {
    background: #13171c;
    border: 1px solid #262c33;
    border-radius: 6px;
}
QLabel#MetricValue {
    font-size: 20px;
    font-weight: 700;
}
QLabel#MetricLabel {
    color: #89939b;
    font-size: 12px;
}
QPushButton {
    min-height: 38px;
    padding: 0 14px;
    border: 1px solid #343b44;
    border-radius: 5px;
    background: #20262d;
    color: #eef2f1;
    font-weight: 600;
}
QPushButton:hover { background: #282f37; border-color: #46505a; }
QPushButton:pressed { background: #191e24; }
QPushButton:disabled { color: #69727a; background: #181c21; border-color: #282e34; }
QPushButton[role="primary"] {
    color: #07130f;
    background: #38d39f;
    border-color: #38d39f;
}
QPushButton[role="primary"]:hover { background: #53dfb0; border-color: #53dfb0; }
QPushButton[active="true"] {
    color: #ffd98b;
    background: #302817;
    border-color: #7b612a;
}
QPushButton#IconButton {
    min-width: 38px;
    max-width: 38px;
    padding: 0;
}
QPushButton#Segment {
    min-height: 34px;
    background: #11151a;
    color: #98a2aa;
}
QPushButton#Segment:checked {
    color: #eaf8f2;
    background: #21483a;
    border-color: #397d64;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    min-height: 36px;
    padding: 0 10px;
    background: #0f1318;
    color: #edf1f0;
    border: 1px solid #343b44;
    border-radius: 5px;
    selection-background-color: #327d63;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #43bc91;
}
QComboBox::drop-down { border: none; width: 26px; }
QComboBox QAbstractItemView {
    background: #151a20;
    color: #edf1f0;
    border: 1px solid #343b44;
    selection-background-color: #285b49;
}
QCheckBox { spacing: 9px; color: #dbe0df; }
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #46505a;
    border-radius: 4px;
    background: #0f1318;
}
QCheckBox::indicator:checked { background: #38d39f; border-color: #38d39f; }
QCheckBox:disabled { color: #68727a; }
QCheckBox::indicator:disabled { background: #181c21; border-color: #30363d; }
QCheckBox::indicator:checked:disabled { background: #315f50; border-color: #315f50; }
QSlider::groove:horizontal { height: 4px; background: #303740; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #38d39f; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 16px;
    margin: -6px 0;
    background: #eef4f1;
    border: 2px solid #38d39f;
    border-radius: 8px;
}
QSlider::sub-page:horizontal:disabled { background: #3b4b46; }
QSlider::handle:horizontal:disabled { background: #69727a; border-color: #3b4b46; }
QComboBox:disabled { color: #68727a; background: #181c21; border-color: #282e34; }
QTabWidget::pane { border: none; top: -1px; }
QTabBar::tab {
    min-width: 92px;
    min-height: 34px;
    color: #879199;
    border-bottom: 2px solid transparent;
    background: transparent;
}
QTabBar::tab:selected { color: #f2f5f4; border-bottom-color: #38d39f; }
QProgressBar {
    min-height: 8px;
    max-height: 8px;
    border: none;
    border-radius: 4px;
    background: #2a3037;
    text-align: center;
}
QProgressBar::chunk { background: #f2b84b; border-radius: 4px; }
QToolTip {
    background: #252b32;
    color: #f2f5f4;
    border: 1px solid #3b444d;
    padding: 6px;
}
"""

_APPLICATION_FONT_FAMILY: str | None = None


def _refresh_style(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def application_icon() -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#38d39f"))
    painter.drawRoundedRect(QRectF(2, 2, 60, 60), 8, 8)
    painter.setPen(QColor("#07120e"))
    font = QFont("Microsoft YaHei UI", 20, QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(QRectF(2, 2, 60, 58), Qt.AlignmentFlag.AlignCenter, "BL")
    painter.end()
    return QIcon(pixmap)


def load_application_fonts() -> str:
    global _APPLICATION_FONT_FAMILY
    if _APPLICATION_FONT_FAMILY is not None:
        return _APPLICATION_FONT_FAMILY
    windows_directory = Path(os.environ.get("WINDIR", r"C:\Windows"))
    candidates = (
        windows_directory / "Fonts" / "msyh.ttc",
        windows_directory / "Fonts" / "segoeui.ttf",
    )
    loaded_families: list[str] = []
    for path in candidates:
        if not path.exists():
            continue
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id >= 0:
            loaded_families.extend(QFontDatabase.applicationFontFamilies(font_id))
    if "Microsoft YaHei UI" in loaded_families:
        _APPLICATION_FONT_FAMILY = "Microsoft YaHei UI"
    else:
        _APPLICATION_FONT_FAMILY = loaded_families[0] if loaded_families else "Segoe UI"
    return _APPLICATION_FONT_FAMILY


class PreviewLabel(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 360)
        self._source = self._placeholder(QSize(960, 540))
        self._render_source()

    def _placeholder(self, size: QSize) -> QPixmap:
        pixmap = QPixmap(size)
        pixmap.fill(QColor("#080a0d"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setPen(QPen(QColor("#12171c"), 1))
        for x in range(0, size.width(), 80):
            painter.drawLine(x, 0, x, size.height())
        for y in range(0, size.height(), 80):
            painter.drawLine(0, y, size.width(), y)

        cx = size.width() / 2
        cy = size.height() / 2 - 12
        joints = {
            "head": QPointF(cx, cy - 126),
            "neck": QPointF(cx, cy - 86),
            "ls": QPointF(cx - 58, cy - 70),
            "rs": QPointF(cx + 58, cy - 70),
            "le": QPointF(cx - 92, cy - 4),
            "re": QPointF(cx + 92, cy - 4),
            "lw": QPointF(cx - 112, cy + 66),
            "rw": QPointF(cx + 112, cy + 66),
            "hip": QPointF(cx, cy + 48),
            "lh": QPointF(cx - 30, cy + 50),
            "rh": QPointF(cx + 30, cy + 50),
            "lk": QPointF(cx - 40, cy + 132),
            "rk": QPointF(cx + 40, cy + 132),
            "lf": QPointF(cx - 48, cy + 216),
            "rf": QPointF(cx + 48, cy + 216),
        }
        links = (
            ("head", "neck"), ("ls", "rs"), ("neck", "hip"),
            ("ls", "le"), ("le", "lw"), ("rs", "re"), ("re", "rw"),
            ("lh", "rh"), ("lh", "lk"), ("lk", "lf"),
            ("rh", "rk"), ("rk", "rf"),
        )
        painter.setPen(QPen(QColor("#315f50"), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for start, end in links:
            painter.drawLine(joints[start], joints[end])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#57c99f"))
        for point in joints.values():
            painter.drawEllipse(point, 5, 5)
        painter.setBrush(QColor("#f2b84b"))
        for key in ("hip", "lf", "rf"):
            painter.drawEllipse(joints[key], 9, 9)

        painter.setPen(QColor("#cbd2d1"))
        font = QFont("Microsoft YaHei UI", 16)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.drawText(QRectF(0, size.height() - 60, size.width(), 24), Qt.AlignmentFlag.AlignCenter, "等待摄像头")
        painter.end()
        return pixmap

    def set_frame(self, image: QImage) -> None:
        self._source = QPixmap.fromImage(image)
        self._render_source()

    def reset_frame(self) -> None:
        self._source = self._placeholder(QSize(960, 540))
        self._render_source()

    def _render_source(self) -> None:
        if self._source.isNull():
            return
        self.setPixmap(
            self._source.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._render_source()


class MetricTile(QFrame):
    def __init__(self, label: str, value: str) -> None:
        super().__init__()
        self.setObjectName("MetricTile")
        self.setMinimumHeight(70)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        self.value_label = QLabel(value)
        self.value_label.setObjectName("MetricValue")
        title = QLabel(label)
        title.setObjectName("MetricLabel")
        layout.addWidget(self.value_label)
        layout.addWidget(title)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        preferred_font = load_application_fonts()
        application = QApplication.instance()
        if application is not None:
            application.setFont(QFont(preferred_font, 10))
            application.setWindowIcon(application_icon())
        self.config = replace(config).normalized()
        self.worker: TrackingWorker | None = None
        self.probe_thread: CameraProbeThread | None = None
        self.is_calibrated = False
        self.is_sending = False
        self._countdown = 0
        self._calibration_active = False
        self._closing_pending = False
        self._last_pose_state = "lost"
        self._camera_names: dict[int, str] = {}

        self.setWindowTitle(f"BodyLink v{__version__} - VRChat OSC 全身追踪")
        self.setWindowIcon(application_icon())
        self.setMinimumSize(1120, 700)
        self.resize(1320, 800)
        self.setStyleSheet(APP_STYLESHEET)
        self._build_ui()
        self._load_controls()
        self._bind_controls()
        QTimer.singleShot(350, self.scan_cameras)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        page = QVBoxLayout(root)
        page.setContentsMargins(22, 18, 22, 20)
        page.setSpacing(14)

        header = QHBoxLayout()
        header.setSpacing(10)
        mark = QLabel("BL")
        mark.setObjectName("BrandMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(mark)
        brand = QLabel("BodyLink")
        brand.setObjectName("Brand")
        header.addWidget(brand)
        self.version_label = QLabel(f"v{__version__}")
        self.version_label.setObjectName("Version")
        header.addWidget(self.version_label, 0, Qt.AlignmentFlag.AlignBottom)
        header.addStretch(1)
        self.header_status = QLabel("摄像头关闭")
        self.header_status.setObjectName("Subtle")
        header.addWidget(self.header_status)
        page.addLayout(header)

        workspace = QHBoxLayout()
        workspace.setSpacing(14)
        page.addLayout(workspace, 1)

        left = QVBoxLayout()
        left.setSpacing(12)
        workspace.addLayout(left, 1)

        preview_frame = QFrame()
        preview_frame.setObjectName("PreviewFrame")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(1, 1, 1, 1)
        self.preview = PreviewLabel()
        preview_layout.addWidget(self.preview)
        left.addWidget(preview_frame, 1)

        preview_status = QHBoxLayout()
        self.pose_badge = QLabel("未检测到人体")
        self.pose_badge.setObjectName("PoseBadge")
        self.pose_badge.setProperty("state", "lost")
        preview_status.addWidget(self.pose_badge)
        self.face_badge = QLabel("面捕未启用")
        self.face_badge.setObjectName("PoseBadge")
        self.face_badge.setProperty("state", "disabled")
        preview_status.addWidget(self.face_badge)
        self.vr_badge = QLabel("等待 SteamVR / Pico")
        self.vr_badge.setObjectName("PoseBadge")
        self.vr_badge.setProperty("state", "loading")
        preview_status.addWidget(self.vr_badge)
        preview_status.addStretch(1)
        self.resolution_label = QLabel("--")
        self.resolution_label.setProperty("muted", True)
        preview_status.addWidget(self.resolution_label)
        left.addLayout(preview_status)

        metrics = QHBoxLayout()
        metrics.setSpacing(10)
        self.metric_fps = MetricTile("实时帧率", "0 FPS")
        self.metric_pose = MetricTile("姿态置信度", "0%")
        self.metric_trackers = MetricTile("有效追踪点", "0 / 3")
        self.metric_inference = MetricTile("GPU 推理", "--")
        self.metric_packets = MetricTile("OSC 数据包", "0")
        for tile in (
            self.metric_fps,
            self.metric_pose,
            self.metric_trackers,
            self.metric_inference,
            self.metric_packets,
        ):
            metrics.addWidget(tile, 1)
        left.addLayout(metrics)

        panel = QFrame()
        panel.setObjectName("ToolPanel")
        panel.setMinimumWidth(350)
        panel.setMaximumWidth(390)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(18, 16, 18, 18)
        panel_layout.setSpacing(12)
        workspace.addWidget(panel)

        panel_title = QLabel("追踪控制")
        panel_title.setObjectName("SectionTitle")
        panel_layout.addWidget(panel_title)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_basic_tab(), "基础")
        self.tabs.addTab(self._build_advanced_tab(), "高级")
        self.tabs.addTab(self._build_face_tab(), "面捕")
        panel_layout.addWidget(self.tabs, 1)

        self.calibration_bar = QProgressBar()
        self.calibration_bar.setRange(0, 100)
        self.calibration_bar.setTextVisible(False)
        self.calibration_bar.hide()
        panel_layout.addWidget(self.calibration_bar)

        self.action_status = QLabel("等待启动")
        self.action_status.setWordWrap(True)
        self.action_status.setProperty("muted", True)
        panel_layout.addWidget(self.action_status)

        self.camera_button = QPushButton("开启摄像头")
        self.camera_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.camera_button.setProperty("role", "primary")
        panel_layout.addWidget(self.camera_button)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.calibrate_button = QPushButton("校准身体")
        self.calibrate_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        )
        self.calibrate_button.setEnabled(False)
        action_row.addWidget(self.calibrate_button, 1)
        self.send_button = QPushButton("发送到 VRChat")
        self.send_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        )
        self.send_button.setEnabled(False)
        action_row.addWidget(self.send_button, 1)
        panel_layout.addLayout(action_row)

    def _build_basic_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(14)

        layout.addWidget(self._field_label("摄像头"))
        camera_row = QHBoxLayout()
        camera_row.setSpacing(7)
        self.camera_combo = QComboBox()
        self.camera_combo.addItem(f"摄像头 {self.config.camera_index}", self.config.camera_index)
        camera_row.addWidget(self.camera_combo, 1)
        self.refresh_camera_button = QPushButton()
        self.refresh_camera_button.setObjectName("IconButton")
        self.refresh_camera_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        )
        self.refresh_camera_button.setToolTip("重新扫描摄像头")
        camera_row.addWidget(self.refresh_camera_button)
        layout.addLayout(camera_row)

        layout.addWidget(self._field_label("真实身高"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1.20, 2.20)
        self.height_spin.setDecimals(2)
        self.height_spin.setSingleStep(0.01)
        self.height_spin.setSuffix(" m")
        layout.addWidget(self.height_spin)

        layout.addWidget(self._field_label("追踪模式"))
        mode_row = QHBoxLayout()
        mode_row.setSpacing(7)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.stable_mode_button = QPushButton("稳定 3 点")
        self.full_mode_button = QPushButton("全身 8 点")
        for button in (self.stable_mode_button, self.full_mode_button):
            button.setObjectName("Segment")
            button.setCheckable(True)
            self.mode_group.addButton(button)
            mode_row.addWidget(button, 1)
        layout.addLayout(mode_row)

        smoothing_header = QHBoxLayout()
        smoothing_header.addWidget(self._field_label("动作平滑"))
        smoothing_header.addStretch(1)
        self.smoothing_value = QLabel("68%")
        self.smoothing_value.setProperty("muted", True)
        smoothing_header.addWidget(self.smoothing_value)
        layout.addLayout(smoothing_header)
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(0, 95)
        layout.addWidget(self.smoothing_slider)

        self.mirror_check = QCheckBox("镜像预览")
        layout.addWidget(self.mirror_check)

        layout.addSpacing(4)
        layout.addWidget(self._field_label("VRChat OSC"))
        target_row = QHBoxLayout()
        target_row.setSpacing(7)
        self.host_edit = QLineEdit()
        self.host_edit.setPlaceholderText("127.0.0.1")
        target_row.addWidget(self.host_edit, 1)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setFixedWidth(100)
        target_row.addWidget(self.port_spin)
        layout.addLayout(target_row)
        layout.addStretch(1)
        return tab

    def _build_advanced_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(14)

        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.fov_spin = QDoubleSpinBox()
        self.fov_spin.setRange(35.0, 110.0)
        self.fov_spin.setDecimals(1)
        self.fov_spin.setSuffix(" deg")
        form.addRow("水平视场角", self.fov_spin)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("1280 x 720", (1280, 720))
        self.resolution_combo.addItem("1920 x 1080", (1920, 1080))
        self.resolution_combo.addItem("640 x 480", (640, 480))
        form.addRow("采集分辨率", self.resolution_combo)

        self.fps_combo = QComboBox()
        self.fps_combo.addItem("30 FPS", 30)
        self.fps_combo.addItem("60 FPS", 60)
        self.fps_combo.addItem("24 FPS", 24)
        form.addRow("目标帧率", self.fps_combo)

        self.capture_format_combo = QComboBox()
        self.capture_format_combo.addItem("MJPEG · 高帧率/低带宽（推荐）", "mjpg")
        self.capture_format_combo.addItem("YUY2 · 无压缩/高带宽", "yuy2")
        self.capture_format_combo.addItem("自动 · 驱动协商", "auto")
        form.addRow("采集格式", self.capture_format_combo)
        layout.addLayout(form)

        self.capture_format_hint = QLabel()
        self.capture_format_hint.setProperty("muted", True)
        self.capture_format_hint.setWordWrap(True)
        layout.addWidget(self.capture_format_hint)

        confidence_header = QHBoxLayout()
        confidence_header.addWidget(self._field_label("最低置信度"))
        confidence_header.addStretch(1)
        self.confidence_value = QLabel("55%")
        self.confidence_value.setProperty("muted", True)
        confidence_header.addWidget(self.confidence_value)
        layout.addLayout(confidence_header)
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(20, 95)
        layout.addWidget(self.confidence_slider)

        self.align_yaw_check = QCheckBox("校准时对齐头显朝向")
        layout.addWidget(self.align_yaw_check)

        self.vr_assist_check = QCheckBox("使用 SteamVR / Pico 头手辅助")
        self.vr_assist_check.setToolTip(
            "Pico 直接驱动头部与双手；BodyLink 仅融合姿态并输出身体 Tracker"
        )
        layout.addWidget(self.vr_assist_check)
        layout.addStretch(1)
        return tab

    def _build_face_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(14)

        self.face_enable_check = QCheckBox("启用 MediaPipe 面捕")
        layout.addWidget(self.face_enable_check)

        self.face_native_eyes_check = QCheckBox("使用 VRChat 原生眼动")
        layout.addWidget(self.face_native_eyes_check)

        layout.addWidget(self._field_label("面捕更新率"))
        self.face_fps_combo = QComboBox()
        self.face_fps_combo.addItem("15 FPS", 15)
        self.face_fps_combo.addItem("20 FPS", 20)
        self.face_fps_combo.addItem("30 FPS", 30)
        layout.addWidget(self.face_fps_combo)

        smoothing_header = QHBoxLayout()
        smoothing_header.addWidget(self._field_label("表情平滑"))
        smoothing_header.addStretch(1)
        self.face_smoothing_value = QLabel("55%")
        self.face_smoothing_value.setProperty("muted", True)
        smoothing_header.addWidget(self.face_smoothing_value)
        layout.addLayout(smoothing_header)
        self.face_smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.face_smoothing_slider.setRange(0, 95)
        layout.addWidget(self.face_smoothing_slider)
        layout.addStretch(1)
        return tab

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("muted", True)
        return label

    def _load_controls(self) -> None:
        self.height_spin.setValue(self.config.user_height_m)
        self.smoothing_slider.setValue(round(self.config.smoothing * 100))
        self.mirror_check.setChecked(self.config.mirror_preview)
        self.host_edit.setText(self.config.target_host)
        self.port_spin.setValue(self.config.target_port)
        self.fov_spin.setValue(self.config.horizontal_fov_deg)
        self.confidence_slider.setValue(round(self.config.min_confidence * 100))
        self.align_yaw_check.setChecked(self.config.align_yaw_on_calibrate)
        self.vr_assist_check.setChecked(self.config.vr_assist_enabled)
        self.stable_mode_button.setChecked(self.config.tracker_mode == "stable")
        self.full_mode_button.setChecked(self.config.tracker_mode == "full")
        self.face_enable_check.setChecked(self.config.face_enabled)
        self.face_native_eyes_check.setChecked(self.config.face_native_eyes)
        self.face_smoothing_slider.setValue(round(self.config.face_smoothing * 100))
        self._select_combo_data(
            self.resolution_combo, (self.config.camera_width, self.config.camera_height)
        )
        self._select_combo_data(self.fps_combo, self.config.camera_fps)
        self._select_combo_data(self.capture_format_combo, self.config.camera_format)
        self._select_combo_data(self.face_fps_combo, self.config.face_fps)
        self._update_slider_labels()
        self._update_capture_format_hint()
        self._update_face_controls_enabled()
        if self.config.face_enabled:
            self._face_state_changed("loading", "面捕待启动")
        self._vr_state_changed(
            "loading" if self.config.vr_assist_enabled else "disabled",
            "等待 SteamVR / Pico" if self.config.vr_assist_enabled else "VR 辅助未启用",
        )

    @staticmethod
    def _select_combo_data(combo: QComboBox, value: object) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _bind_controls(self) -> None:
        self.camera_button.clicked.connect(self.toggle_camera)
        self.calibrate_button.clicked.connect(self.begin_calibration_countdown)
        self.send_button.clicked.connect(self.toggle_sending)
        self.refresh_camera_button.clicked.connect(self.scan_cameras)
        self.height_spin.valueChanged.connect(lambda _: self._settings_changed(True))
        self.fov_spin.valueChanged.connect(lambda _: self._settings_changed(True))
        self.smoothing_slider.valueChanged.connect(lambda _: self._settings_changed(False))
        self.confidence_slider.valueChanged.connect(lambda _: self._settings_changed(False))
        self.mirror_check.toggled.connect(lambda _: self._settings_changed(False))
        self.align_yaw_check.toggled.connect(lambda _: self._settings_changed(False))
        self.vr_assist_check.toggled.connect(lambda _: self._settings_changed(True))
        self.host_edit.editingFinished.connect(lambda: self._settings_changed(False))
        self.port_spin.valueChanged.connect(lambda _: self._settings_changed(False))
        self.resolution_combo.currentIndexChanged.connect(lambda _: self._settings_changed(False))
        self.fps_combo.currentIndexChanged.connect(lambda _: self._settings_changed(False))
        self.capture_format_combo.currentIndexChanged.connect(
            lambda _: self._capture_format_changed()
        )
        self.camera_combo.currentIndexChanged.connect(lambda _: self._settings_changed(False))
        self.stable_mode_button.toggled.connect(lambda _: self._mode_changed())
        self.full_mode_button.toggled.connect(lambda _: self._mode_changed())
        self.face_enable_check.toggled.connect(lambda _: self._face_settings_changed())
        self.face_native_eyes_check.toggled.connect(lambda _: self._settings_changed(False))
        self.face_fps_combo.currentIndexChanged.connect(lambda _: self._settings_changed(False))
        self.face_smoothing_slider.valueChanged.connect(
            lambda _: self._settings_changed(False)
        )

    def _sync_config(self) -> None:
        camera_data = self.camera_combo.currentData()
        resolution = self.resolution_combo.currentData() or (1280, 720)
        fps = self.fps_combo.currentData() or 30
        self.config.camera_index = int(camera_data if camera_data is not None else 0)
        self.config.camera_width, self.config.camera_height = map(int, resolution)
        self.config.camera_fps = int(fps)
        self.config.camera_format = str(self.capture_format_combo.currentData() or "mjpg")
        self.config.user_height_m = self.height_spin.value()
        self.config.horizontal_fov_deg = self.fov_spin.value()
        self.config.smoothing = self.smoothing_slider.value() / 100.0
        self.config.min_confidence = self.confidence_slider.value() / 100.0
        self.config.mirror_preview = self.mirror_check.isChecked()
        self.config.target_host = self.host_edit.text().strip() or "127.0.0.1"
        self.config.target_port = self.port_spin.value()
        self.config.tracker_mode = "full" if self.full_mode_button.isChecked() else "stable"
        self.config.align_yaw_on_calibrate = self.align_yaw_check.isChecked()
        self.config.vr_assist_enabled = self.vr_assist_check.isChecked()
        self.config.face_enabled = self.face_enable_check.isChecked()
        self.config.face_native_eyes = self.face_native_eyes_check.isChecked()
        self.config.face_fps = int(self.face_fps_combo.currentData() or 20)
        self.config.face_smoothing = self.face_smoothing_slider.value() / 100.0
        self.config.normalized()

    def _settings_changed(self, calibration_sensitive: bool) -> None:
        self._sync_config()
        self._update_slider_labels()
        if self.worker is not None:
            self.worker.update_config(self.config)
        if calibration_sensitive and self.is_calibrated:
            self._invalidate_calibration_ui("参数已更改，请重新校准")

    def _mode_changed(self) -> None:
        if not self.stable_mode_button.isChecked() and not self.full_mode_button.isChecked():
            return
        self._settings_changed(False)
        total = 8 if self.config.tracker_mode == "full" else 3
        self.metric_trackers.set_value(f"0 / {total}")

    def _update_slider_labels(self) -> None:
        self.smoothing_value.setText(f"{self.smoothing_slider.value()}%")
        self.confidence_value.setText(f"{self.confidence_slider.value()}%")
        self.face_smoothing_value.setText(f"{self.face_smoothing_slider.value()}%")

    def _capture_format_changed(self) -> None:
        self._update_capture_format_hint()
        self._settings_changed(False)

    def _update_capture_format_hint(self) -> None:
        capture_format = str(self.capture_format_combo.currentData() or "mjpg")
        self.capture_format_hint.setText(CAPTURE_FORMAT_HINTS[capture_format])

    def _face_settings_changed(self) -> None:
        self._update_face_controls_enabled()
        self._settings_changed(False)
        if self.worker is None:
            if self.face_enable_check.isChecked():
                self._face_state_changed("loading", "面捕待启动")
            else:
                self._face_state_changed("disabled", "面捕未启用")

    def _update_face_controls_enabled(self) -> None:
        enabled = self.face_enable_check.isChecked()
        self.face_native_eyes_check.setEnabled(enabled)
        self.face_fps_combo.setEnabled(enabled)
        self.face_smoothing_slider.setEnabled(enabled)

    def scan_cameras(self) -> None:
        if self.worker is not None or (self.probe_thread is not None and self.probe_thread.isRunning()):
            return
        self.refresh_camera_button.setEnabled(False)
        self.refresh_camera_button.setToolTip("正在扫描")
        self.probe_thread = CameraProbeThread()
        self.probe_thread.cameras_found.connect(self._cameras_found)
        self.probe_thread.finished.connect(self._probe_finished)
        self.probe_thread.start()

    def _cameras_found(self, devices: list[CameraDevice]) -> None:
        current = self.config.camera_index
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self._camera_names = {
            device.index: device.name.strip()
            for device in devices
            if device.name.strip()
        }
        available = devices or [CameraDevice(current, "")]
        for device in available:
            self.camera_combo.addItem(device.label, device.index)
        selected = self.camera_combo.findData(current)
        self.camera_combo.setCurrentIndex(max(0, selected))
        self.camera_combo.blockSignals(False)

    def _probe_finished(self) -> None:
        self.refresh_camera_button.setEnabled(True)
        self.refresh_camera_button.setToolTip("重新扫描摄像头")
        if self.probe_thread is not None:
            self.probe_thread.deleteLater()
            self.probe_thread = None

    def toggle_camera(self) -> None:
        if self.worker is not None:
            self.stop_camera()
            return
        self.start_camera()

    def start_camera(self) -> None:
        self._sync_config()
        missing = missing_model_paths()
        if missing:
            names = "、".join(path.name for path in missing)
            QMessageBox.warning(
                self,
                "缺少 RTMW3D 模型",
                f"缺少 {names}，请先运行 install.bat 完成安装。",
            )
            return

        self.worker = TrackingWorker(self.config)
        self.worker.frame_ready.connect(self.preview.set_frame)
        self.worker.metrics_ready.connect(self._metrics_updated)
        self.worker.camera_ready.connect(self._camera_ready)
        self.worker.pose_state.connect(self._pose_state_changed)
        self.worker.face_state.connect(self._face_state_changed)
        self.worker.vr_state.connect(self._vr_state_changed)
        self.worker.calibration_progress.connect(self._calibration_progress)
        self.worker.calibration_succeeded.connect(self._calibration_succeeded)
        self.worker.calibration_failed.connect(self._calibration_failed)
        self.worker.runtime_error.connect(self._runtime_error)
        self.worker.osc_error.connect(self._osc_error)
        self.worker.worker_stopped.connect(self._worker_stopped)
        self.worker.start()

        self.camera_button.setText("关闭摄像头")
        self.camera_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.camera_button.setProperty("role", "")
        _refresh_style(self.camera_button)
        self.header_status.setText("正在启动摄像头")
        self.action_status.setText("正在加载 RTMW3D CUDA 模型")
        self._set_capture_controls_enabled(False)

    def stop_camera(self) -> None:
        if self.worker is None:
            return
        self.action_status.setText("正在关闭摄像头")
        self.camera_button.setEnabled(False)
        self.worker.stop()

    def _camera_ready(self, info: dict[str, object]) -> None:
        width = int(info["width"])
        height = int(info["height"])
        fps = float(info["fps"])
        capture_format = str(info.get("format", "AUTO"))
        self.resolution_label.setText(
            f"{width} x {height} · {capture_format} · {fps:.0f} FPS"
        )
        backend = str(info.get("backend", "CUDA"))
        camera_index = int(info["camera_index"])
        camera_name = self._camera_names.get(camera_index, f"摄像头 {camera_index}")
        self.header_status.setText(f"{camera_name} · {backend}")
        requested_format = str(info.get("requested_format", "AUTO"))
        if requested_format != "AUTO" and capture_format != requested_format:
            self.action_status.setText(
                f"格式回退：请求 {requested_format}，实际 {capture_format}"
            )
        else:
            self.action_status.setText("等待肩部至双脚入镜")
        self.calibrate_button.setEnabled(False)

    def _pose_state_changed(self, state: str, text: str) -> None:
        self._last_pose_state = state
        self.pose_badge.setText(text)
        self.pose_badge.setProperty("state", state)
        _refresh_style(self.pose_badge)
        if self.worker is not None and not self._calibration_active:
            self.calibrate_button.setEnabled(state == "ready")

    def _face_state_changed(self, state: str, text: str) -> None:
        self.face_badge.setText("面捕不可用" if state == "error" else text)
        self.face_badge.setToolTip(text if state == "error" else "")
        self.face_badge.setProperty("state", state)
        _refresh_style(self.face_badge)
        if state == "error":
            self.action_status.setText(f"{text}；身体追踪仍在运行")

    def _vr_state_changed(self, state: str, text: str) -> None:
        self.vr_badge.setText(text)
        self.vr_badge.setProperty("state", state)
        _refresh_style(self.vr_badge)

    def begin_calibration_countdown(self) -> None:
        if self.worker is None:
            return
        if self._last_pose_state != "ready":
            self.action_status.setText("请先让肩部至双脚完整入镜")
            return
        self.set_sending(False)
        self.is_calibrated = False
        self._calibration_active = True
        self.send_button.setEnabled(False)
        self.calibrate_button.setEnabled(False)
        self._countdown = 3
        self.calibration_bar.setValue(0)
        self.calibration_bar.show()
        self.action_status.setText("站直，双手自然放在身体两侧 · 3")
        QTimer.singleShot(1000, self._calibration_tick)

    def _calibration_tick(self) -> None:
        if self.worker is None:
            return
        self._countdown -= 1
        if self._countdown > 0:
            self.action_status.setText(
                f"站直，双手自然放在身体两侧 · {self._countdown}"
            )
            QTimer.singleShot(1000, self._calibration_tick)
            return
        self.action_status.setText("正在采集校准姿态")
        self.worker.request_calibration()

    def _calibration_progress(self, value: int) -> None:
        self.calibration_bar.show()
        self.calibration_bar.setValue(value)

    def _calibration_succeeded(self, result: dict[str, object]) -> None:
        self._calibration_active = False
        self.is_calibrated = True
        self.calibration_bar.hide()
        self.calibrate_button.setEnabled(True)
        self.send_button.setEnabled(True)
        if bool(result.get("vr_assisted", False)):
            alignment_cm = float(result.get("vr_alignment_error_m", 0.0)) * 100.0
            self.action_status.setText(
                f"校准完成 · VR 辅助 · 头手对齐误差 {alignment_cm:.1f} cm"
            )
        elif str(result.get("vr_error", "")):
            self.action_status.setText(
                f"校准完成 · 摄像头模式（{result['vr_error']}）"
            )
        else:
            self.action_status.setText(
                f"校准完成 · 空间误差 {result['error_px']:.1f} px"
            )

    def _calibration_failed(self, message: str) -> None:
        self._calibration_active = False
        self.is_calibrated = False
        self.calibration_bar.hide()
        self.calibrate_button.setEnabled(True)
        self.send_button.setEnabled(False)
        self.action_status.setText(f"校准失败：{message}")

    def toggle_sending(self) -> None:
        self.set_sending(not self.is_sending)

    def set_sending(self, enabled: bool) -> None:
        enabled = bool(enabled and self.is_calibrated and self.worker is not None)
        if enabled:
            try:
                validate_target(self.config.target_host, self.config.target_port)
            except (OSError, ValueError) as exc:
                self.action_status.setText(f"OSC 目标无效：{exc}")
                enabled = False
        self.is_sending = enabled
        if self.worker is not None:
            self.worker.set_sending(enabled)
        self.send_button.setText("停止发送" if enabled else "发送到 VRChat")
        self.send_button.setProperty("active", "true" if enabled else "false")
        _refresh_style(self.send_button)
        if enabled:
            self.action_status.setText(
                f"正在发送到 {self.config.target_host}:{self.config.target_port}"
            )
        elif self.is_calibrated:
            self.action_status.setText("已校准，OSC 发送已停止")

    def _invalidate_calibration_ui(self, message: str) -> None:
        self.set_sending(False)
        self.is_calibrated = False
        self.send_button.setEnabled(False)
        self.action_status.setText(message)

    def _metrics_updated(self, metrics: dict[str, object]) -> None:
        self.metric_fps.set_value(f"{float(metrics['fps']):.0f} FPS")
        self.metric_pose.set_value(f"{float(metrics['pose_score']) * 100:.0f}%")
        total = 8 if self.config.tracker_mode == "full" else 3
        self.metric_trackers.set_value(f"{int(metrics['tracker_count'])} / {total}")
        self.metric_inference.set_value(f"{float(metrics['inference_ms']):.0f} ms")
        self.metric_packets.set_value(f"{int(metrics['packets']):,}")

    def _runtime_error(self, message: str) -> None:
        self.header_status.setText("运行错误")
        self.action_status.setText(message)
        self.pose_badge.setText("追踪已停止")
        self.pose_badge.setProperty("state", "lost")
        _refresh_style(self.pose_badge)

    def _osc_error(self, message: str) -> None:
        self.is_sending = False
        self.send_button.setText("发送到 VRChat")
        self.send_button.setProperty("active", "false")
        _refresh_style(self.send_button)
        self.action_status.setText(f"OSC 发送失败：{message}")

    def _worker_stopped(self) -> None:
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        self.is_calibrated = False
        self.is_sending = False
        self._calibration_active = False
        self.preview.reset_frame()
        self.header_status.setText("摄像头关闭")
        if self.action_status.text() == "正在关闭摄像头":
            self.action_status.setText("等待启动")
        self.resolution_label.setText("--")
        self.camera_button.setEnabled(True)
        self.camera_button.setText("开启摄像头")
        self.camera_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.camera_button.setProperty("role", "primary")
        _refresh_style(self.camera_button)
        self.calibrate_button.setEnabled(False)
        self.send_button.setEnabled(False)
        self.send_button.setText("发送到 VRChat")
        self.send_button.setProperty("active", "false")
        _refresh_style(self.send_button)
        self.calibration_bar.hide()
        self._set_capture_controls_enabled(True)
        self._pose_state_changed("lost", "未检测到人体")
        if self.config.face_enabled:
            self._face_state_changed("loading", "面捕待启动")
        else:
            self._face_state_changed("disabled", "面捕未启用")
        self._vr_state_changed(
            "loading" if self.config.vr_assist_enabled else "disabled",
            "等待 SteamVR / Pico" if self.config.vr_assist_enabled else "VR 辅助未启用",
        )
        self.metric_fps.set_value("0 FPS")
        self.metric_pose.set_value("0%")
        self.metric_inference.set_value("--")
        total = 8 if self.config.tracker_mode == "full" else 3
        self.metric_trackers.set_value(f"0 / {total}")
        if self._closing_pending:
            QTimer.singleShot(0, self.close)

    def _set_capture_controls_enabled(self, enabled: bool) -> None:
        self.camera_combo.setEnabled(enabled)
        self.refresh_camera_button.setEnabled(enabled)
        self.resolution_combo.setEnabled(enabled)
        self.fps_combo.setEnabled(enabled)
        self.capture_format_combo.setEnabled(enabled)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._sync_config()
        save_config(self.config)
        if self.worker is not None and self.worker.isRunning():
            self._closing_pending = True
            self.worker.stop()
            event.ignore()
            return
        if self.probe_thread is not None and self.probe_thread.isRunning():
            self.probe_thread.requestInterruption()
            self.probe_thread.wait(1500)
        event.accept()
