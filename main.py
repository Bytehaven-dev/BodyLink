from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--runtime-report":
        from bodylink.diagnostics import write_runtime_report

        return write_runtime_report(Path(sys.argv[2]))

    from PySide6.QtCore import QLockFile, QStandardPaths, Qt
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication, QMessageBox

    from bodylink.config import load_config
    from bodylink.ui import MainWindow, load_application_fonts

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("BodyLink")
    app.setOrganizationName("BodyLink")
    app.setFont(QFont(load_application_fonts(), 10))

    lock_path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.TempLocation)
    instance_lock = QLockFile(f"{lock_path}/bodylink-instance.lock")
    if not instance_lock.tryLock(100):
        QMessageBox.information(None, "BodyLink", "BodyLink 已经在运行。")
        return 0

    window = MainWindow(load_config())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
