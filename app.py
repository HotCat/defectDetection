"""
Structural Defect Detector — PySide6 UI with industrial camera + PaDiM anomaly detection.

Modes:
  - Live View: continuous camera capture, adjustable settings
  - Inspection: single-frame or auto-inspection with PaDiM inference + heatmap overlay
"""

import sys
import os

# Hide CUDA to force CPU mode — your driver (535.x / CUDA 12.2) is too old for PyTorch 2.11 (needs CUDA 12.4+).
# Remove this line after upgrading your NVIDIA driver to 550+ or installing PyTorch with CUDA 12.2 support.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import yaml
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QComboBox,
    QPushButton, QToolBar, QStatusBar, QSlider, QSpinBox,
    QCheckBox, QFormLayout, QHBoxLayout, QFileDialog,
    QMessageBox, QDoubleSpinBox,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QShortcut, QKeySequence

from camera import MindVisionCamera, CameraSettings, CameraSettingRanges
from defect_detection import DefectDetector


def _app_dir() -> str:
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


# ════════════════════════════════════════════════════════════════════════
#  Camera Settings Window
# ════════════════════════════════════════════════════════════════════════

class CameraSettingsWindow(QWidget):
    """Floating window for camera parameter adjustments."""

    settings_changed = Signal(CameraSettings)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Settings")
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setMinimumWidth(320)
        self._block_signals = False
        self._build_ui()

    def _build_ui(self):
        layout = QFormLayout(self)

        self._ae_check = QCheckBox("Auto Exposure")
        self._ae_check.stateChanged.connect(self._on_setting_changed)
        layout.addRow(self._ae_check)

        self._exposure_slider = QSlider(Qt.Horizontal)
        self._exposure_spin = QSpinBox()
        self._exposure_spin.setSuffix(" us")
        self._exposure_spin.setMinimum(100)
        self._exposure_spin.setMaximum(1000000)
        self._exposure_spin.setSingleStep(100)
        self._exposure_slider.valueChanged.connect(self._exposure_spin.setValue)
        self._exposure_spin.valueChanged.connect(self._exposure_slider.setValue)
        self._exposure_spin.valueChanged.connect(self._on_setting_changed)
        exp_row = QHBoxLayout()
        exp_row.addWidget(self._exposure_slider, 1)
        exp_row.addWidget(self._exposure_spin)
        layout.addRow("Exposure:", exp_row)

        self._gamma_slider = QSlider(Qt.Horizontal)
        self._gamma_spin = QSpinBox()
        self._gamma_slider.valueChanged.connect(self._gamma_spin.setValue)
        self._gamma_spin.valueChanged.connect(self._gamma_slider.setValue)
        self._gamma_spin.valueChanged.connect(self._on_setting_changed)
        gamma_row = QHBoxLayout()
        gamma_row.addWidget(self._gamma_slider, 1)
        gamma_row.addWidget(self._gamma_spin)
        layout.addRow("Gamma:", gamma_row)

        self._contrast_slider = QSlider(Qt.Horizontal)
        self._contrast_spin = QSpinBox()
        self._contrast_slider.valueChanged.connect(self._contrast_spin.setValue)
        self._contrast_spin.valueChanged.connect(self._contrast_slider.setValue)
        self._contrast_spin.valueChanged.connect(self._on_setting_changed)
        contrast_row = QHBoxLayout()
        contrast_row.addWidget(self._contrast_slider, 1)
        contrast_row.addWidget(self._contrast_spin)
        layout.addRow("Contrast:", contrast_row)

        self._gain_slider = QSlider(Qt.Horizontal)
        self._gain_spin = QSpinBox()
        self._gain_slider.valueChanged.connect(self._gain_spin.setValue)
        self._gain_spin.valueChanged.connect(self._gain_slider.setValue)
        self._gain_spin.valueChanged.connect(self._on_setting_changed)
        gain_row = QHBoxLayout()
        gain_row.addWidget(self._gain_slider, 1)
        gain_row.addWidget(self._gain_spin)
        layout.addRow("Analog Gain:", gain_row)

        self._reverse_x_check = QCheckBox("Reverse X (Horizontal Mirror)")
        self._reverse_x_check.stateChanged.connect(self._on_setting_changed)
        layout.addRow(self._reverse_x_check)

        self._reverse_y_check = QCheckBox("Reverse Y (Vertical Mirror)")
        self._reverse_y_check.stateChanged.connect(self._on_setting_changed)
        layout.addRow(self._reverse_y_check)

    def set_ranges(self, ranges: CameraSettingRanges):
        self._block_signals = True
        self._exposure_slider.setRange(ranges.exposure_min_us, ranges.exposure_max_us)
        self._exposure_slider.setSingleStep(ranges.exposure_step_us)
        self._exposure_spin.setRange(ranges.exposure_min_us, ranges.exposure_max_us)
        self._exposure_spin.setSingleStep(ranges.exposure_step_us)
        self._gamma_slider.setRange(ranges.gamma_min, ranges.gamma_max)
        self._gamma_spin.setRange(ranges.gamma_min, ranges.gamma_max)
        self._contrast_slider.setRange(ranges.contrast_min, ranges.contrast_max)
        self._contrast_spin.setRange(ranges.contrast_min, ranges.contrast_max)
        self._gain_slider.setRange(ranges.analog_gain_min, ranges.analog_gain_max)
        self._gain_spin.setRange(ranges.analog_gain_min, ranges.analog_gain_max)
        self._block_signals = False

    def set_values(self, settings: CameraSettings):
        self._block_signals = True
        self._ae_check.setChecked(settings.ae_enabled)
        self._exposure_slider.setValue(settings.exposure_us)
        self._exposure_spin.setValue(settings.exposure_us)
        self._exposure_slider.setEnabled(not settings.ae_enabled)
        self._exposure_spin.setEnabled(not settings.ae_enabled)
        self._gamma_slider.setValue(settings.gamma)
        self._gamma_spin.setValue(settings.gamma)
        self._contrast_slider.setValue(settings.contrast)
        self._contrast_spin.setValue(settings.contrast)
        self._gain_slider.setValue(settings.analog_gain)
        self._gain_spin.setValue(settings.analog_gain)
        self._reverse_x_check.setChecked(settings.reverse_x)
        self._reverse_y_check.setChecked(settings.reverse_y)
        self._block_signals = False

    def _on_setting_changed(self):
        if self._block_signals:
            return
        settings = CameraSettings(
            exposure_us=self._exposure_spin.value(),
            gamma=self._gamma_spin.value(),
            contrast=self._contrast_spin.value(),
            analog_gain=self._gain_spin.value(),
            ae_enabled=self._ae_check.isChecked(),
            reverse_x=self._reverse_x_check.isChecked(),
            reverse_y=self._reverse_y_check.isChecked(),
        )
        self._exposure_slider.setEnabled(not settings.ae_enabled)
        self._exposure_spin.setEnabled(not settings.ae_enabled)
        self.settings_changed.emit(settings)


# ════════════════════════════════════════════════════════════════════════
#  Background Workers
# ════════════════════════════════════════════════════════════════════════

class _TrainWorker(QThread):
    """Background worker for PaDiM training."""
    done = Signal()
    error = Signal(str)

    def __init__(self, detector: DefectDetector):
        super().__init__()
        self._detector = detector

    def run(self):
        try:
            self._detector.train()
            self.done.emit()
        except Exception as e:
            self.error.emit(str(e))


class _InferenceWorker(QThread):
    """Background worker for single-frame inference."""
    done = Signal(dict)
    error = Signal(str)

    def __init__(self, detector: DefectDetector, frame: np.ndarray, threshold: float):
        super().__init__()
        self._detector = detector
        self._frame = frame
        self._threshold = threshold

    def run(self):
        try:
            result = self._detector.infer(self._frame, self._threshold)
            self.done.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ════════════════════════════════════════════════════════════════════════
#  Main Window
# ════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self, camera: MindVisionCamera, config: dict):
        super().__init__()
        self._camera = camera
        self._config = config
        self._detector = DefectDetector(config)

        self._current_mode: str = "live"
        self._last_live_frame: np.ndarray | None = None
        self._current_frame: np.ndarray | None = None
        self._display_pixmap: QPixmap | None = None
        self._active_worker: QThread | None = None
        self._auto_inspect: bool = False
        self._show_roi: bool = False
        self._last_inference_result: dict | None = None

        self._build_ui()
        self._connect_signals()

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("Structural Defect Detector")
        self.setMinimumSize(800, 600)

        # Central image display
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet("background-color: #1a1a1a;")
        self._image_label.setCursor(Qt.CrossCursor)
        self._image_label.setMouseTracking(True)
        self.setCentralWidget(self._image_label)

        # Toolbar
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Live View", "Inspection"])
        self._mode_combo.setMinimumWidth(140)
        toolbar.addWidget(self._mode_combo)

        self._toggle_shortcut = QShortcut(QKeySequence(Qt.Key_T), self)
        self._toggle_shortcut.setContext(Qt.ApplicationShortcut)

        toolbar.addSeparator()

        self._grab_btn = QPushButton("Grab")
        self._grab_btn.setEnabled(False)
        toolbar.addWidget(self._grab_btn)

        self._load_btn = QPushButton("Load Image")
        toolbar.addWidget(self._load_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setEnabled(False)
        toolbar.addWidget(self._save_btn)

        toolbar.addSeparator()

        self._set_template_btn = QPushButton("Set Template")
        self._set_template_btn.setEnabled(False)
        self._set_template_btn.setToolTip("Use current frame as the perfect-product template")
        toolbar.addWidget(self._set_template_btn)

        self._train_btn = QPushButton("Train")
        self._train_btn.setEnabled(False)
        self._train_btn.setToolTip("Augment template and train PaDiM model")
        toolbar.addWidget(self._train_btn)

        self._auto_inspect_btn = QPushButton("Auto Inspect")
        self._auto_inspect_btn.setCheckable(True)
        self._auto_inspect_btn.setEnabled(False)
        self._auto_inspect_btn.setToolTip("Toggle continuous inspection on live frames")
        toolbar.addWidget(self._auto_inspect_btn)

        toolbar.addSeparator()

        self._roi_check = QCheckBox("Show ROI")
        self._roi_check.setEnabled(False)
        toolbar.addWidget(self._roi_check)

        toolbar.addSeparator()

        # Threshold slider
        threshold_label = QLabel(" Threshold:")
        toolbar.addWidget(threshold_label)

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.0, 500.0)
        self._threshold_spin.setSingleStep(5.0)
        self._threshold_spin.setDecimals(1)
        self._threshold_spin.setValue(self._config.get("detection", {}).get("threshold", 90.0))
        self._threshold_spin.setToolTip("Anomaly score threshold for defect detection")
        self._threshold_spin.setFixedWidth(80)
        toolbar.addWidget(self._threshold_spin)

        toolbar.addSeparator()

        self._settings_btn = QPushButton("Settings")
        toolbar.addWidget(self._settings_btn)

        # Status bar
        self._status_label = QLabel("No camera connected")
        self.statusBar().addWidget(self._status_label, 1)

        # Floating windows
        self._settings_window = CameraSettingsWindow(self)

    # ── Signal wiring ───────────────────────────────────────────────

    def _connect_signals(self):
        # Camera -> UI
        self._camera.signals.frame_ready.connect(self._on_live_frame)
        self._camera.signals.grab_done.connect(self._on_grab_frame)
        self._camera.signals.error.connect(self._on_camera_error)

        # Mode & buttons
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._toggle_shortcut.activated.connect(self._on_toggle_mode)
        self._grab_btn.clicked.connect(self._on_grab_clicked)
        self._load_btn.clicked.connect(self._on_load_image)
        self._save_btn.clicked.connect(self._on_save)
        self._set_template_btn.clicked.connect(self._on_set_template)
        self._train_btn.clicked.connect(self._on_train)
        self._auto_inspect_btn.toggled.connect(self._on_auto_inspect_toggled)
        self._roi_check.stateChanged.connect(self._on_roi_check_changed)
        self._settings_btn.clicked.connect(self._settings_window.show)

        # Settings -> camera + config save
        self._settings_window.settings_changed.connect(self._on_camera_settings_changed)

    # ── Slots: camera frames ────────────────────────────────────────

    @Slot(np.ndarray)
    def _on_live_frame(self, frame: np.ndarray):
        self._last_live_frame = self._ensure_bgr(frame)

        if self._auto_inspect and self._detector.trained:
            self._run_inference(self._last_live_frame)
        else:
            display = self._last_live_frame.copy()
            if self._show_roi and self._detector.has_template():
                display = self._draw_roi_overlay(display)
            self._display_image(display)

    @Slot(np.ndarray)
    def _on_grab_frame(self, frame: np.ndarray):
        self._current_frame = self._ensure_bgr(frame)
        if self._auto_inspect and self._detector.trained:
            self._run_inference(self._current_frame)
        else:
            display = self._current_frame.copy()
            if self._show_roi and self._detector.has_template():
                display = self._draw_roi_overlay(display)
            self._display_image(display)
            self._status_label.setText("Frame captured")

    @Slot(str)
    def _on_camera_error(self, msg: str):
        self._status_label.setText(f"Camera error: {msg}")
        self._stop_auto_inspect()

    @Slot(CameraSettings)
    def _on_camera_settings_changed(self, settings: CameraSettings):
        self._camera.apply_settings(settings)
        cam_cfg = self._config.setdefault("camera", {})
        cam_cfg["exposure_us"] = settings.exposure_us
        cam_cfg["gamma"] = settings.gamma
        cam_cfg["contrast"] = settings.contrast
        cam_cfg["analog_gain"] = settings.analog_gain
        cam_cfg["ae_enabled"] = settings.ae_enabled
        cam_cfg["reverse_x"] = settings.reverse_x
        cam_cfg["reverse_y"] = settings.reverse_y
        self._save_config()

    # ── Slots: mode and buttons ─────────────────────────────────────

    @Slot()
    def _on_toggle_mode(self):
        idx = self._mode_combo.currentIndex()
        self._mode_combo.setCurrentIndex(1 - idx)

    @Slot(int)
    def _on_mode_changed(self, index: int):
        if index == 0:
            self._switch_to_live()
        else:
            self._switch_to_inspection()

    def _switch_to_live(self):
        self._current_mode = "live"
        self._grab_btn.setEnabled(False)
        self._set_template_btn.setEnabled(False)
        if self._camera.is_open:
            self._camera.set_live_mode()

    def _switch_to_inspection(self):
        self._current_mode = "inspection"
        self._grab_btn.setEnabled(True)
        self._set_template_btn.setEnabled(True)
        if self._camera.is_open:
            self._camera.set_trigger_mode()
            if self._last_live_frame is not None:
                self._current_frame = self._last_live_frame.copy()
                display = self._current_frame.copy()
                if self._show_roi and self._detector.has_template():
                    display = self._draw_roi_overlay(display)
                self._display_image(display)

    @Slot()
    def _on_grab_clicked(self):
        if self._auto_inspect:
            return
        self._status_label.setText("Grabbing frame...")
        self._camera.software_trigger()

    @Slot()
    def _on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", "Images (*.png *.jpg *.bmp *.tif)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self._status_label.setText(f"Failed to load: {path}")
            return
        self._current_frame = img
        display = img.copy()
        if self._show_roi and self._detector.has_template():
            display = self._draw_roi_overlay(display)
        self._display_image(display)
        self._set_template_btn.setEnabled(True)
        self._status_label.setText(f"Loaded: {os.path.basename(path)}")

    @Slot()
    def _on_save(self):
        if self._current_frame is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "result.png", "Images (*.png *.jpg *.bmp)"
        )
        if not path:
            return
        if self._last_inference_result and "overlay" in self._last_inference_result:
            cv2.imwrite(path, self._last_inference_result["overlay"])
        else:
            cv2.imwrite(path, self._current_frame)
        self._status_label.setText(f"Saved to {path}")

    # ── Template & training ─────────────────────────────────────────

    @Slot()
    def _on_set_template(self):
        if self._current_frame is None:
            self._status_label.setText("No frame available — Grab first")
            return

        roi_mask = self._detector.set_template(self._current_frame)
        self._roi_check.setEnabled(True)
        self._train_btn.setEnabled(True)

        display = self._draw_roi_overlay(self._current_frame.copy())
        self._display_image(display)
        self._status_label.setText(
            f"Template set | ROI bbox: {self._detector.roi_bbox}"
        )

    @Slot()
    def _on_train(self):
        if not self._detector.has_template():
            self._status_label.setText("Set a template first")
            return

        self._train_btn.setEnabled(False)
        self._train_btn.setText("Training...")
        self._status_label.setText("Augmenting template & training PaDiM...")

        self._active_worker = _TrainWorker(self._detector)
        self._active_worker.done.connect(self._on_train_done)
        self._active_worker.error.connect(self._on_worker_error)
        self._active_worker.start()

    @Slot()
    def _on_train_done(self):
        self._active_worker = None
        self._train_btn.setText("Train")
        self._train_btn.setEnabled(True)
        self._auto_inspect_btn.setEnabled(True)
        self._status_label.setText("PaDiM trained — ready for inspection")

    @Slot(str)
    def _on_worker_error(self, msg: str):
        self._active_worker = None
        self._train_btn.setText("Train")
        self._train_btn.setEnabled(True)
        self._status_label.setText(f"Error: {msg}")

    # ── Inference ───────────────────────────────────────────────────

    def _run_inference(self, frame: np.ndarray):
        if self._active_worker is not None:
            return  # previous inference still running

        threshold = self._threshold_spin.value()
        self._active_worker = _InferenceWorker(self._detector, frame, threshold)
        self._active_worker.done.connect(self._on_inference_done)
        self._active_worker.error.connect(self._on_inference_error)
        self._active_worker.start()

    @Slot(dict)
    def _on_inference_done(self, result: dict):
        self._active_worker = None
        self._last_inference_result = result

        overlay = result["overlay"]
        score = result["score"]
        is_anomalous = result["is_anomalous"]

        # Draw pass/fail indicator
        self._draw_pass_fail(overlay, is_anomalous)
        self._display_image(overlay)

        status = "DEFECT" if is_anomalous else "PASS"
        color = "red" if is_anomalous else "green"
        self._status_label.setText(
            f"{status} | Score: {score:.2f} | Threshold: {self._threshold_spin.value():.1f}"
        )

        # Auto-inspect: trigger next grab
        if self._auto_inspect and self._camera.is_open:
            QApplication.processEvents()
            try:
                self._camera.software_trigger()
            except Exception:
                pass

    @Slot(str)
    def _on_inference_error(self, msg: str):
        self._active_worker = None
        self._status_label.setText(f"Inference error: {msg}")
        if self._auto_inspect:
            # Try to continue
            try:
                self._camera.software_trigger()
            except Exception:
                self._stop_auto_inspect()

    # ── Auto inspect ────────────────────────────────────────────────

    @Slot(bool)
    def _on_auto_inspect_toggled(self, checked: bool):
        if checked:
            self._start_auto_inspect()
        else:
            self._stop_auto_inspect()

    def _start_auto_inspect(self):
        if not self._detector.trained:
            self._auto_inspect_btn.setChecked(False)
            self._status_label.setText("Train model first")
            return

        self._auto_inspect = True
        self._auto_inspect_btn.setText("Stop Inspect")

        if self._current_mode != "inspection":
            self._mode_combo.setCurrentIndex(1)

        if self._camera.is_open:
            self._camera.set_trigger_mode()
            self._camera.software_trigger()
        elif self._current_frame is not None:
            self._run_inference(self._current_frame)

        self._status_label.setText("Auto inspecting...")

    def _stop_auto_inspect(self):
        self._auto_inspect = False
        self._auto_inspect_btn.setChecked(False)
        self._auto_inspect_btn.setText("Auto Inspect")

    # ── ROI overlay ─────────────────────────────────────────────────

    @Slot(int)
    def _on_roi_check_changed(self, state: int):
        self._show_roi = bool(state)
        if self._current_frame is not None and not self._auto_inspect:
            display = self._current_frame.copy()
            if self._show_roi and self._detector.has_template():
                display = self._draw_roi_overlay(display)
            self._display_image(display)

    def _draw_roi_overlay(self, image_bgr: np.ndarray) -> np.ndarray:
        """Draw green ROI contour and bbox on image."""
        if not self._detector.has_template():
            return image_bgr
        result = image_bgr.copy()

        # Resize ROI mask to match image if needed
        mask = self._detector.roi_mask
        if mask.shape[:2] != result.shape[:2]:
            mask = cv2.resize(mask, (result.shape[1], result.shape[0]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        if self._detector.roi_bbox:
            x, y, w, h = self._detector.roi_bbox
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 200, 0), 1)

        return result

    @staticmethod
    def _draw_pass_fail(image_bgr: np.ndarray, is_anomalous: bool) -> None:
        """Draw pass/fail indicator border on the image in-place."""
        h, w = image_bgr.shape[:2]
        thickness = max(4, h // 80)
        color = (0, 0, 255) if is_anomalous else (0, 200, 0)
        cv2.rectangle(image_bgr, (0, 0), (w - 1, h - 1), color, thickness)

        label = "DEFECT" if is_anomalous else "PASS"
        font_scale = max(0.8, h / 400)
        thickness_text = max(2, int(font_scale * 2))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)
        cv2.putText(image_bgr, label, (w - tw - 15, th + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness_text)

    # ── Image display ───────────────────────────────────────────────

    @staticmethod
    def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
        """Convert grayscale/mono frame to BGR if needed."""
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[2] == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame.copy()

    def _display_image(self, image_bgr: np.ndarray):
        """Display a BGR numpy array in the central QLabel."""
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.shape[2] == 1:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._display_pixmap = QPixmap.fromImage(qimg)

        scaled = self._display_pixmap.scaled(
            self._image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._image_label.setPixmap(scaled)

    # ── Config persistence ──────────────────────────────────────────

    def _save_config(self):
        config_path = os.path.join(_app_dir(), "config.yaml")
        det_cfg = self._config.setdefault("detection", {})
        det_cfg["threshold"] = self._threshold_spin.value()
        try:
            with open(config_path, "w") as f:
                yaml.dump(self._config, f, default_flow_style=False)
        except Exception:
            pass

    def closeEvent(self, event):
        self._stop_auto_inspect()
        self._save_config()
        super().closeEvent(event)


# ════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)

    config_path = os.path.join(_app_dir(), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    camera = MindVisionCamera()
    window = MainWindow(camera, config)

    # Connect camera
    devices = camera.enumerate_devices()
    if devices:
        try:
            camera.open(devices[0]["dev_info"])
            cam_cfg = config.get("camera", {})
            default_settings = CameraSettings(
                exposure_us=cam_cfg.get("exposure_us", 30000),
                gamma=cam_cfg.get("gamma", 100),
                contrast=cam_cfg.get("contrast", 100),
                analog_gain=cam_cfg.get("analog_gain", 16),
                ae_enabled=cam_cfg.get("ae_enabled", False),
                reverse_x=cam_cfg.get("reverse_x", False),
                reverse_y=cam_cfg.get("reverse_y", False),
            )
            camera.apply_settings(default_settings)

            ranges = camera.get_setting_ranges()
            window._settings_window.set_ranges(ranges)
            window._settings_window.set_values(camera.get_current_settings())

            camera.set_live_mode()
            window._status_label.setText(
                f"Camera connected: {devices[0]['name']} ({devices[0]['sn']})"
            )
        except Exception as e:
            window._status_label.setText(f"Camera init failed: {e}")
    else:
        window._status_label.setText(
            "No camera found — use Load Image to test with files"
        )

    window.show()
    ret = app.exec()

    camera.close()
    sys.exit(ret)


if __name__ == "__main__":
    main()
