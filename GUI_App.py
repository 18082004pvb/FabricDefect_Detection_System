import sys
import cv2
import os
import time
import threading
from PIL import Image
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *

from ultralytics import YOLO
import torch


# ============================
# AUTO SELECT DEVICE
# ============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(">>> Đang chạy bằng:", device)


# ============================
# LABEL CONVERTdevice=device
# ============================
def convert_label(original):
    original = original.lower()
    if original in ["loi_ban", "stain"]:
        return "loi_ban"
    elif original in ["loi_xuoc", "scratch"]:
        return "loi_xuoc"
    return original


# ==============================================================
# THREAD XỬ LÝ VIDEO + YOLO
# ==============================================================
# ==============================================================
# THREAD XỬ LÝ VIDEO + YOLO (TRACK ID - KHÔNG TRÙNG LỖI)
# ==============================================================
class VideoThread(QThread):
    frame_signal = Signal(QImage)
    info_signal = Signal(int, int, int, float, str)
    save_signal = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.cap = None
        self.save_dir = None

        self.error_dirty = 0
        self.error_scratch = 0
        self.saved_count = 0

        self.model = None

        # 🔥 Danh sách Track ID để tránh đếm trùng
        self.tracked_ids = set()

    def set_source(self, cap):
        self.cap = cap

    def set_save_dir(self, folder):
        self.save_dir = folder

    def set_model(self, model):
        self.model = model

    def run(self):
        prev_time = time.time()

        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

            detected = False
            new_error_detected = False

            # 🔥 Dùng track() thay vì predict()
            results = self.model.track(frame, persist=True, device=0, verbose=False)[0]

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < 0.5:
                    continue

                # 🔥 Tracking ID
                track_id = int(box.id[0]) if box.id is not None else None
                if track_id is None:
                    continue

                detected = True
                label_name = convert_label(self.model.names[cls])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ============================
                # 🔥 Đếm lỗi theo TRACK ID
                # ============================
                if track_id not in self.tracked_ids:
                    self.tracked_ids.add(track_id)
                    new_error_detected = True

                    if label_name == "loi_ban":
                        self.error_dirty += 1
                    else:
                        self.error_scratch += 1

                    self.saved_count += 1

                    # Lưu ảnh
                    timestamp = time.strftime("%Y-%m-%d %H_%M_%S")
                    file_name = f"{label_name}_{timestamp}_ID{track_id}.jpg"
                    save_path = os.path.join(self.save_dir, file_name)

                    save_frame = frame.copy()
                    cv2.rectangle(save_frame, (x1, y1), (x2, y2), (0,0,255), 2)
                    cv2.imwrite(save_path, save_frame)
                    cv2.putText(save_frame, f"{label_name}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,0,255), 2)

                    self.save_signal.emit(file_name, timestamp)

                # Vẽ box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, f"{label_name} ID:{track_id}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,0,255), 2)

            # STATUS
            if new_error_detected:
                status = "Phát hiện lỗi mới!"
            elif detected:
                status = "Đã thấy lỗi (không đếm)."
            else:
                status = "Không có lỗi."

            # Convert frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img_q = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_signal.emit(img_q)

            # FPS
            now = time.time()
            fps = 1 / (now - prev_time)
            prev_time = now

            self.info_signal.emit(
                self.error_dirty,
                self.error_scratch,
                self.saved_count,
                fps,
                status
            )

# ==============================================================
# MAIN WINDOW
# ==============================================================
class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("PHÁT HIỆN LỖI VẢI")
        self.resize(1350, 750)

        layout = QHBoxLayout(self)
        self.setLayout(layout)

        # ========================
        # VIDEO AREA
        # ========================
        self.video_label = QLabel("VIDEO")
        self.video_label.setFixedSize(900, 650)
        self.video_label.setStyleSheet("background: black;")
        self.video_label.setAlignment(Qt.AlignCenter)

        # ========================
        # CONTROL PANEL
        # ========================
        panel = QVBoxLayout()
        panel.setAlignment(Qt.AlignCenter)

        def style_button(btn):
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    font-size: 15px;
                    padding: 8px 20px;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #1f5f8a;
                }
            """)
            btn.setMinimumHeight(30)
            btn.setMinimumWidth(150)

        # ===== BUTTONS =====
        btn_video = QPushButton("🎥 Chọn video")
        btn_camera = QPushButton("📸 Mở camera")
        btn_folder = QPushButton("📁 Chọn thư mục lưu")
        btn_start = QPushButton("▶ Bắt đầu")
        btn_stop = QPushButton("⏸ Dừng")

        btn_video.clicked.connect(self.select_video)
        btn_camera.clicked.connect(self.open_camera)
        btn_folder.clicked.connect(self.select_folder)
        btn_start.clicked.connect(self.start_detection)
        btn_stop.clicked.connect(self.stop_detection)

        for b in (btn_video, btn_camera, btn_folder, btn_start, btn_stop):
            style_button(b)

        # ===== TẠO 3 HÀNG =====
        row1 = QHBoxLayout()
        row1.addWidget(btn_video)
        row1.addWidget(btn_camera)

        row2 = QHBoxLayout()
        row2.addWidget(btn_folder, alignment=Qt.AlignCenter)

        row3 = QHBoxLayout()
        row3.addWidget(btn_start)
        row3.addWidget(btn_stop)

        panel.addLayout(row1)
        panel.addSpacing(5)
        panel.addLayout(row2)
        panel.addSpacing(5)
        panel.addLayout(row3)
        panel.addSpacing(5)

        # ========== THỐNG KÊ ==========
        self.lbl_err_dirty = QLabel("Lỗi bẩn: 0")
        self.lbl_err_dirty.setAlignment(Qt.AlignCenter)

        self.lbl_err_scratch = QLabel("Lỗi xước: 0")
        self.lbl_err_scratch.setAlignment(Qt.AlignCenter)

        self.lbl_save = QLabel("")
        self.lbl_save.setAlignment(Qt.AlignCenter)

        self.lbl_fps = QLabel("FPS: --")
        self.lbl_fps.setAlignment(Qt.AlignCenter)

        self.lbl_status = QLabel("Trạng thái: Đang chờ...")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setWordWrap(True)

        stats_row = QHBoxLayout()
        stats_row.setAlignment(Qt.AlignCenter)
        stats_row.addWidget(self.lbl_err_dirty)
        stats_row.addWidget(self.lbl_err_scratch)
        stats_row.addWidget(self.lbl_save)

        panel.addSpacing(5)
        panel.addLayout(stats_row)
        panel.addWidget(self.lbl_fps, alignment=Qt.AlignCenter)
        panel.addWidget(self.lbl_status, alignment=Qt.AlignCenter)

        # ========================
        # MODEL SELECT COMBOBOX
        # ========================
        model_row = QHBoxLayout()
        model_row.setAlignment(Qt.AlignCenter)

        self.model_label = QLabel("Model đang chọn:")
        self.model_label.setAlignment(Qt.AlignCenter)

        self.model_select = QComboBox()
        self.model_select.setMinimumWidth(180)

        # Thêm model
        self.model_paths = {
            "Model Modelwhite20112025": "Modelwhite20112025.pt",
            "Model Modelgreen27112025": "Modelgreen27112025.pt",
            "Model Modelgreen07122025": "Modelgreen07122025.pt",
            "Model Modelgreen09122025": "Modelgreen09122025.pt",
            "Model Modelwhite09122025": "Modelwhite09122025.pt",
            "Model Modelgreen10122025": "Modelgreen10122025.pt"
        }

        # Load sẵn tất cả model
        self.models = {}
        for name, path in self.model_paths.items():
            try:
                self.models[name] = YOLO(path)
            except:
                QMessageBox.warning(self, "Model lỗi", f"Không thể load model: {path}")

        for name in self.model_paths.keys():
            self.model_select.addItem(name)

        self.current_model_name = "Model Modelgreen10122025"
        self.current_model = self.models[self.current_model_name]
        self.model_label.setText(f"Model đang chọn: {self.current_model_name}")

        self.model_select.currentTextChanged.connect(self.on_model_selected)

        model_row.addWidget(self.model_select)
        model_row.addWidget(self.model_label)

        panel.addSpacing(10)
        panel.addLayout(model_row)

        # ========================
        # LIST VIEW
        # ========================
        self.listview = QListWidget()
        self.listview.setFixedHeight(200)

        panel.addWidget(QLabel("Danh sách ảnh lưu:"), alignment=Qt.AlignCenter)
        panel.addWidget(self.listview)

        layout.addWidget(self.video_label)
        layout.addLayout(panel)

        # ========================
        # THREAD
        # ========================
        self.thread = VideoThread()
        self.thread.frame_signal.connect(self.update_video)
        self.thread.info_signal.connect(self.update_info)
        self.thread.save_signal.connect(self.add_list_item)

        # Set default model cho thread
        self.thread.set_model(self.current_model)

        self.cap_source = None
        self.save_dir = None
        self.camera_opened = False

    # ============================
    # CALLBACKS
    # ============================
    def on_model_selected(self, name):
        if not self.model_select.isEnabled():
            return

        self.current_model_name = name
        self.current_model = self.models[name]
        self.thread.set_model(self.current_model)
        self.model_label.setText(f"Model đang chọn: {name}")

    def select_video(self):
        self.thread.error_dirty = 0
        self.thread.error_scratch = 0
        self.thread.saved_count = 0
        self.thread.tracked_ids.clear()


        self.lbl_err_dirty.setText("Lỗi bẩn: 0")
        self.lbl_err_scratch.setText("Lỗi xước: 0")
        self.lbl_save.setText("")

        file = QFileDialog.getOpenFileName(self, "Chọn video", "", "Video (*.mp4 *.avi *.mkv)")[0]
        if file:
            self.cap_source = cv2.VideoCapture(file)
            self.lbl_status.setText("Đã chọn video.")

    def scan_cameras(self, max_test=10):
        available = []
        for i in range(max_test):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def open_camera(self):
        if self.thread.running:
            QMessageBox.warning(self, "Cảnh báo", "Hãy DỪNG trước khi mở camera!")
            return

        if self.camera_opened:
            QMessageBox.warning(self, "Cảnh báo", "Camera đã bật!")
            return

        cameras = self.scan_cameras()
        if not cameras:
            QMessageBox.critical(self, "Lỗi", "Không tìm thấy camera.")
            return

        items = [f"Camera {i}" for i in cameras]
        cam_str, ok = QInputDialog.getItem(
            self, "Chọn camera", "Danh sách camera:", items, 0, False
        )

        if not ok:
            return

        cam_index = int(cam_str.split()[-1])

        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            QMessageBox.critical(self, "Lỗi", f"Không thể mở camera {cam_index}.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.cap_source = cap
        self.camera_opened = True
        self.lbl_status.setText(f"Đã bật camera {cam_index}.")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục")
        if folder:
            self.save_dir = folder
            self.lbl_status.setText(f"Thư mục lưu: {folder}")

    def start_detection(self):
        self.thread.tracked_ids.clear()
        self.thread.error_dirty = 0
        self.thread.error_scratch = 0
        self.thread.saved_count = 0

        if self.cap_source is None:
            self.lbl_status.setText("Chưa chọn nguồn video!")
            return

        if self.save_dir is None:
            self.lbl_status.setText("Chưa chọn thư mục lưu!")
            return

        self.lbl_status.setText("Đang chạy...")
        self.lbl_fps.setText("FPS: --")

        self.model_select.setEnabled(False)

        self.thread.set_source(self.cap_source)
        self.thread.set_save_dir(self.save_dir)
        self.thread.set_model(self.current_model)

        self.thread.running = True
        self.thread.start()

    def stop_detection(self):
        self.thread.running = False
        self.thread.wait()

        self.model_select.setEnabled(True)

        self.lbl_status.setText("Đã dừng.")
        self.camera_opened = False

    # ============================
    # SIGNAL EVENTS
    # ============================
    def update_video(self, img_q):
        pix = QPixmap.fromImage(img_q).scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)

    def update_info(self, err_dirty, err_scratch, saved, fps, status):
        self.lbl_err_dirty.setText(f"Lỗi bẩn: {err_dirty}")
        self.lbl_err_scratch.setText(f"Lỗi xước: {err_scratch}")
        self.lbl_save.setText(f"")
        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_status.setText(status)

    def add_list_item(self, file_name, time_str):
        self.listview.addItem(f"{file_name} | {time_str}")
        self.listview.scrollToBottom()


# ==============================================================
# RUN APP
# ==============================================================
app = QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec())
