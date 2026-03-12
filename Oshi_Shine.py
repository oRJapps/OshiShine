import sys
import os
import ctypes
import time
import multiprocessing
import numpy as np
import cv2
import torch
import dxcam
from ultralytics import YOLO

# ★ MiDaSのパスを追加（これでもう迷子にならないぜ）
current_dir = os.path.dirname(os.path.abspath(__file__))
midas_root = os.path.join(current_dir, "MiDaS")
if midas_root not in sys.path:
    sys.path.append(midas_root)

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                             QSlider, QPushButton, QLabel, QColorDialog, 
                             QListWidget, QComboBox, QCheckBox, QMessageBox)
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QPainter, QColor, QImage

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def check_single_instance():
    mutex_name = "OshiShine_GOD_Edition_Final_Fixed_V2"
    kernel32 = ctypes.windll.kernel32
    mutex = kernel32.CreateMutexW(None, False, mutex_name)
    if kernel32.GetLastError() == 183: return None
    return mutex

# ══════════════════════════════════════════
# データクラス
# ══════════════════════════════════════════
class LightSource:
    def __init__(self, name, type_name="Point", x=160, y=90):
        self.name = name
        self.type = type_name
        self.pos = QPoint(x, y)
        self.color = QColor(255, 200, 150)
        self.intensity = 150
        self.radius = 400
        self.z_depth = 128
        self.blend_mode = "加算"

# ══════════════════════════════════════════
# AI & レンダリングスレッド
# ══════════════════════════════════════════
class DepthEstimationThread(QThread):
    render_ready = pyqtSignal(QImage)

    def __init__(self, overlay):
        super().__init__()
        self.overlay = overlay
        self.running = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        from midas.model_loader import load_model
        # ★ 軽量モデルに変更: 344Mパラメータ → ~6Mパラメータ（50倍以上高速）
        m_path = os.path.join(current_dir, "model", "midas_v21_small_256.pt")
        self.model, self.transform, _, _ = load_model(self.device, m_path, "midas_v21_small_256", optimize=True)
        if self.device.type == "cuda": self.model = self.model.half()
        self.model.eval()

        self.tracker = YOLO("yolov8n.pt")
        self.cached_depth = None
        self.last_ai_time = 0
        self.calc_w, self.calc_h = 320, 180  # ★軽量化: 480x270→320x180
        self.Y_grid, self.X_grid = np.mgrid[:self.calc_h, :self.calc_w].astype(np.float32)
        # ★軽量化: バッファ事前確保（毎フレームのメモリ確保を回避）
        self._canvas_buf = np.zeros((self.calc_h, self.calc_w, 3), dtype=np.float32)
        self._target_interval = 1.0 / 15  # ★軽量化: 15FPSキャップ
        self._midas_input_size = 256  # ★ midas_v21_small_256に最適

    def run(self):
        import warnings
        warnings.filterwarnings("ignore")
        camera = dxcam.create(output_color="BGR", max_buffer_len=2)
        camera.start(video_mode=True)
        try:
            while self.running:
                frame_start = time.time()
                frame = camera.get_latest_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                now = time.time()
                # ★軽量化: 軽量モデルなので2秒間隔でOK
                if self.cached_depth is None or now - self.last_ai_time > 2.0:
                    inp = cv2.resize(frame, (self._midas_input_size, self._midas_input_size))
                    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB) / 255.0
                    blob = self.transform({"image": rgb})["image"]
                    blob = torch.from_numpy(blob).unsqueeze(0).to(self.device)
                    if self.device.type == "cuda": blob = blob.half()
                    with torch.no_grad():
                        out = self.model(blob)
                        out = torch.nn.functional.interpolate(out.unsqueeze(1), size=(self.calc_h, self.calc_w), mode="bilinear").squeeze()
                    self.cached_depth = cv2.normalize(out.float().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
                    # ★軽量化: char_maskもAI推論時だけ計算してキャッシュ
                    self._cached_char_mask = cv2.GaussianBlur(np.clip((self.cached_depth - 100.0) / 80.0, 0, 1), (7, 7), 0)
                    self.last_ai_time = now

                lights = self.overlay.lights
                # ★軽量化: ライトが無い場合は合成処理をスキップ
                if not lights or self.cached_depth is None:
                    self.render_ready.emit(QImage(frame.data, 1920, 1080, 1920*3, QImage.Format_BGR888).copy())
                else:
                    # バッファをゼロクリア（新規確保より高速）
                    self._canvas_buf[:] = 0
                    char_mask_2d = self._cached_char_mask

                    for l in lights:
                        dx = self.X_grid - l.pos.x()
                        dy = self.Y_grid - l.pos.y()
                        dist = np.sqrt(dx * dx + dy * dy)
                        r_scaled = max(l.radius * self.calc_w / 1920, 1.0)
                        f_att = np.clip(1.0 - dist / r_scaled, 0, 1)
                        alpha = f_att * char_mask_2d * (l.intensity / 255.0) * 0.5
                        light_bgr = np.array([l.color.blue(), l.color.green(), l.color.red()], dtype=np.float32)
                        self._canvas_buf += alpha[:, :, None] * light_bgr

                    # ★軽量化: ぼかしカーネル縮小 31→15
                    canvas_blurred = cv2.GaussianBlur(self._canvas_buf, (15, 15), 0)
                    # ★軽量化: INTER_LINEAR（高速）
                    canvas_1080 = cv2.resize(canvas_blurred, (1920, 1080), interpolation=cv2.INTER_LINEAR)
                    canvas_u8 = np.clip(canvas_1080, 0, 255).astype(np.uint8)

                    # ★軽量化: スクリーン合成(float)→加算合成(uint8)で大幅高速化
                    final_u8 = cv2.add(frame, canvas_u8)
                    self.render_ready.emit(QImage(final_u8.data, 1920, 1080, 1920*3, QImage.Format_BGR888).copy())

                # ★軽量化: 15FPSキャップ（余った時間はスリープ）
                elapsed = time.time() - frame_start
                sleep_time = self._target_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            camera.stop()

# ══════════════════════════════════════════
# オーバーレイ & コントロールパネル
# ══════════════════════════════════════════
class LightingOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(0, 0, 1920, 1080)
        self.setMouseTracking(True)  # ★ マウストラッキング有効化
        
        # ★【光学迷彩モード】カメラ（dxcam）に自分を映さない魔法だ！
        try:
            hwnd = int(self.winId())
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x00000011) # WDA_EXCLUDEFROMCAPTURE
        except Exception as e:
            print(f"迷彩失敗: {e}")

        self.lights = []
        self.rendered_image = None
        self.placement_mode = False
        self.ctrl = None
        self._last_move_time = 0  # ★ マウス移動スロットリング用

    def update_render(self, img):
        self.rendered_image = img
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.rendered_image:
            painter.drawImage(self.rect(), self.rendered_image)
        if self.placement_mode:
            painter.setPen(QColor(0, 255, 255))
            painter.drawText(20, 40, "★ 配置モード(F9) - クリックで移動 / Sで4K保存")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F9: self.toggle_mode()
        if event.key() == Qt.Key_S: self.ctrl.take_screenshot()

    def toggle_mode(self):
        self.placement_mode = not self.placement_mode
        self.ctrl.chk_move.setChecked(self.placement_mode)
        if not self.placement_mode:
            # ★ 配置モード解除時にコントロールパネルを前面に復帰
            self.ctrl.raise_()
            self.ctrl.activateWindow()
        self.update()

    def mousePressEvent(self, event):
        if self.placement_mode:
            self.move_light(event.pos())
            self.activateWindow()  # ★ オーバーレイにフォーカスを確保
            self.ctrl.raise_()     # ★ コントロールパネルも前面に保つ

    def mouseMoveEvent(self, event):
        if not self.placement_mode:
            return
        # ★ ドラッグ中またはマウス移動のどちらでも追従
        now = time.time()
        if now - self._last_move_time < 0.03:  # 30msスロットリング
            return
        self._last_move_time = now
        self.move_light(event.pos())

    def move_light(self, pos):
        r = self.ctrl.list.currentRow()
        if r >= 0 and r < len(self.lights):
            self.lights[r].pos = QPoint(int(pos.x() * 320 / 1920), int(pos.y() * 180 / 1080))

class ControlPanel(QWidget):
    def __init__(self, overlay):
        super().__init__()
        self.overlay = overlay
        self.overlay.ctrl = self
        self.setWindowTitle("OshiShine GOD Edition")
        self.resize(320, 700)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        # ★【光学迷彩モード】コントロールパネルもカメラから隠す！
        try:
            hwnd = int(self.winId())
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x00000011)
        except:
            pass
        
        layout = QVBoxLayout(self)
        self.chk_move = QCheckBox("ライト配置モード (F9)")
        self.chk_move.toggled.connect(lambda v: setattr(self.overlay, 'placement_mode', v))
        layout.addWidget(self.chk_move)

        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.sync_ui)
        layout.addWidget(QLabel("📌 ライトリスト"))
        layout.addWidget(self.list)

        self.btn_del = QPushButton("🗑 選択ライト削除")
        self.btn_del.setStyleSheet("background:#a00; color:white; font-weight:bold; height:32px;")
        self.btn_del.clicked.connect(self.delete_light)
        layout.addWidget(self.btn_del)

        for t in ["Sunlight", "Rim", "Point", "Area"]:
            btn = QPushButton(f"＋ {t} 追加")
            btn.clicked.connect(lambda _, x=t: self.add_light(x))
            layout.addWidget(btn)

        self.s_int = self.add_s(layout, "光量", 0, 255, 150)
        self.s_rad = self.add_s(layout, "範囲", 50, 2000, 400)
        
        self.btn_col = QPushButton("🎨 色選択")
        self.btn_col.clicked.connect(self.pick_color)
        layout.addWidget(self.btn_col)

        self.btn_snap = QPushButton("📸 4K高画質保存 (S)")
        self.btn_snap.setStyleSheet("background:#05a; color:white; font-weight:bold; height:40px;")
        self.btn_snap.clicked.connect(self.take_screenshot)
        layout.addWidget(self.btn_snap)

    def add_s(self, layout, name, mn, mx, dv):
        layout.addWidget(QLabel(name))
        s = QSlider(Qt.Horizontal)
        s.setRange(mn, mx); s.setValue(dv)
        s.valueChanged.connect(self.update_params)
        layout.addWidget(s); return s

    def add_light(self, t):
        nl = LightSource(f"{t}_{self.list.count()+1}", t)
        self.overlay.lights.append(nl)
        self.list.addItem(nl.name)
        self.list.setCurrentRow(self.list.count()-1)

    def sync_ui(self, r):
        if 0 <= r < len(self.overlay.lights):
            l = self.overlay.lights[r]
            self.s_int.setValue(l.intensity)
            self.s_rad.setValue(l.radius)

    def update_params(self):
        r = self.list.currentRow()
        if 0 <= r < len(self.overlay.lights):
            l = self.overlay.lights[r]
            l.intensity = self.s_int.value()
            l.radius = self.s_rad.value()

    def pick_color(self):
        r = self.list.currentRow()
        if 0 <= r < len(self.overlay.lights):
            c = QColorDialog.getColor(self.overlay.lights[r].color, self, "ライトの色を選択")
            if c.isValid(): self.overlay.lights[r].color = c

    def delete_light(self):
        r = self.list.currentRow()
        if 0 <= r < len(self.overlay.lights):
            self.overlay.lights.pop(r)
            self.list.takeItem(r)
            self.overlay.update()

    def take_screenshot(self):
        img = self.overlay.rendered_image
        if img:
            save_path = os.path.join(os.path.dirname(__file__), "Screenshots")
            os.makedirs(save_path, exist_ok=True)
            fname = os.path.join(save_path, f"4K_{int(time.time())}.png")
            img.scaled(3840, 2160, Qt.IgnoreAspectRatio, Qt.SmoothTransformation).save(fname, "PNG")
            self.btn_snap.setText("✅ SAVED!")
            QTimer.singleShot(1000, lambda: self.btn_snap.setText("📸 4K高画質保存 (S)"))

# ══════════════════════════════════════════
# 実行
# ══════════════════════════════════════════
if __name__ == '__main__':
    multiprocessing.freeze_support()
    if check_single_instance():
        app = QApplication(sys.argv)
        overlay = LightingOverlay()
        ctrl = ControlPanel(overlay)
        thread = DepthEstimationThread(overlay)
        thread.render_ready.connect(overlay.update_render)
        overlay.show(); ctrl.show(); thread.start()
        sys.exit(app.exec_())