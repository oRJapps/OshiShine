import sys
import os

# ★これを追加！
try:
    import timm
except ImportError:
    pass

# ==========================================
# ★【最強のnoconsole対策
# ==========================================
class DummyStream:
    def write(self, text): pass
    def flush(self): pass
    def isatty(self): return False

if sys.stdout is None:
    sys.stdout = DummyStream()
if sys.stderr is None:
    sys.stderr = DummyStream()

import time
import dxcam
import cv2
import torch
import numpy as np
import ctypes
import gc
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QPushButton, 
                             QLabel, QCheckBox, QColorDialog, QGroupBox, QListWidget, QComboBox, QScrollArea)
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QObject, QThread, QTimer
from PyQt5.QtGui import QPainter, QColor, QRadialGradient, QBrush, QImage

class KeySignal(QObject):
    toggle_mode = pyqtSignal()
    take_ss = pyqtSignal() 

# ==========================================
# ★ ライト情報
# ==========================================
class LightSource:
    def __init__(self, name, x=960, y=540):
        self.name = name
        self.pos = QPoint(x, y)
        self.color = QColor(255, 200, 150)
        self.intensity = 150
        self.radius = 400
        self.z_depth = 128    
        self.softness = 20    
        self.light_type = "ポイントライト"
        self.placement = "背面 (バック)"
        self.scale_x = 1.0  
        self.scale_y = 1.0  
        self.rotation = 0   
        self.blend_mode = "加算 (くっきり発光)"
        self.sharpness = 1.5 
        self.is_active = True

# ==========================================
# 0. 究極のAI深度推定 ＆ 3D光演算エンジン
# ==========================================
class DepthEstimationThread(QThread):
    render_ready = pyqtSignal(QImage)

    def __init__(self, overlay):
        super().__init__()
        self.is_running = True
        self.overlay = overlay 
        self.cached_depth = None 
        self.cached_game_img = None
        self.last_state = ""
        
        torch.set_num_threads(1) 
        self.last_ai_time = 0 
        
        print("OshiShine: AIモデルをロード中...")
        self.device = torch.device("cpu")
        
        # ★DummyStreamでエラー対策は完了しているので、普通に読み込めばキャッシュからオフライン起動します！
        print("OshiShine: ローカルモデルをロード中...")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform

        self.Y, self.X = np.ogrid[:1080, :1920]
        print("OshiShine: システム準備完了！")

    def run(self):
        camera = dxcam.create(output_color="BGR")
        camera.start(video_mode=True)
        try:
            while self.is_running:
                if getattr(self.overlay, 'is_transitioning', False):
                    self.msleep(50); continue
                
                try:
                    is_locked = getattr(self.overlay, 'lock_depth', False)
                    hq_mode = getattr(self.overlay, 'hq_mode', False)
                    is_rendering = getattr(self.overlay, 'is_rendering_ss', False)
                    scale = 1.0 if (not getattr(self.overlay, 'fast_preview', True) or is_rendering) else 0.25

                    current_state = f"{hq_mode}_{scale}_{self.overlay.active_light_index}_{self.overlay.is_dragging}_"
                    for l in self.overlay.lights:
                        if l.is_active:
                            current_state += f"{l.pos.x()},{l.pos.y()},{l.radius},{l.intensity},{l.color.name()},{l.z_depth},{l.rotation},{l.scale_x},{l.scale_y},{l.blend_mode},{l.sharpness}_"
                    
                    if is_locked and not is_rendering and current_state == self.last_state:
                        self.msleep(100); continue
                    
                    self.last_state = current_state

                    current_time = time.time()
                    need_ai_update = (current_time - self.last_ai_time) > 0.5 
                    
                    if not is_locked or is_rendering:
                        frame = camera.get_latest_frame()
                        if frame is None:
                            self.msleep(5)
                            continue
                        game_img_raw = frame
                        self.cached_game_img = game_img_raw.astype(np.float32)
                        
                        if need_ai_update or self.cached_depth is None or is_rendering:
                            img_input = self.transform(game_img_raw).to(self.device)
                            with torch.no_grad():
                                prediction = self.model(img_input)
                                prediction = torch.nn.functional.interpolate(
                                    prediction.unsqueeze(1), size=game_img_raw.shape[:2], mode="bicubic", align_corners=False
                                ).squeeze()
                            self.cached_depth = cv2.normalize(prediction.cpu().numpy(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_32F)
                            self.last_ai_time = current_time 
                            
                            if is_locked and not is_rendering:
                                torch.cuda.empty_cache()

                    if scale != 1.0:
                        calc_img = cv2.resize(self.cached_game_img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                        calc_depth = cv2.resize(self.cached_depth, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                    else:
                        calc_img = self.cached_game_img
                        calc_depth = self.cached_depth

                    h, w = calc_depth.shape
                    
                    if hq_mode:
                        game_b, game_g, game_r = calc_img[:,:,0].copy(), calc_img[:,:,1].copy(), calc_img[:,:,2].copy()
                    
                    canvas_b, canvas_g, canvas_r = np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

                    for light in self.overlay.lights:
                        if not light.is_active or light.intensity <= 0: continue

                        lx = light.pos.x() * scale
                        ly = light.pos.y() * scale
                        radius = light.radius * scale
                        
                        theta = np.radians(light.rotation)
                        cos_t, sin_t = np.cos(theta), np.sin(theta)
                        dx = self.X[:h, :w] - lx
                        dy = self.Y[:h, :w] - ly
                        rx = dx * cos_t - dy * sin_t
                        ry = dx * sin_t + dy * cos_t
                        sx = max(light.scale_x, 0.01)
                        sy = max(light.scale_y, 0.01)

                        if light.light_type == "サンライト":
                            falloff = np.ones((h, w), dtype=np.float32)
                        elif light.light_type == "エリアライト":
                            dist = np.maximum(np.abs(rx / sx), np.abs(ry / sy))
                            falloff = np.clip(1.0 - (dist / radius), 0, 1)
                        else: 
                            dist = np.sqrt((rx / sx)**2 + (ry / sy)**2)
                            falloff = np.clip(1.0 - (dist / radius), 0, 1)

                        if light.sharpness != 1.0:
                            falloff = falloff ** light.sharpness

                        S = max(light.softness, 1)
                        if light.light_type == "リムライト":
                            diff = np.abs(light.z_depth - calc_depth)
                            depth_mask = np.clip(1.0 - (diff / S), 0, 1)
                        else:
                            if light.placement == "前面 (フロント)":
                                depth_mask = np.ones((h, w), dtype=np.float32)
                            else:
                                diff = light.z_depth - calc_depth
                                depth_mask = np.clip(diff / S + 0.5, 0, 1)

                        alpha = falloff * depth_mask * (light.intensity / 255.0)
                        light_b = alpha * light.color.blue()
                        light_g = alpha * light.color.green()
                        light_r = alpha * light.color.red()

                        if hq_mode:
                            if light.blend_mode == "加算 (くっきり発光)":
                                game_b = np.clip(game_b + light_b, 0, 255)
                                game_g = np.clip(game_g + light_g, 0, 255)
                                game_r = np.clip(game_r + light_r, 0, 255)
                            elif light.blend_mode == "覆い焼きカラー (鮮やか)":
                                blend_b = np.clip(light_b / 255.0, 0, 0.99)
                                blend_g = np.clip(light_g / 255.0, 0, 0.99)
                                blend_r = np.clip(light_r / 255.0, 0, 0.99)
                                game_b = np.where(blend_b == 0, game_b, np.clip(game_b / (1.0 - blend_b), 0, 255))
                                game_g = np.where(blend_g == 0, game_g, np.clip(game_g / (1.0 - blend_g), 0, 255))
                                game_r = np.where(blend_r == 0, game_r, np.clip(game_r / (1.0 - blend_r), 0, 255))
                            else: 
                                game_b = 255.0 - ((255.0 - game_b) * (255.0 - np.clip(light_b, 0, 255)) / 255.0)
                                game_g = 255.0 - ((255.0 - game_g) * (255.0 - np.clip(light_g, 0, 255)) / 255.0)
                                game_r = 255.0 - ((255.0 - game_r) * (255.0 - np.clip(light_r, 0, 255)) / 255.0)
                        else:
                            canvas_b += light_b
                            canvas_g += light_g
                            canvas_r += light_r

                    if hq_mode:
                        rgb_img = np.dstack((np.clip(game_b, 0, 255).astype(np.uint8), 
                                             np.clip(game_g, 0, 255).astype(np.uint8), 
                                             np.clip(game_r, 0, 255).astype(np.uint8)))
                    else:
                        canvas_b, canvas_g, canvas_r = np.clip(canvas_b, 0, 255).astype(np.uint8), np.clip(canvas_g, 0, 255).astype(np.uint8), np.clip(canvas_r, 0, 255).astype(np.uint8)
                        canvas_a = np.clip(canvas_b.astype(int) + canvas_g.astype(int) + canvas_r.astype(int), 0, 255).astype(np.uint8)
                        rgb_img = np.dstack((canvas_b, canvas_g, canvas_r, canvas_a))

                    if scale != 1.0:
                        rgb_img = cv2.resize(rgb_img, (1920, 1080), interpolation=cv2.INTER_LINEAR)

                    if is_rendering:
                        cv2.imwrite("oshikatsu_best_shot.png", rgb_img)
                        self.overlay.is_rendering_ss = False
                        self.overlay.ss_finished.emit()
                    else:
                        fmt = QImage.Format_BGR888 if hq_mode else QImage.Format_ARGB32
                        q_img = QImage(rgb_img.data, 1920, 1080, 1920 * (3 if hq_mode else 4), fmt).copy()
                        self.render_ready.emit(q_img)
                    
                    self.msleep(80) 
                    
                except Exception as e:
                    self.msleep(100)
        finally:
            if camera is not None:
                camera.stop()
                del camera

    def stop(self):
        self.is_running = False
        self.wait()

# ==========================================
# 1. 透明なキャンバス
# ==========================================
class LightingOverlay(QWidget):
    ss_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setGeometry(0, 0, 1920, 1080)
        
        try:
            hwnd = int(self.winId())
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x11)
        except Exception:
            pass
            
        self.lights = []
        self.active_light_index = -1
        self.vignette_intensity = 180
        self.is_dragging = False
        self.is_edit_mode = False
        self.is_transitioning = False
        
        self.hq_mode = True 
        self.lock_depth = False 
        self.fast_preview = True 
        self.is_rendering_ss = False
        self.rendered_image = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.rendered_image:
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.drawImage(self.rect(), self.rendered_image)
        if self.vignette_intensity > 0:
            vignette = QRadialGradient(960, 540, 1100)
            vignette.setColorAt(0, QColor(0, 0, 0, 0))
            vignette.setColorAt(1, QColor(0, 0, 0, self.vignette_intensity))
            painter.fillRect(self.rect(), QBrush(vignette))

    def update_render(self, q_img):
        self.rendered_image = q_img
        self.update()

    def mousePressEvent(self, event):
        if not self.is_edit_mode or self.active_light_index == -1: return 
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.lights[self.active_light_index].pos = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if not self.is_edit_mode or self.active_light_index == -1: return
        if self.is_dragging:
            self.lights[self.active_light_index].pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if not self.is_edit_mode: return
        if event.button() == Qt.LeftButton:
            self.is_dragging = False

    def set_edit_mode(self, enabled):
        self.is_transitioning = True
        time.sleep(0.1) 
        self.is_edit_mode = enabled
        self.is_dragging = False
        flags = self.windowFlags()
        if enabled:
            flags &= ~Qt.WindowTransparentForInput
            self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        else:
            flags |= Qt.WindowTransparentForInput
            self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setWindowFlags(flags)
        self.show() 
        self.is_transitioning = False

# ==========================================
# 2. 操作パネル
# ==========================================
class ControlPanel(QWidget):
    def __init__(self, overlay):
        super().__init__()
        self.overlay = overlay
        self.light_count = 0
        
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setWindowTitle("OshiShine")
        self.setGeometry(50, 50, 440, 900)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        self.hotkey_timer = QTimer(self)
        self.hotkey_timer.timeout.connect(self.poll_hotkeys)
        self.hotkey_timer.start(30)
        self.f9_pressed = False
        self.f10_pressed = False

        self.fast_check = QCheckBox("⚡ プレビュー極限軽量化 (負荷軽減)")
        self.fast_check.setStyleSheet("font-weight: bold; color: #ffff55;")
        self.fast_check.setChecked(True)
        self.fast_check.stateChanged.connect(self.toggle_fast_mode)
        main_layout.addWidget(self.fast_check)

        self.lock_check = QCheckBox("🔒 深度(空間)をロックして負荷軽減")
        self.lock_check.setStyleSheet("font-weight: bold; color: #ff5555;")
        self.lock_check.stateChanged.connect(self.toggle_lock_mode)
        main_layout.addWidget(self.lock_check)

        self.hq_check = QCheckBox("✨ 高画質スクリーン合成 (推奨)")
        self.hq_check.setStyleSheet("font-weight: bold; color: #00ffcc;")
        self.hq_check.setChecked(True)
        self.hq_check.stateChanged.connect(self.toggle_hq_mode)
        main_layout.addWidget(self.hq_check)

        self.mode_check = QCheckBox("💡 ライト移動モード [F9キーでON/OFF]")
        self.mode_check.setStyleSheet("font-weight: bold; color: #ffaa00;")
        self.mode_check.stateChanged.connect(self.toggle_mode_checkbox)
        main_layout.addWidget(self.mode_check)
        
        list_group = QGroupBox("▼ ライト一覧")
        list_layout = QVBoxLayout()
        self.light_list = QListWidget()
        self.light_list.setMaximumHeight(100)
        self.light_list.currentRowChanged.connect(self.select_light)
        list_layout.addWidget(self.light_list)

        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("➕ 追加")
        self.add_btn.clicked.connect(self.add_light)
        self.del_btn = QPushButton("➖ 削除")
        self.del_btn.clicked.connect(self.remove_light)
        btn_layout.addWidget(self.add_btn); btn_layout.addWidget(self.del_btn)
        list_layout.addLayout(btn_layout); list_group.setLayout(list_layout)
        main_layout.addWidget(list_group)

        prop_group = QGroupBox("▼ 選択中のライト (トランスフォーム)")
        self.prop_layout = QVBoxLayout()

        combo_layout = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["ポイントライト", "サンライト", "エリアライト", "リムライト"])
        self.type_combo.currentTextChanged.connect(self.update_light_prop)
        self.place_combo = QComboBox()
        self.place_combo.addItems(["背面 (バック)", "前面 (フロント)"])
        self.place_combo.currentTextChanged.connect(self.update_light_prop)
        combo_layout.addWidget(QLabel("種類:")); combo_layout.addWidget(self.type_combo)
        combo_layout.addWidget(QLabel("配置:")); combo_layout.addWidget(self.place_combo)
        self.prop_layout.addLayout(combo_layout)

        blend_layout = QHBoxLayout()
        self.blend_combo = QComboBox()
        self.blend_combo.addItems(["加算 (くっきり発光)", "覆い焼きカラー (鮮やか)", "スクリーン (ふんわり)"])
        self.blend_combo.currentTextChanged.connect(self.update_light_prop)
        blend_layout.addWidget(QLabel("合成:")); blend_layout.addWidget(self.blend_combo)
        self.prop_layout.addLayout(blend_layout)

        self.color_btn = QPushButton("🎨 色を変更")
        self.color_btn.clicked.connect(self.choose_color)
        self.prop_layout.addWidget(self.color_btn)

        self.rad_slider = self.create_slider("基本サイズ (Radius)", 100, 2000, 400, self.update_light_prop, self.prop_layout)
        
        scale_layout = QHBoxLayout()
        self.sx_slider = self.create_slider("幅 (X Scale)", 10, 300, 100, self.update_light_prop, scale_layout, is_horizontal=False)
        self.sy_slider = self.create_slider("高さ (Y Scale)", 10, 300, 100, self.update_light_prop, scale_layout, is_horizontal=False)
        self.prop_layout.addLayout(scale_layout)
        
        self.rot_slider = self.create_slider("回転 (Rotation)", 0, 360, 0, self.update_light_prop, self.prop_layout)
        self.int_slider = self.create_slider("明るさ (Intensity)", 0, 255, 150, self.update_light_prop, self.prop_layout)
        self.sharp_slider = self.create_slider("光の芯 (シャープさ) 1.0~5.0", 10, 50, 15, self.update_light_prop, self.prop_layout)
        self.sharp_slider.setStyleSheet("background-color: #113333;")

        self.z_slider = self.create_slider("Z軸・奥行き (0=奥, 255=手前)", 0, 255, 128, self.update_light_prop, self.prop_layout)
        self.z_slider.setStyleSheet("background-color: #331111;")
        self.soft_slider = self.create_slider("回り込み・柔らかさ", 1, 100, 20, self.update_light_prop, self.prop_layout)

        prop_group.setLayout(self.prop_layout)
        main_layout.addWidget(prop_group)
        
        self.v_slider = self.create_slider("周辺光量落ち (ビネット)", 0, 255, 180, self.update_vignette, main_layout)

        self.ss_btn = QPushButton("📸 スクリーンショット撮影 (F10)")
        self.ss_btn.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold; padding: 10px; margin-top: 10px;")
        self.ss_btn.clicked.connect(self.take_screenshot)
        main_layout.addWidget(self.ss_btn)

        scroll_area.setWidget(main_widget)
        outer_layout.addWidget(scroll_area)
        self.add_light()
        
        self.overlay.ss_finished.connect(self.on_ss_finished)

    def poll_hotkeys(self):
        f9_state = ctypes.windll.user32.GetAsyncKeyState(0x78) & 0x8000
        f10_state = ctypes.windll.user32.GetAsyncKeyState(0x79) & 0x8000
        if f9_state and not self.f9_pressed:
            self.toggle_mode_from_hotkey()
            self.f9_pressed = True
        elif not f9_state:
            self.f9_pressed = False
        if f10_state and not self.f10_pressed:
            self.take_screenshot()
            self.f10_pressed = True
        elif not f10_state:
            self.f10_pressed = False

    def create_slider(self, label_text, min_val, max_val, default_val, connect_func, layout, is_horizontal=True):
        container = QWidget()
        l = QVBoxLayout(container) if not is_horizontal else QHBoxLayout(container)
        l.setContentsMargins(0,0,0,0)
        l.addWidget(QLabel(label_text))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(connect_func)
        l.addWidget(slider)
        layout.addWidget(container)
        return slider

    def toggle_fast_mode(self, state): self.overlay.fast_preview = (state == Qt.Checked)
    def toggle_lock_mode(self, state): self.overlay.lock_depth = (state == Qt.Checked)
    def toggle_hq_mode(self, state): self.overlay.hq_mode = (state == Qt.Checked)
    def toggle_mode_checkbox(self, state): self.overlay.set_edit_mode(state == Qt.Checked); self.raise_(); self.activateWindow()
    def toggle_mode_from_hotkey(self): self.mode_check.setChecked(not self.mode_check.isChecked())

    def add_light(self):
        self.light_count += 1
        name = f"ライト {self.light_count}"
        self.overlay.lights.append(LightSource(name))
        self.light_list.addItem(name)
        self.light_list.setCurrentRow(len(self.overlay.lights) - 1)

    def remove_light(self):
        idx = self.light_list.currentRow()
        if idx >= 0:
            self.overlay.lights.pop(idx); self.light_list.takeItem(idx)

    def select_light(self, idx):
        self.overlay.active_light_index = idx
        if idx >= 0:
            light = self.overlay.lights[idx]
            for s in [self.type_combo, self.place_combo, self.blend_combo, self.rad_slider, self.int_slider, 
                      self.z_slider, self.soft_slider, self.sx_slider, self.sy_slider, self.rot_slider, self.sharp_slider]:
                s.blockSignals(True)
            
            self.type_combo.setCurrentText(light.light_type)
            self.place_combo.setCurrentText(light.placement)
            self.blend_combo.setCurrentText(light.blend_mode)
            self.rad_slider.setValue(light.radius)
            self.sx_slider.setValue(int(light.scale_x * 100))
            self.sy_slider.setValue(int(light.scale_y * 100))
            self.rot_slider.setValue(light.rotation)
            self.int_slider.setValue(light.intensity)
            self.sharp_slider.setValue(int(light.sharpness * 10))
            self.z_slider.setValue(light.z_depth)
            self.soft_slider.setValue(light.softness)
            
            for s in [self.type_combo, self.place_combo, self.blend_combo, self.rad_slider, self.int_slider, 
                      self.z_slider, self.soft_slider, self.sx_slider, self.sy_slider, self.rot_slider, self.sharp_slider]:
                s.blockSignals(False)

    def update_light_prop(self, *args):
        idx = self.light_list.currentRow()
        if idx >= 0:
            light = self.overlay.lights[idx]
            light.light_type = self.type_combo.currentText()
            light.placement = self.place_combo.currentText()
            light.blend_mode = self.blend_combo.currentText()
            light.radius = self.rad_slider.value()
            light.scale_x = self.sx_slider.value() / 100.0
            light.scale_y = self.sy_slider.value() / 100.0
            light.rotation = self.rot_slider.value()
            light.intensity = self.int_slider.value()
            light.sharpness = self.sharp_slider.value() / 10.0
            light.z_depth = self.z_slider.value()
            light.softness = self.soft_slider.value()

    def choose_color(self):
        idx = self.light_list.currentRow()
        if idx >= 0:
            color = QColorDialog.getColor(self.overlay.lights[idx].color, self, "色を選択")
            if color.isValid(): self.overlay.lights[idx].color = color
    def update_vignette(self, v): self.overlay.vignette_intensity = v

    def take_screenshot(self):
        self.ss_btn.setText("⏳ レンダリング中...")
        self.ss_btn.setEnabled(False)
        self.hide() 
        if self.mode_check.isChecked(): 
            self.mode_check.setChecked(False)
        
        QTimer.singleShot(800, self._start_rendering)

    def _start_rendering(self):
        self.overlay.is_rendering_ss = True

    def on_ss_finished(self):
        self.ss_btn.setText("📸 スクリーンショット撮影 (F10)")
        self.ss_btn.setEnabled(True)
        self.show()

    def closeEvent(self, event):
        self.overlay.close(); event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    palette = app.palette()
    palette.setColor(palette.Window, QColor(30, 30, 30)); palette.setColor(palette.WindowText, Qt.white)
    palette.setColor(palette.Base, QColor(15, 15, 15)); palette.setColor(palette.Button, QColor(50, 50, 50))
    palette.setColor(palette.ButtonText, Qt.white); palette.setColor(palette.Text, Qt.white)
    app.setPalette(palette)
    
    overlay_window = LightingOverlay()
    overlay_window.show()
    
    depth_thread = DepthEstimationThread(overlay_window)
    depth_thread.start(QThread.LowestPriority)
    depth_thread.render_ready.connect(overlay_window.update_render)
    
    control_panel = ControlPanel(overlay_window)
    control_panel.show()
    
    app.aboutToQuit.connect(depth_thread.stop)
    sys.exit(app.exec_())