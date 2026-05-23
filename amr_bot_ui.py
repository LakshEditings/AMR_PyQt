import sys
import math
import json
import os
import threading
import csv
import datetime
from collections import deque

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QLabel, QPushButton, QFrame, QSizePolicy, QDialog,
    QFileDialog,
)
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, Signal, QObject
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont,
    QPolygonF, QRadialGradient, QLinearGradient,
    QKeyEvent, QPalette,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
PIXELS_PER_METRE   = 40.0
GRID_SIZE          = 40.0  # CHANGED: 40 pixels = Exactly 1.0 Metre spatial separation
BOT_W, BOT_H       = 30, 20
BASE_LINEAR_SPEED  = 0.5
BASE_ANGULAR_SPEED = 0.8
SPEED_STEP         = 5          
WHEEL_BASE         = 0.40
WAYPOINT_REACH_DIST = 0.15
# ─────────────────────────────────────────────────────────────────────────────
#  Map Canvas
# ─────────────────────────────────────────────────────────────────────────────
class MapCanvas(QWidget):
    waypoint_clicked = Signal(float, float)

    MODE_FOLLOW   = "follow"
    MODE_OVERVIEW = "overview"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.bot_x     = 0.0
        self.bot_y     = 0.0
        self.bot_angle = 0.0

        self.home_x = 0.0
        self.home_y = 0.0
        self.home_set = False          

        self.trail: list[tuple[float, float]] = []

        self.scale    = PIXELS_PER_METRE
        self._view_ox = 0.0
        self._view_oy = 0.0

        self.mode = self.MODE_FOLLOW

        self.lidar_points: list[tuple[float, float]] = []

        self.waypoints: list[tuple[float, float]] = []
        self.current_waypoint_idx = 0
        self.waypoint_mode = False

        self._manual_view = False
        self._pan_active  = False
        self._pan_last    = QPointF(0, 0)
        self.setMouseTracking(False)

    def _w2c(self, wx, wy):
        cx = self.width()  / 2 + (wx - self._view_ox) * self.scale
        cy = self.height() / 2 - (wy - self._view_oy) * self.scale
        return QPointF(cx, cy)

    def _c2w(self, cx, cy):
        wx = (cx - self.width()  / 2) / self.scale + self._view_ox
        wy = (self.height() / 2 - cy) / self.scale + self._view_oy
        return wx, wy

    def set_home(self, x, y):
        self.home_x   = x
        self.home_y   = y
        self.home_set = True
        self.trail    = [(x, y)]       
        self.update()

    def update_bot(self, x, y, angle):
        self.bot_x, self.bot_y, self.bot_angle = x, y, angle
        self.trail.append((x, y))
        if len(self.trail) > 5000:
            self.trail = self.trail[-5000:]
        self.update()

    def set_mode_follow(self):
        self._manual_view = False
        self.mode = self.MODE_FOLLOW
        self.update()

    def set_mode_overview(self):
        self._manual_view = False
        self.mode = self.MODE_OVERVIEW
        self.update()

    def fit_trail_view(self):
        if len(self.trail) < 2:
            self.set_mode_overview()
            return
        self._manual_view = False
        self.mode = self.MODE_OVERVIEW
        xs = [p[0] for p in self.trail]
        ys = [p[1] for p in self.trail]
        for wx, wy in self.waypoints:
            xs.append(wx); ys.append(wy)
        xs.append(self.home_x); ys.append(self.home_y)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        self._view_ox = (x_min + x_max) / 2
        self._view_oy = (y_min + y_max) / 2
        span_x = max(x_max - x_min, 0.5)
        span_y = max(y_max - y_min, 0.5)
        W, H = self.width() or 500, self.height() or 420
        pad = 0.80
        self.scale = max(min(W * pad / span_x, H * pad / span_y), 2.0)
        self.update()

    def set_waypoint_mode(self, active):
        self.waypoint_mode = active
        self.setCursor(Qt.CrossCursor if active else Qt.ArrowCursor)
        self.update()

    def clear_waypoints(self):
        self.waypoints.clear()
        self.current_waypoint_idx = 0
        self.update()

    ZOOM_FACTOR = 1.25

    def zoom_in(self):
        self._manual_view = True
        self.scale = min(self.scale * self.ZOOM_FACTOR, 400.0)
        self.update()

    def zoom_out(self):
        self._manual_view = True
        self.scale = max(self.scale / self.ZOOM_FACTOR, 1.5)
        self.update()

    def reset_view(self):
        self._manual_view = False
        self.scale = PIXELS_PER_METRE
        self.update()

    def mousePressEvent(self, event):
        if self.waypoint_mode and event.button() == Qt.LeftButton:
            wx, wy = self._c2w(event.position().x(), event.position().y())
            self.waypoints.append((wx, wy))
            self.waypoint_clicked.emit(wx, wy)
            self.update()
            return
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._pan_active  = True
            self._pan_last    = event.position()
            self._manual_view = True
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._pan_active:
            delta = event.position() - self._pan_last
            self._pan_last = event.position()
            self._view_ox -= delta.x() / self.scale
            self._view_oy += delta.y() / self.scale
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._pan_active = False
            self.setCursor(Qt.CrossCursor if self.waypoint_mode else Qt.ArrowCursor)

    def wheelEvent(self, event):
        self._manual_view = True
        cx, cy = event.position().x(), event.position().y()
        wx_before, wy_before = self._c2w(cx, cy)
        delta  = event.angleDelta().y()
        factor = self.ZOOM_FACTOR if delta > 0 else 1.0 / self.ZOOM_FACTOR
        self.scale = max(1.5, min(400.0, self.scale * factor))
        wx_after, wy_after = self._c2w(cx, cy)
        self._view_ox -= (wx_after - wx_before)
        self._view_oy -= (wy_after - wy_before)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        self._compute_view()
        self._draw_bg(p)
        self._draw_grid(p)
        self._draw_lidar(p)
        self._draw_trail(p)
        self._draw_waypoints(p)
        self._draw_home(p)
        self._draw_bot(p)
        self._draw_mode_label(p)
        if not self.home_set:
            p.setPen(QPen(QColor(255, 100, 100, 180)))
            p.setFont(QFont("Courier New", 13, QFont.Bold))
            p.drawText(self.rect(), Qt.AlignCenter, "⚠  CLICK  ⌂  TO SET HOME FIRST")
        p.end()

    def _compute_view(self):
        if self._manual_view:
            return
        W, H = self.width(), self.height()
        if self.mode == self.MODE_FOLLOW:
            self._view_ox = self.bot_x
            self._view_oy = self.bot_y
            if len(self.trail) > 1:
                max_reach = max(
                    math.hypot(tx - self.bot_x, ty - self.bot_y)
                    for tx, ty in self.trail
                )
                half_vp     = min(W, H) / 2 * 0.70
                needed_scale = half_vp / max_reach if max_reach > 0 else PIXELS_PER_METRE
                self.scale  = min(self.scale, max(needed_scale, 4.0))
        else:
            all_pts = self.trail + self.waypoints + [(self.bot_x, self.bot_y)]
            if len(all_pts) < 2:
                self._view_ox = self.bot_x
                self._view_oy = self.bot_y
                self.scale    = PIXELS_PER_METRE
                return
            xs = [t[0] for t in all_pts]; ys = [t[1] for t in all_pts]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            self._view_ox = (x_min + x_max) / 2
            self._view_oy = (y_min + y_max) / 2
            span_x = (x_max - x_min) or 0.1
            span_y = (y_max - y_min) or 0.1
            sx = W * 0.85 / span_x; sy = H * 0.85 / span_y
            self.scale = max(min(sx, sy), 2.0)

    def _draw_bg(self, p):
        g = QLinearGradient(0, 0, 0, self.height())
        g.setColorAt(0, QColor("#035ae6"))
        g.setColorAt(1, QColor("#0C64E0"))
        p.fillRect(self.rect(), g)

    def _draw_grid(self, p):
        W, H = self.width(), self.height()
        
        # Calculate dynamic pixel size based on user zoom interaction scale profile
        step = self.scale
        if step < 6:
            return  # Prevents rendering dense grid lines when zoomed out way too far
            
        p.setPen(QPen(QColor(255, 255, 255, 60), 1, Qt.DotLine))
        
        # Calculate viewport bounds in real-world meters
        left_world = self._view_ox - (W / 2) / self.scale
        right_world = self._view_ox + (W / 2) / self.scale
        bot_world = self._view_oy - (H / 2) / self.scale
        top_world = self._view_oy + (H / 2) / self.scale
        
        # Draw Vertical Grid Lines (1 meter intervals)
        x_start = math.ceil(left_world)
        x_end = math.floor(right_world)
        for wx in range(x_start, x_end + 1):
            cp = self._w2c(float(wx), 0.0)
            p.drawLine(int(cp.x()), 0, int(cp.x()), H)
            
        # Draw Horizontal Grid Lines (1 meter intervals)
        y_start = math.ceil(bot_world)
        y_end = math.floor(top_world)
        for wy in range(y_start, y_end + 1):
            cp = self._w2c(0.0, float(wy))
            p.drawLine(0, int(cp.y()), W, int(cp.y()))
            
        # Draw Highlighted Origin Axes (Red for X, Green for Y)
        origin = self._w2c(0.0, 0.0)
        p.setPen(QPen(QColor(220, 50, 50, 210), 1.5))  # X-Axis Line
        p.drawLine(0, int(origin.y()), W, int(origin.y()))
        p.setPen(QPen(QColor(50, 220, 80, 210), 1.5))  # Y-Axis Line
        p.drawLine(int(origin.x()), 0, int(origin.x()), H)

    def _draw_lidar(self, p):
        if not self.lidar_points: return
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 230, 255, 150)))
        for wx, wy in self.lidar_points:
            p.drawEllipse(self._w2c(wx, wy), 2, 2)

    def _draw_trail(self, p):
        if len(self.trail) < 2: return
        n   = len(self.trail)
        pen = QPen(QColor(0, 200, 100, 180), 2, Qt.SolidLine)
        pen.setCapStyle(Qt.RoundCap)
        pts = [self._w2c(x, y) for x, y in self.trail]
        lw  = 2 if self.mode == self.MODE_FOLLOW else 3
        pen.setWidth(lw)
        for i in range(1, n):
            alpha = max(40, int(180 * i / n))
            pen.setColor(QColor(0, 200, 100, alpha))
            p.setPen(pen)
            p.drawLine(pts[i - 1], pts[i])
        if self.mode == self.MODE_OVERVIEW:
            sp = pts[0]
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor("#FFD700")))
            p.drawEllipse(sp, 7, 7)
            p.setPen(QPen(QColor("#FFD700")))
            p.setFont(QFont("Courier New", 7, QFont.Bold))
            p.drawText(int(sp.x()) + 10, int(sp.y()) - 4, "START")

    def _draw_waypoints(self, p):
        if not self.waypoints: return
        pts = [self._w2c(x, y) for x, y in self.waypoints]
        if len(pts) > 1:
            p.setPen(QPen(QColor(255, 165, 0, 160), 2, Qt.DashLine))
            bp = self._w2c(self.bot_x, self.bot_y)
            p.drawLine(bp, pts[0])
            for i in range(1, len(pts)):
                p.drawLine(pts[i - 1], pts[i])
        for i, pt in enumerate(pts):
            is_current = (i == self.current_waypoint_idx)
            is_done    = (i <  self.current_waypoint_idx)
            if is_done:
                p.setBrush(QBrush(QColor(100, 100, 100, 180)))
                p.setPen(QPen(QColor(180, 180, 180), 1))
            elif is_current:
                p.setBrush(QBrush(QColor(255, 200, 0, 230)))
                p.setPen(QPen(QColor("#FFD700"), 2))
            else:
                p.setBrush(QBrush(QColor(255, 120, 30, 200)))
                p.setPen(QPen(QColor("#ff9800"), 2))
            r = 10 if is_current else 8
            p.drawEllipse(pt, r, r)
            p.setPen(QPen(QColor("#000") if not is_done else QColor("#aaa")))
            p.setFont(QFont("Courier New", 7, QFont.Bold))
            p.drawText(int(pt.x()) - 4, int(pt.y()) + 4, str(i + 1))
        if self.waypoint_mode:
            p.setPen(QPen(QColor("#ff9800")))
            p.setFont(QFont("Courier New", 9, QFont.Bold))
            p.drawText(8, self.height() - 10,
                       f"✦  WAYPOINT MODE  —  CLICK TO ADD  |  {len(self.waypoints)} set")

    def _draw_home(self, p):
        if not self.home_set:
            return
        hp  = self._w2c(self.home_x, self.home_y)
        pen = QPen(QColor("#FFD700"), 2)
        p.setPen(pen)
        r = 12
        p.drawLine(int(hp.x()) - r, int(hp.y()), int(hp.x()) + r, int(hp.y()))
        p.drawLine(int(hp.x()), int(hp.y()) - r, int(hp.x()), int(hp.y()) + r)
        p.setBrush(Qt.NoBrush)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.drawEllipse(hp, 10, 10)
        p.setPen(QPen(QColor("#FFD700")))
        p.setFont(QFont("Courier New", 8, QFont.Bold))
        p.drawText(int(hp.x()) + 14, int(hp.y()) - 6, "HOME")

    def _draw_bot(self, p):
        bp = self._w2c(self.bot_x, self.bot_y)
        p.save()
        p.translate(bp)
        p.rotate(-self.bot_angle)
        glow = QRadialGradient(QPointF(0, 0), 30)
        glow.setColorAt(0, QColor(0, 180, 255, 60))
        glow.setColorAt(1, QColor(0, 180, 255, 0))
        p.setBrush(QBrush(glow)); p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(0, 0), 30, 30)
        bw, bh = BOT_W, BOT_H
        p.setBrush(QBrush(QColor("#e76018")))
        p.setPen(QPen(QColor("#e76016"), 1.5))
        p.drawRoundedRect(QRectF(-bw/2, -bh/2, bw, bh), 4, 4)
        p.setBrush(QBrush(QColor("#000000"))); p.setPen(Qt.NoPen)
        for wx_, wy_ in [(-bw/2+3, -bh/2-2), (bw/2-3, -bh/2-2),
                          (-bw/2+3,  bh/2+2), (bw/2-3,  bh/2+2)]:
            p.drawEllipse(QPointF(wx_, wy_), 4, 3)
        pen = QPen(QColor("#00e676"), 2); pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen); p.setBrush(QBrush(QColor("#00e676")))
        p.drawPolyline(QPolygonF([QPointF(bw/2+4, 0), QPointF(bw/2+14, 0)]))
        p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygonF([QPointF(bw/2+14,-4),
                                  QPointF(bw/2+20,  0),
                                  QPointF(bw/2+14,  4)]))
        p.restore()

    def _draw_mode_label(self, p):
        if self._manual_view:
            p.setPen(QPen(QColor("#26c6da")))
            p.setFont(QFont("Courier New", 9, QFont.Bold))
            p.drawText(8, 18, "◎  MANUAL VIEW  —  SCROLL/DRAG TO PAN·ZOOM")
        elif self.mode == self.MODE_OVERVIEW:
            p.setPen(QPen(QColor("#ff9800")))
            p.setFont(QFont("Courier New", 9, QFont.Bold))
            p.drawText(8, 18, "◎  ROUTE OVERVIEW  —  FULL PATH")
        else:
            p.setPen(QPen(QColor("#39d353")))
            p.setFont(QFont("Courier New", 9, QFont.Bold))
            p.drawText(8, 18, "◎  FOLLOW MODE  —  BOT CENTRED")


# ─────────────────────────────────────────────────────────────────────────────
#  Professional Telemetry Panel
# ─────────────────────────────────────────────────────────────────────────────
class TelemetryPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumWidth(500)
        self.setMinimumHeight(520)

        self.setStyleSheet("""
            QFrame {
                background:#111827;
                border:1px solid #374151;
                border-radius:10px;
            }
            QLabel#title {
                color:#60a5fa;
                font-family:'Segoe UI';
                font-size:18px;
                font-weight:700;
                border:none;
            }
            QLabel#label {
                color:#d1d5db;
                font-family:'Segoe UI';
                font-size:14px;
                font-weight:600;
                letter-spacing:1px;
                border:none;
            }
            QLabel#value {
                color:#22c55e;
                font-family:'Consolas';
                font-size:22px;
                font-weight:700;
                min-width:130px;
                background:#0b1220;
                border:1px solid #374151;
                border-radius:6px;
                padding:6px 14px;
            }
            QLabel#unit {
                color:#9ca3af;
                font-family:'Segoe UI';
                font-size:13px;
                font-weight:600;
                min-width:38px;
                border:none;
            }
            QWidget#row {
                background:#0f172a;
                border:1px solid #1f2937;
                border-radius:8px;
            }
        """)

        main = QVBoxLayout(self)
        main.setContentsMargins(22, 22, 22, 22)
        main.setSpacing(10)

        title = QLabel("◉  AMR TELEMETRY")
        title.setObjectName("title")
        main.addWidget(title)

        self.fields = {}

        telemetry_items = [
            ("X DIST FROM HOME", "m",   "x_dist"),
            ("Y DIST FROM HOME", "m",   "y_dist"),
            ("ANGLE FROM HOME",  "deg", "angle"),
            ("SPEED",            "m/s", "speed"),
            ("ANGULAR SPEED",    "°/s", "ang_speed"),
            ("LINEAR SPD MOD",   "%",   "lin_mod"),
            ("TURN SPD MOD",     "%",   "trn_mod"),
            ("WAYPOINT",         "",    "waypoint"),
        ]

        for label, unit, key in telemetry_items:
            row = QWidget()
            row.setObjectName("row")

            rl = QHBoxLayout(row)
            rl.setContentsMargins(16, 13, 16, 13)
            rl.setSpacing(16)

            lb = QLabel(label)
            lb.setObjectName("label")

            vl = QLabel("0.00")
            vl.setMinimumHeight(46)
            vl.setObjectName("value")
            vl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            ul = QLabel(unit)
            ul.setObjectName("unit")

            rl.addWidget(lb)
            rl.addStretch()
            rl.addWidget(vl)
            rl.addWidget(ul)

            main.addWidget(row)
            self.fields[key] = vl

        main.addStretch()

    def update_telemetry(self, x, y, angle, speed, ang_speed, lin_mod, trn_mod, waypoint_info="—"):
        self.fields["x_dist"].setText(f"{x:+.2f}")
        self.fields["y_dist"].setText(f"{y:+.2f}")
        self.fields["angle"].setText(f"{angle:.1f}")
        self.fields["speed"].setText(f"{speed:.2f}")
        self.fields["ang_speed"].setText(f"{ang_speed:.1f}")
        self.fields["lin_mod"].setText(f"{lin_mod:.0f}")
        self.fields["trn_mod"].setText(f"{trn_mod:.0f}")
        self.fields["waypoint"].setText(waypoint_info)
        

# ─────────────────────────────────────────────────────────────────────────────
#  Speed Value Display
# ─────────────────────────────────────────────────────────────────────────────
class SpeedValueDisplay(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(145, 105)
        self.setStyleSheet("""
            QFrame {
                background:qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #180028,
                    stop:1 #2b0040
                );
                border:2px solid #9333ea;
                border-radius:16px;
            }
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)

        top = QLabel("SPEED")
        top.setAlignment(Qt.AlignCenter)
        top.setStyleSheet("""
            color:#d8b4fe;
            font-family:'Segoe UI';
            font-size:11px;
            font-weight:700;
            border:none;
        """)
        lay.addWidget(top)

        self._val_lbl = QLabel("100")
        self._val_lbl.setAlignment(Qt.AlignCenter)
        self._val_lbl.setStyleSheet("""
            color:#f5d0fe;
            font-family:'JetBrains Mono';
            font-size:42px;
            font-weight:900;
            letter-spacing:2px;
            border:none;
            padding-top:0px;
            padding-bottom:0px;
        """)
        lay.addWidget(self._val_lbl)

        bot = QLabel("%")
        bot.setAlignment(Qt.AlignCenter)
        bot.setStyleSheet("""
            color:#c084fc;
            font-family:'Segoe UI';
            font-size:11px;
            font-weight:700;
            border:none;
        """)
        lay.addWidget(bot)

    def set_value(self, v: float):
        self._val_lbl.setText(f"{v:.0f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Control Pad
# ─────────────────────────────────────────────────────────────────────────────
class ControlPad(QFrame):
    GAP_PX = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame { background:#0d1117; border:1px solid #30363d; border-radius:10px; }"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        ttl = QLabel("◈  DRIVE CONTROLS")
        ttl.setStyleSheet(
            "color:#58a6ff; font-family:'Courier New'; font-size:11px; "
            "font-weight:bold; letter-spacing:2px;"
        )
        outer.addWidget(ttl, alignment=Qt.AlignCenter)

        top_body = QHBoxLayout()
        top_body.setSpacing(10)
        top_body.setContentsMargins(0, 0, 0, 0)

        SZ = 68
        self.btn_fwd      = ControlButton("▲", "red",    size=SZ)
        self.btn_bwd      = ControlButton("▼", "red",    size=SZ)
        self.btn_left     = ControlButton("◄", "red",    size=SZ)
        self.btn_right    = ControlButton("►", "red",    size=SZ)
        self.btn_set_home = ControlButton("⌂", "yellow", size=SZ, font_size=26)

        dpad = QGridLayout()
        dpad.setSpacing(self.GAP_PX)
        dpad.setContentsMargins(0, 0, 0, 0)
        dpad.addWidget(QWidget(),         0, 0)
        dpad.addWidget(self.btn_fwd,      0, 1, alignment=Qt.AlignCenter)
        dpad.addWidget(QWidget(),         0, 2)
        dpad.addWidget(self.btn_left,     1, 0, alignment=Qt.AlignCenter)
        dpad.addWidget(self.btn_set_home, 1, 1, alignment=Qt.AlignCenter)
        dpad.addWidget(self.btn_right,    1, 2, alignment=Qt.AlignCenter)
        dpad.addWidget(QWidget(),         2, 0)
        dpad.addWidget(self.btn_bwd,      2, 1, alignment=Qt.AlignCenter)
        dpad.addWidget(QWidget(),         2, 2)
        left_panel = QWidget()
        left_panel.setLayout(dpad)
        top_body.addWidget(left_panel)

        div = QFrame(); div.setFrameShape(QFrame.VLine)
        div.setStyleSheet("color:#30363d; background:#30363d;")
        div.setFixedWidth(1)
        top_body.addWidget(div)

        self.speed_display = SpeedValueDisplay()
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setFixedSize(120, 44)
        self.btn_stop.setFocusPolicy(Qt.NoFocus)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background:#1a0700; color:#ff9800; border:2px solid #ff6f00;
                border-radius:8px; font-family:'Courier New'; font-size:16px;
                font-weight:bold;
            }
            QPushButton:hover   { background:#ff6f00; color:white; }
            QPushButton:pressed { background:white;   color:#1a0700; }
        """)

        right_stack = QVBoxLayout()
        right_stack.setSpacing(8)
        right_stack.setContentsMargins(4, 0, 0, 0)
        right_stack.addStretch()
        right_stack.addWidget(self.speed_display, alignment=Qt.AlignHCenter)
        right_stack.addWidget(self.btn_stop,       alignment=Qt.AlignHCenter)
        right_stack.addStretch()
        top_body.addLayout(right_stack)

        outer.addLayout(top_body)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#1e2d3d; background:#1e2d3d;")
        sep.setFixedHeight(1)
        outer.addWidget(sep)

        BH = 46   

        def _gb(text, bg, hover, color="white"):
            b = QPushButton(text)
            b.setFixedHeight(BH)
            b.setFocusPolicy(Qt.NoFocus)
            b.setStyleSheet(f"""
                QPushButton {{
                    background:{bg}; color:{color};
                    border:2px solid {hover};
                    border-radius:9px;
                    font-family:'Segoe UI';
                    font-size:12px; font-weight:700;
                }}
                QPushButton:hover   {{ background:{hover}; color:white; }}
                QPushButton:pressed {{ background:#000; }}
            """)
            return b

        self.btn_spd_up   = _gb("Speed  +",        "#004d40", "#26a69a")
        self.btn_spd_dn   = _gb("Speed  −",        "#004d40", "#26a69a")
        self.btn_r1       = _gb("R1  ＋5",          "#1a003a", "#7c3aed", "#d8b4fe")
        self.btn_l1       = _gb("L1  −5",           "#1a003a", "#7c3aed", "#d8b4fe")
        self.btn_waypoint = _gb("Waypoint",         "#1a1200", "#f9a825", "#f9a825")
        self.btn_clear_wp = _gb("Clear WPs",        "#1a1200", "#f9a825", "#f9a825")
        self.btn_save_wp  = _gb("💾  Save WP",      "#3b0000", "#c62828", "#ff8a80")
        self.btn_load_wp  = _gb("📂  Load WP",      "#003060", "#1565c0", "#82b1ff")
        self.btn_encoder_graph = _gb("📊  Encoder Graph", "#003d1a", "#00c853", "#39d353")

        grid = QGridLayout()
        grid.setSpacing(self.GAP_PX)
        grid.setContentsMargins(0, 0, 0, 0)

        grid.addWidget(self.btn_spd_up,      0, 0)
        grid.addWidget(self.btn_spd_dn,      0, 1)
        grid.addWidget(self.btn_r1,          1, 0)   
        grid.addWidget(self.btn_l1,          1, 1)   
        grid.addWidget(self.btn_waypoint,    2, 0)   
        grid.addWidget(self.btn_clear_wp,    2, 1)   
        grid.addWidget(self.btn_save_wp,     3, 0)   
        grid.addWidget(self.btn_load_wp,     3, 1)   
        grid.addWidget(self.btn_encoder_graph, 4, 0, 1, 2)   

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        outer.addLayout(grid)

        def leg_lbl(txt, clr):
            l = QLabel(txt)
            l.setStyleSheet(f"color:{clr}; font-family:'Courier New'; font-size:9px;")
            return l

        lg = QHBoxLayout(); lg.setSpacing(10); lg.setContentsMargins(2, 0, 2, 0)
        lg.addWidget(leg_lbl("🔴 MOVE  🟡 HOME", "#ff7070"))
        lg.addWidget(leg_lbl("🔵 SPD ±5", "#00bcd4"))
        lg.addStretch()
        outer.addLayout(lg)
        

# ─────────────────────────────────────────────────────────────────────────────
#  Encoder Graph Popup
# ─────────────────────────────────────────────────────────────────────────────
class EncoderGraphDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📊  Encoder Graph  —  /odom  |  Left (Green) & Right (Blue)")
        self.resize(900, 540)
        self.setStyleSheet("background:#0d1117;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        hdr = QHBoxLayout()
        ttl = QLabel("◈  WHEEL ENCODER  vs  DISTANCE")
        ttl.setStyleSheet(
            "color:#58a6ff; font-family:'Courier New'; font-size:12px; "
            "font-weight:bold; letter-spacing:2px;"
        )
        hdr.addWidget(ttl); hdr.addStretch()
        src_color = "#39d353" if HAS_ROS else "#f0a500"
        src_text  = "● ROS2 /odom LIVE" if HAS_ROS else "● SIMULATION MODE"
        src_lbl   = QLabel(src_text)
        src_lbl.setStyleSheet(f"color:{src_color}; font-family:'Courier New'; font-size:10px; font-weight:bold;")
        hdr.addWidget(src_lbl)
        layout.addLayout(hdr)

        leg = QHBoxLayout()
        for clr, txt in [("#00c853", "━━  LEFT MOTOR"), ("#2979ff", "━━  RIGHT MOTOR")]:
            l = QLabel(txt)
            l.setStyleSheet(f"color:{clr}; font-family:'Courier New'; font-size:10px; font-weight:bold; padding-right:20px;")
            leg.addWidget(l)
        leg.addStretch()
        layout.addLayout(leg)

        self.figure = Figure(figsize=(8, 4.5), dpi=100, facecolor="#0d1117")
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#161b22")
        for spine in self.ax.spines.values():
            spine.set_color("#30363d")
        self.ax.tick_params(colors="#8b949e", labelsize=8)
        self.ax.xaxis.label.set_color("#8b949e")
        self.ax.yaxis.label.set_color("#8b949e")
        self.ax.set_xlabel("Distance Moved (m)")
        self.ax.set_ylabel("Encoder Value (m)")
        self.ax.set_title("Real-time Wheel Encoder Stream", color="#58a6ff", fontsize=11)
        self.ax.grid(True, alpha=0.15, color="gray")

        self.dist_data  = deque(maxlen=500)
        self.left_data  = deque(maxlen=500)
        self.right_data = deque(maxlen=500)

        vrow = QHBoxLayout(); vrow.setSpacing(8)
        self._lbl_l = self._val_box("LEFT ENC",  "#00c853")
        self._lbl_r = self._val_box("RIGHT ENC", "#2979ff")
        self._lbl_d = self._val_box("DISTANCE",  "#ffd740")
        for w in [self._lbl_l, self._lbl_r, self._lbl_d]:
            vrow.addWidget(w)
        vrow.addStretch()
        layout.addLayout(vrow)

        self._plot_timer = QTimer(self)
        self._plot_timer.timeout.connect(self._refresh)
        self._plot_timer.start(120)

    def _val_box(self, name, color):
        w = QLabel(f"{name}\n—")
        w.setAlignment(Qt.AlignCenter)
        w.setStyleSheet(f"""
            color:{color}; font-family:'Courier New'; font-size:10px; font-weight:bold;
            background:#161b22; border:1px solid #30363d; border-radius:6px; padding:5px 14px;
        """)
        return w

    def push(self, left_enc, right_enc, distance):
        self.dist_data.append(distance)
        self.left_data.append(left_enc)
        self.right_data.append(right_enc)
        self._lbl_l.setText(f"LEFT ENC\n{left_enc:+.4f} m")
        self._lbl_r.setText(f"RIGHT ENC\n{right_enc:+.4f} m")
        self._lbl_d.setText(f"DISTANCE\n{distance:.4f} m")

    def _refresh(self):
        if len(self.dist_data) < 2: return
        d = list(self.dist_data); l = list(self.left_data); r = list(self.right_data)
        self.ax.clear()
        self.ax.set_facecolor("#161b22")
        self.ax.plot(d, l, color="#00c853", linewidth=2.2, label="Left Motor")
        self.ax.plot(d, r, color="#2979ff", linewidth=2.2, label="Right Motor")
        self.ax.set_xlabel("Distance Moved (m)", color="#8b949e")
        self.ax.set_ylabel("Encoder Value (m)",  color="#8b949e")
        self.ax.set_title("Real-time Wheel Encoder Stream", color="#58a6ff", fontsize=11)
        self.ax.tick_params(colors="#8b949e", labelsize=8)
        self.ax.grid(True, alpha=0.15, color="gray")
        self.ax.legend(facecolor="#161b22", labelcolor="white", fontsize=9, loc="upper left")
        self.canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
#  ROS2 Node
# ─────────────────────────────────────────────────────────────────────────────
class RealOdomSubscriber:
    def __init__(self, main_window):
        self.mw           = main_window
        self.left_enc     = 0.0
        self.right_enc    = 0.0
        self.total_dist   = 0.0
        self._last_x      = None
        self._last_y      = None
        self._node        = None
        self._thread      = None
        self.last_command = "Stop"
        if not HAS_ROS:
            return
        self._log = open("encoder_log.csv", "w", newline="")
        self._csv = csv.writer(self._log)
        self._csv.writerow(["timestamp", "command", "cdistance", "left_enc", "right_enc"])
        rclpy.init(args=sys.argv)
        self._node = _OdomNode(self)
        self._thread = threading.Thread(target=lambda: rclpy.spin(self._node), daemon=True)
        self._thread.start()

    def send_cmd(self, linear_x, angular_z, command=""):
        if command:
            self.last_command = command
        if self._node:
            self._node.send_cmd(linear_x, angular_z)

    def shutdown(self):
        if self._node:
            self._node.destroy_node()
            rclpy.shutdown()
        if hasattr(self, "_log"):
            self._log.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Control Buttons
# ─────────────────────────────────────────────────────────────────────────────
class ControlButton(QPushButton):
    _C = {
        "red":    ("#e53935", "#ff5252"),
        "blue":   ("#1565c0", "#42a5f5"),
        "pink":   ("#ad1457", "#f06292"),
        "yellow": ("#f9a825", "#fdd835"),
        "orange": ("#e65100", "#ff9800"),
        "green":  ("#1b5e20", "#43a047"),
        "teal":   ("#004d40", "#26a69a"),
        "purple": ("#4a148c", "#ab47bc"),
        "cyan":   ("#006064", "#00bcd4"),
    }

    def __init__(self, label, color="red", parent=None, size=52, font_size=None):
        super().__init__(label, parent)
        base, hover = self._C.get(color, self._C["red"])
        fs = font_size or max(11, size // 3)
        self.setFixedSize(size, size)
        self.setStyleSheet(f"""
            QPushButton {{
                background:{base}; border-radius:12px; color:white;
                font-size:{fs}px; font-weight:bold; border:2px solid {hover};
            }}
            QPushButton:hover {{ background:{hover}; }}
            QPushButton:pressed {{ background:white; color:{base}; }}
        """)
        self.setFocusPolicy(Qt.NoFocus)

class _OdomNode(Node):
    def __init__(self, subscriber):
        super().__init__("amr_ui_odom")
        self.sub_ref = subscriber
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Odometry, "/odom", self._cb, 10)
        # Fix: Subscribe to cmd_vel to read controller labels in real time
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, 10)

    def _cmd_vel_cb(self, msg):
        vx = msg.linear.x
        wz = msg.angular.z
        
        if vx == 0.0 and wz == 0.0:
            self.sub_ref.last_command = "Stop"
        elif vx > 0.0:
            self.sub_ref.last_command = "Forward"
        elif vx < 0.0:
            self.sub_ref.last_command = "Reverse"
        elif wz > 0.0:
            self.sub_ref.last_command = "Left"
        elif wz < 0.0:
            self.sub_ref.last_command = "Right"

    def _cb(self, msg):
        mw = self.sub_ref.mw
        cx = msg.pose.pose.position.x
        cy = msg.pose.pose.position.y

        if self.sub_ref._last_x is None:
            self.sub_ref._last_x = cx
            self.sub_ref._last_y = cy
            return                        

        dx = cx - self.sub_ref._last_x
        dy = cy - self.sub_ref._last_y
        dist = math.hypot(dx, dy)

        if dist < 0.5:                    
            self.sub_ref.total_dist += dist

        self.sub_ref._last_x = cx
        self.sub_ref._last_y = cy

        v  = msg.twist.twist.linear.x
        w  = msg.twist.twist.angular.z
        hw = WHEEL_BASE / 2.0

        now_ns = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
        if not hasattr(self, '_last_stamp_ns'):
            self._last_stamp_ns = now_ns
        dt = max(0.001, min(0.1, (now_ns - self._last_stamp_ns) / 1e9))
        self._last_stamp_ns = now_ns
        self.sub_ref.left_enc  += (v - w * hw) * dt
        self.sub_ref.right_enc += (v + w * hw) * dt

        q   = msg.pose.pose.orientation
        yaw = math.degrees(math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z)
        ))
        mw.bot_x = cx; mw.bot_y = cy; mw.bot_angle = yaw
        mw.current_speed = abs(v)
        mw.current_ang_speed = abs(math.degrees(w))
        mw.map_canvas.update_bot(cx, cy, yaw)

        if mw._enc_dlg.isVisible():
            mw._enc_dlg.push(
                self.sub_ref.left_enc,
                self.sub_ref.right_enc,
                self.sub_ref.total_dist
            )

        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.sub_ref._csv.writerow([
            ts,
            self.sub_ref.last_command,
            round(self.sub_ref.total_dist, 4),
            round(self.sub_ref.left_enc,   4),
            round(self.sub_ref.right_enc,  4),
        ])
        self.sub_ref._log.flush()
        
    def send_cmd(self, lx, az):
        msg = Twist(); msg.linear.x = float(lx); msg.angular.z = float(az)
        self.cmd_pub.publish(msg)

# ─────────────────────────────────────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────────────────────────────────────
class AMRMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AMR Bot – 2D Mapping & Control  |  PySide6")
        self.setMinimumSize(1160, 700)
        self.setStyleSheet("background:#010409;")

        # ── State ────────────────────────────────────────────────────────────
        self.bot_x             = 0.0
        self.bot_y             = 0.0
        self.bot_angle         = 0.0
        self.current_speed     = 0.0
        self.current_ang_speed = 0.0
        self.lin_speed_mod     = 100.0    
        self.trn_speed_mod     = 100.0
        self._moving           = {"fwd": False, "bwd": False, "left": False, "right": False}
        self._hard_stop        = False
        self._home_set         = False

        self._wp_mode_active      = False
        self._following_waypoints = False
        self._returning_home      = False
        self._last_sent_v         = 0.0
        self._last_sent_w         = 0.0

        self._wp_smooth_v  = 0.0
        self._wp_smooth_w  = 0.0
        self._WP_LIN_RAMP  = 0.015
        self._WP_ANG_RAMP  = 0.04

        self._sim_l = self._sim_r = self._sim_d = 0.0
        self._ros     = RealOdomSubscriber(self)
        self._enc_dlg = EncoderGraphDialog(self)

        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10); root.setSpacing(10)

        # ── Map panel ────────────────────────────────────────────────────────
        lf = QFrame()
        lf.setStyleSheet("QFrame{background:#0d1117;border:1px solid #30363d;border-radius:10px;}")
        lv = QVBoxLayout(lf); lv.setContentsMargins(8, 8, 8, 8); lv.setSpacing(4)
        mt = QLabel("◉  LIVE 2D MAP   —   AMR POSITION TRACKING")
        mt.setStyleSheet("color:#58a6ff;font-family:'Courier New';font-size:11px;font-weight:bold;letter-spacing:2px;padding:4px;")
        lv.addWidget(mt)
        self.map_canvas = MapCanvas()
        lv.addWidget(self.map_canvas)

        ztb = QHBoxLayout(); ztb.setSpacing(6); ztb.setContentsMargins(0, 2, 0, 2)

        def _zb(label, tip, color_pair, slot, w=36, h=28):
            bg, hv = color_pair
            b = QPushButton(label)
            b.setFixedSize(w, h); b.setToolTip(tip); b.setFocusPolicy(Qt.NoFocus)
            b.setStyleSheet(f"""
                QPushButton {{
                    background:{bg}; color:white; border:1px solid {hv};
                    border-radius:6px; font-family:'Courier New'; font-size:11px; font-weight:bold;
                }}
                QPushButton:hover   {{ background:{hv}; }}
                QPushButton:pressed {{ background:white; color:{bg}; }}
            """)
            b.clicked.connect(slot)
            return b

        lbl_map = QLabel("🔍  MAP VIEW")
        lbl_map.setStyleSheet("color:#58a6ff; font-family:'Courier New'; font-size:10px; font-weight:bold;")
        ztb.addWidget(lbl_map); ztb.addStretch()
        self._btn_zoom_in  = _zb("＋ ZOOM IN",  "Zoom in  [+]",   ("#0d4f1c","#43a047"), self.map_canvas.zoom_in,   w=84)
        self._btn_zoom_out = _zb("－ ZOOM OUT", "Zoom out  [−]",  ("#1a1a4f","#42a5f5"), self.map_canvas.zoom_out,  w=84)
        self._btn_zoom_rst = _zb("⊙ RESET",    "Reset view  [R]", ("#3d0000","#e53935"), self.map_canvas.reset_view, w=72)
        pan_hint = QLabel("  RMB/Middle-drag or scroll to pan·zoom")
        pan_hint.setStyleSheet("color:#8b949e; font-family:'Courier New'; font-size:9px;")
        for w in [self._btn_zoom_in, self._btn_zoom_out, self._btn_zoom_rst, pan_hint]:
            ztb.addWidget(w)
        lv.addLayout(ztb)
        root.addWidget(lf, stretch=3)

        # ── Right panel ───────────────────────────────────────────────────────
        rv = QVBoxLayout(); rv.setSpacing(10)
        self.telemetry   = TelemetryPanel()
        rv.addWidget(self.telemetry, stretch=5)
        self.control_pad = ControlPad()
        rv.addWidget(self.control_pad, stretch=4)
        root.addLayout(rv, stretch=2)

        # ── Wire buttons ─────────────────────────────────────────────────────
        cp = self.control_pad
        cp.btn_fwd.clicked.connect(lambda: self._timed_move_1m("fwd"))
        cp.btn_bwd.clicked.connect(lambda: self._timed_move_1m("bwd"))
        cp.btn_left.pressed.connect(lambda:   self._set_move("left",  True))
        cp.btn_left.released.connect(lambda:  self._set_move("left",  False))
        cp.btn_right.pressed.connect(lambda:  self._set_move("right", True))
        cp.btn_right.released.connect(lambda: self._set_move("right", False))

        cp.btn_set_home.clicked.connect(self._set_home)
        cp.btn_stop.clicked.connect(self._stop_and_fit_path)

        cp.btn_spd_up.clicked.connect(lambda: self._adj_lin(+SPEED_STEP))
        cp.btn_spd_dn.clicked.connect(lambda: self._adj_lin(-SPEED_STEP))
        
        cp.btn_r1.clicked.connect(lambda: (self._adj_lin(+5), self._adj_trn(+5)))
        cp.btn_l1.clicked.connect(lambda: (self._adj_lin(-5), self._adj_trn(-5)))

        cp.btn_encoder_graph.clicked.connect(self._show_enc_graph)
        cp.btn_waypoint.clicked.connect(self._toggle_waypoint_mode)
        cp.btn_clear_wp.clicked.connect(self._clear_waypoints)
        cp.btn_save_wp.clicked.connect(self._save_waypoints)   
        cp.btn_load_wp.clicked.connect(self._load_waypoints)   

        self.map_canvas.waypoint_clicked.connect(self._on_waypoint_added)
        self._show_home_prompt()

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(20)
        self.setFocusPolicy(Qt.StrongFocus); self.setFocus()

    # ─────────────────────────────────────────────────────────────────────────
    #  Home
    # ─────────────────────────────────────────────────────────────────────────
    def _show_home_prompt(self):
        self.control_pad.btn_set_home.setStyleSheet("""
            QPushButton {
                background:#f9a825; border-radius:8px; color:#000;
                font-size:20px; font-weight:bold; border:3px solid #fff176;
            }
            QPushButton:hover { background:#fdd835; }
        """)

    def _set_home(self):
        self._home_set = True
        self.map_canvas.set_home(self.bot_x, self.bot_y)
        base, hover = ControlButton._C["yellow"]
        self.control_pad.btn_set_home.setStyleSheet(f"""
            QPushButton {{ background:{base}; border-radius:8px; color:white;
                font-size:20px; font-weight:bold; border:2px solid {hover}; }}
            QPushButton:hover   {{ background:{hover}; }}
            QPushButton:pressed {{ background:white; color:{base}; }}
        """)
        self.control_pad.btn_set_home.setText("✓")
        QTimer.singleShot(800, lambda: self.control_pad.btn_set_home.setText("⌂"))

    # ─────────────────────────────────────────────────────────────────────────
    #  Stop → auto-fit path  
    # ─────────────────────────────────────────────────────────────────────────
    def _stop_and_fit_path(self):
        self._moving           = {k: False for k in self._moving}
        self._following_waypoints = False
        self._returning_home   = False
        self._wp_mode_active   = False
        self._wp_smooth_v      = 0.0
        self._wp_smooth_w      = 0.0
        self.map_canvas.set_waypoint_mode(False)
        self._ros.send_cmd(0.0, 0.0, "Stop")
        self._last_sent_v = 0.0
        self._last_sent_w = 0.0
        self.map_canvas.fit_trail_view()

        self.control_pad.btn_stop.setText("▶ Resume")
        self.control_pad.btn_stop.clicked.disconnect()
        self.control_pad.btn_stop.clicked.connect(self._stop_and_fit_path)

    def _resume_follow(self):
        self.map_canvas.set_mode_follow()
        self.control_pad.btn_stop.setText("Stop")
        self.control_pad.btn_stop.clicked.disconnect()
        self.control_pad.btn_stop.clicked.connect(self._stop_and_fit_path)

    # ─────────────────────────────────────────────────────────────────────────
    #  Waypoints
    # ─────────────────────────────────────────────────────────────────────────
    def _toggle_waypoint_mode(self):
        if not self._home_set:
            self.control_pad.btn_waypoint.setText("SET HOME FIRST!")
            QTimer.singleShot(1500, lambda: self.control_pad.btn_waypoint.setText("Waypoint"))
            return
        self._wp_mode_active = not self._wp_mode_active
        self.map_canvas.set_waypoint_mode(self._wp_mode_active)
        if self._wp_mode_active:
            self.control_pad.btn_waypoint.setText("CLICK MAP →")
            self._following_waypoints = True
        else:
            self.control_pad.btn_waypoint.setText("Waypoint")
            self._following_waypoints = len(self.map_canvas.waypoints) > 0

    def _on_waypoint_added(self, wx, wy):
        n = len(self.map_canvas.waypoints)
        self.control_pad.btn_waypoint.setText(f"WP: {n} set")
        self._following_waypoints = True

    def _clear_waypoints(self):
        self.map_canvas.clear_waypoints()
        self._following_waypoints = False
        self._wp_smooth_v = 0.0
        self._wp_smooth_w = 0.0
        self.control_pad.btn_waypoint.setText("Waypoint")

    # ─────────────────────────────────────────────────────────────────────────
    #  Save / Load Waypoints  
    # ─────────────────────────────────────────────────────────────────────────
    def _save_waypoints(self):
        wps = self.map_canvas.waypoints
        if not wps:
            self.control_pad.btn_save_wp.setText("No WPs!")
            QTimer.singleShot(1200, lambda: self.control_pad.btn_save_wp.setText("💾  Save Waypoint"))
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"waypoints_{timestamp}.json"
        script_dir   = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(script_dir, default_name)
        path, _ = QFileDialog.getSaveFileName(self, "Save Waypoints", default_path, "JSON Files (*.json);;All Files (*)")
        if not path:
            return
        data = {
            "saved_at": datetime.datetime.now().isoformat(),
            "home":     {"x": self.map_canvas.home_x, "y": self.map_canvas.home_y},
            "waypoints": [{"x": wx, "y": wy} for wx, wy in wps],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        short = os.path.basename(path)
        self.control_pad.btn_save_wp.setText(f"✓ {short[:16]}")
        QTimer.singleShot(2000, lambda: self.control_pad.btn_save_wp.setText("💾  Save Waypoint"))

    def _load_waypoints(self):
        if not self._home_set:
            self.control_pad.btn_load_wp.setText("SET HOME FIRST!")
            QTimer.singleShot(1500, lambda: self.control_pad.btn_load_wp.setText("📂  Load Waypoint"))
            return
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path, _ = QFileDialog.getOpenFileName(self, "Load Waypoints", script_dir, "JSON Files (*.json);;All Files (*)")
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            pts = [(p["x"], p["y"]) for p in data.get("waypoints", [])]
            self.map_canvas.waypoints = pts
            self.map_canvas.current_waypoint_idx = 0
            self._following_waypoints = len(pts) > 0
            self.map_canvas.update()
            self.control_pad.btn_waypoint.setText(f"WP: {len(pts)} loaded")
            QTimer.singleShot(2000, lambda: self.control_pad.btn_waypoint.setText("Waypoint"))
        except Exception as e:
            self.control_pad.btn_load_wp.setText("Load Error!")
            QTimer.singleShot(2000, lambda: self.control_pad.btn_load_wp.setText("📂  Load Waypoint"))

    # ─────────────────────────────────────────────────────────────────────────
    #  Misc
    # ─────────────────────────────────────────────────────────────────────────
    def _show_enc_graph(self):
        if self._enc_dlg.isVisible():
            self._enc_dlg.raise_(); self._enc_dlg.activateWindow()
        else:
            self._enc_dlg.show()

    def _adj_lin(self, d: int):
        self.lin_speed_mod = max(10.0, min(200.0, self.lin_speed_mod + d))
        self.control_pad.speed_display.set_value(self.lin_speed_mod)

    def _adj_trn(self, d: int):
        self.trn_speed_mod = max(10.0, min(200.0, self.trn_speed_mod + d))

    def _set_move(self, direction, state):
        if not self._hard_stop and self._home_set:
            self._moving[direction] = state
        elif not self._home_set and state:
            self._show_home_prompt()

    def _timed_move_1m(self, direction: str):
        if self._hard_stop or not self._home_set:
            return
        if hasattr(self, '_timed_move_timer') and self._timed_move_timer.isActive():
            self._timed_move_timer.stop()
            self._moving = {k: False for k in self._moving}

        speed = BASE_LINEAR_SPEED * self.lin_speed_mod / 100.0
        duration_ms = int((1.0 / speed) * 1000)   

        self._moving[direction] = True

        self._timed_move_timer = QTimer()
        self._timed_move_timer.setSingleShot(True)
        self._timed_move_timer.timeout.connect(lambda: self._end_timed_move(direction))
        self._timed_move_timer.start(duration_ms)

    def _end_timed_move(self, direction: str):
        self._moving[direction] = False
        self._ros.send_cmd(0.0, 0.0, "Stop")
        self._last_sent_v = 0.0
        self._last_sent_w = 0.0

    def _navigate_to(self, target_x, target_y, dt):
        dx   = target_x - self.bot_x
        dy   = target_y - self.bot_y
        dist = math.hypot(dx, dy)
        if dist < WAYPOINT_REACH_DIST:
            return 0.0, 0.0, True
        target_angle = math.degrees(math.atan2(dy, dx))
        angle_diff   = target_angle - self.bot_angle
        while angle_diff >  180: angle_diff -= 360
        while angle_diff < -180: angle_diff += 360
        lin_max = BASE_LINEAR_SPEED  * self.lin_speed_mod / 100.0
        ang_max = BASE_ANGULAR_SPEED * self.trn_speed_mod / 100.0
        KP_ANG  = ang_max / 90.0
        w = max(-ang_max, min(ang_max, KP_ANG * angle_diff))
        abs_err = abs(angle_diff)
        if abs_err > 90.0:
            v = 0.0
        elif abs_err > 30.0:
            v = lin_max * 0.20
        else:
            forward_frac = math.cos(math.radians(abs_err))
            slowdown     = min(1.0, dist / 0.5)
            v = lin_max * forward_frac * slowdown
        return v, w, False

    def _smooth_wp_vel(self, current, target, ramp):
        diff = target - current
        if abs(diff) <= ramp:
            return target
        return current + math.copysign(ramp, diff)

    # ─────────────────────────────────────────────────────────────────────────
    #  20 Hz tick
    # ─────────────────────────────────────────────────────────────────────────
    def _tick(self):
        dt = 0.02
        v = w = 0.0
        waypoint_info = "—"
        _cmd_label    = "Stop"

        if not self._hard_stop and self._home_set:
            mc = self.map_canvas

            if self._returning_home:
                v_nav, w_nav, reached = self._navigate_to(mc.home_x, mc.home_y, dt)
                waypoint_info = "RTH"
                self._wp_smooth_v = self._smooth_wp_vel(self._wp_smooth_v, v_nav, self._WP_LIN_RAMP)
                self._wp_smooth_w = self._smooth_wp_vel(self._wp_smooth_w, w_nav, self._WP_ANG_RAMP)
                v, w = self._wp_smooth_v, self._wp_smooth_w
                _cmd_label = "RTH"
                if reached:
                    self._returning_home  = False
                    self._wp_smooth_v     = 0.0
                    self._wp_smooth_w     = 0.0
                    self._ros.send_cmd(0.0, 0.0, "Stop")
                    self._last_sent_v = 0.0; self._last_sent_w = 0.0
                    self.map_canvas.fit_trail_view()

            elif self._following_waypoints and mc.waypoints:
                idx = mc.current_waypoint_idx
                if idx < len(mc.waypoints):
                    tx, ty = mc.waypoints[idx]
                    v_nav, w_nav, reached = self._navigate_to(tx, ty, dt)
                    waypoint_info = f"{idx+1}/{len(mc.waypoints)}"
                    _cmd_label    = f"WP{idx+1}"
                    self._wp_smooth_v = self._smooth_wp_vel(self._wp_smooth_v, v_nav, self._WP_LIN_RAMP)
                    self._wp_smooth_w = self._smooth_wp_vel(self._wp_smooth_w, w_nav, self._WP_ANG_RAMP)
                    v, w = self._wp_smooth_v, self._wp_smooth_w
                    if reached:
                        mc.current_waypoint_idx += 1
                        if mc.current_waypoint_idx >= len(mc.waypoints):
                            self._following_waypoints = False
                            mc.current_waypoint_idx   = len(mc.waypoints)
                            waypoint_info = "DONE"
                else:
                    self._wp_smooth_v = self._smooth_wp_vel(self._wp_smooth_v, 0.0, self._WP_LIN_RAMP)
                    self._wp_smooth_w = self._smooth_wp_vel(self._wp_smooth_w, 0.0, self._WP_ANG_RAMP)
                    v, w = self._wp_smooth_v, self._wp_smooth_w
                    waypoint_info = "DONE"; _cmd_label = "Stop"

            else:
                self._wp_smooth_v = 0.0; self._wp_smooth_w = 0.0
                lin = BASE_LINEAR_SPEED  * self.lin_speed_mod / 100.
                ang = BASE_ANGULAR_SPEED * self.trn_speed_mod / 100.
                
                if self._moving.get("fwd"):   v += lin
                if self._moving.get("bwd"):   v -= lin
                
                if self._moving.get("left"):  w -= ang
                if self._moving.get("right"): w += ang
                
                parts = []
                if self._moving.get("fwd"):   parts.append("Forward")
                if self._moving.get("bwd"):   parts.append("Reverse")
                if self._moving.get("left"):  parts.append("Left")
                if self._moving.get("right"): parts.append("Right")
                _cmd_label = "+".join(parts) if parts else "Stop"

        final_v =  v
        final_w =  w

        if (v != 0.0 or w != 0.0) and not self._hard_stop:
            self._ros.send_cmd(final_v, final_w, _cmd_label)
            self._last_sent_v = final_v; self._last_sent_w = final_w
        else:
            if not self._hard_stop:
                if self._last_sent_v != 0.0 or self._last_sent_w != 0.0:
                    self._ros.send_cmd(0.0, 0.0, "Stop")
                    self._last_sent_v = 0.0; self._last_sent_w = 0.0

        if not HAS_ROS and not self._hard_stop and self._home_set:
            self.bot_angle += math.degrees(-w) * dt
            rad = math.radians(self.bot_angle)
            self.bot_x += (-v) * math.cos(rad) * dt
            self.bot_y += (-v) * math.sin(rad) * dt
            self.current_speed     = abs(v)
            self.current_ang_speed = abs(math.degrees(w))
            if self.map_canvas.mode == MapCanvas.MODE_FOLLOW:
                self.map_canvas.update_bot(self.bot_x, self.bot_y, self.bot_angle)
            if v != 0.0 or w != 0.0:
                hw = WHEEL_BASE / 2.0
                self._sim_l += (v - w * hw) * dt
                self._sim_r += (v + w * hw) * dt
                self._sim_d += abs(v) * dt
                if self._enc_dlg.isVisible():
                    self._enc_dlg.push(self._sim_l, self._sim_r, self._sim_d)

        self.control_pad.speed_display.set_value(self.lin_speed_mod)

        home_x = self.map_canvas.home_x
        home_y = self.map_canvas.home_y
        dx = self.bot_x - home_x
        dy = self.bot_y - home_y
        self.telemetry.update_telemetry(
            x             = dx,
            y             = dy,
            angle         = math.degrees(math.atan2(dy, dx)) % 360,
            speed         = self.current_speed,
            ang_speed     = self.current_ang_speed,
            lin_mod       = self.lin_speed_mod,
            trn_mod       = self.trn_speed_mod,
            waypoint_info = waypoint_info,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Keyboard shortcuts
    # ─────────────────────────────────────────────────────────────────────────
    def keyPressEvent(self, e: QKeyEvent):
        k = e.key()
        if k in (Qt.Key_Up,    Qt.Key_W): self._set_move("fwd",   True)
        if k in (Qt.Key_Down,  Qt.Key_S): self._set_move("bwd",   True)
        if k in (Qt.Key_Left,  Qt.Key_A): self._set_move("left",  True)
        if k in (Qt.Key_Right, Qt.Key_D): self._set_move("right", True)
        if k == Qt.Key_Space: self._stop_and_fit_path()
        if k == Qt.Key_H:     self._set_home()
        if k == Qt.Key_G:     self._show_enc_graph()
        if k == Qt.Key_E:     self._adj_lin(+SPEED_STEP)
        if k == Qt.Key_Q:     self._adj_lin(-SPEED_STEP)
        if k == Qt.Key_X:     self._adj_trn(+SPEED_STEP)
        if k == Qt.Key_Z:     self._adj_trn(-SPEED_STEP)
        if k == Qt.Key_P:     self._toggle_waypoint_mode()
        if k == Qt.Key_C:     self._clear_waypoints()
        if k in (Qt.Key_Plus, Qt.Key_Equal):       self.map_canvas.zoom_in()
        if k in (Qt.Key_Minus, Qt.Key_Underscore): self.map_canvas.zoom_out()
        if k == Qt.Key_R:     self.map_canvas.reset_view()

    def keyReleaseEvent(self, e: QKeyEvent):
        k = e.key()
        if k in (Qt.Key_Up,    Qt.Key_W): self._set_move("fwd",   False)
        if k in (Qt.Key_Down,  Qt.Key_S): self._set_move("bwd",   False)
        if k in (Qt.Key_Left,  Qt.Key_A): self._set_move("left",  False)
        if k in (Qt.Key_Right, Qt.Key_D): self._set_move("right", False)

    def closeEvent(self, e):
        self._ros.shutdown()
        self._enc_dlg.close()
        super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor("#2d6ece"))
    pal.setColor(QPalette.WindowText,      QColor("#c9d1d9"))
    pal.setColor(QPalette.Base,            QColor("#161b22"))
    pal.setColor(QPalette.AlternateBase,   QColor("#21262d"))
    pal.setColor(QPalette.ToolTipBase,     QColor("#161b22"))
    pal.setColor(QPalette.ToolTipText,     QColor("#c9d1d9"))
    pal.setColor(QPalette.Text,            QColor("#c9d1d9"))
    pal.setColor(QPalette.Button,          QColor("#21262d"))
    pal.setColor(QPalette.ButtonText,      QColor("#c9d1d9"))
    pal.setColor(QPalette.Highlight,       QColor("#1f6feb"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)
    win = AMRMainWindow()
    win.show()
    sys.exit(app.exec())