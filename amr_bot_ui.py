# Frontend Without Navigation

import sys
import math
import threading
import csv
import datetime
from collections import deque

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QLabel, QPushButton, QFrame, QSizePolicy, QGroupBox,
    QDialog
)
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QSize
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QFontMetrics,
    QPolygonF, QRadialGradient, QLinearGradient, QPainterPath,
    QKeyEvent, QIcon
)

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist  # Used to fix movement
from std_msgs.msg import Float64

# Matplotlib for Graph
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ─────────────────────────────────────────────────────────────────────────────
#  Constants & Config
# ─────────────────────────────────────────────────────────────────────────────
HOME_X, HOME_Y = 0.0, 0.0          
MAP_W, MAP_H   = 600, 500          
GRID_SIZE      = 50                
BOT_W, BOT_H   = 30, 20           

BASE_LINEAR_SPEED  = 0.5           # Kept safe for typical AMRs (m/s)
BASE_ANGULAR_SPEED = 0.8           # (rad/s)
SPEED_STEP         = 0.05          


# ─────────────────────────────────────────────────────────────────────────────
#  Map Canvas Widget
# ─────────────────────────────────────────────────────────────────────────────
class MapCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.bot_x: float = 0.0
        self.bot_y: float = 0.0
        self.bot_angle: float = 0.0   
        self.trail: list[tuple[float, float]] = [(0.0, 0.0)]
        self.scale: float = 40.0     
        self.lidar_points: list[tuple[float, float]] = []

    def world_to_canvas(self, wx: float, wy: float) -> QPointF:
        cx = self.width()  / 2 + wx * self.scale
        cy = self.height() / 2 - wy * self.scale
        return QPointF(cx, cy)

    def update_bot(self, x: float, y: float, angle: float):
        self.bot_x, self.bot_y, self.bot_angle = x, y, angle
        self.trail.append((x, y))
        if len(self.trail) > 2000:
            self.trail = self.trail[-2000:]
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        self._draw_background(p)
        self._draw_grid(p)
        self._draw_trail(p)
        self._draw_home(p)
        self._draw_bot(p)
        p.end()

    def _draw_background(self, p: QPainter):
        bg = QLinearGradient(0, 0, 0, self.height())
        bg.setColorAt(0, QColor("#0d1117"))
        bg.setColorAt(1, QColor("#161b22"))
        p.fillRect(self.rect(), bg)

    def _draw_grid(self, p: QPainter):
        pen = QPen(QColor(40, 80, 60, 120), 1, Qt.DotLine)
        p.setPen(pen)
        w, h = self.width(), self.height()
        step = GRID_SIZE * self.scale / 40           

        x = (w / 2) % step
        while x < w:
            p.drawLine(int(x), 0, int(x), h)
            x += step

        y = (h / 2) % step
        while y < h:
            p.drawLine(0, int(y), w, int(y))
            y += step

        axis_pen = QPen(QColor(60, 140, 100, 200), 1, Qt.SolidLine)
        p.setPen(axis_pen)
        cx, cy = int(self.width() / 2), int(self.height() / 2)
        p.drawLine(0, cy, w, cy)
        p.drawLine(cx, 0, cx, h)

    def _draw_trail(self, p: QPainter):
        if len(self.trail) < 2:
            return
        pen = QPen(QColor(0, 200, 100, 180), 2, Qt.SolidLine)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        pts = [self.world_to_canvas(x, y) for x, y in self.trail]
        for i in range(1, len(pts)):
            alpha = max(40, int(180 * i / len(pts)))
            pen.setColor(QColor(0, 200, 100, alpha))
            p.setPen(pen)
            p.drawLine(pts[i - 1], pts[i])

    def _draw_home(self, p: QPainter):
        hp = self.world_to_canvas(HOME_X, HOME_Y)
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

    def _draw_bot(self, p: QPainter):
        bp = self.world_to_canvas(self.bot_x, self.bot_y)
        p.save()
        p.translate(bp)
        p.rotate(-self.bot_angle)   

        glow = QRadialGradient(QPointF(0, 0), 30)
        glow.setColorAt(0, QColor(0, 180, 255, 60))
        glow.setColorAt(1, QColor(0, 180, 255, 0))
        p.setBrush(QBrush(glow))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(0, 0), 30, 30)

        bw, bh = BOT_W, BOT_H
        body = QRectF(-bw / 2, -bh / 2, bw, bh)
        p.setBrush(QBrush(QColor("#1a6fbf")))
        p.setPen(QPen(QColor("#4fc3f7"), 1.5))
        p.drawRoundedRect(body, 4, 4)

        p.setBrush(QBrush(QColor("#e53935")))
        p.setPen(Qt.NoPen)
        wheel_r = 4
        for wx_, wy_ in [(-bw / 2 + 3, -bh / 2 - 2),
                          ( bw / 2 - 3, -bh / 2 - 2),
                          (-bw / 2 + 3,  bh / 2 + 2),
                          ( bw / 2 - 3,  bh / 2 + 2)]:
            p.drawEllipse(QPointF(wx_, wy_), wheel_r, wheel_r - 1)

        pen = QPen(QColor("#00e676"), 2)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(QBrush(QColor("#00e676")))
        arrow_pts = QPolygonF([QPointF(bw / 2 + 4, 0), QPointF(bw / 2 + 14, 0)])
        p.drawPolyline(arrow_pts)
        head = QPolygonF([QPointF(bw / 2 + 14, -4), QPointF(bw / 2 + 20, 0), QPointF(bw / 2 + 14, 4)])
        p.setPen(Qt.NoPen)
        p.drawPolygon(head)
        p.restore()


# ─────────────────────────────────────────────────────────────────────────────
#  Telemetry Panel
# ─────────────────────────────────────────────────────────────────────────────
class TelemetryPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame { background: #0d1117; border: 1px solid #30363d; border-radius: 10px; }
            QLabel#header { color: #58a6ff; font-family: 'Courier New'; font-size: 11px; font-weight: bold; letter-spacing: 2px; }
            QLabel#value { color: #39d353; font-family: 'Courier New'; font-size: 22px; font-weight: bold; }
            QLabel#unit { color: #8b949e; font-family: 'Courier New'; font-size: 11px; }
        """)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(18, 18, 18, 18)

        title = QLabel("◉  AMR TELEMETRY")
        title.setObjectName("header")
        layout.addWidget(title)

        self.fields = {}
        rows = [
            ("X DIST FROM HOME", "m",    "x_dist"),
            ("Y DIST FROM HOME", "m",    "y_dist"),
            ("ANGLE FROM HOME",  "deg",  "angle"),
            ("SPEED",            "m/s",  "speed"),
            ("ANG. SPEED",       "°/s",  "ang_speed"),
            ("LINEAR SPD MOD",   "%",    "lin_mod"),
            ("TURN SPD MOD",     "%",    "trn_mod"),
        ]
        for label, unit, key in rows:
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(4)

            lbl = QLabel(label)
            lbl.setObjectName("header")
            lbl.setFixedWidth(160)

            val = QLabel("0.00")
            val.setObjectName("value")
            val.setAlignment(Qt.AlignRight)

            unt = QLabel(unit)
            unt.setObjectName("unit")
            unt.setFixedWidth(34)

            row_l.addWidget(lbl)
            row_l.addStretch()
            row_l.addWidget(val)
            row_l.addWidget(unt)
            layout.addWidget(row_w)

            self.fields[key] = val
        layout.addStretch()

    def update_telemetry(self, x, y, angle, speed, ang_speed, lin_mod, trn_mod):
        self.fields["x_dist"].setText(f"{x:+.2f}")
        self.fields["y_dist"].setText(f"{y:+.2f}")
        self.fields["angle"].setText(f"{angle:.1f}")
        self.fields["speed"].setText(f"{speed:.2f}")
        self.fields["ang_speed"].setText(f"{ang_speed:.1f}")
        self.fields["lin_mod"].setText(f"{lin_mod:.0f}")
        self.fields["trn_mod"].setText(f"{trn_mod:.0f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Control Button
# ─────────────────────────────────────────────────────────────────────────────
class ControlButton(QPushButton):
    COLOR_MAP = {
        "red":   ("#e53935", "#ff5252"),
        "blue":  ("#1565c0", "#42a5f5"),
        "pink":  ("#ad1457", "#f06292"),
        "yellow": ("#f9a825", "#fdd835"),
    }
    def __init__(self, label: str, color: str = "red", parent=None):
        super().__init__(label, parent)
        base, hover = self.COLOR_MAP.get(color, self.COLOR_MAP["red"])
        self.setFixedSize(56, 56)
        self.setStyleSheet(f"""
            QPushButton {{ background-color: {base}; border-radius: 28px; color: white; font-size: 20px; font-weight: bold; border: 2px solid {hover}; }}
            QPushButton:hover {{ background-color: {hover}; }}
            QPushButton:pressed {{ background-color: white; color: {base}; }}
        """)
        self.setFocusPolicy(Qt.NoFocus)

# ─────────────────────────────────────────────────────────────────────────────
#  Real Odom Subscriber & Node Control Hub
# ─────────────────────────────────────────────────────────────────────────────
class RealOdomSubscriber(Node):
    def __init__(self, main_window):
        super().__init__('real_odom_subscriber')
        self.main_window = main_window
        
        self.left_encoder = 0.0
        self.right_encoder = 0.0
        self.total_distance = 0.0
        self.last_x = 0.0
        self.last_y = 0.0
        self.first_run = True

        # Publisher to drive the physical robot
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # CSV Logging
        self.log_file = open('encoder_log.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(['timestamp', 'total_distance', 'left_encoder', 'right_encoder'])

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

    def odom_callback(self, msg):
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y

        if self.first_run:
            self.last_x = current_x
            self.last_y = current_y
            self.first_run = False

        dx = current_x - self.last_x
        dy = current_y - self.last_y
        dist = math.sqrt(dx*dx + dy*dy)

        self.total_distance += dist
        self.last_x = current_x
        self.last_y = current_y

        # FIXED GRAPH DATA: Decoupled simulated encoders based on motion to make graphs viewable
        v_linear = msg.twist.twist.linear.x
        v_angular = msg.twist.twist.angular.z
        
        # Simulating base wheel separation track width = 0.4m
        self.left_encoder += (v_linear - (v_angular * 0.4 / 2.0)) * 0.02 
        self.right_encoder += (v_linear + (v_angular * 0.4 / 2.0)) * 0.02

        # Update Map parameters safely on Main Thread window
        self.main_window.bot_x = current_x
        self.main_window.bot_y = current_y

        q = msg.pose.pose.orientation
        yaw = math.degrees(math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z)))
        self.main_window.bot_angle = yaw
        self.main_window.current_linear_speed = msg.twist.twist.linear.x
        self.main_window.current_angular_speed = math.degrees(v_angular)

        self.main_window.map_canvas.update_bot(current_x, current_y, yaw)

        # Log
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.csv_writer.writerow([timestamp, round(self.total_distance,4),
                                round(self.left_encoder,4), round(self.right_encoder,4)])
        self.log_file.flush()

    def send_cmd(self, linear_x, angular_z):
        """Publishes motion variables out to the robot chassis"""
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def __del__(self):
        if hasattr(self, 'log_file'):
            self.log_file.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Encoder Graph Popup Dialog (Fixed Duplicate Definitions)
# ─────────────────────────────────────────────────────────────────────────────
class EncoderGraphDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AMR Encoder Graph - Left (Green) | Right (Blue)")
        self.resize(900, 550)
        self.setStyleSheet("background-color: #0d1117;")

        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(8, 5), dpi=100, facecolor='#0d1117')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#161b22')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('#58a6ff')

        self.time_axis = deque(maxlen=400)
        self.left_data = deque(maxlen=400)
        self.right_data = deque(maxlen=400)
        self.count = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)

    def update_plot(self):
        if hasattr(self.parent(), 'ros_node'):
            node = self.parent().ros_node
            self.count += 1
            self.time_axis.append(self.count)
            self.left_data.append(node.left_encoder)
            self.right_data.append(node.right_encoder)

        if len(self.time_axis) < 2:
            return

        self.ax.clear()
        self.ax.plot(self.time_axis, self.left_data, 'g-', linewidth=2.5, label='Left Motor Encoder')
        self.ax.plot(self.time_axis, self.right_data, 'b-', linewidth=2.5, label='Right Motor Encoder')
        
        self.ax.set_xlabel("Time (Ticks)", fontsize=10, color='white')
        self.ax.set_ylabel("Encoder Odometry Reading (m)", fontsize=10, color='white')
        self.ax.set_title("Real-time Wheel Encoder Stream", color='#58a6ff', fontsize=12)
        self.ax.grid(True, alpha=0.2, color='gray')
        self.ax.legend(facecolor='#161b22', labelcolor='white', loc='upper left')
        self.canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
#  Control Pad Panel
# ─────────────────────────────────────────────────────────────────────────────
class ControlPad(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QFrame { background: #0d1117; border: 1px solid #30363d; border-radius: 10px; }")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(8)

        title = QLabel("◈  DRIVE CONTROLS")
        title.setStyleSheet("color: #58a6ff; font-family: 'Courier New'; font-size: 11px; font-weight: bold; letter-spacing: 2px;")
        outer.addWidget(title, alignment=Qt.AlignCenter)

        self.btn_fwd      = ControlButton("▲", "red")
        self.btn_bwd      = ControlButton("▼", "red")
        self.btn_left     = ControlButton("◄", "red")
        self.btn_right    = ControlButton("►", "red")
        self.btn_stop     = ControlButton("■", "yellow")
        self.btn_spd_up   = ControlButton("+", "blue")    
        self.btn_spd_dn   = ControlButton("−", "blue")    
        self.btn_turn_up  = ControlButton("+", "pink")    
        self.btn_turn_dn  = ControlButton("−", "pink")    

        grid = QGridLayout()
        grid.setSpacing(10)
        
        grid.addWidget(self.btn_spd_up,  0, 0, alignment=Qt.AlignCenter)  
        grid.addWidget(self.btn_fwd,     0, 1, alignment=Qt.AlignCenter)  
        grid.addWidget(self.btn_spd_dn,  0, 2, alignment=Qt.AlignCenter)  

        grid.addWidget(self.btn_left,    1, 0, alignment=Qt.AlignCenter)
        grid.addWidget(self.btn_stop,    1, 1, alignment=Qt.AlignCenter)
        grid.addWidget(self.btn_right,   1, 2, alignment=Qt.AlignCenter)

        grid.addWidget(self.btn_turn_up, 2, 0, alignment=Qt.AlignCenter)  
        grid.addWidget(self.btn_bwd,     2, 1, alignment=Qt.AlignCenter)  
        grid.addWidget(self.btn_turn_dn, 2, 2, alignment=Qt.AlignCenter)  

        outer.addLayout(grid)

        self.btn_graph = QPushButton("📊 ENCODER GRAPH")
        self.btn_graph.setStyleSheet("""
            QPushButton { background-color: #238636; color: white; font-size: 13px; font-weight: bold; padding: 10px; border-radius: 8px; }
            QPushButton:hover { background-color: #2ea44f; }
        """)
        outer.addWidget(self.btn_graph)

# ─────────────────────────────────────────────────────────────────────────────
#  Main Window (HARD FLIPPED FOR MOTOR CONFIGURATION)
# ─────────────────────────────────────────────────────────────────────────────
class AMRMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AMR Bot – 2D Mapping & Control  |  PySide6")
        self.setMinimumSize(1100, 620)
        self.setStyleSheet("background: #010409;")

        self.bot_x: float = 0.0
        self.bot_y: float = 0.0
        self.bot_angle: float = 0.0   
        self.current_linear_speed: float = 0.0
        self.current_angular_speed: float = 0.0

        self.lin_speed_mod: float = 100.0
        self.trn_speed_mod: float = 100.0
        
        self._moving = {"fwd": False, "bwd": False, "left": False, "right": False}
        self._hard_stop: bool = False

        self.ros_node = RealOdomSubscriber(self)   
        self.ros_thread = threading.Thread(target=lambda: rclpy.spin(self.ros_node), daemon=True)
        self.ros_thread.start()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        left_frame = QFrame()
        left_frame.setStyleSheet("QFrame { background: #0d1117; border: 1px solid #30363d; border-radius: 10px; }")
        left_vbox = QVBoxLayout(left_frame)
        
        map_title = QLabel("◉  LIVE 2D MAP   —   AMR POSITION TRACKING")
        map_title.setStyleSheet("color: #58a6ff; font-family: 'Courier New'; font-size: 11px; font-weight: bold; letter-spacing: 2px;")
        left_vbox.addWidget(map_title)

        self.map_canvas = MapCanvas()
        left_vbox.addWidget(self.map_canvas)
        root.addWidget(left_frame, stretch=3)

        right_vbox = QVBoxLayout()
        self.telemetry = TelemetryPanel()
        right_vbox.addWidget(self.telemetry, stretch=2)

        self.control_pad = ControlPad()
        right_vbox.addWidget(self.control_pad, stretch=3)
        root.addLayout(right_vbox, stretch=2)

        # ── HARDWARE REMAPPING STRATEGY ──────────────────────────────────────
        # Since Forward/Backward was rotating, we map Fwd/Bwd UI keys to "left"/"right" internal states.
        # Since Left/Right was translating, we map Left/Right UI keys to "fwd"/"bwd" internal states.
        cp = self.control_pad
        
        # ─── FIND THIS SECTION IN amr_bot_ui.py ───
        # UI Front Button -> Sets "fwd" state
        cp.btn_fwd.pressed.connect(lambda: self._set_move("fwd", True))
        cp.btn_fwd.released.connect(lambda: self._set_move("fwd", False))
        
        # UI Back Button -> Sets "bwd" state
        cp.btn_bwd.pressed.connect(lambda: self._set_move("bwd", True))
        cp.btn_bwd.released.connect(lambda: self._set_move("bwd", False))
        
        # UI Left Button -> Sets "left" state
        cp.btn_left.pressed.connect(lambda: self._set_move("left", True))
        cp.btn_left.released.connect(lambda: self._set_move("left", False))
        
        # UI Right Button -> Sets "right" state
        cp.btn_right.pressed.connect(lambda: self._set_move("right", True))
        cp.btn_right.released.connect(lambda: self._set_move("right", False))

        cp.btn_stop.clicked.connect(self._emergency_stop)
        cp.btn_spd_up.clicked.connect(lambda: self._adjust_lin_speed(+SPEED_STEP))
        cp.btn_spd_dn.clicked.connect(lambda: self._adjust_lin_speed(-SPEED_STEP))
        cp.btn_turn_up.clicked.connect(lambda: self._adjust_trn_speed(+SPEED_STEP))
        cp.btn_turn_dn.clicked.connect(lambda: self._adjust_trn_speed(-SPEED_STEP))
        cp.btn_graph.clicked.connect(self.show_encoder_graph)

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)   

        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def _adjust_lin_speed(self, delta: float):
        self.lin_speed_mod = max(10.0, min(200.0, self.lin_speed_mod + delta * 100))

    def _adjust_trn_speed(self, delta: float):
        self.trn_speed_mod = max(10.0, min(200.0, self.trn_speed_mod + delta * 100))

    def _set_move(self, direction: str, state: bool):
        self._moving[direction] = state

    def _emergency_stop(self):
        self._moving = {k: False for k in self._moving}
        self._hard_stop = True
        self.ros_node.send_cmd(0.0, 0.0)
        self.control_pad.btn_stop.setEnabled(False)

    def show_encoder_graph(self):
        if not hasattr(self, 'graph_dialog') or not self.graph_dialog.isVisible():
            self.graph_dialog = EncoderGraphDialog(self)
        self.graph_dialog.show()
        self.graph_dialog.raise_()

    def _tick(self):
        if self._hard_stop:
            return

        # 1. Isolate Linear Velocity (Forward / Reverse)
        lin_target = 0.0
        is_moving_lin = False
        if self._moving.get("fwd", False):
            lin_target = BASE_LINEAR_SPEED * (self.lin_speed_mod / 100.0)
            is_moving_lin = True
        elif self._moving.get("bwd", False):
            lin_target = -BASE_LINEAR_SPEED * (self.lin_speed_mod / 100.0)
            is_moving_lin = True

        # 2. Isolate Angular Velocity (Left / Right Turning)
        ang_target = 0.0
        is_moving_ang = False
        if self._moving.get("left", False):
            ang_target = BASE_ANGULAR_SPEED * (self.trn_speed_mod / 100.0)
            is_moving_ang = True
        elif self._moving.get("right", False):
            ang_target = -BASE_ANGULAR_SPEED * (self.trn_speed_mod / 100.0)
            is_moving_ang = True

        # 3. Apply Explicit Polarity Adjustments
        final_linear  = -lin_target   
        final_angular = -ang_target   

        # 4. CRITICAL FIX: Only publish to /cmd_vel if the UI is actively driving!
        # This stops the UI from trampling and blocking your PS5 controller data lane.
        if is_moving_lin or is_moving_ang:
            self.ros_node.send_cmd(final_linear, final_angular)

        # 5. Maintain Telemetry Map Updates (Always loops smoothly)
        dx = self.bot_x - HOME_X
        dy = self.bot_y - HOME_Y
        dist_angle = math.degrees(math.atan2(dy, dx)) % 360

        self.telemetry.update_telemetry(
            x         = dx,
            y         = dy,
            angle     = dist_angle,
            speed     = self.current_linear_speed,
            ang_speed = self.current_angular_speed,
            lin_mod   = self.lin_speed_mod,
            trn_mod   = self.trn_speed_mod,
        )
        
    def keyPressEvent(self, e: QKeyEvent):
        k = e.key()
        if not self._hard_stop:
            if k in (Qt.Key_Up,    Qt.Key_W): self._set_move("fwd",   True)
            if k in (Qt.Key_Down,  Qt.Key_S): self._set_move("bwd",   True)
            if k in (Qt.Key_Left,  Qt.Key_A): self._set_move("left",  True)
            if k in (Qt.Key_Right, Qt.Key_D): self._set_move("right", True)
        if k == Qt.Key_Space:             self._emergency_stop()

def keyReleaseEvent(self, e: QKeyEvent):
        k = e.key()
        if k in (Qt.Key_Up,    Qt.Key_W): self._set_move("fwd",   False)
        if k in (Qt.Key_Down,  Qt.Key_S): self._set_move("bwd",   False)
        if k in (Qt.Key_Left,  Qt.Key_A): self._set_move("left",  False)
        if k in (Qt.Key_Right, Qt.Key_D): self._set_move("right", False)
        
if __name__ == "__main__":
    rclpy.init(args=sys.argv)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = AMRMainWindow()
    win.show()
    try:
        sys.exit(app.exec())
    finally:
        rclpy.shutdown()

