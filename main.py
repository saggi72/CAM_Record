import sys
import os
import serial
import serial.tools.list_ports
import cv2
import numpy as np
import csv
import json
from datetime import datetime
import time

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QTextEdit, QLineEdit, QLabel, QStatusBar,
    QGridLayout, QGroupBox, QFileDialog, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView, QCheckBox, QSpinBox, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QSettings
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette
from PyQt5.QtMultimedia import QSoundEffect # For sound alerts

# --- Constants ---
APP_NAME = "Serial & Camera Monitor"
ORG_NAME = "MyCompany" # Change if needed for QSettings
DEFAULT_BAUDRATE = 9600
DEFAULT_LOG_FILE = "command_log.txt"
DEFAULT_ALERT_SOUND = "alert.wav" # Ensure this file exists or change path

# --- Helper Functions ---
def get_timestamp():
    """Returns the current timestamp in a readable format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def validate_serial_settings(baudrate_str, data_bits_str, parity_str, stop_bits_str):
    """Validates serial settings before attempting connection."""
    try:
        baudrate = int(baudrate_str)
        data_bits_map = {"5": serial.FIVEBITS, "6": serial.SIXBITS, "7": serial.SEVENBITS, "8": serial.EIGHTBITS}
        parity_map = {"None": serial.PARITY_NONE, "Even": serial.PARITY_EVEN, "Odd": serial.PARITY_ODD, "Mark": serial.PARITY_MARK, "Space": serial.PARITY_SPACE}
        stop_bits_map = {"1": serial.STOPBITS_ONE, "1.5": serial.STOPBITS_ONE_POINT_FIVE, "2": serial.STOPBITS_TWO}

        data_bits = data_bits_map.get(data_bits_str)
        parity = parity_map.get(parity_str)
        stop_bits = stop_bits_map.get(stop_bits_str)

        if not all([data_bits, parity, stop_bits]):
            raise ValueError("Invalid serial setting selected.")

        return baudrate, data_bits, parity, stop_bits
    except ValueError as e:
        print(f"Error validating settings: {e}")
        return None, None, None, None

# --- Serial Communication Thread ---
class SerialWorker(QThread):
    """
    Worker thread for handling serial communication (reading).
    Avoids blocking the main GUI thread.
    """
    data_received = pyqtSignal(bytes)
    connection_error = pyqtSignal(str)
    disconnected = pyqtSignal()

    def __init__(self, port, baudrate, bytesize, parity, stopbits, read_timeout=0.1):
        super().__init__()
        self.serial_port = None
        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self.read_timeout = read_timeout
        self._is_running = False

    def run(self):
        self._is_running = True
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=self.bytesize,
                parity=self.parity,
                stopbits=self.stopbits,
                timeout=self.read_timeout
            )
            print(f"SerialWorker: Connected to {self.port}")
            while self._is_running and self.serial_port and self.serial_port.is_open:
                if self.serial_port.in_waiting > 0:
                    try:
                        # Read available data (adjust reading method if needed)
                        # data = self.serial_port.readline() # Use readline if expecting line endings
                        data = self.serial_port.read(self.serial_port.in_waiting)
                        if data:
                            self.data_received.emit(data)
                    except serial.SerialException as read_err:
                        print(f"SerialWorker: Read error: {read_err}")
                        # Optionally emit an error signal here
                        time.sleep(0.1) # Avoid busy-waiting on error
                    except Exception as general_read_err:
                        print(f"SerialWorker: Unexpected read error: {general_read_err}")
                        time.sleep(0.1)
                else:
                    # Small sleep to prevent high CPU usage when no data
                    self.msleep(50) # Sleep for 50 milliseconds

        except serial.SerialException as e:
            print(f"SerialWorker: Failed to open port {self.port}: {e}")
            self.connection_error.emit(f"Failed to connect: {e}")
        except Exception as ex:
            print(f"SerialWorker: Unexpected error: {ex}")
            self.connection_error.emit(f"An unexpected error occurred: {ex}")
        finally:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
                print(f"SerialWorker: Port {self.port} closed.")
            self._is_running = False
            self.disconnected.emit()
            print("SerialWorker: Thread finished.")

    def stop(self):
        self._is_running = False
        print("SerialWorker: Stop requested.")
        # No need to close port here, finally block handles it

# --- Camera Handling Thread ---
class CameraWorker(QThread):
    """
    Worker thread for handling camera frame grabbing and processing.
    Avoids blocking the main GUI thread.
    """
    frame_ready = pyqtSignal(np.ndarray) # Emit raw frame for flexibility
    camera_error = pyqtSignal(str)
    camera_stopped = pyqtSignal()

    def __init__(self, camera_index_or_url):
        super().__init__()
        self.camera_source = camera_index_or_url
        self.cap = None
        self._is_running = False
        self.do_grayscale = False
        self.do_motion_detection = False
        self.show_overlay = False
        self.last_frame = None
        self.motion_threshold = 500 # Adjust sensitivity

    def run(self):
        self._is_running = True
        print(f"CameraWorker: Trying to open camera: {self.camera_source}")
        try:
            # Attempt to convert to int if possible (for USB cameras)
            try:
                source = int(self.camera_source)
            except ValueError:
                source = self.camera_source # Assume it's a URL/path

            self.cap = cv2.VideoCapture(source) #, cv2.CAP_DSHOW for Windows maybe

            if not self.cap.isOpened():
                raise IOError(f"Cannot open camera: {self.camera_source}")

            print(f"CameraWorker: Camera {self.camera_source} opened successfully.")

            while self._is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("CameraWorker: Failed to grab frame or stream ended.")
                    # self.camera_error.emit("Failed to grab frame.") # Can be noisy
                    self.msleep(100) # Wait a bit before retrying or stopping
                    continue # Try again or exit loop

                processed_frame = frame.copy()

                # --- Image Processing ---
                if self.do_motion_detection:
                    gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (21, 21), 0) # Blur to reduce noise

                    if self.last_frame is None:
                        self.last_frame = gray
                        continue

                    frame_delta = cv2.absdiff(self.last_frame, gray)
                    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                    thresh = cv2.dilate(thresh, None, iterations=2)
                    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    motion_detected = False
                    for contour in contours:
                        if cv2.contourArea(contour) < self.motion_threshold:
                            continue
                        motion_detected = True
                        (x, y, w, h) = cv2.boundingRect(contour)
                        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    if motion_detected and self.show_overlay:
                       cv2.putText(processed_frame, "Motion Detected", (10, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    self.last_frame = gray # Update last frame

                if self.do_grayscale:
                    # If already gray from motion detection, don't reconvert
                    if len(processed_frame.shape) == 3: # Check if it's color
                         processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                    # Need to convert back to BGR for consistent display if overlay is used
                    if len(processed_frame.shape) == 2: # Check if it's grayscale
                        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)


                if self.show_overlay:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(processed_frame, f"CAM: {self.camera_source} {timestamp}", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    status_text = "LIVE"
                    if self.do_grayscale: status_text += " GRAY"
                    # Motion status added within motion detection block

                    cv2.putText(processed_frame, status_text, (processed_frame.shape[1] - 150, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


                self.frame_ready.emit(processed_frame)
                self.msleep(30) # Limit frame rate slightly (~30 FPS), adjust as needed

        except (IOError, cv2.error) as e:
            print(f"CameraWorker: Error: {e}")
            self.camera_error.emit(f"Camera error: {e}")
        except Exception as ex:
            print(f"CameraWorker: Unexpected error: {ex}")
            self.camera_error.emit(f"An unexpected error occurred: {ex}")
        finally:
            if self.cap:
                self.cap.release()
                print(f"CameraWorker: Camera {self.camera_source} released.")
            self._is_running = False
            self.last_frame = None # Reset motion detection state
            self.camera_stopped.emit()
            print("CameraWorker: Thread finished.")

    def stop(self):
        self._is_running = False
        print("CameraWorker: Stop requested.")

    def set_grayscale(self, enabled):
        self.do_grayscale = enabled

    def set_motion_detection(self, enabled):
        self.do_motion_detection = enabled
        if not enabled:
            self.last_frame = None # Reset when disabling

    def set_overlay(self, enabled):
        self.show_overlay = enabled


# --- Main Application Window ---
class SerialMonitorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setGeometry(100, 100, 1200, 800) # Increased size

        # --- Member Variables ---
        self.serial_worker = None
        self.camera_worker = None
        self.serial_port = None # Keep the pyserial object reference
        self.is_connected = False
        self.log_file_path = DEFAULT_LOG_FILE
        self.command_history = [] # Store {'ts', 'dir', 'cmd', 'status'}
        self.auto_send_timer = QTimer(self)
        self.settings = QSettings(ORG_NAME, APP_NAME) # For persistent settings

        # --- Sound Effects ---
        self.alert_sound = QSoundEffect(self)
        self.alert_sound_path = self.settings.value("alertSoundPath", DEFAULT_ALERT_SOUND)
        if os.path.exists(self.alert_sound_path):
             self.alert_sound.setSource(QUrl.fromLocalFile(self.alert_sound_path))
             self.alert_sound.setVolume(0.8) # Adjust volume 0.0 to 1.0
        else:
            print(f"Warning: Alert sound file not found at {self.alert_sound_path}")


        # --- UI Elements ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Create main sections using Grid Layout for better organization
        grid_layout = QGridLayout()
        self.main_layout.addLayout(grid_layout)

        # --- Section 1: Serial Configuration & Control ---
        serial_group = QGroupBox("Serial Communication (COM)")
        serial_layout = QGridLayout()
        serial_group.setLayout(serial_layout)
        grid_layout.addWidget(serial_group, 0, 0) # Row 0, Col 0

        serial_layout.addWidget(QLabel("Port:"), 0, 0)
        self.combo_ports = QComboBox()
        self.combo_ports.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        serial_layout.addWidget(self.combo_ports, 0, 1)

        self.btn_refresh_ports = QPushButton("Refresh")
        self.btn_refresh_ports.clicked.connect(self.populate_ports)
        serial_layout.addWidget(self.btn_refresh_ports, 0, 2)

        serial_layout.addWidget(QLabel("Baud Rate:"), 1, 0)
        self.combo_baud = QComboBox()
        self.combo_baud.addItems(['9600', '19200', '38400', '57600', '115200', '230400', '460800', '921600'])
        self.combo_baud.setCurrentText(str(DEFAULT_BAUDRATE))
        serial_layout.addWidget(self.combo_baud, 1, 1, 1, 2) # Span 2 cols

        # More Serial Settings
        serial_layout.addWidget(QLabel("Data Bits:"), 2, 0)
        self.combo_databits = QComboBox()
        self.combo_databits.addItems(['8', '7', '6', '5'])
        self.combo_databits.setCurrentText('8')
        serial_layout.addWidget(self.combo_databits, 2, 1)

        serial_layout.addWidget(QLabel("Parity:"), 2, 2)
        self.combo_parity = QComboBox()
        self.combo_parity.addItems(['None', 'Even', 'Odd', 'Mark', 'Space'])
        serial_layout.addWidget(self.combo_parity, 2, 3)

        serial_layout.addWidget(QLabel("Stop Bits:"), 3, 0)
        self.combo_stopbits = QComboBox()
        self.combo_stopbits.addItems(['1', '1.5', '2'])
        serial_layout.addWidget(self.combo_stopbits, 3, 1)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self.toggle_serial_connection)
        serial_layout.addWidget(self.btn_connect, 4, 0, 1, 2)

        self.label_status = QLabel("Status: Disconnected")
        font = QFont()
        font.setBold(True)
        self.label_status.setFont(font)
        serial_layout.addWidget(self.label_status, 4, 2, 1, 2, alignment=Qt.AlignRight)


        # --- Section 2: Serial Data Display ---
        data_group = QGroupBox("Received Serial Data")
        data_layout = QVBoxLayout()
        data_group.setLayout(data_layout)
        grid_layout.addWidget(data_group, 1, 0) # Row 1, Col 0

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self.entry_rx_filter = QLineEdit()
        self.entry_rx_filter.setPlaceholderText("Search received data...")
        # self.entry_rx_filter.textChanged.connect(self.filter_received_data) # Implement if needed
        filter_layout.addWidget(self.entry_rx_filter)
        self.btn_clear_rx = QPushButton("Clear")
        self.btn_clear_rx.clicked.connect(lambda: self.text_serial_output.clear())
        filter_layout.addWidget(self.btn_clear_rx)
        data_layout.addLayout(filter_layout)

        self.text_serial_output = QTextEdit()
        self.text_serial_output.setReadOnly(True)
        self.text_serial_output.setFontFamily("Consolas") # Monospaced font
        data_layout.addWidget(self.text_serial_output)


        # --- Section 3: Serial Command Sending ---
        send_group = QGroupBox("Send Serial Command")
        send_layout = QGridLayout()
        send_group.setLayout(send_layout)
        grid_layout.addWidget(send_group, 2, 0) # Row 2, Col 0

        send_layout.addWidget(QLabel("Manual Cmd:"), 0, 0)
        self.entry_command = QLineEdit()
        self.entry_command.setPlaceholderText("Enter command to send")
        send_layout.addWidget(self.entry_command, 0, 1, 1, 2)

        self.btn_send = QPushButton("Send")
        self.btn_send.clicked.connect(self.send_manual_command)
        send_layout.addWidget(self.btn_send, 0, 3)

        send_layout.addWidget(QLabel("Preset Cmd:"), 1, 0)
        self.combo_preset_commands = QComboBox()
        self.combo_preset_commands.addItems(["quay", "tam dung", "luu", "STATUS?", "RESET"]) # Add more
        send_layout.addWidget(self.combo_preset_commands, 1, 1)

        self.btn_send_preset = QPushButton("Send Preset")
        self.btn_send_preset.clicked.connect(self.send_preset_command)
        send_layout.addWidget(self.btn_send_preset, 1, 2)

        # Auto Send
        self.check_auto_send = QCheckBox("Auto Send Every")
        send_layout.addWidget(self.check_auto_send, 2, 0)
        self.spin_auto_interval = QSpinBox()
        self.spin_auto_interval.setRange(1, 3600) # 1 sec to 1 hour
        self.spin_auto_interval.setValue(5)
        self.spin_auto_interval.setSuffix(" sec")
        send_layout.addWidget(self.spin_auto_interval, 2, 1)
        self.btn_toggle_auto_send = QPushButton("Start Auto")
        self.btn_toggle_auto_send.setCheckable(True)
        self.btn_toggle_auto_send.clicked.connect(self.toggle_auto_send)
        send_layout.addWidget(self.btn_toggle_auto_send, 2, 2)

        # Disable send controls initially
        self.entry_command.setEnabled(False)
        self.btn_send.setEnabled(False)
        self.combo_preset_commands.setEnabled(False)
        self.btn_send_preset.setEnabled(False)
        self.check_auto_send.setEnabled(False)
        self.spin_auto_interval.setEnabled(False)
        self.btn_toggle_auto_send.setEnabled(False)


        # --- Section 4: Camera ---
        camera_group = QGroupBox("Camera")
        camera_layout = QVBoxLayout()
        camera_group.setLayout(camera_layout)
        grid_layout.addWidget(camera_group, 0, 1, 2, 1) # Row 0, Col 1, Span 2 rows

        cam_select_layout = QHBoxLayout()
        cam_select_layout.addWidget(QLabel("Select Camera:"))
        self.combo_cameras = QComboBox()
        cam_select_layout.addWidget(self.combo_cameras)
        self.btn_refresh_cameras = QPushButton("Refresh")
        self.btn_refresh_cameras.clicked.connect(self.populate_cameras)
        cam_select_layout.addWidget(self.btn_refresh_cameras)
        camera_layout.addLayout(cam_select_layout)

        self.label_camera_feed = QLabel("Camera feed will appear here")
        self.label_camera_feed.setAlignment(Qt.AlignCenter)
        self.label_camera_feed.setMinimumSize(320, 240) # Minimum size
        self.label_camera_feed.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_camera_feed.setStyleSheet("background-color: black; color: grey;")
        camera_layout.addWidget(self.label_camera_feed)

        cam_control_layout = QGridLayout()
        self.btn_start_camera = QPushButton("Start Camera")
        self.btn_start_camera.clicked.connect(self.toggle_camera_feed)
        cam_control_layout.addWidget(self.btn_start_camera, 0, 0)

        self.btn_capture_image = QPushButton("Capture Image")
        self.btn_capture_image.clicked.connect(self.capture_image)
        self.btn_capture_image.setEnabled(False)
        cam_control_layout.addWidget(self.btn_capture_image, 0, 1)

        self.check_grayscale = QCheckBox("Grayscale")
        self.check_grayscale.stateChanged.connect(self.update_camera_processing)
        cam_control_layout.addWidget(self.check_grayscale, 1, 0)

        self.check_motion = QCheckBox("Motion Detect")
        self.check_motion.stateChanged.connect(self.update_camera_processing)
        cam_control_layout.addWidget(self.check_motion, 1, 1)

        self.check_overlay = QCheckBox("Show Overlay")
        self.check_overlay.setChecked(True) # Enabled by default
        self.check_overlay.stateChanged.connect(self.update_camera_processing)
        cam_control_layout.addWidget(self.check_overlay, 1, 2)

        camera_layout.addLayout(cam_control_layout)


        # --- Section 5: Command History / Log ---
        log_group = QGroupBox("Command Log & History")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        grid_layout.addWidget(log_group, 2, 1, 2, 1) # Row 2, Col 1, Span 2 rows

        log_filter_layout = QHBoxLayout()
        log_filter_layout.addWidget(QLabel("Filter Log:"))
        self.entry_log_filter = QLineEdit()
        self.entry_log_filter.setPlaceholderText("Search timestamp, direction, or command...")
        # self.entry_log_filter.textChanged.connect(self.filter_log_table) # Implement filtering
        log_filter_layout.addWidget(self.entry_log_filter)
        log_layout.addLayout(log_filter_layout)

        self.table_log = QTableWidget()
        self.table_log.setColumnCount(4)
        self.table_log.setHorizontalHeaderLabels(["Timestamp", "Direction", "Command", "Status"])
        self.table_log.setEditTriggers(QAbstractItemView.NoEditTriggers) # Read-only
        self.table_log.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_log.verticalHeader().setVisible(False)
        header = self.table_log.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents) # Timestamp
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents) # Direction
        header.setSectionResizeMode(2, QHeaderView.Stretch)          # Command
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents) # Status
        log_layout.addWidget(self.table_log)

        log_button_layout = QHBoxLayout()
        self.btn_export_log = QPushButton("Export Log")
        self.btn_export_log.clicked.connect(self.export_log)
        log_button_layout.addWidget(self.btn_export_log)

        self.btn_clear_log_table = QPushButton("Clear Table")
        self.btn_clear_log_table.clicked.connect(self.clear_log_table)
        log_button_layout.addWidget(self.btn_clear_log_table)
        log_layout.addLayout(log_button_layout)


        # --- Section 6: Settings (Could be a dialog or separate tab) ---
        settings_group = QGroupBox("Settings")
        settings_layout = QGridLayout()
        settings_group.setLayout(settings_layout)
        grid_layout.addWidget(settings_group, 3, 0) # Row 3, Col 0

        self.check_dark_mode = QCheckBox("Dark Mode")
        self.check_dark_mode.stateChanged.connect(self.toggle_dark_mode)
        settings_layout.addWidget(self.check_dark_mode, 0, 0)

        self.check_debug_mode = QCheckBox("Debug Mode (Log Verbose)")
        # self.check_debug_mode.stateChanged.connect(self.set_debug_mode) # Implement if needed
        settings_layout.addWidget(self.check_debug_mode, 0, 1)

        self.check_sound_alert = QCheckBox("Sound Alert on Command")
        self.check_sound_alert.setChecked(self.settings.value("soundAlertEnabled", False, type=bool))
        self.check_sound_alert.stateChanged.connect(self.toggle_sound_alert_setting)
        settings_layout.addWidget(self.check_sound_alert, 1, 0)

        self.btn_select_sound = QPushButton("Select Alert Sound")
        self.btn_select_sound.clicked.connect(self.select_alert_sound)
        settings_layout.addWidget(self.btn_select_sound, 1, 1)

        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # --- Final Touches ---
        self.populate_ports()
        self.populate_cameras()
        self.load_settings() # Load theme preference
        self.update_ui_state() # Set initial enabled/disabled states

        # Connect auto-send timer
        self.auto_send_timer.timeout.connect(self.execute_auto_send)


    # --- UI State Management ---
    def update_ui_state(self):
        """Enable/disable UI elements based on connection status."""
        is_disconnected = not self.is_connected
        is_streaming = self.camera_worker is not None and self.camera_worker.isRunning()

        # Serial Config
        self.combo_ports.setEnabled(is_disconnected)
        self.btn_refresh_ports.setEnabled(is_disconnected)
        self.combo_baud.setEnabled(is_disconnected)
        self.combo_databits.setEnabled(is_disconnected)
        self.combo_parity.setEnabled(is_disconnected)
        self.combo_stopbits.setEnabled(is_disconnected)
        self.btn_connect.setText("Disconnect" if self.is_connected else "Connect")

        # Serial Send
        can_send = self.is_connected and not self.btn_toggle_auto_send.isChecked()
        self.entry_command.setEnabled(can_send)
        self.btn_send.setEnabled(can_send)
        self.combo_preset_commands.setEnabled(can_send)
        self.btn_send_preset.setEnabled(can_send)

        # Auto Send (only enable controls if connected)
        self.check_auto_send.setEnabled(self.is_connected)
        self.spin_auto_interval.setEnabled(self.is_connected and self.check_auto_send.isChecked())
        self.btn_toggle_auto_send.setEnabled(self.is_connected)
        if not self.is_connected and self.btn_toggle_auto_send.isChecked():
             self.btn_toggle_auto_send.setChecked(False) # Stop auto send if disconnected
             self.toggle_auto_send(False)

        # Camera Controls
        self.combo_cameras.setEnabled(not is_streaming)
        self.btn_refresh_cameras.setEnabled(not is_streaming)
        self.btn_start_camera.setText("Stop Camera" if is_streaming else "Start Camera")
        self.btn_capture_image.setEnabled(is_streaming)
        # Processing checkboxes can be enabled always or only when streaming
        self.check_grayscale.setEnabled(is_streaming)
        self.check_motion.setEnabled(is_streaming)
        self.check_overlay.setEnabled(is_streaming)


    # --- Serial Port Methods ---
    def populate_ports(self):
        """Fills the COM port dropdown."""
        self.combo_ports.clear()
        ports = serial.tools.list_ports.comports()
        if not ports:
            self.combo_ports.addItem("No ports found")
            self.combo_ports.setEnabled(False)
            self.btn_connect.setEnabled(False)
        else:
            for port in sorted(ports):
                self.combo_ports.addItem(f"{port.device} - {port.description}")
            self.combo_ports.setEnabled(True)
            self.btn_connect.setEnabled(True)
            # Try to select previously used port
            last_port = self.settings.value("lastSerialPort", "")
            index = self.combo_ports.findText(last_port, Qt.MatchContains)
            if index >= 0:
                self.combo_ports.setCurrentIndex(index)


    def toggle_serial_connection(self):
        """Connects or disconnects the serial port."""
        if self.is_connected:
            self.disconnect_serial()
        else:
            self.connect_serial()
        self.update_ui_state()


    def connect_serial(self):
        """Establishes the serial connection in a separate thread."""
        selected_port_text = self.combo_ports.currentText()
        if not selected_port_text or "No ports found" in selected_port_text:
            self.show_error_message("No serial port selected.")
            return

        port = selected_port_text.split(" - ")[0] # Get device name
        baudrate_str = self.combo_baud.currentText()
        databits_str = self.combo_databits.currentText()
        parity_str = self.combo_parity.currentText()
        stopbits_str = self.combo_stopbits.currentText()

        baudrate, bytesize, parity, stopbits = validate_serial_settings(
            baudrate_str, databits_str, parity_str, stopbits_str
        )

        if baudrate is None:
            self.show_error_message("Invalid serial port settings.")
            return

        # --- Start Worker Thread ---
        self.status_bar.showMessage(f"Attempting to connect to {port}...")
        self.serial_worker = SerialWorker(port, baudrate, bytesize, parity, stopbits)
        self.serial_worker.data_received.connect(self.handle_serial_data)
        self.serial_worker.connection_error.connect(self.handle_serial_error)
        self.serial_worker.disconnected.connect(self.handle_serial_disconnect_signal) # Use signal
        self.serial_worker.finished.connect(self.on_serial_worker_finished) # Cleanup thread object
        self.serial_worker.start()

        # Assume connection success for now, update on error/success signal
        # (Better: wait for a success signal from the worker)
        self.is_connected = True # Optimistic, corrected by error signal if fails
        self.label_status.setText(f"Status: Connecting...")
        self.label_status.setStyleSheet("color: orange;")
        self.settings.setValue("lastSerialPort", selected_port_text) # Save port choice


    def disconnect_serial(self):
        """Stops the serial worker thread and closes the port."""
        if self.serial_worker and self.serial_worker.isRunning():
            self.status_bar.showMessage("Disconnecting...")
            self.serial_worker.stop()
            # Don't set is_connected = False here, wait for disconnected signal
        else:
            # If worker wasn't running but UI thought it was connected
            self.handle_serial_disconnect_signal()

    def handle_serial_data(self, data):
        """Processes and displays incoming serial data."""
        try:
            # Try decoding as UTF-8, replace errors
            text = data.decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Error decoding data: {e}")
            text = repr(data) # Show raw bytes representation on error

        timestamp = get_timestamp()
        self.text_serial_output.moveCursor(QTextCursor.End)
        self.text_serial_output.insertPlainText(f"[{timestamp}] RX: {text}\n")
        self.text_serial_output.moveCursor(QTextCursor.End) # Scroll to bottom

        # Log command
        self.log_command("RX", text.strip()) # Log stripped text

        # Process specific commands
        command = text.strip().lower() # Normalize command
        if command == "quay":
            self.status_bar.showMessage("Received 'quay' command.", 3000)
            self.play_alert_sound()
            # Add specific action for 'quay' if needed
        elif command == "tam dung" or command == "tạm dừng":
             self.status_bar.showMessage("Received 'tạm dừng' command.", 3000)
             self.play_alert_sound()
             # Add specific action for 'tạm dừng' if needed
        elif command == "luu" or command == "lưu":
             self.status_bar.showMessage("Received 'lưu' command. Prompting for save...", 3000)
             self.save_received_data_dialog()


    def handle_serial_error(self, error_message):
        """Handles errors reported by the serial worker."""
        self.show_error_message(f"Serial Connection Error: {error_message}")
        self.is_connected = False
        self.label_status.setText("Status: Error")
        self.label_status.setStyleSheet("color: red;")
        self.status_bar.showMessage(f"Error: {error_message}", 5000)
        self.update_ui_state()
        self.serial_worker = None # Worker likely stopped


    def handle_serial_disconnect_signal(self):
        """Handles the disconnected signal from the worker."""
        if self.is_connected: # Only update if we thought we were connected
            print("Handle disconnect signal: Updating UI")
            self.is_connected = False
            self.label_status.setText("Status: Disconnected")
            self.label_status.setStyleSheet("color: red;")
            self.status_bar.showMessage("Disconnected", 3000)
            self.update_ui_state()


    def on_serial_worker_finished(self):
        """Called when the serial worker thread has completely finished."""
        print("Serial worker thread finished.")
        # Ensure UI reflects disconnected state if not already done
        if self.is_connected:
            self.handle_serial_disconnect_signal()
        self.serial_worker = None # Allow garbage collection


    def send_serial_command(self, command_str):
        """Sends a command string over the serial port."""
        if not self.is_connected or not self.serial_worker or not self.serial_worker.serial_port:
            self.show_error_message("Not connected to serial port.")
            return False

        if not command_str:
            return False # Don't send empty commands

        try:
            # Add newline or specific ending character if required by device
            # command_to_send = (command_str + '\n').encode('utf-8')
            command_to_send = command_str.encode('utf-8') # Send as is
            self.serial_worker.serial_port.write(command_to_send)

            timestamp = get_timestamp()
            # Display sent command in the output window for clarity
            self.text_serial_output.moveCursor(QTextCursor.End)
            self.text_serial_output.insertPlainText(f"[{timestamp}] TX: {command_str}\n")
            self.text_serial_output.moveCursor(QTextCursor.End)

            self.log_command("TX", command_str, "Success")
            self.status_bar.showMessage(f"Sent: {command_str}", 2000)
            return True
        except serial.SerialException as e:
            error_msg = f"Error sending command: {e}"
            self.show_error_message(error_msg)
            self.log_command("TX", command_str, f"Error: {e}")
            self.status_bar.showMessage(error_msg, 5000)
            # Consider disconnecting if write fails consistently
            # self.disconnect_serial()
            return False
        except Exception as ex:
            error_msg = f"Unexpected error sending command: {ex}"
            self.show_error_message(error_msg)
            self.log_command("TX", command_str, f"Error: {ex}")
            self.status_bar.showMessage(error_msg, 5000)
            return False


    def send_manual_command(self):
        """Sends the command entered in the manual command entry."""
        command = self.entry_command.text()
        if self.send_serial_command(command):
            self.entry_command.clear() # Clear on success


    def send_preset_command(self):
        """Sends the selected preset command."""
        command = self.combo_preset_commands.currentText()
        self.send_serial_command(command)


    def toggle_auto_send(self, checked):
        """Starts or stops the auto-send timer."""
        if checked and self.is_connected:
            interval_ms = self.spin_auto_interval.value() * 1000
            if interval_ms <= 0:
                self.show_error_message("Auto-send interval must be positive.")
                self.btn_toggle_auto_send.setChecked(False)
                return

            command_type = "preset" if self.check_auto_send.isChecked() else "manual" # Determine which cmd
            self.auto_send_timer.setInterval(interval_ms)
            self.auto_send_timer.start()
            self.btn_toggle_auto_send.setText("Stop Auto")
            self.status_bar.showMessage(f"Auto-send started (every {self.spin_auto_interval.value()}s).", 3000)
        else:
            self.auto_send_timer.stop()
            self.btn_toggle_auto_send.setChecked(False) # Ensure it's unchecked
            self.btn_toggle_auto_send.setText("Start Auto")
            if self.is_connected: # Only show message if we were connected
                 self.status_bar.showMessage("Auto-send stopped.", 3000)

        # Update UI to enable/disable manual send buttons during auto-send
        self.update_ui_state()


    def execute_auto_send(self):
        """Called by the QTimer to send the command."""
        if not self.is_connected or not self.check_auto_send.isChecked():
            self.toggle_auto_send(False) # Stop if disconnected or checkbox unchecked
            return

        # Decide which command to send (preset is typical for auto)
        command = self.combo_preset_commands.currentText()
        # Optionally, add logic to send from manual entry if needed
        # command = self.entry_command.text()

        if not command:
            print("Auto-send: No command selected/entered.")
            return

        print(f"Auto-sending: {command}")
        self.send_serial_command(command)


    # --- Camera Methods ---
    def populate_cameras(self):
        """Detects available USB cameras and populates the dropdown."""
        self.combo_cameras.clear()
        available_cameras = []
        # Check standard indices (0, 1, 2, etc.) - adjust range if needed
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # CAP_DSHOW might help on Windows
            if cap is not None and cap.isOpened():
                available_cameras.append((i, f"Camera {i}"))
                cap.release()
            else:
                 # If index 0 fails, often no more cameras
                 if i == 0: break

        if not available_cameras:
            self.combo_cameras.addItem("No cameras detected")
            self.combo_cameras.setEnabled(False)
            self.btn_start_camera.setEnabled(False)
        else:
            for index, name in available_cameras:
                self.combo_cameras.addItem(name, userData=index) # Store index as data
            self.combo_cameras.setEnabled(True)
            self.btn_start_camera.setEnabled(True)
            # Add option for IP Camera URL?
            # self.combo_cameras.addItem("IP Camera (Enter URL)", userData="IP")


    def toggle_camera_feed(self):
        """Starts or stops the camera feed."""
        if self.camera_worker and self.camera_worker.isRunning():
            self.stop_camera_feed()
        else:
            self.start_camera_feed()
        self.update_ui_state()

    def start_camera_feed(self):
        """Starts the camera worker thread."""
        selected_index = self.combo_cameras.currentIndex()
        if selected_index < 0:
            self.show_error_message("No camera selected.")
            return

        camera_source = self.combo_cameras.itemData(selected_index)
        if camera_source is None:
             # Handle case where "No cameras detected" might be selected
             self.show_error_message("Invalid camera selection.")
             return

        print(f"Starting camera: {camera_source}")
        self.status_bar.showMessage(f"Starting camera {camera_source}...")
        self.label_camera_feed.setText(f"Starting Camera {camera_source}...")
        self.label_camera_feed.setStyleSheet("background-color: black; color: yellow;")

        self.camera_worker = CameraWorker(camera_source)
        self.camera_worker.frame_ready.connect(self.update_camera_frame)
        self.camera_worker.camera_error.connect(self.handle_camera_error)
        self.camera_worker.camera_stopped.connect(self.handle_camera_stopped_signal)
        self.camera_worker.finished.connect(self.on_camera_worker_finished)

        # Apply initial processing settings from checkboxes
        self.update_camera_processing()

        self.camera_worker.start()
        # UI state updated once started/stopped signal received or error


    def stop_camera_feed(self):
        """Stops the camera worker thread."""
        if self.camera_worker and self.camera_worker.isRunning():
            self.status_bar.showMessage("Stopping camera...")
            self.camera_worker.stop()
        else:
            self.handle_camera_stopped_signal() # Ensure UI is reset


    def update_camera_frame(self, frame_np):
        """Updates the camera feed label with a new frame."""
        try:
            # Convert numpy array (OpenCV frame) to QImage/QPixmap
            if frame_np is None or frame_np.size == 0:
                print("Received empty frame")
                return

            height, width = frame_np.shape[:2]
            bytes_per_line = frame_np.strides[0] # More robust way to get bytes per line

            if len(frame_np.shape) == 3: # Color BGR
                image_format = QImage.Format_BGR888 # OpenCV uses BGR
            elif len(frame_np.shape) == 2: # Grayscale
                image_format = QImage.Format_Grayscale8
            else:
                print("Unsupported frame format")
                return

            q_image = QImage(frame_np.data, width, height, bytes_per_line, image_format)

            # Scale pixmap to fit the label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.label_camera_feed.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label_camera_feed.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"Error updating camera frame: {e}")
            # Optionally display an error on the label
            self.label_camera_feed.setText(f"Frame Error: {e}")
            self.label_camera_feed.setStyleSheet("background-color: black; color: red;")


    def handle_camera_error(self, error_message):
        """Handles errors reported by the camera worker."""
        self.show_error_message(f"Camera Error: {error_message}")
        self.label_camera_feed.setText(f"Camera Error:\n{error_message}")
        self.label_camera_feed.setStyleSheet("background-color: black; color: red;")
        # Ensure the worker is considered stopped
        self.handle_camera_stopped_signal()


    def handle_camera_stopped_signal(self):
        """Handles the stopped signal from the camera worker."""
        print("Handle camera stopped signal: Resetting UI")
        self.label_camera_feed.setText("Camera feed stopped.")
        self.label_camera_feed.setStyleSheet("background-color: black; color: grey;")
        self.status_bar.showMessage("Camera stopped.", 3000)
        # No need to set self.camera_worker = None here, wait for finished signal
        self.update_ui_state()

    def on_camera_worker_finished(self):
        """Called when the camera worker thread has completely finished."""
        print("Camera worker thread finished.")
        # Ensure UI is reset if not already done
        if self.camera_worker is not None: # Check if it wasn't already cleared by error
             if self.btn_start_camera.text() == "Stop Camera": # If UI still thinks it's running
                 self.handle_camera_stopped_signal()
        self.camera_worker = None # Allow garbage collection


    def capture_image(self):
        """Captures the current frame from the camera feed and saves it."""
        if not self.camera_worker or not self.camera_worker.isRunning():
            self.show_error_message("Camera is not running.")
            return

        # Get the current pixmap displayed on the label
        current_pixmap = self.label_camera_feed.pixmap()
        if not current_pixmap or current_pixmap.isNull():
             self.show_error_message("No image frame available to capture.")
             return

        # Ask user for save location
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"capture_{timestamp}.png"
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Image", default_filename,
                                                  "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)

        if fileName:
            if not current_pixmap.save(fileName):
                self.show_error_message(f"Failed to save image to {fileName}")
            else:
                self.status_bar.showMessage(f"Image saved to {fileName}", 3000)


    def update_camera_processing(self):
        """Updates the processing flags in the camera worker based on checkboxes."""
        if self.camera_worker:
            self.camera_worker.set_grayscale(self.check_grayscale.isChecked())
            self.camera_worker.set_motion_detection(self.check_motion.isChecked())
            self.camera_worker.set_overlay(self.check_overlay.isChecked())


    # --- Logging Methods ---
    def log_command(self, direction, command, status=""):
        """Adds a command to the history list and the log table/file."""
        timestamp = get_timestamp()
        log_entry = {
            "ts": timestamp,
            "dir": direction, # RX or TX
            "cmd": command,
            "status": status
        }
        self.command_history.append(log_entry)

        # Add to table widget
        row_position = self.table_log.rowCount()
        self.table_log.insertRow(row_position)
        self.table_log.setItem(row_position, 0, QTableWidgetItem(timestamp))
        self.table_log.setItem(row_position, 1, QTableWidgetItem(direction))
        # Prevent overly long commands from messing up table display too much
        display_cmd = command if len(command) < 200 else command[:197] + "..."
        self.table_log.setItem(row_position, 2, QTableWidgetItem(display_cmd))
        self.table_log.setItem(row_position, 3, QTableWidgetItem(status))
        self.table_log.scrollToBottom() # Keep latest visible

        # Append to persistent log file
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"{timestamp}|{direction}|{command}|{status}\n")
        except Exception as e:
            print(f"Error writing to log file '{self.log_file_path}': {e}")


    def export_log(self):
        """Exports the current log table content to CSV or TXT."""
        if self.table_log.rowCount() == 0:
            self.show_error_message("Log is empty, nothing to export.")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, selected_filter = QFileDialog.getSaveFileName(self, "Export Log", "command_log_export",
                                                  "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)", options=options)

        if fileName:
            try:
                with open(fileName, 'w', newline='', encoding='utf-8') as outfile:
                    if selected_filter.startswith("CSV"):
                        writer = csv.writer(outfile)
                        # Write header
                        headers = [self.table_log.horizontalHeaderItem(i).text() for i in range(self.table_log.columnCount())]
                        writer.writerow(headers)
                        # Write data rows
                        for row in range(self.table_log.rowCount()):
                            rowData = [self.table_log.item(row, col).text() if self.table_log.item(row, col) else ""
                                       for col in range(self.table_log.columnCount())]
                            writer.writerow(rowData)
                    else: # TXT or All Files (treat as TXT)
                        # Write header
                        headers = [self.table_log.horizontalHeaderItem(i).text() for i in range(self.table_log.columnCount())]
                        outfile.write("\t".join(headers) + "\n")
                        # Write data rows
                        for row in range(self.table_log.rowCount()):
                            rowData = [self.table_log.item(row, col).text() if self.table_log.item(row, col) else ""
                                       for col in range(self.table_log.columnCount())]
                            outfile.write("\t".join(rowData) + "\n")

                self.status_bar.showMessage(f"Log exported successfully to {fileName}", 3000)

            except Exception as e:
                self.show_error_message(f"Error exporting log: {e}")


    def clear_log_table(self):
        """Clears the log table in the UI (doesn't affect the file log)."""
        self.table_log.setRowCount(0)
        self.command_history.clear() # Also clear the in-memory list if table is cleared
        self.status_bar.showMessage("Log table cleared.", 2000)


    # --- Data Saving ("luu" command) ---
    def save_received_data_dialog(self):
        """Shows a dialog to save recently received data."""
        # Decide what data to save. Let's save the entire content of the RX window
        data_to_save = self.text_serial_output.toPlainText()

        if not data_to_save.strip():
            self.show_error_message("No received data to save.")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"serial_data_{timestamp}"
        fileName, selected_filter = QFileDialog.getSaveFileName(self, "Save Received Data", default_filename,
                                                  "Text Files (*.txt);;CSV Files (*.csv);;JSON Files (*.json);;All Files (*)", options=options)

        if fileName:
            try:
                # Determine format based on extension or filter
                file_ext = os.path.splitext(fileName)[1].lower()
                save_format = "txt" # Default
                if selected_filter.startswith("CSV") or file_ext == ".csv":
                    save_format = "csv"
                elif selected_filter.startswith("JSON") or file_ext == ".json":
                    save_format = "json"

                # Add timestamp to the data itself before saving
                content_with_ts = f"--- Data saved at {get_timestamp()} ---\n{data_to_save}"

                with open(fileName, 'w', encoding='utf-8', newline='') as f:
                    if save_format == "csv":
                        # Simple CSV: each line from QTextEdit becomes a row in one column
                        # Or parse lines if they have a structure (e.g., comma-separated)
                        writer = csv.writer(f)
                        writer.writerow(["Timestamp", "ReceivedData"]) # Header
                        for line in data_to_save.splitlines():
                           # Extract timestamp and data if possible (example assumes format "[ts] RX: data")
                           ts_part = ""
                           data_part = line
                           if line.startswith("[") and "] RX:" in line:
                               try:
                                   ts_part = line[1:line.find("]")]
                                   data_part = line[line.find("RX:")+4:]
                               except Exception:
                                    pass # Keep original line if parsing fails
                           writer.writerow([ts_part.strip(), data_part.strip()])

                    elif save_format == "json":
                        # Save as a list of lines or a single string block
                        # Example: list of dictionaries with timestamp extracted
                        json_data = []
                        for line in data_to_save.splitlines():
                            ts_part = ""
                            data_part = line
                            if line.startswith("[") and "] RX:" in line:
                                try:
                                    ts_part = line[1:line.find("]")]
                                    data_part = line[line.find("RX:")+4:]
                                except Exception:
                                     pass
                            json_data.append({"timestamp": ts_part.strip(), "data": data_part.strip()})
                        json.dump(json_data, f, indent=4) # Pretty print JSON

                    else: # TXT format
                        f.write(content_with_ts)

                self.status_bar.showMessage(f"Data saved successfully to {fileName}", 3000)

            except Exception as e:
                self.show_error_message(f"Error saving data: {e}")


    # --- Settings & Appearance ---
    def load_settings(self):
        """Load persistent settings."""
        # Theme
        use_dark_mode = self.settings.value("darkMode", False, type=bool)
        self.check_dark_mode.setChecked(use_dark_mode)
        self.apply_theme(use_dark_mode)

        # Sound alert enabled state
        sound_enabled = self.settings.value("soundAlertEnabled", False, type=bool)
        self.check_sound_alert.setChecked(sound_enabled)

        # Log file path (optional, if you want to make it configurable)
        # self.log_file_path = self.settings.value("logFilePath", DEFAULT_LOG_FILE)

        # Last used serial port (already handled in populate_ports)
        baud_index = self.combo_baud.findText(self.settings.value("lastBaudRate", str(DEFAULT_BAUDRATE)))
        if baud_index >= 0: self.combo_baud.setCurrentIndex(baud_index)
        # Load other serial settings similarly if desired

    def save_settings(self):
        """Save persistent settings."""
        self.settings.setValue("darkMode", self.check_dark_mode.isChecked())
        self.settings.setValue("soundAlertEnabled", self.check_sound_alert.isChecked())
        self.settings.setValue("alertSoundPath", self.alert_sound_path)

        if self.is_connected:
            self.settings.setValue("lastSerialPort", self.combo_ports.currentText())
            self.settings.setValue("lastBaudRate", self.combo_baud.currentText())
            # Save other serial settings if desired

    def toggle_dark_mode(self, checked):
        """Applies or removes the dark theme."""
        self.apply_theme(checked)
        self.settings.setValue("darkMode", checked) # Save preference

    def apply_theme(self, dark_mode):
        """Sets the application style sheet."""
        if dark_mode:
            # Basic Dark Theme using QPalette (simpler)
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.WindowText, Qt.white)
            dark_palette.setColor(QPalette.Base, QColor(35, 35, 35)) # Edits, lists
            dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
            dark_palette.setColor(QPalette.ToolTipText, Qt.white)
            dark_palette.setColor(QPalette.Text, Qt.white)
            dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ButtonText, Qt.white)
            dark_palette.setColor(QPalette.BrightText, Qt.red)
            dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.HighlightedText, Qt.black)
            # Disabled state
            dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
            dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
            dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
            dark_palette.setColor(QPalette.Disabled, QPalette.Base, QColor(60, 60, 60))

            QApplication.instance().setPalette(dark_palette)
            # You might need more specific styling via StyleSheet for some widgets
            # E.g., QApplication.instance().setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")
            self.text_serial_output.setStyleSheet("background-color: #232323; color: #E0E0E0;")
            self.table_log.setStyleSheet("background-color: #232323; color: #E0E0E0; gridline-color: #5A5A5A;")

        else:
            # Reset to default system palette
            QApplication.instance().setPalette(QApplication.style().standardPalette())
            QApplication.instance().setStyleSheet("") # Clear stylesheet
            self.text_serial_output.setStyleSheet("")
            self.table_log.setStyleSheet("")

        # Re-apply status colors as palette changes might override them
        if self.is_connected:
             self.label_status.setStyleSheet("color: green;" if self.label_status.text() == "Status: Connected" else self.label_status.styleSheet())
        else:
             self.label_status.setStyleSheet("color: red;" if "Disconnected" in self.label_status.text() or "Error" in self.label_status.text() else self.label_status.styleSheet())


    def toggle_sound_alert_setting(self, checked):
        """Enable/disable the sound alert feature via settings."""
        self.settings.setValue("soundAlertEnabled", checked)
        if checked and not os.path.exists(self.alert_sound_path):
             self.show_error_message(f"Sound alerts enabled, but alert file not found:\n{self.alert_sound_path}\nPlease select a valid .wav file in Settings.")
        elif checked:
             self.status_bar.showMessage("Sound alerts enabled.", 2000)
        else:
             self.status_bar.showMessage("Sound alerts disabled.", 2000)


    def select_alert_sound(self):
        """Opens a dialog to choose the alert sound file."""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Alert Sound File", "",
                                                  "WAV Sound Files (*.wav);;All Files (*)", options=options)
        if fileName:
            if os.path.exists(fileName):
                self.alert_sound_path = fileName
                self.alert_sound.setSource(QUrl.fromLocalFile(self.alert_sound_path))
                self.settings.setValue("alertSoundPath", self.alert_sound_path)
                self.status_bar.showMessage(f"Alert sound set to: {os.path.basename(fileName)}", 3000)
            else:
                self.show_error_message("Selected sound file does not exist.")


    def play_alert_sound(self):
        """Plays the alert sound if enabled and file exists."""
        if self.check_sound_alert.isChecked() and self.alert_sound.isLoaded():
            self.alert_sound.play()


    # --- Utility Methods ---
    def show_error_message(self, message):
        """Displays an error message box."""
        QMessageBox.critical(self, "Error", message)

    def show_info_message(self, message):
        """Displays an informational message box."""
        QMessageBox.information(self, "Information", message)

    # --- Window Closing ---
    def closeEvent(self, event):
        """Handle window close event."""
        print("Close event triggered")
        # Stop threads gracefully
        if self.serial_worker and self.serial_worker.isRunning():
            print("Stopping serial worker...")
            self.serial_worker.stop()
            # self.serial_worker.wait(2000) # Wait max 2 seconds for thread to finish

        if self.camera_worker and self.camera_worker.isRunning():
            print("Stopping camera worker...")
            self.camera_worker.stop()
            # self.camera_worker.wait(2000) # Wait max 2 seconds

        # Save settings before closing
        self.save_settings()
        print("Settings saved.")

        # It's generally better practice to let threads finish via signals
        # rather than forced waits in the closeEvent, but wait() can be
        # used if absolutely necessary.

        event.accept() # Accept the close event
        print("Exiting application.")


# --- Main Execution ---
if __name__ == '__main__':
    # Enable high DPI scaling for better look on modern displays
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationName(APP_NAME)

    # Apply initial theme based on saved settings before creating window
    settings = QSettings(ORG_NAME, APP_NAME)
    use_dark_mode = settings.value("darkMode", False, type=bool)
    if use_dark_mode:
         # Apply dark palette directly at startup (same code as in apply_theme)
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
        dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
        dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
        dark_palette.setColor(QPalette.Disabled, QPalette.Base, QColor(60, 60, 60))
        app.setPalette(dark_palette)

    mainWin = SerialMonitorApp()
    mainWin.show()
    sys.exit(app.exec_())