import math
import os
import json
import threading
import time
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk


PROJECT_DIR = Path(__file__).resolve().parents[1]
CAMERA_CALIBRATION_FILE = PROJECT_DIR / "data" / "camera_calibration.npz"
FIXED_SCENE_FILE = PROJECT_DIR / "data" / "camera_fixed_scene.json"


# ============================================================
# CAMERA CAPTURE SERVICE
# ============================================================

class CameraCapture:
    """
    Small capture service.

    Responsibilities:
    - open / close VideoCapture
    - continuously grab frames in a background thread
    - expose only the latest raw frame
    - stop cleanly after repeated read failures

    No Tkinter calls are made from this worker thread.
    """

    MAX_CONSECUTIVE_READ_FAILURES = 10

    def __init__(self, log):
        self.log = log

        self.cap = None
        self.thread = None
        self.running = False

        self.frame_lock = threading.Lock()
        self.latest_raw_frame = None

        self.frame_size = None
        self.last_frame_time = 0.0

        self.status = "stopped"
        self.last_error = None
        self.backend_name = "—"

    def _open_capture(self, index):
        """
        Prefer DirectShow on Windows because MSMF can repeatedly
        emit grabFrame errors on some USB cameras.
        """

        candidates = []

        if os.name == "nt":
            candidates.append(("DirectShow", cv2.CAP_DSHOW))

        candidates.append(("Default", None))

        for name, backend in candidates:
            cap = (
                cv2.VideoCapture(index, backend)
                if backend is not None
                else cv2.VideoCapture(index)
            )

            if not cap.isOpened():
                cap.release()
                continue

            # Try a real frame immediately. A camera can report isOpened()
            # while the backend still cannot deliver frames.
            ok, frame = cap.read()

            if ok and frame is not None:
                self.backend_name = name
                return cap, frame

            cap.release()

        return None, None

    def start(self, index):
        if self.running:
            return True, "Camera already running."

        self.last_error = None
        self.status = "opening"

        cap, first_frame = self._open_capture(int(index))

        if cap is None:
            self.status = "error"
            self.last_error = (
                f"Could not open camera index {index} "
                f"with DirectShow/default backend."
            )
            return False, self.last_error

        self.cap = cap

        with self.frame_lock:
            self.latest_raw_frame = first_frame.copy()

        h, w = first_frame.shape[:2]
        self.frame_size = (w, h)
        self.last_frame_time = time.monotonic()

        self.running = True
        self.status = "running"

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )
        self.thread.start()

        return True, (
            f"Camera running | index {index} | "
            f"backend {self.backend_name}"
        )

    def _loop(self):
        failures = 0

        while self.running:
            cap = self.cap

            if cap is None:
                break

            try:
                ok, frame = cap.read()
            except Exception as error:
                ok = False
                frame = None
                self.last_error = str(error)

            if not ok or frame is None:
                failures += 1

                if failures >= self.MAX_CONSECUTIVE_READ_FAILURES:
                    self.last_error = (
                        "Camera stopped after repeated frame read failures."
                    )
                    self.status = "read error"
                    self.running = False
                    break

                time.sleep(0.05)
                continue

            failures = 0

            h, w = frame.shape[:2]

            with self.frame_lock:
                self.latest_raw_frame = frame.copy()

            self.frame_size = (w, h)
            self.last_frame_time = time.monotonic()

            time.sleep(0.005)

        self._release_capture()

    def get_frame(self):
        with self.frame_lock:
            if self.latest_raw_frame is None:
                return None
            return self.latest_raw_frame.copy()

    def stop(self):
        self.running = False
        self._release_capture()

        if self.status != "read error":
            self.status = "stopped"

    def _release_capture(self):
        cap = self.cap
        self.cap = None

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def shutdown(self):
        self.stop()



# ============================================================
# YOLO DETECTION SERVICE
# ============================================================

class YoloDetectionService:
    """
    Background YOLO inference service.

    Design goals:
    - Ultralytics is imported lazily only when a model is loaded.
    - inference never blocks Tkinter / ArUco processing
    - only the newest submitted frame is processed
    - model can be loaded / unloaded at runtime
    """

    def __init__(self, log):
        self.log = log

        self.model = None
        self.model_path = None
        self.model_names = {}

        self.running = True
        self.enabled = False

        self.confidence = 0.40
        self.max_fps = 10.0

        self.input_lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_id = 0
        self.last_consumed_frame_id = -1

        self.result_lock = threading.Lock()
        self.latest_result = None
        self.last_inference_ms = None
        self.last_result_time = 0.0

        self.status = "no model"
        self.last_error = None

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )
        self.thread.start()

    def load_model(self, path):
        path = Path(path)

        if not path.exists():
            return False, f"Model not found: {path}"

        try:
            from ultralytics import YOLO

            model = YOLO(str(path))

            self.model = model
            self.model_path = str(path)
            self.model_names = dict(model.names)
            self.status = "loaded"
            self.last_error = None

            return True, (
                f"YOLO loaded: {path.name} | "
                f"{len(self.model_names)} classes"
            )

        except Exception as error:
            self.model = None
            self.model_path = None
            self.model_names = {}
            self.status = "error"
            self.last_error = str(error)

            return False, f"YOLO load error: {error}"

    def unload_model(self):
        self.enabled = False
        self.model = None
        self.model_path = None
        self.model_names = {}
        self.status = "no model"

        with self.result_lock:
            self.latest_result = None
            self.last_inference_ms = None
            self.last_result_time = 0.0

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)

    def set_confidence(self, value):
        self.confidence = max(
            0.01,
            min(0.99, float(value)),
        )

    def set_max_fps(self, value):
        self.max_fps = max(
            1.0,
            min(30.0, float(value)),
        )

    def submit(self, frame):
        if (
            not self.enabled
            or self.model is None
            or frame is None
        ):
            return

        with self.input_lock:
            self.latest_frame = frame.copy()
            self.latest_frame_id += 1

    def get_result(self):
        with self.result_lock:
            if self.latest_result is None:
                return None

            result = dict(self.latest_result)
            result["detections"] = [
                dict(item)
                for item in self.latest_result.get(
                    "detections",
                    []
                )
            ]
            return result

    def _loop(self):
        last_inference_start = 0.0

        while self.running:
            if not self.enabled or self.model is None:
                time.sleep(0.05)
                continue

            interval = 1.0 / max(
                1.0,
                float(self.max_fps),
            )

            now = time.monotonic()

            if (
                now - last_inference_start
                < interval
            ):
                time.sleep(0.005)
                continue

            frame = None
            frame_id = None

            with self.input_lock:
                if (
                    self.latest_frame is not None
                    and self.latest_frame_id
                    != self.last_consumed_frame_id
                ):
                    frame = self.latest_frame.copy()
                    frame_id = self.latest_frame_id

            if frame is None:
                time.sleep(0.005)
                continue

            last_inference_start = time.monotonic()

            try:
                started = time.perf_counter()

                try:
                    import torch

                    device = (
                        0
                        if torch.cuda.is_available()
                        else "cpu"
                    )
                except Exception:
                    device = "cpu"

                predictions = self.model.predict(
                    source=frame,
                    conf=float(self.confidence),
                    verbose=False,
                    device=device,
                )

                elapsed_ms = (
                    time.perf_counter()
                    - started
                ) * 1000.0

                prediction = predictions[0]

                detections = []

                if prediction.boxes is not None:
                    for box in prediction.boxes:
                        cls_id = int(
                            box.cls[0].item()
                        )

                        confidence = float(
                            box.conf[0].item()
                        )

                        x1, y1, x2, y2 = (
                            box.xyxy[0]
                            .detach()
                            .cpu()
                            .tolist()
                        )

                        label = self.model_names.get(
                            cls_id,
                            str(cls_id),
                        )

                        detections.append(
                            {
                                "class_id": cls_id,
                                "class": str(label),
                                "confidence": confidence,
                                "bbox_px": (
                                    float(x1),
                                    float(y1),
                                    float(x2),
                                    float(y2),
                                ),
                                "center_px": (
                                    float(
                                        (x1 + x2) / 2.0
                                    ),
                                    float(
                                        (y1 + y2) / 2.0
                                    ),
                                ),
                            }
                        )

                result = {
                    "frame_id": int(frame_id),
                    "detections": detections,
                    "inference_ms": float(
                        elapsed_ms
                    ),
                    "timestamp": float(
                        time.monotonic()
                    ),
                }

                with self.result_lock:
                    self.latest_result = result
                    self.last_inference_ms = (
                        elapsed_ms
                    )
                    self.last_result_time = (
                        time.monotonic()
                    )

                self.last_consumed_frame_id = (
                    frame_id
                )

                self.status = "running"
                self.last_error = None

            except Exception as error:
                self.status = "error"
                self.last_error = str(error)

                self.log(
                    f"YOLO inference error: {error}"
                )

                time.sleep(0.20)

    def shutdown(self):
        self.running = False
        self.enabled = False
        self.model = None

# ============================================================
# CAMERA PANEL
# ============================================================

class ModulePanel:
    title = "Top Camera + ArUco"

    # --------------------------------------------------------
    # Main scene markers
    # --------------------------------------------------------

    ARUCO_DICTIONARY = aruco.DICT_4X4_50

    WORKSPACE_IDS = [0, 1, 2, 3]
    ROBOT_BASE_ID = 4
    CONTAINER_IDS = [5, 6, 7, 19]
    TARGET_ID = 8

    SCENE_IDS = [
        0, 1, 2, 3,
        4,
        5, 6, 7, 19,
        8,
    ]

    REFERENCE_ID = 0
    # Physical size of the workspace reference marker (ID0)
    REFERENCE_MARKER_SIZE_MM = 31.5

    # --------------------------------------------------------
    # Camera calibration board
    # --------------------------------------------------------

    CALIBRATION_IDS = list(range(10, 19))

    CALIB_MARKER_SIZE_MM = 30.0
    CALIB_GAP_MM = 14.5
    CALIB_STEP_MM = CALIB_MARKER_SIZE_MM + CALIB_GAP_MM
    CALIB_BOARD_SIZE_MM = (
        3 * CALIB_MARKER_SIZE_MM
        + 2 * CALIB_GAP_MM
    )

    CALIB_MIN_VISIBLE_MARKERS = 5
    CALIB_RECOMMENDED_SAMPLES = 20

    FIXABLE_IDS = [
        0, 1, 2, 3,
        4,
        5, 6, 7, 19,
    ]

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    UI_UPDATE_MS = 33
    SCENE_PROCESS_INTERVAL = 1.0 / 30.0

    # Marker memory is useful when a fixed scene marker is briefly hidden,
    # but it should not live forever in Live mode.
    MARKER_STALE_TIMEOUT_S = 2.0

    CONTAINER_SIZE_MM = 50.0
    CONTAINER_EXCLUSION_MARGIN_MM = 12.0

    def __init__(self, parent, app_context):
        self.app_context = app_context
        self.frame = ttk.Frame(parent, padding=10)

        # ====================================================
        # Capture
        # ====================================================

        self.capture = CameraCapture(
            log=self.app_context.log,
        )

        self.yolo = YoloDetectionService(
            log=self.app_context.log,
        )

        self.last_processed_raw_time = 0.0
        self.last_display_frame = None

        # ====================================================
        # ArUco
        # ====================================================

        dictionary = aruco.getPredefinedDictionary(
            self.ARUCO_DICTIONARY
        )
        parameters = aruco.DetectorParameters()

        self.detector = aruco.ArucoDetector(
            dictionary,
            parameters,
        )

        self.detected_markers = {}
        self.marker_memory = {}

        # ====================================================
        # Homography / scene
        # ====================================================

        self.live_H = None

        self.fixed_H = None
        self.fixed_marker_world = {}
        self.fixed_base_angle_deg = None

        self.last_workspace_polygon_world = None
        self.last_robot_base_world = None
        self.last_robot_base_marker_angle = None

        # ====================================================
        # Mouse target
        # ====================================================

        self.selected_target_pixel = None
        self.selected_target_world = None

        # ====================================================
        # Lens calibration
        # ====================================================

        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_image_size = None
        self.calibration_rms = None
        self.calibration_mean_error = None

        self.new_camera_matrix = None
        self.undistort_map1 = None
        self.undistort_map2 = None
        self.undistort_map_size = None

        # Captured calibration frames
        self.calibration_object_points = []
        self.calibration_image_points = []
        self.calibration_sample_info = []

        self.calibration_window = None
        self.calibration_panel = None

        # ====================================================
        # GUI vars
        # ====================================================

        self.camera_index_var = tk.IntVar(value=0)
        self.marker_mode_var = tk.StringVar(value="Live")
        self.enable_undistortion_var = tk.BooleanVar(value=True)

        self.camera_status_var = tk.StringVar(value="Camera: stopped")
        self.metric_status_var = tk.StringVar(value="Metric map: —")
        self.mode_status_var = tk.StringVar(value="Mode: LIVE")
        self.fixed_status_var = tk.StringVar(value="Fixed markers: none")
        self.lens_status_var = tk.StringVar(value="Lens: not calibrated")

        self.workspace_status_var = tk.StringVar(value="Workspace: —")
        self.base_status_var = tk.StringVar(value="ID4 Base: —")
        self.base_angle_var = tk.StringVar(value="ID4 heading: —")
        self.target8_var = tk.StringVar(value="ID8 Target: —")

        self.mouse_world_var = tk.StringVar(value="Mouse world: —")
        self.target_world_var = tk.StringVar(value="Mouse target: —")

        # YOLO
        self.yolo_model_path_var = tk.StringVar(
            value=str(
                self.app_context.shared_data.get(
                    "yolo_model_path",
                    "",
                )
            )
        )
        self.yolo_enabled_var = tk.BooleanVar(
            value=bool(
                self.app_context.shared_data.get(
                    "yolo_enabled",
                    False,
                )
            )
        )
        self.yolo_confidence_var = tk.DoubleVar(
            value=float(
                self.app_context.shared_data.get(
                    "yolo_confidence",
                    0.40,
                )
            )
        )
        self.yolo_fps_var = tk.DoubleVar(
            value=float(
                self.app_context.shared_data.get(
                    "yolo_max_fps",
                    10.0,
                )
            )
        )
        self.yolo_status_var = tk.StringVar(
            value="YOLO: no model"
        )
        self.yolo_detection_var = tk.StringVar(
            value="Detections: —"
        )
        self.show_exclusion_zones_var = tk.BooleanVar(value=False)

        self.latest_yolo_detections = []

        self.marker_vars = {
            marker_id: {
                "state": tk.StringVar(value="no"),
                "world": tk.StringVar(value="—"),
            }
            for marker_id in self.SCENE_IDS
        }

        # Video canvas state
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.displayed_w = 1
        self.displayed_h = 1

        self.photo = None
        self.canvas_image_id = None

        # ====================================================
        # Shared data canonical keys
        # ====================================================

        shared = self.app_context.shared_data

        shared.setdefault("workspace_polygon_world", None)
        shared.setdefault("workspace_bounds", None)
        shared.setdefault("workspace_camera_shape", None)

        shared.setdefault("robot_base_world", None)
        shared.setdefault("robot_base_marker_angle_deg", None)

        shared.setdefault("containers_world", {})

        shared.setdefault("marker_target_world", None)

        shared.setdefault("target_world", None)
        shared.setdefault("target_inside_workspace", False)

        shared.setdefault("yolo_model_path", "")
        shared.setdefault("yolo_enabled", False)
        shared.setdefault("yolo_confidence", 0.40)
        shared.setdefault("yolo_max_fps", 10.0)

        shared.setdefault("object_detections", [])
        shared.setdefault("cube_detections", [])
        shared.setdefault("hand_detections", [])
        shared.setdefault("hand_detected", False)
        shared.setdefault("hand_inside_workspace", False)
        shared.setdefault("yolo_inference_ms", None)
        shared.setdefault("container_exclusion_zones_world", {})

        # ====================================================

        self.load_camera_calibration()
        self.load_fixed_coordinates()
        self.build_ui()

        if self.fixed_H is not None:
            self.update_workspace()
            self.update_robot_base()
            self.update_containers()

        remembered_model = (
            self.yolo_model_path_var.get().strip()
        )

        if remembered_model:
            ok, message = self.yolo.load_model(
                remembered_model
            )

            if ok:
                self.apply_yolo_settings()

                if self.yolo_enabled_var.get():
                    self.yolo.set_enabled(
                        True
                    )
            else:
                self.yolo_enabled_var.set(
                    False
                )
                self.app_context.log(
                    message
                )

        self.frame.after(
            self.UI_UPDATE_MS,
            self.update_loop,
        )

        self.app_context.log(
            "Camera Panel loaded."
        )

    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):
        header = ttk.Frame(self.frame)
        header.pack(fill="x", pady=(0, 8))

        ttk.Label(
            header,
            text="CAMERA + ARUCO",
            font=("Segoe UI", 18, "bold"),
        ).pack(side="left")

        ttk.Label(
            header,
            textvariable=self.camera_status_var,
        ).pack(side="right")

        camera_bar = ttk.LabelFrame(
            self.frame,
            text="Camera",
            padding=8,
        )
        camera_bar.pack(fill="x", pady=(0, 8))

        ttk.Label(camera_bar, text="Index").pack(side="left")

        ttk.Entry(
            camera_bar,
            textvariable=self.camera_index_var,
            width=5,
        ).pack(side="left", padx=4)

        ttk.Button(
            camera_bar,
            text="Start",
            command=self.start_camera,
        ).pack(side="left", padx=3)

        ttk.Button(
            camera_bar,
            text="Stop",
            command=self.stop_camera,
        ).pack(side="left", padx=3)

        ttk.Button(
            camera_bar,
            text="Clear Scene",
            command=self.clear_scene,
        ).pack(side="left", padx=3)

        ttk.Checkbutton(
            camera_bar,
            text="Lens correction",
            variable=self.enable_undistortion_var,
            command=self.on_undistortion_changed,
        ).pack(side="left", padx=12)

        ttk.Button(
            camera_bar,
            text="Camera Calibration",
            command=self.open_calibration_window,
        ).pack(side="left", padx=5)

        ttk.Label(
            camera_bar,
            textvariable=self.metric_status_var,
        ).pack(side="right", padx=8)

        ttk.Label(
            camera_bar,
            textvariable=self.lens_status_var,
        ).pack(side="right", padx=8)

        body = ttk.PanedWindow(
            self.frame,
            orient="horizontal",
        )
        body.pack(fill="both", expand=True)

        video_frame = ttk.Frame(body)
        sidebar_outer = ttk.Frame(body)

        body.add(video_frame, weight=5)
        body.add(sidebar_outer, weight=2)

        self.video_canvas = tk.Canvas(
            video_frame,
            bg="black",
            highlightthickness=0,
        )
        self.video_canvas.pack(fill="both", expand=True)

        self.video_canvas.bind(
            "<Motion>",
            self.on_mouse_move,
        )
        self.video_canvas.bind(
            "<Button-1>",
            self.on_mouse_click,
        )

        self._build_sidebar(sidebar_outer)

    def _build_sidebar(self, parent):
        canvas = tk.Canvas(
            parent,
            width=390,
            highlightthickness=0,
        )

        scrollbar = ttk.Scrollbar(
            parent,
            orient="vertical",
            command=canvas.yview,
        )

        sidebar = ttk.Frame(
            canvas,
            padding=8,
        )

        window_id = canvas.create_window(
            (0, 0),
            window=sidebar,
            anchor="nw",
        )

        sidebar.bind(
            "<Configure>",
            lambda event: canvas.configure(
                scrollregion=canvas.bbox("all")
            ),
        )

        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(
                window_id,
                width=event.width,
            ),
        )

        canvas.configure(
            yscrollcommand=scrollbar.set
        )

        canvas.pack(
            side="left",
            fill="both",
            expand=True,
        )

        scrollbar.pack(
            side="right",
            fill="y",
        )

        # --------------------------------------------------
        # Scene Coordinates
        # --------------------------------------------------

        mode = ttk.LabelFrame(
            sidebar,
            text="Scene Coordinates",
            padding=8,
        )
        mode.pack(fill="x", pady=(0, 8))

        ttk.Radiobutton(
            mode,
            text="Live markers",
            value="Live",
            variable=self.marker_mode_var,
            command=self.on_marker_mode_changed,
        ).pack(anchor="w")

        ttk.Radiobutton(
            mode,
            text="Fixed workspace / base / containers",
            value="Fixed",
            variable=self.marker_mode_var,
            command=self.on_marker_mode_changed,
        ).pack(anchor="w")

        ttk.Separator(mode).pack(
            fill="x",
            pady=6,
        )

        ttk.Button(
            mode,
            text="Fix Current Coordinates",
            command=self.fix_coordinates,
        ).pack(fill="x", pady=2)

        ttk.Button(
            mode,
            text="Update Fixed Coordinates",
            command=self.fix_coordinates,
        ).pack(fill="x", pady=2)

        ttk.Button(
            mode,
            text="Update Container Positions",
            command=self.update_container_positions_only,
        ).pack(fill="x", pady=2)

        ttk.Button(
            mode,
            text="Clear Fixed Coordinates",
            command=self.clear_fixed_coordinates,
        ).pack(fill="x", pady=2)

        ttk.Label(
            mode,
            textvariable=self.mode_status_var,
        ).pack(anchor="w", pady=(6, 0))

        ttk.Label(
            mode,
            textvariable=self.fixed_status_var,
            wraplength=350,
        ).pack(anchor="w")

        ttk.Label(
            mode,
            text="ID8 always stays live.",
        ).pack(anchor="w", pady=(5, 0))

        # --------------------------------------------------
        # Scene
        # --------------------------------------------------

        scene = ttk.LabelFrame(
            sidebar,
            text="Scene",
            padding=8,
        )
        scene.pack(fill="x", pady=(0, 8))

        for variable in [
            self.workspace_status_var,
            self.base_status_var,
            self.base_angle_var,
            self.target8_var,
        ]:
            ttk.Label(
                scene,
                textvariable=variable,
                wraplength=350,
            ).pack(anchor="w", pady=1)

        # --------------------------------------------------
        # YOLO
        # --------------------------------------------------

        yolo_box = ttk.LabelFrame(
            sidebar,
            text="YOLO Object Detection",
            padding=8,
        )
        yolo_box.pack(fill="x", pady=(0, 8))

        ttk.Label(
            yolo_box,
            textvariable=self.yolo_model_path_var,
            wraplength=350,
        ).pack(anchor="w")

        model_row = ttk.Frame(yolo_box)
        model_row.pack(fill="x", pady=(5, 3))

        ttk.Button(
            model_row,
            text="Browse Model",
            command=self.browse_yolo_model,
        ).pack(side="left", padx=(0, 3))

        ttk.Button(
            model_row,
            text="Load",
            command=self.load_yolo_model,
        ).pack(side="left", padx=3)

        ttk.Button(
            model_row,
            text="Unload",
            command=self.unload_yolo_model,
        ).pack(side="left", padx=3)

        ttk.Checkbutton(
            yolo_box,
            text="Enable detection",
            variable=self.yolo_enabled_var,
            command=self.on_yolo_enabled_changed,
        ).pack(anchor="w", pady=(4, 3))

        settings_row = ttk.Frame(yolo_box)
        settings_row.pack(fill="x", pady=2)

        ttk.Label(
            settings_row,
            text="Confidence:",
        ).pack(side="left")

        conf_entry = ttk.Entry(
            settings_row,
            textvariable=self.yolo_confidence_var,
            width=7,
        )
        conf_entry.pack(side="left", padx=(4, 10))
        conf_entry.bind(
            "<Return>",
            lambda event: self.apply_yolo_settings(),
        )

        ttk.Label(
            settings_row,
            text="Max FPS:",
        ).pack(side="left")

        fps_entry = ttk.Entry(
            settings_row,
            textvariable=self.yolo_fps_var,
            width=6,
        )
        fps_entry.pack(side="left", padx=4)
        fps_entry.bind(
            "<Return>",
            lambda event: self.apply_yolo_settings(),
        )

        ttk.Button(
            yolo_box,
            text="Apply Settings",
            command=self.apply_yolo_settings,
        ).pack(fill="x", pady=(4, 3))

        ttk.Label(
            yolo_box,
            textvariable=self.yolo_status_var,
            wraplength=350,
        ).pack(anchor="w")

        ttk.Label(
            yolo_box,
            textvariable=self.yolo_detection_var,
            wraplength=350,
        ).pack(anchor="w")

        ttk.Checkbutton(
            yolo_box,
            text="Show container exclusion zones",
            variable=self.show_exclusion_zones_var,
        ).pack(anchor="w", pady=(4, 0))

        # --------------------------------------------------
        # Mouse Target
        # --------------------------------------------------

        mouse = ttk.LabelFrame(
            sidebar,
            text="Mouse Target",
            padding=8,
        )
        mouse.pack(fill="x", pady=(0, 8))

        ttk.Label(
            mouse,
            textvariable=self.mouse_world_var,
        ).pack(anchor="w")

        ttk.Label(
            mouse,
            textvariable=self.target_world_var,
        ).pack(anchor="w")

        ttk.Button(
            mouse,
            text="Clear Mouse Target",
            command=self.clear_mouse_target,
        ).pack(fill="x", pady=(5, 0))

        # --------------------------------------------------
        # Marker table
        # --------------------------------------------------

        table = ttk.LabelFrame(
            sidebar,
            text="Markers",
            padding=8,
        )
        table.pack(fill="x")

        ttk.Label(
            table,
            text="ID",
            font=("Segoe UI", 9, "bold"),
        ).grid(row=0, column=0, padx=4, pady=3)

        ttk.Label(
            table,
            text="State",
            font=("Segoe UI", 9, "bold"),
        ).grid(row=0, column=1, padx=4, pady=3)

        ttk.Label(
            table,
            text="World X/Y",
            font=("Segoe UI", 9, "bold"),
        ).grid(row=0, column=2, padx=4, pady=3)

        for row, marker_id in enumerate(
            self.SCENE_IDS,
            start=1,
        ):
            ttk.Label(
                table,
                text=str(marker_id),
            ).grid(row=row, column=0, padx=4, pady=2)

            ttk.Label(
                table,
                textvariable=self.marker_vars[
                    marker_id
                ]["state"],
            ).grid(row=row, column=1, padx=4)

            ttk.Label(
                table,
                textvariable=self.marker_vars[
                    marker_id
                ]["world"],
            ).grid(row=row, column=2, padx=4)

    # ========================================================
    # YOLO control
    # ========================================================

    def browse_yolo_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO model",
            filetypes=[
                ("PyTorch model", "*.pt"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        self.yolo_model_path_var.set(
            path
        )

    def load_yolo_model(self):
        path = self.yolo_model_path_var.get().strip()

        if not path:
            self.yolo_status_var.set(
                "YOLO: select a model first"
            )
            return

        ok, message = self.yolo.load_model(
            path
        )

        self.app_context.log(
            message
        )

        if not ok:
            self.yolo_status_var.set(
                "YOLO: ERROR"
            )
            return

        self.apply_yolo_settings()

        enabled = bool(
            self.yolo_enabled_var.get()
        )

        self.yolo.set_enabled(
            enabled
        )

        shared = self.app_context.shared_data
        shared["yolo_model_path"] = path
        shared["yolo_enabled"] = enabled

        self.yolo_status_var.set(
            message
        )

    def unload_yolo_model(self):
        self.yolo.unload_model()
        self.yolo_enabled_var.set(
            False
        )

        self.latest_yolo_detections = []

        shared = self.app_context.shared_data
        shared["yolo_enabled"] = False
        shared["object_detections"] = []
        shared["cube_detections"] = []
        shared["hand_detections"] = []
        shared["hand_detected"] = False
        shared["hand_inside_workspace"] = False
        shared["yolo_inference_ms"] = None

        self.yolo_status_var.set(
            "YOLO: no model"
        )

        self.yolo_detection_var.set(
            "Detections: —"
        )

    def apply_yolo_settings(self):
        try:
            confidence = float(
                self.yolo_confidence_var.get()
            )
            max_fps = float(
                self.yolo_fps_var.get()
            )

            self.yolo.set_confidence(
                confidence
            )
            self.yolo.set_max_fps(
                max_fps
            )

            # Snap UI to effective clamped values.
            self.yolo_confidence_var.set(
                self.yolo.confidence
            )
            self.yolo_fps_var.set(
                self.yolo.max_fps
            )

            shared = self.app_context.shared_data
            shared["yolo_confidence"] = (
                self.yolo.confidence
            )
            shared["yolo_max_fps"] = (
                self.yolo.max_fps
            )

        except Exception as error:
            self.yolo_status_var.set(
                f"YOLO settings error: {error}"
            )

    def on_yolo_enabled_changed(self):
        enabled = bool(
            self.yolo_enabled_var.get()
        )

        if enabled and self.yolo.model is None:
            self.yolo_enabled_var.set(
                False
            )
            self.yolo_status_var.set(
                "YOLO: load a model first"
            )
            return

        self.apply_yolo_settings()
        self.yolo.set_enabled(
            enabled
        )

        self.app_context.shared_data[
            "yolo_enabled"
        ] = enabled

        if not enabled:
            self.publish_yolo_detections(
                []
            )

    def process_yolo_result(self):
        result = self.yolo.get_result()
        if result is None:
            return

        detections = []
        H = self.get_active_H()

        for item in result.get("detections", []):
            detection = dict(item)
            cx, cy = detection["center_px"]
            world = None
            if H is not None:
                try:
                    mapped = self.pixel_to_world((cx, cy), H)
                    world = (float(mapped[0]), float(mapped[1]))
                except Exception:
                    pass

            detection["world"] = world
            detection["inside_workspace"] = (False if world is None else self.point_inside_workspace(world[0], world[1]))
            label = str(detection.get("class", ""))
            detection["intersects_workspace"] = (
                self.bbox_intersects_workspace(detection.get("bbox_px"), H)
                if label == "hand" else detection["inside_workspace"]
            )
            detection["inside_container_exclusion"] = (
                self.cube_bbox_in_container_exclusion(detection.get("bbox_px"), H)
                if label.startswith("cube_") else False
            )
            detections.append(detection)

        self.latest_yolo_detections = detections
        self.publish_yolo_detections(detections, inference_ms=result.get("inference_ms"))

    def publish_yolo_detections(self, detections, inference_ms=None):
        shared = self.app_context.shared_data
        object_detections = [dict(item) for item in detections]
        cube_detections = [
            dict(item) for item in detections
            if str(item.get("class", "")).startswith("cube_")
            and bool(item.get("inside_workspace", False))
            and not bool(item.get("inside_container_exclusion", False))
        ]
        hand_detections = [dict(item) for item in detections if item.get("class") == "hand"]
        shared["object_detections"] = object_detections
        shared["cube_detections"] = cube_detections
        shared["hand_detections"] = hand_detections
        shared["hand_detected"] = bool(hand_detections)
        shared["hand_inside_workspace"] = any(bool(item.get("intersects_workspace", False)) for item in hand_detections)
        shared["yolo_inference_ms"] = None if inference_ms is None else float(inference_ms)

    def draw_yolo_detections(
        self,
        frame,
    ):
        if not self.yolo_enabled_var.get():
            return

        class_colors = {
            # OpenCV uses BGR.
            "cube_red": (0, 0, 255),
            "cube_green": (0, 200, 0),
            "cube_blue": (255, 0, 0),
            "cube_yellow": (0, 220, 255),
            "hand": (180, 0, 180),
        }

        frame_h, frame_w = frame.shape[:2]

        for detection in self.latest_yolo_detections:
            x1, y1, x2, y2 = detection[
                "bbox_px"
            ]

            p1 = (
                int(round(x1)),
                int(round(y1)),
            )
            p2 = (
                int(round(x2)),
                int(round(y2)),
            )

            label = str(
                detection.get(
                    "class",
                    "?",
                )
            )

            confidence = float(
                detection.get(
                    "confidence",
                    0.0,
                )
            )

            inside = bool(
                detection.get(
                    "inside_workspace",
                    False,
                )
            )

            color = class_colors.get(
                label,
                (255, 255, 255),
            )

            cv2.rectangle(
                frame,
                p1,
                p2,
                color,
                2,
            )

            text = (
                f"{label} "
                f"{confidence:.2f}"
            )

            if inside:
                text += " | IN"
            if bool(detection.get("inside_container_exclusion", False)):
                text += " | IGNORE"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.44
            thickness = 1

            (
                (text_w, text_h),
                baseline,
            ) = cv2.getTextSize(
                text,
                font,
                font_scale,
                thickness,
            )

            text_x = max(
                0,
                min(
                    frame_w - text_w - 6,
                    p1[0],
                ),
            )

            # Prefer above the box. If there is no room,
            # place the label just inside/below the top edge.
            if p1[1] - text_h - baseline - 8 >= 0:
                bg_y1 = (
                    p1[1]
                    - text_h
                    - baseline
                    - 8
                )
                bg_y2 = p1[1] - 2
                text_y = p1[1] - baseline - 5
            else:
                bg_y1 = p1[1] + 2
                bg_y2 = (
                    p1[1]
                    + text_h
                    + baseline
                    + 8
                )
                text_y = (
                    p1[1]
                    + text_h
                    + 5
                )

            bg_y1 = max(
                0,
                min(frame_h - 1, bg_y1),
            )
            bg_y2 = max(
                0,
                min(frame_h - 1, bg_y2),
            )

            cv2.rectangle(
                frame,
                (
                    text_x,
                    bg_y1,
                ),
                (
                    min(
                        frame_w - 1,
                        text_x + text_w + 6,
                    ),
                    bg_y2,
                ),
                color,
                -1,
            )

            # Black text remains readable on all chosen class colors.
            cv2.putText(
                frame,
                text,
                (
                    text_x + 3,
                    text_y,
                ),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

            world = detection.get(
                "world"
            )

            if world is not None:
                cx, cy = detection[
                    "center_px"
                ]

                cv2.circle(
                    frame,
                    (
                        int(round(cx)),
                        int(round(cy)),
                    ),
                    4,
                    color,
                    -1,
                )


        if self.show_exclusion_zones_var.get():
            self.draw_container_exclusion_zones(frame)

    # ========================================================
    # Camera control
    # ========================================================

    def start_camera(self):
        ok, message = self.capture.start(
            self.camera_index_var.get()
        )

        self.camera_status_var.set(
            "Camera: running"
            if ok
            else "Camera: ERROR"
        )

        self.app_context.log(
            message
        )

    def stop_camera(self):
        self.capture.stop()
        self.camera_status_var.set(
            "Camera: stopped"
        )

    # ========================================================
    # Scene processing
    # ========================================================

    def process_latest_frame(self):
        raw = self.capture.get_frame()

        if raw is None:
            return

        frame = self.apply_undistortion(
            raw
        )

        # YOLO uses the same corrected image geometry as ArUco,
        # so bbox centers and homography coordinates stay consistent.
        self.yolo.submit(
            frame
        )

        display = frame.copy()

        corners, ids, _ = self.detector.detectMarkers(
            frame
        )

        now = time.monotonic()
        detected = {}

        if ids is not None:
            aruco.drawDetectedMarkers(
                display,
                corners,
                ids,
            )

            for index, marker_id in enumerate(
                ids.flatten()
            ):
                marker_id = int(marker_id)

                marker_corners = np.asarray(
                    corners[index][0],
                    dtype=np.float32,
                )

                center = np.mean(
                    marker_corners,
                    axis=0,
                )

                marker_data = {
                    "corners": marker_corners,
                    "center": center,
                    "time": now,
                }

                detected[marker_id] = marker_data

                if marker_id in self.SCENE_IDS:
                    self.marker_memory[
                        marker_id
                    ] = marker_data

        self.detected_markers = detected

        self._expire_stale_markers(now)

        self.update_live_homography()
        self.update_workspace()
        self.update_robot_base()
        self.update_id8_target()
        self.update_containers()

        self.draw_workspace(display)
        self.draw_base_heading(display)
        self.draw_id8(display)
        self.draw_mouse_target(display)

        self.process_yolo_result()
        self.draw_yolo_detections(
            display
        )

        calibration_count = sum(
            1
            for marker_id in self.CALIBRATION_IDS
            if marker_id in detected
        )

        if calibration_count:
            cv2.putText(
                display,
                f"CALIB BOARD: {calibration_count}/9",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2,
            )

        self.last_display_frame = display

    def _expire_stale_markers(self, now):
        if self.marker_mode_var.get() == "Fixed":
            # Fixed coordinates intentionally survive occlusion.
            return

        stale_ids = []

        for marker_id, data in self.marker_memory.items():
            if marker_id == self.TARGET_ID:
                # ID8 is never read from memory anyway.
                continue

            age = now - float(
                data.get("time", now)
            )

            if age > self.MARKER_STALE_TIMEOUT_S:
                stale_ids.append(marker_id)

        for marker_id in stale_ids:
            self.marker_memory.pop(
                marker_id,
                None,
            )

    # ========================================================
    # Lens calibration
    # ========================================================

    def load_camera_calibration(self):
        if not CAMERA_CALIBRATION_FILE.exists():
            self.camera_matrix = None
            self.dist_coeffs = None
            self.calibration_rms = None
            self.calibration_mean_error = None
            return

        try:
            data = np.load(
                CAMERA_CALIBRATION_FILE
            )

            self.camera_matrix = np.asarray(
                data["camera_matrix"],
                dtype=np.float64,
            )

            self.dist_coeffs = np.asarray(
                data["dist_coeffs"],
                dtype=np.float64,
            )

            if (
                "image_width" in data
                and "image_height" in data
            ):
                self.calibration_image_size = (
                    int(data["image_width"]),
                    int(data["image_height"]),
                )

            if "rms_error" in data:
                self.calibration_rms = float(
                    data["rms_error"]
                )

            if "mean_reprojection_error" in data:
                self.calibration_mean_error = float(
                    data["mean_reprojection_error"]
                )

        except Exception as error:
            self.camera_matrix = None
            self.dist_coeffs = None

            self.app_context.log(
                "Camera calibration load error: "
                f"{error}"
            )

    def update_lens_status(self):
        if (
            self.camera_matrix is None
            or self.dist_coeffs is None
        ):
            self.lens_status_var.set(
                "Lens: not calibrated"
            )
            return

        state = (
            "ON"
            if self.enable_undistortion_var.get()
            else "OFF"
        )

        if self.calibration_rms is None:
            self.lens_status_var.set(
                f"Lens correction: {state}"
            )
        else:
            self.lens_status_var.set(
                f"Lens: {state} | "
                f"RMS {self.calibration_rms:.3f}px"
            )

    def on_undistortion_changed(self):
        # Fixed coordinates were calculated in a specific image geometry.
        self.clear_fixed_coordinates()
        self.update_lens_status()

    def reset_undistortion_maps(self):
        self.new_camera_matrix = None
        self.undistort_map1 = None
        self.undistort_map2 = None
        self.undistort_map_size = None

    def apply_undistortion(self, frame):
        if not self.enable_undistortion_var.get():
            return frame

        if (
            self.camera_matrix is None
            or self.dist_coeffs is None
        ):
            return frame

        h, w = frame.shape[:2]
        size = (w, h)

        if (
            self.undistort_map1 is None
            or self.undistort_map_size != size
        ):
            self.build_undistortion_maps(
                w,
                h,
            )

        if self.undistort_map1 is None:
            return frame

        return cv2.remap(
            frame,
            self.undistort_map1,
            self.undistort_map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def build_undistortion_maps(
        self,
        width,
        height,
    ):
        size = (width, height)

        try:
            new_matrix, _ = (
                cv2.getOptimalNewCameraMatrix(
                    self.camera_matrix,
                    self.dist_coeffs,
                    size,
                    1.0,
                    size,
                )
            )

            map1, map2 = (
                cv2.initUndistortRectifyMap(
                    self.camera_matrix,
                    self.dist_coeffs,
                    None,
                    new_matrix,
                    size,
                    cv2.CV_16SC2,
                )
            )

            self.new_camera_matrix = new_matrix
            self.undistort_map1 = map1
            self.undistort_map2 = map2
            self.undistort_map_size = size

        except Exception as error:
            self.app_context.log(
                f"Undistortion map error: {error}"
            )
            self.reset_undistortion_maps()

    # ========================================================
    # Homography
    # ========================================================

    def update_live_homography(self):
        marker = self.marker_memory.get(
            self.REFERENCE_ID
        )

        if marker is None:
            self.live_H = None
            return

        corners = np.asarray(
            marker["corners"],
            dtype=np.float32,
        )

        half = (
            self.REFERENCE_MARKER_SIZE_MM
            / 2.0
        )

        metric = np.asarray(
            [
                [-half, -half],
                [ half, -half],
                [ half,  half],
                [-half,  half],
            ],
            dtype=np.float32,
        )

        self.live_H = cv2.getPerspectiveTransform(
            corners,
            metric,
        )

    def get_active_H(self):
        if (
            self.marker_mode_var.get() == "Fixed"
            and self.fixed_H is not None
        ):
            return self.fixed_H

        return self.live_H

    @staticmethod
    def pixel_to_world(point, H):
        array = np.asarray(
            [[[
                float(point[0]),
                float(point[1]),
            ]]],
            dtype=np.float32,
        )

        result = cv2.perspectiveTransform(
            array,
            H,
        )

        return result[0][0]

    def get_marker_world(self, marker_id):
        # ID8 is strictly live.
        if marker_id == self.TARGET_ID:
            marker = self.detected_markers.get(
                marker_id
            )

            H = self.get_active_H()

            if marker is None or H is None:
                return None

            result = self.pixel_to_world(
                marker["center"],
                H,
            )

            return (
                float(result[0]),
                float(result[1]),
            )

        if self.marker_mode_var.get() == "Fixed":
            fixed = self.fixed_marker_world.get(
                marker_id
            )

            if fixed is not None:
                return fixed

        marker = self.marker_memory.get(
            marker_id
        )

        if marker is None:
            return None

        H = self.live_H

        if H is None:
            return None

        result = self.pixel_to_world(
            marker["center"],
            H,
        )

        return (
            float(result[0]),
            float(result[1]),
        )

    # ========================================================
    # Persistent fixed scene
    # ========================================================

    def save_fixed_coordinates(self):
        if self.fixed_H is None or not self.fixed_marker_world:
            return False
        try:
            FIXED_SCENE_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "undistortion_enabled": bool(self.enable_undistortion_var.get()),
                "fixed_H": np.asarray(self.fixed_H, dtype=float).tolist(),
                "fixed_marker_world": {str(k): [float(v[0]), float(v[1])] for k, v in self.fixed_marker_world.items()},
                "fixed_base_angle_deg": None if self.fixed_base_angle_deg is None else float(self.fixed_base_angle_deg),
            }
            FIXED_SCENE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.app_context.log(f"Fixed camera scene saved: {FIXED_SCENE_FILE.name}")
            return True
        except Exception as error:
            self.app_context.log(f"Fixed scene save error: {error}")
            return False

    def load_fixed_coordinates(self):
        if not FIXED_SCENE_FILE.exists():
            return False
        try:
            payload = json.loads(FIXED_SCENE_FILE.read_text(encoding="utf-8"))
            if bool(payload.get("undistortion_enabled", True)) != bool(self.enable_undistortion_var.get()):
                self.app_context.log("Saved fixed scene not loaded: lens-correction mode differs.")
                return False
            H = np.asarray(payload["fixed_H"], dtype=np.float64)
            if H.shape != (3, 3):
                raise ValueError("fixed_H must be 3x3")
            markers = {}
            for key, value in payload.get("fixed_marker_world", {}).items():
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    markers[int(key)] = (float(value[0]), float(value[1]))
            if not markers:
                raise ValueError("no saved fixed markers")
            self.fixed_H = H
            self.fixed_marker_world = markers
            angle = payload.get("fixed_base_angle_deg")
            self.fixed_base_angle_deg = None if angle is None else float(angle)
            self.marker_mode_var.set("Fixed")
            self.mode_status_var.set("Mode: FIXED")
            ids_text = ", ".join(str(i) for i in sorted(markers))
            self.fixed_status_var.set(f"Fixed markers: {ids_text} | loaded from disk")
            self.app_context.log(f"Fixed camera scene loaded: {FIXED_SCENE_FILE.name}")
            return True
        except Exception as error:
            self.app_context.log(f"Fixed scene load error: {error}")
            self.fixed_H = None
            self.fixed_marker_world = {}
            self.fixed_base_angle_deg = None
            return False

    # ========================================================
    # Fixed coordinates
    # ========================================================

    def fix_coordinates(self):
        if self.live_H is None:
            self.fixed_status_var.set(
                "Cannot fix: ID0 missing"
            )
            return

        fixed = {}

        # Important:
        # calculate values while still using live coordinates.
        previous_mode = self.marker_mode_var.get()
        self.marker_mode_var.set("Live")

        try:
            for marker_id in self.FIXABLE_IDS:
                world = self.get_marker_world(
                    marker_id
                )

                if world is not None:
                    fixed[marker_id] = world
        finally:
            self.marker_mode_var.set(
                previous_mode
            )

        if not fixed:
            self.fixed_status_var.set(
                "No markers available"
            )
            return

        self.fixed_H = self.live_H.copy()
        self.fixed_marker_world = fixed

        self.fixed_base_angle_deg = (
            self.calculate_live_base_angle(
                self.fixed_H
            )
        )

        self.marker_mode_var.set("Fixed")
        self.mode_status_var.set(
            "Mode: FIXED"
        )

        ids_text = ", ".join(
            str(marker_id)
            for marker_id in sorted(fixed)
        )

        self.fixed_status_var.set(
            f"Fixed markers: {ids_text}"
        )

        # Publish immediately using the new fixed scene.
        self.update_workspace()
        self.update_robot_base()
        self.update_containers()
        self.save_fixed_coordinates()

    def update_container_positions_only(self):
        """Refresh only visible container markers without changing workspace/base."""

        H = self.get_active_H()

        if H is None:
            self.fixed_status_var.set(
                "Cannot update containers: metric map missing"
            )
            return

        updated = {}
        missing = []

        for marker_id in self.CONTAINER_IDS:
            marker = self.detected_markers.get(marker_id)

            if marker is None:
                missing.append(marker_id)
                continue

            try:
                mapped = self.pixel_to_world(
                    marker["center"],
                    H,
                )

                updated[marker_id] = (
                    float(mapped[0]),
                    float(mapped[1]),
                )

            except Exception:
                missing.append(marker_id)

        if not updated:
            self.fixed_status_var.set(
                "No visible container markers to update"
            )
            return

        if self.marker_mode_var.get() == "Fixed":
            for marker_id, world in updated.items():
                self.fixed_marker_world[marker_id] = world

        current = dict(
            self.app_context.shared_data.get(
                "containers_world",
                {},
            )
            or {}
        )
        current.update(updated)

        self.app_context.shared_data[
            "containers_world"
        ] = current

        ids_text = ", ".join(
            str(marker_id)
            for marker_id in sorted(updated)
        )

        if missing:
            missing_text = ", ".join(
                str(marker_id)
                for marker_id in missing
            )

            self.fixed_status_var.set(
                f"Containers updated: {ids_text} | "
                f"not visible: {missing_text}"
            )
        else:
            self.fixed_status_var.set(
                f"Containers updated: {ids_text}"
            )

        if self.marker_mode_var.get() == "Fixed":
            self.save_fixed_coordinates()

    def clear_fixed_coordinates(self, delete_saved=True):
        self.fixed_H = None
        self.fixed_marker_world = {}
        self.fixed_base_angle_deg = None

        self.marker_mode_var.set("Live")
        self.mode_status_var.set(
            "Mode: LIVE"
        )
        self.fixed_status_var.set(
            "Fixed markers: none"
        )
        if delete_saved:
            try:
                if FIXED_SCENE_FILE.exists():
                    FIXED_SCENE_FILE.unlink()
            except Exception as error:
                self.app_context.log(f"Fixed scene delete error: {error}")

    def on_marker_mode_changed(self):
        mode = self.marker_mode_var.get()

        if mode == "Fixed" and self.fixed_H is None:
            self.marker_mode_var.set("Live")
            self.mode_status_var.set(
                "Mode: LIVE"
            )
            self.fixed_status_var.set(
                "Fix coordinates first."
            )
            return

        self.mode_status_var.set(
            f"Mode: {mode.upper()}"
        )

    # ========================================================
    # Scene publication
    # ========================================================

    def update_workspace(self):
        polygon = []
        camera_points = []

        for marker_id in self.WORKSPACE_IDS:
            world = self.get_marker_world(
                marker_id
            )

            if world is None:
                self.app_context.shared_data[
                    "workspace_polygon_world"
                ] = None
                self.app_context.shared_data[
                    "workspace_bounds"
                ] = None
                return

            polygon.append(world)

            if self.marker_mode_var.get() == "Fixed" and self.fixed_H is not None:
                pixel = self.world_to_pixel(world, self.fixed_H)
                if pixel is not None:
                    camera_points.append((float(pixel[0]), float(pixel[1])))
            else:
                marker = self.marker_memory.get(marker_id)
                if marker is not None:
                    center = marker["center"]
                    camera_points.append((float(center[0]), float(center[1])))

        self.last_workspace_polygon_world = polygon

        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]

        bounds = {
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(ys),
            "max_y": max(ys),
            "width": max(xs) - min(xs),
            "height": max(ys) - min(ys),
        }

        shared = self.app_context.shared_data

        shared[
            "workspace_polygon_world"
        ] = polygon

        shared[
            "workspace_bounds"
        ] = bounds

        if len(camera_points) == 4:
            shared[
                "workspace_camera_shape"
            ] = camera_points

    def calculate_live_base_angle(self, H=None):
        marker = self.marker_memory.get(
            self.ROBOT_BASE_ID
        )

        if marker is None:
            return None

        if H is None:
            H = self.get_active_H()

        if H is None:
            return None

        center_world = self.pixel_to_world(
            marker["center"],
            H,
        )

        corners = marker["corners"]

        forward_pixel = (
            corners[0]
            + corners[1]
        ) / 2.0

        forward_world = self.pixel_to_world(
            forward_pixel,
            H,
        )

        dx = (
            float(forward_world[0])
            - float(center_world[0])
        )

        dy = (
            float(forward_world[1])
            - float(center_world[1])
        )

        return self.normalize_angle_deg(
            math.degrees(
                math.atan2(
                    dy,
                    dx,
                )
            )
        )

    def update_robot_base(self):
        base = self.get_marker_world(
            self.ROBOT_BASE_ID
        )

        if base is None:
            self.app_context.shared_data[
                "robot_base_world"
            ] = None
            self.app_context.shared_data[
                "robot_base_marker_angle_deg"
            ] = None
            return

        if (
            self.marker_mode_var.get() == "Fixed"
            and self.fixed_base_angle_deg is not None
        ):
            angle = self.fixed_base_angle_deg
        else:
            angle = self.calculate_live_base_angle()

        if angle is None:
            return

        self.last_robot_base_world = base
        self.last_robot_base_marker_angle = angle

        shared = self.app_context.shared_data

        shared[
            "robot_base_world"
        ] = base

        shared[
            "robot_base_marker_angle_deg"
        ] = angle

    def update_id8_target(self):
        target = self.get_marker_world(
            self.TARGET_ID
        )

        self.app_context.shared_data[
            "marker_target_world"
        ] = target

    def update_containers(self):
        world_data = {}

        for marker_id in self.CONTAINER_IDS:
            world = self.get_marker_world(
                marker_id
            )

            if world is not None:
                world_data[
                    marker_id
                ] = world

        self.app_context.shared_data[
            "containers_world"
        ] = world_data

    # ========================================================
    # Workspace utilities
    # ========================================================

    def point_inside_workspace(
        self,
        x,
        y,
    ):
        polygon = self.last_workspace_polygon_world

        if (
            polygon is None
            or len(polygon) < 3
        ):
            return False

        contour = np.asarray(
            polygon,
            dtype=np.float32,
        )

        return (
            cv2.pointPolygonTest(
                contour,
                (float(x), float(y)),
                False,
            )
            >= 0
        )

    def bbox_intersects_workspace(self, bbox_px, H=None):
        if bbox_px is None or self.last_workspace_polygon_world is None:
            return False
        H = H if H is not None else self.get_active_H()
        if H is None:
            return False
        x1, y1, x2, y2 = [float(v) for v in bbox_px]
        samples = [(x1,y1),(x2,y1),(x2,y2),(x1,y2),((x1+x2)/2,y1),(x2,(y1+y2)/2),((x1+x2)/2,y2),(x1,(y1+y2)/2),((x1+x2)/2,(y1+y2)/2)]
        for p in samples:
            try:
                w = self.pixel_to_world(p, H)
                if self.point_inside_workspace(float(w[0]), float(w[1])):
                    return True
            except Exception:
                pass
        try:
            inv_H = np.linalg.inv(H)
            for wx, wy in self.last_workspace_polygon_world:
                p = self.pixel_to_world((wx, wy), inv_H)
                if x1 <= float(p[0]) <= x2 and y1 <= float(p[1]) <= y2:
                    return True
        except Exception:
            pass
        return False

    def get_container_exclusion_zones_world(self):
        containers = self.app_context.shared_data.get("containers_world", {}) or {}
        half = self.CONTAINER_SIZE_MM / 2.0 + self.CONTAINER_EXCLUSION_MARGIN_MM
        zones = {}
        for marker_id in self.CONTAINER_IDS:
            center = containers.get(marker_id)
            if center is None:
                continue
            cx, cy = float(center[0]), float(center[1])
            zones[marker_id] = {"min_x": cx-half, "max_x": cx+half, "min_y": cy-half, "max_y": cy+half}
        self.app_context.shared_data["container_exclusion_zones_world"] = zones
        return zones

    def cube_bbox_in_container_exclusion(self, bbox_px, H=None):
        if bbox_px is None:
            return False
        H = H if H is not None else self.get_active_H()
        if H is None:
            return False
        zones = self.get_container_exclusion_zones_world()
        if not zones:
            return False
        x1, y1, x2, y2 = [float(v) for v in bbox_px]
        samples = [((x1+x2)/2,(y1+y2)/2),(x1,y1),(x2,y1),(x2,y2),(x1,y2),((x1+x2)/2,y1),((x1+x2)/2,y2),(x1,(y1+y2)/2),(x2,(y1+y2)/2)]
        hits = 0; valid = 0; center_hit = False
        for idx, p in enumerate(samples):
            try:
                w = self.pixel_to_world(p, H); wx, wy = float(w[0]), float(w[1]); valid += 1
                hit = any(z["min_x"] <= wx <= z["max_x"] and z["min_y"] <= wy <= z["max_y"] for z in zones.values())
                if hit:
                    hits += 1
                    if idx == 0: center_hit = True
            except Exception:
                pass
        return center_hit or (valid > 0 and hits / valid >= 0.34)

    def world_to_pixel(self, point, H=None):
        H = H if H is not None else self.get_active_H()
        if H is None:
            return None
        try:
            p = self.pixel_to_world(point, np.linalg.inv(H))
            return (int(round(p[0])), int(round(p[1])))
        except Exception:
            return None

    def draw_container_exclusion_zones(self, frame):
        for marker_id, z in self.get_container_exclusion_zones_world().items():
            wp = [(z["min_x"],z["min_y"]),(z["max_x"],z["min_y"]),(z["max_x"],z["max_y"]),(z["min_x"],z["max_y"])]
            points = [self.world_to_pixel(p) for p in wp]
            if any(p is None for p in points):
                continue
            contour = np.asarray(points, dtype=np.int32)
            cv2.polylines(frame, [contour], True, (0,165,255), 2)
            cv2.putText(frame, f"IGNORE ID{marker_id}", tuple(contour[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,165,255), 1, cv2.LINE_AA)

    # ========================================================
    # Frame drawing
    # ========================================================

    def draw_workspace(self, frame):
        points = []
        if self.marker_mode_var.get() == "Fixed" and self.fixed_H is not None:
            for marker_id in self.WORKSPACE_IDS:
                world = self.fixed_marker_world.get(marker_id)
                if world is None:
                    return
                pixel = self.world_to_pixel(world, self.fixed_H)
                if pixel is None:
                    return
                points.append(pixel)
        else:
            for marker_id in self.WORKSPACE_IDS:
                marker = self.marker_memory.get(marker_id)
                if marker is None:
                    return
                points.append(tuple(marker["center"]))
        cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], True, (0,255,0), 3)

    def draw_base_heading(self, frame):
        marker = self.detected_markers.get(
            self.ROBOT_BASE_ID
        )

        if marker is None:
            return

        center = marker["center"]
        corners = marker["corners"]

        forward = (
            corners[0]
            + corners[1]
        ) / 2.0

        cv2.arrowedLine(
            frame,
            tuple(center.astype(int)),
            tuple(forward.astype(int)),
            (255, 0, 255),
            3,
            tipLength=0.25,
        )

    def draw_id8(self, frame):
        marker = self.detected_markers.get(
            self.TARGET_ID
        )

        if marker is None:
            return

        point = tuple(
            marker["center"].astype(int)
        )

        cv2.drawMarker(
            frame,
            point,
            (0, 0, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=35,
            thickness=3,
        )

        cv2.putText(
            frame,
            "TARGET ID8",
            (
                point[0] + 10,
                point[1] - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

    def draw_mouse_target(self, frame):
        if self.selected_target_pixel is None:
            return

        cv2.drawMarker(
            frame,
            self.selected_target_pixel,
            (0, 180, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=25,
            thickness=2,
        )

    # ========================================================
    # Mouse interaction
    # ========================================================

    def display_to_frame_xy(self, event):
        frame = self.last_display_frame

        if frame is None:
            return 0, 0

        frame_h, frame_w = frame.shape[:2]

        x_display = (
            event.x
            - self.display_offset_x
        )
        y_display = (
            event.y
            - self.display_offset_y
        )

        x_display = max(
            0,
            min(
                self.displayed_w - 1,
                x_display,
            ),
        )

        y_display = max(
            0,
            min(
                self.displayed_h - 1,
                y_display,
            ),
        )

        x = int(
            x_display
            * frame_w
            / self.displayed_w
        )

        y = int(
            y_display
            * frame_h
            / self.displayed_h
        )

        return x, y

    def on_mouse_move(self, event):
        H = self.get_active_H()

        if (
            H is None
            or self.last_display_frame is None
        ):
            return

        x, y = self.display_to_frame_xy(
            event
        )

        world = self.pixel_to_world(
            (x, y),
            H,
        )

        self.mouse_world_var.set(
            "Mouse world: "
            f"{float(world[0]):.1f}, "
            f"{float(world[1]):.1f} mm"
        )

    def on_mouse_click(self, event):
        H = self.get_active_H()

        if (
            H is None
            or self.last_display_frame is None
        ):
            return

        x, y = self.display_to_frame_xy(
            event
        )

        world = self.pixel_to_world(
            (x, y),
            H,
        )

        wx = float(world[0])
        wy = float(world[1])

        self.selected_target_pixel = (
            x,
            y,
        )
        self.selected_target_world = (
            wx,
            wy,
        )

        shared = self.app_context.shared_data

        shared[
            "target_world"
        ] = (
            wx,
            wy,
        )

        shared[
            "target_inside_workspace"
        ] = self.point_inside_workspace(
            wx,
            wy,
        )

    def clear_mouse_target(self):
        self.selected_target_pixel = None
        self.selected_target_world = None

        shared = self.app_context.shared_data

        shared["target_world"] = None
        shared["target_inside_workspace"] = False

    # ========================================================
    # Calibration board
    # ========================================================

    def get_calibration_marker_object_points(
        self,
        marker_id,
    ):
        if marker_id not in self.CALIBRATION_IDS:
            return None

        index = marker_id - 10

        row = index // 3
        col = index % 3

        x0 = col * self.CALIB_STEP_MM
        y0 = row * self.CALIB_STEP_MM

        x1 = x0 + self.CALIB_MARKER_SIZE_MM
        y1 = y0 + self.CALIB_MARKER_SIZE_MM

        return np.asarray(
            [
                [x0, y0, 0.0],
                [x1, y0, 0.0],
                [x1, y1, 0.0],
                [x0, y1, 0.0],
            ],
            dtype=np.float32,
        )

    def capture_calibration_sample(self):
        raw = self.capture.get_frame()

        if raw is None:
            return {
                "ok": False,
                "message": "Camera is not running.",
            }

        # Calibration intentionally uses RAW image.
        corners, ids, _ = self.detector.detectMarkers(
            raw
        )

        if ids is None:
            return {
                "ok": False,
                "message": "No ArUco markers detected.",
            }

        object_points = []
        image_points = []
        visible_ids = []

        for index, marker_id in enumerate(
            ids.flatten()
        ):
            marker_id = int(marker_id)

            if marker_id not in self.CALIBRATION_IDS:
                continue

            object_points.extend(
                self.get_calibration_marker_object_points(
                    marker_id
                )
            )

            image_points.extend(
                np.asarray(
                    corners[index][0],
                    dtype=np.float32,
                )
            )

            visible_ids.append(marker_id)

        if (
            len(visible_ids)
            < self.CALIB_MIN_VISIBLE_MARKERS
        ):
            return {
                "ok": False,
                "message": (
                    f"Visible {len(visible_ids)}/9 board markers, "
                    f"need at least "
                    f"{self.CALIB_MIN_VISIBLE_MARKERS}."
                ),
            }

        object_points = np.asarray(
            object_points,
            dtype=np.float32,
        )

        image_points = np.asarray(
            image_points,
            dtype=np.float32,
        )

        self.calibration_object_points.append(
            object_points
        )

        self.calibration_image_points.append(
            image_points
        )

        self.calibration_sample_info.append(
            {
                "visible_ids": sorted(
                    visible_ids
                ),
                "points": len(
                    image_points
                ),
            }
        )

        return {
            "ok": True,
            "message": (
                f"Sample captured: "
                f"{len(visible_ids)}/9 markers, "
                f"{len(image_points)} corners."
            ),
        }

    def clear_calibration_samples(self):
        self.calibration_object_points = []
        self.calibration_image_points = []
        self.calibration_sample_info = []

    def calculate_camera_calibration(self):
        sample_count = len(
            self.calibration_image_points
        )

        if sample_count < 8:
            return {
                "ok": False,
                "message": "Capture at least 8 different samples.",
            }

        frame_size = self.capture.frame_size

        if frame_size is None:
            return {
                "ok": False,
                "message": "Camera image size unknown.",
            }

        try:
            (
                rms,
                camera_matrix,
                dist_coeffs,
                rvecs,
                tvecs,
            ) = cv2.calibrateCamera(
                self.calibration_object_points,
                self.calibration_image_points,
                frame_size,
                None,
                None,
            )

            total_error = 0.0
            total_points = 0

            for index in range(
                len(self.calibration_object_points)
            ):
                projected, _ = cv2.projectPoints(
                    self.calibration_object_points[
                        index
                    ],
                    rvecs[index],
                    tvecs[index],
                    camera_matrix,
                    dist_coeffs,
                )

                projected = projected.reshape(
                    -1,
                    2,
                )

                measured = (
                    self.calibration_image_points[
                        index
                    ]
                )

                errors = np.linalg.norm(
                    measured - projected,
                    axis=1,
                )

                total_error += float(
                    np.sum(errors)
                )
                total_points += len(errors)

            mean_error = (
                total_error / total_points
                if total_points
                else 0.0
            )

            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.calibration_image_size = frame_size
            self.calibration_rms = float(rms)
            self.calibration_mean_error = float(
                mean_error
            )

            CAMERA_CALIBRATION_FILE.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.savez(
                CAMERA_CALIBRATION_FILE,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                image_width=frame_size[0],
                image_height=frame_size[1],
                rms_error=float(rms),
                mean_reprojection_error=float(
                    mean_error
                ),
                board_marker_size_mm=self.CALIB_MARKER_SIZE_MM,
                board_gap_mm=self.CALIB_GAP_MM,
                board_step_mm=self.CALIB_STEP_MM,
                board_size_mm=self.CALIB_BOARD_SIZE_MM,
            )

            self.reset_undistortion_maps()
            self.enable_undistortion_var.set(
                True
            )

            self.clear_fixed_coordinates()
            self.update_lens_status()

            return {
                "ok": True,
                "message": "Calibration calculated and saved.",
                "rms": float(rms),
                "mean_error": float(mean_error),
                "samples": sample_count,
            }

        except Exception as error:
            return {
                "ok": False,
                "message": f"Calibration failed: {error}",
            }

    def delete_camera_calibration(self):
        try:
            if CAMERA_CALIBRATION_FILE.exists():
                CAMERA_CALIBRATION_FILE.unlink()
        except Exception as error:
            return {
                "ok": False,
                "message": str(error),
            }

        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_image_size = None
        self.calibration_rms = None
        self.calibration_mean_error = None

        self.reset_undistortion_maps()
        self.enable_undistortion_var.set(
            False
        )

        self.clear_fixed_coordinates()
        self.update_lens_status()

        return {
            "ok": True,
            "message": "Saved camera calibration deleted.",
        }

    # ========================================================
    # Calibration popup
    # ========================================================

    def open_calibration_window(self):
        if (
            self.calibration_window is not None
            and self.calibration_window.winfo_exists()
        ):
            self.calibration_window.lift()
            self.calibration_window.focus_force()
            return

        self.calibration_window = tk.Toplevel(
            self.frame
        )

        self.calibration_window.title(
            "Camera Calibration — ArUco 3×3"
        )

        self.calibration_window.geometry(
            "720x760"
        )

        self.calibration_panel = CameraCalibrationPanel(
            self.calibration_window,
            self,
        )

        self.calibration_panel.frame.pack(
            fill="both",
            expand=True,
        )

        def close():
            if self.calibration_panel:
                self.calibration_panel.shutdown()

            try:
                self.calibration_window.destroy()
            except Exception:
                pass

            self.calibration_window = None
            self.calibration_panel = None

        self.calibration_window.protocol(
            "WM_DELETE_WINDOW",
            close,
        )

    # ========================================================
    # UI update
    # ========================================================

    def update_loop(self):
        try:
            # Process latest camera frame at max ~30 fps.
            now = time.monotonic()

            if (
                self.capture.running
                and now - self.last_processed_raw_time
                >= self.SCENE_PROCESS_INTERVAL
            ):
                self.process_latest_frame()
                self.last_processed_raw_time = now

            self._update_status_labels()
            self.update_marker_table()
            self.draw_video_to_canvas()

        except Exception as error:
            self.app_context.log(
                f"Camera UI loop: {error}"
            )

        self.frame.after(
            self.UI_UPDATE_MS,
            self.update_loop,
        )

    def _update_status_labels(self):
        capture_status = self.capture.status

        if capture_status == "running":
            age_ms = (
                (
                    time.monotonic()
                    - self.capture.last_frame_time
                )
                * 1000.0
            )

            self.camera_status_var.set(
                f"Camera: running | "
                f"{self.capture.backend_name} | "
                f"frame age {age_ms:.0f} ms"
            )

        elif capture_status == "read error":
            self.camera_status_var.set(
                "Camera: read error — stopped"
            )

        elif capture_status == "error":
            self.camera_status_var.set(
                "Camera: ERROR"
            )

        else:
            self.camera_status_var.set(
                "Camera: stopped"
            )

        H = self.get_active_H()

        self.metric_status_var.set(
            "Metric map: OK"
            if H is not None
            else "Metric map: —"
        )

        self.update_lens_status()

        bounds = self.app_context.shared_data.get(
            "workspace_bounds"
        )

        if bounds:
            self.workspace_status_var.set(
                "Workspace: "
                f"{bounds['width']:.1f} × "
                f"{bounds['height']:.1f} mm"
            )
        else:
            self.workspace_status_var.set(
                "Workspace: —"
            )

        base = self.app_context.shared_data.get(
            "robot_base_world"
        )

        if base is None:
            self.base_status_var.set(
                "ID4 Base: —"
            )
        else:
            self.base_status_var.set(
                f"ID4 Base: X={base[0]:.1f}, "
                f"Y={base[1]:.1f}"
            )

        angle = self.app_context.shared_data.get(
            "robot_base_marker_angle_deg"
        )

        if angle is None:
            self.base_angle_var.set(
                "ID4 heading: —"
            )
        else:
            self.base_angle_var.set(
                f"ID4 heading: {angle:.1f}°"
            )

        target = self.app_context.shared_data.get(
            "marker_target_world"
        )

        if target is None:
            self.target8_var.set(
                "ID8 Target: —"
            )
        else:
            inside = self.point_inside_workspace(
                target[0],
                target[1],
            )

            self.target8_var.set(
                "ID8 Target: "
                f"X={target[0]:.1f}, "
                f"Y={target[1]:.1f} "
                f"({'inside' if inside else 'outside'})"
            )

        # YOLO status
        if self.yolo.model is None:
            self.yolo_status_var.set(
                "YOLO: no model"
                if self.yolo.status != "error"
                else f"YOLO ERROR: {self.yolo.last_error}"
            )
        else:
            state = (
                "ON"
                if self.yolo_enabled_var.get()
                else "OFF"
            )

            inference = (
                "—"
                if self.yolo.last_inference_ms is None
                else f"{self.yolo.last_inference_ms:.1f} ms"
            )

            model_name = Path(
                self.yolo.model_path
            ).name

            self.yolo_status_var.set(
                f"YOLO: {state} | "
                f"{model_name} | "
                f"{inference}"
            )

        cube_count = len(
            self.app_context.shared_data.get(
                "cube_detections",
                [],
            )
        )

        hand_count = len(
            self.app_context.shared_data.get(
                "hand_detections",
                [],
            )
        )

        hand_inside = bool(
            self.app_context.shared_data.get(
                "hand_inside_workspace",
                False,
            )
        )

        self.yolo_detection_var.set(
            f"Detections: cubes={cube_count} | "
            f"hand={hand_count} | "
            f"hand in workspace="
            f"{'YES' if hand_inside else 'NO'}"
        )

        if self.selected_target_world is None:
            self.target_world_var.set(
                "Mouse target: —"
            )
        else:
            wx, wy = self.selected_target_world

            inside = self.point_inside_workspace(
                wx,
                wy,
            )

            self.target_world_var.set(
                f"Mouse target: "
                f"{wx:.1f}, {wy:.1f} "
                f"({'inside' if inside else 'outside'})"
            )

    def update_marker_table(self):
        now = time.monotonic()

        for marker_id in self.SCENE_IDS:
            world = self.get_marker_world(
                marker_id
            )

            visible = (
                marker_id
                in self.detected_markers
            )

            fixed = (
                marker_id
                in self.fixed_marker_world
            )

            remembered = self.marker_memory.get(
                marker_id
            )

            if marker_id == self.TARGET_ID:
                state = (
                    "live"
                    if visible
                    else "no"
                )

            elif (
                self.marker_mode_var.get() == "Fixed"
                and fixed
            ):
                state = "fixed"

            elif visible:
                state = "live"

            elif remembered is not None:
                age = (
                    now
                    - float(
                        remembered.get(
                            "time",
                            now,
                        )
                    )
                )
                state = (
                    "saved"
                    if age <= self.MARKER_STALE_TIMEOUT_S
                    else "stale"
                )

            else:
                state = "no"

            self.marker_vars[
                marker_id
            ]["state"].set(
                state
            )

            if world is None:
                self.marker_vars[
                    marker_id
                ]["world"].set(
                    "—"
                )
            else:
                self.marker_vars[
                    marker_id
                ]["world"].set(
                    f"{world[0]:.1f}, "
                    f"{world[1]:.1f}"
                )

    # ========================================================
    # Video canvas
    # ========================================================

    def draw_video_to_canvas(self):
        frame = self.last_display_frame

        if frame is None:
            return

        frame_h, frame_w = frame.shape[:2]

        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()

        if (
            canvas_w <= 1
            or canvas_h <= 1
        ):
            return

        scale = min(
            canvas_w / frame_w,
            canvas_h / frame_h,
        )

        self.displayed_w = max(
            1,
            int(frame_w * scale),
        )
        self.displayed_h = max(
            1,
            int(frame_h * scale),
        )

        self.display_offset_x = (
            canvas_w - self.displayed_w
        ) // 2

        self.display_offset_y = (
            canvas_h - self.displayed_h
        ) // 2

        resized = cv2.resize(
            frame,
            (
                self.displayed_w,
                self.displayed_h,
            ),
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB,
        )

        self.photo = ImageTk.PhotoImage(
            image=Image.fromarray(rgb)
        )

        if self.canvas_image_id is None:
            self.canvas_image_id = (
                self.video_canvas.create_image(
                    self.display_offset_x,
                    self.display_offset_y,
                    image=self.photo,
                    anchor="nw",
                )
            )
        else:
            self.video_canvas.coords(
                self.canvas_image_id,
                self.display_offset_x,
                self.display_offset_y,
            )

            self.video_canvas.itemconfigure(
                self.canvas_image_id,
                image=self.photo,
            )

    # ========================================================
    # Clear / shutdown
    # ========================================================

    def clear_scene(self):
        self.detected_markers = {}
        self.marker_memory = {}

        self.live_H = None

        self.clear_fixed_coordinates(delete_saved=False)

        self.last_workspace_polygon_world = None
        self.last_robot_base_world = None
        self.last_robot_base_marker_angle = None

        self.clear_mouse_target()

        shared = self.app_context.shared_data

        shared[
            "workspace_polygon_world"
        ] = None

        shared[
            "workspace_bounds"
        ] = None

        shared[
            "workspace_camera_shape"
        ] = None

        shared[
            "robot_base_world"
        ] = None

        shared[
            "robot_base_marker_angle_deg"
        ] = None

        shared[
            "containers_world"
        ] = {}

        shared[
            "marker_target_world"
        ] = None

        self.latest_yolo_detections = []
        shared["object_detections"] = []
        shared["cube_detections"] = []
        shared["hand_detections"] = []
        shared["hand_detected"] = False
        shared["hand_inside_workspace"] = False

    @staticmethod
    def normalize_angle_deg(angle):
        return (
            float(angle) + 180.0
        ) % 360.0 - 180.0

    def get_frame(self):
        return self.frame

    def shutdown(self):
        self.capture.shutdown()
        self.yolo.shutdown()

        if (
            self.calibration_window is not None
            and self.calibration_window.winfo_exists()
        ):
            try:
                self.calibration_window.destroy()
            except Exception:
                pass

        self.app_context.log(
            "Camera Panel shutdown."
        )


# ============================================================
# CAMERA CALIBRATION PANEL
# ============================================================

class CameraCalibrationPanel:
    def __init__(
        self,
        parent,
        camera_panel,
    ):
        self.camera = camera_panel

        self.frame = ttk.Frame(
            parent,
            padding=12,
        )

        self.status_var = tk.StringVar(
            value="Ready"
        )

        self.board_status_var = tk.StringVar(
            value="Board detected: 0 / 9"
        )

        self.ids_var = tk.StringVar(
            value="Visible IDs: —"
        )

        self.samples_var = tk.StringVar(
            value="Samples: 0"
        )

        self.points_var = tk.StringVar(
            value="Captured corners: 0"
        )

        self.rms_var = tk.StringVar(
            value="RMS: —"
        )

        self.error_var = tk.StringVar(
            value="Mean reprojection error: —"
        )

        self.matrix_var = tk.StringVar(
            value="Camera matrix: —"
        )

        self.distortion_var = tk.StringVar(
            value="Distortion: —"
        )

        self.build_ui()
        self.update_existing_calibration_info()

        self.frame.after(
            100,
            self.update_loop,
        )

    def build_ui(self):
        ttk.Label(
            self.frame,
            text="CAMERA CALIBRATION",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            self.frame,
            text="3×3 ArUco Board — IDs 10–18",
            font=("Segoe UI", 11, "bold"),
        ).pack(
            anchor="w",
            pady=(2, 10),
        )

        board = ttk.LabelFrame(
            self.frame,
            text="Physical Board",
            padding=8,
        )
        board.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Label(
            board,
            text=(
                "Dictionary: DICT_4X4_50\n"
                "Layout: 3 × 3\n"
                "IDs: 10–18\n"
                "Marker size: 30.0 mm\n"
                "Gap: 14.5 mm\n"
                "Center step: 44.5 mm\n"
                "Total board size: 119.0 × 119.0 mm"
            ),
        ).pack(anchor="w")

        live = ttk.LabelFrame(
            self.frame,
            text="Live Board Detection",
            padding=8,
        )
        live.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Label(
            live,
            textvariable=self.board_status_var,
        ).pack(anchor="w")

        ttk.Label(
            live,
            textvariable=self.ids_var,
        ).pack(anchor="w")

        samples = ttk.LabelFrame(
            self.frame,
            text="Samples",
            padding=8,
        )
        samples.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Label(
            samples,
            textvariable=self.samples_var,
        ).pack(anchor="w")

        ttk.Label(
            samples,
            textvariable=self.points_var,
        ).pack(anchor="w")

        ttk.Label(
            samples,
            text=(
                "Move and tilt the board through the complete image. "
                "Capture center, edges and corners."
            ),
            wraplength=650,
        ).pack(
            anchor="w",
            pady=(6, 8),
        )

        row = ttk.Frame(samples)
        row.pack(fill="x")

        ttk.Button(
            row,
            text="Capture Sample",
            command=self.capture_sample,
        ).pack(side="left", padx=3)

        ttk.Button(
            row,
            text="Clear Samples",
            command=self.clear_samples,
        ).pack(side="left", padx=3)

        calculate = ttk.LabelFrame(
            self.frame,
            text="Calculate",
            padding=8,
        )
        calculate.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Button(
            calculate,
            text="Calculate + Save Calibration",
            command=self.calculate,
        ).pack(fill="x")

        ttk.Label(
            calculate,
            textvariable=self.rms_var,
        ).pack(
            anchor="w",
            pady=(7, 0),
        )

        ttk.Label(
            calculate,
            textvariable=self.error_var,
        ).pack(anchor="w")

        result = ttk.LabelFrame(
            self.frame,
            text="Result",
            padding=8,
        )
        result.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Label(
            result,
            textvariable=self.matrix_var,
            wraplength=650,
        ).pack(anchor="w")

        ttk.Label(
            result,
            textvariable=self.distortion_var,
            wraplength=650,
        ).pack(
            anchor="w",
            pady=5,
        )

        ttk.Button(
            result,
            text="Delete Saved Calibration",
            command=self.delete_calibration,
        ).pack(
            fill="x",
            pady=(5, 0),
        )

        status = ttk.LabelFrame(
            self.frame,
            text="Status",
            padding=8,
        )
        status.pack(fill="x")

        ttk.Label(
            status,
            textvariable=self.status_var,
            wraplength=650,
        ).pack(anchor="w")

    def capture_sample(self):
        result = self.camera.capture_calibration_sample()

        self.status_var.set(
            result["message"]
        )

        self.update_sample_info()

    def clear_samples(self):
        self.camera.clear_calibration_samples()
        self.update_sample_info()

        self.status_var.set(
            "Calibration samples cleared."
        )

    def calculate(self):
        result = self.camera.calculate_camera_calibration()

        self.status_var.set(
            result["message"]
        )

        if not result["ok"]:
            return

        self.rms_var.set(
            f"RMS: {result['rms']:.4f} px"
        )

        self.error_var.set(
            "Mean reprojection error: "
            f"{result['mean_error']:.4f} px"
        )

        self.update_existing_calibration_info()

    def delete_calibration(self):
        result = self.camera.delete_camera_calibration()

        self.status_var.set(
            result["message"]
        )

        self.update_existing_calibration_info()

    def update_sample_info(self):
        samples = len(
            self.camera.calibration_image_points
        )

        total_points = sum(
            len(sample)
            for sample
            in self.camera.calibration_image_points
        )

        self.samples_var.set(
            f"Samples: {samples} / "
            f"{self.camera.CALIB_RECOMMENDED_SAMPLES}"
        )

        self.points_var.set(
            f"Captured corners: {total_points}"
        )

    def update_existing_calibration_info(self):
        matrix = self.camera.camera_matrix
        dist = self.camera.dist_coeffs

        if matrix is None:
            self.matrix_var.set(
                "Camera matrix: —"
            )
            self.distortion_var.set(
                "Distortion: —"
            )
            self.rms_var.set(
                "RMS: —"
            )
            self.error_var.set(
                "Mean reprojection error: —"
            )
            return

        self.matrix_var.set(
            "Camera matrix:\n"
            + np.array2string(
                matrix,
                precision=3,
                suppress_small=True,
            )
        )

        if dist is not None:
            self.distortion_var.set(
                "Distortion:\n"
                + np.array2string(
                    dist.reshape(-1),
                    precision=5,
                    suppress_small=True,
                )
            )

        if self.camera.calibration_rms is not None:
            self.rms_var.set(
                "RMS: "
                f"{self.camera.calibration_rms:.4f} px"
            )

        if (
            self.camera.calibration_mean_error
            is not None
        ):
            self.error_var.set(
                "Mean reprojection error: "
                f"{self.camera.calibration_mean_error:.4f} px"
            )

    def update_loop(self):
        try:
            detected = self.camera.detected_markers

            visible_ids = [
                marker_id
                for marker_id in self.camera.CALIBRATION_IDS
                if marker_id in detected
            ]

            self.board_status_var.set(
                f"Board detected: "
                f"{len(visible_ids)} / 9"
            )

            self.ids_var.set(
                "Visible IDs: "
                + (
                    ", ".join(
                        str(marker_id)
                        for marker_id in visible_ids
                    )
                    if visible_ids
                    else "—"
                )
            )

            self.update_sample_info()

        except Exception as error:
            self.status_var.set(
                f"Calibration UI error: {error}"
            )

        self.frame.after(
            100,
            self.update_loop,
        )

    def shutdown(self):
        pass
