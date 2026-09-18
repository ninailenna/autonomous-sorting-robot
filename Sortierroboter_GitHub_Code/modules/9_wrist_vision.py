from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk


try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
WRIST_CAMERA_CALIBRATION_FILE = PROJECT_DIR / "data" / "wrist_camera_calibration.npz"

WRIST_CAMERA_BACKEND = (
    cv2.CAP_DSHOW
    if os.name == "nt"
    else None
)



# ============================================================
# CAMERA CAPTURE
# ============================================================

class CameraCapture:
    def __init__(self):
        self.cap = None
        self.thread = None
        self.running = False

        self.lock = threading.Lock()
        self.latest_frame = None

        self.frame_size = None
        self.last_frame_time = 0.0

        self.fps = 0.0
        self._fps_started = time.monotonic()
        self._fps_frames = 0

        self.status = "stopped"
        self.backend_name = "—"

    def _open_capture(self, index):
        candidates = []

        if os.name == "nt":
            candidates.append(
                ("DirectShow", cv2.CAP_DSHOW)
            )

        candidates.append(
            ("Default", None)
        )

        for name, backend in candidates:
            cap = (
                cv2.VideoCapture(
                    int(index),
                    backend,
                )
                if backend is not None
                else cv2.VideoCapture(
                    int(index)
                )
            )

            if not cap.isOpened():
                cap.release()
                continue

            # Keep USB bandwidth modest.
            cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(
                    *"MJPG"
                ),
            )
            cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                640,
            )
            cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                480,
            )
            cap.set(
                cv2.CAP_PROP_FPS,
                15,
            )

            ok, frame = cap.read()

            if ok and frame is not None:
                self.backend_name = name
                return cap, frame

            cap.release()

        return None, None

    def start(self, index):
        if self.running:
            return (
                True,
                "Camera already running.",
            )

        cap, first = self._open_capture(
            int(index)
        )

        if cap is None or first is None:
            self.status = "error"
            return (
                False,
                f"Could not open wrist camera index {index}.",
            )

        self.cap = cap

        with self.lock:
            self.latest_frame = first.copy()

        h, w = first.shape[:2]
        self.frame_size = (
            w,
            h,
        )

        self.running = True
        self.status = "running"

        self.last_frame_time = (
            time.monotonic()
        )

        self._fps_started = (
            time.monotonic()
        )
        self._fps_frames = 0
        self.fps = 0.0

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )
        self.thread.start()

        return (
            True,
            f"Wrist camera running | "
            f"index {index} | "
            f"{self.backend_name}",
        )

    def _loop(self):
        failures = 0

        while self.running:
            cap = self.cap

            if cap is None:
                break

            try:
                ok, frame = cap.read()
            except Exception:
                ok = False
                frame = None

            if not ok or frame is None:
                failures += 1

                if failures >= 10:
                    self.status = "read error"
                    self.running = False
                    break

                time.sleep(0.05)
                continue

            failures = 0

            with self.lock:
                self.latest_frame = (
                    frame.copy()
                )

            self.last_frame_time = (
                time.monotonic()
            )

            self._fps_frames += 1
            now = time.monotonic()

            elapsed = (
                now
                - self._fps_started
            )

            if elapsed >= 1.0:
                self.fps = (
                    self._fps_frames
                    / elapsed
                )

                self._fps_frames = 0
                self._fps_started = now

            time.sleep(0.002)

        self._release()

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None

            return (
                self.latest_frame.copy()
            )

    def stop(self):
        self.running = False
        self.status = "stopped"
        self._release()

    def _release(self):
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
# WRIST VISION MODULE
# ============================================================

class ModulePanel:
    """
    Module 9 = wrist-camera vision only.

    Responsibilities:
    - wrist camera
    - lens calibration / undistortion
    - OBB inference
    - requested-class tracking
    - anti-jump target association
    - center + square-angle smoothing
    - stable-observation decision
    - frozen grasp/orientation lock

    IMPORTANT:
    Module 9 NEVER commands the robot and NEVER rotates wrist_roll.
    Module 8 decides when the robot is close enough, requests a stable
    observation, freezes the cube orientation, and only then rotates wrist_roll.

    Inputs through shared_data:
        wrist_requested_class
        wrist_lock_request
        wrist_unlock_request

    Outputs through shared_data:
        wrist_obb_detections
        wrist_tracking_target
        wrist_stable_target
        wrist_grasp_lock
        wrist_frame_size
    """

    title = "Wrist Vision"

    UI_UPDATE_MS = 35

    # Same physical 3x3 ArUco calibration board as Module 3.
    ARUCO_DICTIONARY = aruco.DICT_4X4_50
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

    def __init__(
        self,
        parent,
        app_context,
    ):
        self.app_context = (
            app_context
        )


        self.frame = ttk.Frame(
            parent,
            padding=10,
        )

        self.capture = (
            CameraCapture()
        )

        # ====================================================
        # Wrist-camera lens calibration
        # ====================================================

        dictionary = aruco.getPredefinedDictionary(
            self.ARUCO_DICTIONARY
        )

        self.calibration_detector = aruco.ArucoDetector(
            dictionary,
            aruco.DetectorParameters(),
        )

        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_image_size = None
        self.calibration_rms = None
        self.calibration_mean_error = None

        self.new_camera_matrix = None
        self.undistort_map1 = None
        self.undistort_map2 = None
        self.undistort_map_size = None

        self.calibration_object_points = []
        self.calibration_image_points = []
        self.calibration_sample_info = []

        self.calibration_window = None
        self.calibration_panel = None

        # ====================================================
        # Camera
        # ====================================================

        self.camera_index_var = tk.IntVar(
            value=1
        )

        self.camera_status_var = (
            tk.StringVar(
                value="Camera: stopped"
            )
        )

        self.fps_var = tk.StringVar(
            value="FPS: —"
        )

        self.flip_horizontal_var = (
            tk.BooleanVar(
                value=False
            )
        )

        self.flip_vertical_var = (
            tk.BooleanVar(
                value=False
            )
        )

        self.rotation_var = (
            tk.StringVar(
                value="0"
            )
        )

        self.enable_undistortion_var = (
            tk.BooleanVar(
                value=True
            )
        )

        self.lens_status_var = (
            tk.StringVar(
                value="Lens: not calibrated"
            )
        )

        # ====================================================
        # OBB model
        # ====================================================

        self.model = None

        self.model_path_var = (
            tk.StringVar(
                value=str(
                    self.app_context
                    .shared_data
                    .get(
                        "wrist_obb_model_path",
                        "",
                    )
                )
            )
        )

        self.detection_enabled_var = (
            tk.BooleanVar(
                value=bool(
                    self.app_context
                    .shared_data
                    .get(
                        "wrist_obb_enabled",
                        False,
                    )
                )
            )
        )

        self.conf_var = tk.DoubleVar(
            value=float(
                self.app_context
                .shared_data
                .get(
                    "wrist_obb_confidence",
                    0.35,
                )
            )
        )

        self.max_fps_var = (
            tk.DoubleVar(
                value=float(
                    self.app_context
                    .shared_data
                    .get(
                        "wrist_obb_max_fps",
                        10.0,
                    )
                )
            )
        )

        self.ai_status_var = (
            tk.StringVar(
                value="OBB: no model"
            )
        )

        self.last_inference_time = 0.0

        self.latest_detections = []

        # ====================================================
        # Target tracking
        # ====================================================

        # Internal tracking defaults.
        # These are intentionally not exposed in the GUI.
        self.manual_target_class = "all"
        self.follow_external_target = True
        self.target_gate_px = 120.0

        self.current_target = None

        # Smoothed target is NOT frozen.
        self.smoothed_target = None

        # Frozen grasp target is never updated until unlock.
        self.grasp_lock = None

        self.target_history = deque(
            maxlen=9
        )

        # ====================================================
        # Stability / smoothing
        # ====================================================

        # Stability / smoothing settings.
        # These remain editable from the GUI because wrist-model confidence
        # and jitter can vary noticeably with distance / lighting.
        self.stability_window_var = tk.IntVar(value=7)
        self.stability_min_frames_var = tk.IntVar(value=5)
        self.stability_center_jitter_var = tk.DoubleVar(value=12.0)
        self.stability_angle_jitter_var = tk.DoubleVar(value=4.0)
        self.stability_min_conf_var = tk.DoubleVar(value=0.12)
        self.smooth_center_alpha_var = tk.DoubleVar(value=0.35)

        self.stability_status_var = (
            tk.StringVar(
                value="Stability: —"
            )
        )

        self.lock_status_var = (
            tk.StringVar(
                value="Lock: none"
            )
        )

        self.target_status_var = (
            tk.StringVar(
                value="Target: —"
            )
        )

        # ====================================================
        # Preview
        # ====================================================

        self.photo = None
        self.canvas_image_id = None
        self.last_display_frame = None

        # ====================================================
        # Shared API
        # ====================================================

        shared = (
            self.app_context
            .shared_data
        )

        shared.setdefault(
            "wrist_requested_class",
            "all",
        )


        shared.setdefault(
            "wrist_lock_request",
            False,
        )

        shared.setdefault(
            "wrist_unlock_request",
            False,
        )


        shared.setdefault(
            "wrist_obb_model_path",
            "",
        )

        shared.setdefault(
            "wrist_obb_enabled",
            False,
        )

        shared.setdefault(
            "wrist_obb_confidence",
            0.35,
        )

        shared.setdefault(
            "wrist_obb_max_fps",
            10.0,
        )

        shared.setdefault(
            "wrist_obb_detections",
            [],
        )

        shared.setdefault(
            "wrist_tracking_target",
            None,
        )

        shared.setdefault(
            "wrist_stable_target",
            None,
        )

        shared.setdefault(
            "wrist_grasp_lock",
            None,
        )



        shared.setdefault(
            "wrist_frame_size",
            None,
        )

        self.load_wrist_camera_calibration()
        self.build_ui()

        # Auto-load remembered model.
        remembered = (
            self.model_path_var
            .get()
            .strip()
        )

        if remembered:
            self._load_model_path(
                remembered
            )

        self.update_loop()

        self.app_context.log(
            "Wrist Vision module loaded."
        )

    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):
        header = ttk.Frame(
            self.frame
        )

        header.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Label(
            header,
            text="WRIST VISION + OBB",
            font=(
                "Segoe UI",
                18,
                "bold",
            ),
        ).pack(
            side="left"
        )

        ttk.Label(
            header,
            textvariable=
                self.camera_status_var,
        ).pack(
            side="right"
        )

        camera_bar = ttk.LabelFrame(
            self.frame,
            text="Camera",
            padding=8,
        )

        camera_bar.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Label(
            camera_bar,
            text="Index",
        ).pack(
            side="left"
        )

        ttk.Entry(
            camera_bar,
            textvariable=
                self.camera_index_var,
            width=5,
        ).pack(
            side="left",
            padx=4,
        )

        ttk.Button(
            camera_bar,
            text="Start",
            command=
                self.start_camera,
        ).pack(
            side="left",
            padx=3,
        )

        ttk.Button(
            camera_bar,
            text="Stop",
            command=
                self.stop_camera,
        ).pack(
            side="left",
            padx=3,
        )

        ttk.Checkbutton(
            camera_bar,
            text="Lens correction",
            variable=self.enable_undistortion_var,
            command=self.on_undistortion_changed,
        ).pack(
            side="left",
            padx=(12, 4),
        )

        ttk.Button(
            camera_bar,
            text="Camera Calibration",
            command=self.open_calibration_window,
        ).pack(
            side="left",
            padx=4,
        )


        ttk.Label(
            camera_bar,
            textvariable=self.fps_var,
        ).pack(
            side="right"
        )

        ttk.Label(
            camera_bar,
            textvariable=self.lens_status_var,
        ).pack(
            side="right",
            padx=8,
        )

        body = ttk.PanedWindow(
            self.frame,
            orient="horizontal",
        )

        body.pack(
            fill="both",
            expand=True,
        )

        video_frame = ttk.Frame(
            body
        )

        sidebar_outer = ttk.Frame(
            body
        )

        body.add(
            video_frame,
            weight=5,
        )

        body.add(
            sidebar_outer,
            weight=2,
        )

        self.video_canvas = tk.Canvas(
            video_frame,
            bg="black",
            highlightthickness=0,
        )

        self.video_canvas.pack(
            fill="both",
            expand=True,
        )

        self._build_sidebar(
            sidebar_outer
        )

    def _build_sidebar(
        self,
        parent,
    ):
        canvas = tk.Canvas(
            parent,
            width=390,
            highlightthickness=0,
        )

        scrollbar = ttk.Scrollbar(
            parent,
            orient="vertical",
            command=
                canvas.yview,
        )

        sidebar = ttk.Frame(
            canvas,
            padding=8,
        )

        window_id = (
            canvas.create_window(
                (0, 0),
                window=sidebar,
                anchor="nw",
            )
        )

        sidebar.bind(
            "<Configure>",
            lambda event:
                canvas.configure(
                    scrollregion=
                        canvas.bbox(
                            "all"
                        )
                ),
        )

        canvas.bind(
            "<Configure>",
            lambda event:
                canvas.itemconfigure(
                    window_id,
                    width=
                        event.width,
                ),
        )

        canvas.configure(
            yscrollcommand=
                scrollbar.set
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
        # Image orientation
        # --------------------------------------------------

        image_box = ttk.LabelFrame(
            sidebar,
            text="Image Orientation",
            padding=8,
        )

        image_box.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Checkbutton(
            image_box,
            text="Flip horizontal",
            variable=
                self.flip_horizontal_var,
        ).pack(
            anchor="w"
        )

        ttk.Checkbutton(
            image_box,
            text="Flip vertical",
            variable=
                self.flip_vertical_var,
        ).pack(
            anchor="w"
        )

        row = ttk.Frame(
            image_box
        )

        row.pack(
            fill="x",
            pady=(5, 0),
        )

        ttk.Label(
            row,
            text="Rotation",
        ).pack(
            side="left"
        )

        ttk.Combobox(
            row,
            textvariable=
                self.rotation_var,
            state="readonly",
            width=8,
            values=[
                "0",
                "90",
                "180",
                "270",
            ],
        ).pack(
            side="left",
            padx=5,
        )

        # --------------------------------------------------
        # OBB
        # --------------------------------------------------

        obb = ttk.LabelFrame(
            sidebar,
            text="OBB Detection",
            padding=8,
        )

        obb.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Label(
            obb,
            textvariable=
                self.model_path_var,
            wraplength=350,
        ).pack(
            anchor="w"
        )

        row = ttk.Frame(
            obb
        )

        row.pack(
            fill="x",
            pady=(5, 3),
        )

        ttk.Button(
            row,
            text="Browse Model",
            command=
                self.browse_model,
        ).pack(
            side="left",
            padx=(0, 3),
        )

        ttk.Button(
            row,
            text="Load",
            command=
                self.load_model,
        ).pack(
            side="left",
            padx=3,
        )

        ttk.Checkbutton(
            obb,
            text="Enable detection",
            variable=
                self.detection_enabled_var,
            command=
                self.on_detection_enabled,
        ).pack(
            anchor="w",
            pady=(4, 3),
        )

        settings = ttk.Frame(
            obb
        )

        settings.pack(
            fill="x"
        )

        ttk.Label(
            settings,
            text="Confidence",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        ttk.Entry(
            settings,
            textvariable=
                self.conf_var,
            width=7,
        ).grid(
            row=0,
            column=1,
            padx=(4, 10),
        )

        ttk.Label(
            settings,
            text="Max FPS",
        ).grid(
            row=0,
            column=2,
            sticky="w",
        )

        ttk.Entry(
            settings,
            textvariable=
                self.max_fps_var,
            width=7,
        ).grid(
            row=0,
            column=3,
            padx=4,
        )

        ttk.Button(
            obb,
            text="Apply Settings",
            command=
                self.apply_obb_settings,
        ).pack(
            fill="x",
            pady=(5, 2),
        )

        ttk.Label(
            obb,
            textvariable=
                self.ai_status_var,
            wraplength=350,
        ).pack(
            anchor="w"
        )

        # --------------------------------------------------
        # Vision Status
        # --------------------------------------------------

        stability_box = ttk.LabelFrame(
            sidebar,
            text="Stability Settings",
            padding=8,
        )

        stability_box.pack(
            fill="x",
            pady=(0, 8),
        )

        grid = ttk.Frame(stability_box)
        grid.pack(fill="x")

        fields = [
            ("Window", self.stability_window_var),
            ("Min frames", self.stability_min_frames_var),
            ("Center jitter [px]", self.stability_center_jitter_var),
            ("Angle jitter [deg]", self.stability_angle_jitter_var),
            ("Min confidence", self.stability_min_conf_var),
            ("Smoothing α", self.smooth_center_alpha_var),
        ]

        for index, (label, variable) in enumerate(fields):
            row = index // 2
            col = (index % 2) * 2

            ttk.Label(
                grid,
                text=label,
            ).grid(
                row=row,
                column=col,
                sticky="w",
                pady=2,
            )

            ttk.Entry(
                grid,
                textvariable=variable,
                width=8,
            ).grid(
                row=row,
                column=col + 1,
                padx=(4, 10),
                pady=2,
            )

        ttk.Button(
            stability_box,
            text="Reset Stability Defaults",
            command=self.reset_stability_defaults,
        ).pack(
            fill="x",
            pady=(6, 0),
        )

        vision_status = ttk.LabelFrame(
            sidebar,
            text="Vision Status",
            padding=8,
        )

        vision_status.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Label(
            vision_status,
            textvariable=self.target_status_var,
            wraplength=350,
        ).pack(
            anchor="w"
        )

        ttk.Label(
            vision_status,
            textvariable=self.stability_status_var,
            wraplength=350,
        ).pack(
            anchor="w",
            pady=(3, 0),
        )

        ttk.Label(
            vision_status,
            textvariable=self.lock_status_var,
            wraplength=350,
        ).pack(
            anchor="w",
            pady=(3, 0),
        )


    def reset_stability_defaults(self):
        self.stability_window_var.set(7)
        self.stability_min_frames_var.set(5)
        self.stability_center_jitter_var.set(12.0)
        self.stability_angle_jitter_var.set(4.0)
        self.stability_min_conf_var.set(0.12)
        self.smooth_center_alpha_var.set(0.35)

        self.reset_tracking_history()

        self.stability_status_var.set(
            "Stability: defaults restored"
        )

    # ========================================================
    # Camera
    # ========================================================

    def start_camera(self):
        ok, message = (
            self.capture.start(
                self.camera_index_var.get()
            )
        )

        self.camera_status_var.set(
            message
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
    # Wrist-camera lens calibration
    # ========================================================

    def load_wrist_camera_calibration(self):
        if not WRIST_CAMERA_CALIBRATION_FILE.exists():
            self.camera_matrix = None
            self.dist_coeffs = None
            self.calibration_rms = None
            self.calibration_mean_error = None
            return

        try:
            data = np.load(
                WRIST_CAMERA_CALIBRATION_FILE
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
                f"Wrist camera calibration load error: {error}"
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
        self.reset_undistortion_maps()
        self.reset_tracking_history()
        self.unlock_target()
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
        size = (
            int(width),
            int(height),
        )

        try:
            new_matrix, _ = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.dist_coeffs,
                size,
                1.0,
                size,
            )

            map1, map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.dist_coeffs,
                None,
                new_matrix,
                size,
                cv2.CV_16SC2,
            )

            self.new_camera_matrix = new_matrix
            self.undistort_map1 = map1
            self.undistort_map2 = map2
            self.undistort_map_size = size

        except Exception as error:
            self.app_context.log(
                f"Wrist undistortion map error: {error}"
            )

            self.reset_undistortion_maps()

    def get_calibration_marker_object_points(
        self,
        marker_id,
    ):
        if marker_id not in self.CALIBRATION_IDS:
            return None

        index = (
            marker_id - 10
        )

        row = (
            index // 3
        )

        col = (
            index % 3
        )

        x0 = (
            col
            * self.CALIB_STEP_MM
        )

        y0 = (
            row
            * self.CALIB_STEP_MM
        )

        x1 = (
            x0
            + self.CALIB_MARKER_SIZE_MM
        )

        y1 = (
            y0
            + self.CALIB_MARKER_SIZE_MM
        )

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
                "message":
                    "Wrist camera is not running.",
            }

        # Camera calibration uses the raw image before undistortion
        # and image-orientation transformations.
        corners, ids, _ = (
            self.calibration_detector
            .detectMarkers(
                raw
            )
        )

        if ids is None:
            return {
                "ok": False,
                "message":
                    "No ArUco markers detected.",
            }

        object_points = []
        image_points = []
        visible_ids = []

        for index, marker_id in enumerate(
            ids.flatten()
        ):
            marker_id = int(
                marker_id
            )

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

            visible_ids.append(
                marker_id
            )

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
                "visible_ids":
                    sorted(
                        visible_ids
                    ),
                "points":
                    len(
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

    def calculate_wrist_camera_calibration(self):
        sample_count = len(
            self.calibration_image_points
        )

        if sample_count < 8:
            return {
                "ok": False,
                "message":
                    "Capture at least 8 different samples.",
            }

        frame_size = (
            self.capture.frame_size
        )

        if frame_size is None:
            return {
                "ok": False,
                "message":
                    "Camera image size unknown.",
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
                len(
                    self.calibration_object_points
                )
            ):
                projected, _ = cv2.projectPoints(
                    self.calibration_object_points[
                        index
                    ],
                    rvecs[
                        index
                    ],
                    tvecs[
                        index
                    ],
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
                    np.sum(
                        errors
                    )
                )

                total_points += len(
                    errors
                )

            mean_error = (
                total_error
                / total_points
                if total_points
                else 0.0
            )

            self.camera_matrix = (
                camera_matrix
            )

            self.dist_coeffs = (
                dist_coeffs
            )

            self.calibration_image_size = (
                frame_size
            )

            self.calibration_rms = float(
                rms
            )

            self.calibration_mean_error = float(
                mean_error
            )

            WRIST_CAMERA_CALIBRATION_FILE.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.savez(
                WRIST_CAMERA_CALIBRATION_FILE,
                camera_matrix=
                    camera_matrix,
                dist_coeffs=
                    dist_coeffs,
                image_width=
                    frame_size[0],
                image_height=
                    frame_size[1],
                rms_error=
                    float(
                        rms
                    ),
                mean_reprojection_error=
                    float(
                        mean_error
                    ),
                board_marker_size_mm=
                    self.CALIB_MARKER_SIZE_MM,
                board_gap_mm=
                    self.CALIB_GAP_MM,
                board_step_mm=
                    self.CALIB_STEP_MM,
                board_size_mm=
                    self.CALIB_BOARD_SIZE_MM,
            )

            self.reset_undistortion_maps()
            self.enable_undistortion_var.set(
                True
            )
            self.reset_tracking_history()
            self.unlock_target()
            self.update_lens_status()

            return {
                "ok": True,
                "message":
                    "Wrist calibration calculated and saved.",
                "rms":
                    float(
                        rms
                    ),
                "mean_error":
                    float(
                        mean_error
                    ),
                "samples":
                    sample_count,
            }

        except Exception as error:
            return {
                "ok": False,
                "message":
                    f"Calibration failed: {error}",
            }

    def delete_wrist_camera_calibration(self):
        try:
            if WRIST_CAMERA_CALIBRATION_FILE.exists():
                WRIST_CAMERA_CALIBRATION_FILE.unlink()
        except Exception as error:
            return {
                "ok": False,
                "message":
                    str(
                        error
                    ),
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

        self.reset_tracking_history()
        self.unlock_target()
        self.update_lens_status()

        return {
            "ok": True,
            "message":
                "Saved wrist-camera calibration deleted.",
        }

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
            "Wrist Camera Calibration — ArUco 3×3"
        )

        self.calibration_window.geometry(
            "720x760"
        )

        self.calibration_panel = WristCameraCalibrationPanel(
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
    # OBB model
    # ========================================================

    def browse_model(self):
        path = (
            filedialog.askopenfilename(
                title=
                    "Select wrist OBB model",
                filetypes=[
                    (
                        "PyTorch model",
                        "*.pt",
                    ),
                    (
                        "All files",
                        "*.*",
                    ),
                ],
            )
        )

        if path:
            self.model_path_var.set(
                path
            )

    def load_model(self):
        path = (
            self.model_path_var
            .get()
            .strip()
        )

        self._load_model_path(
            path
        )

    def _load_model_path(
        self,
        path,
    ):
        if YOLO is None:
            self.ai_status_var.set(
                "OBB: ultralytics not installed"
            )
            return False

        if not path:
            self.ai_status_var.set(
                "OBB: select a model first"
            )
            return False

        model_path = Path(
            path
        )

        if not model_path.exists():
            self.ai_status_var.set(
                "OBB: model file not found"
            )
            return False

        try:
            self.model = YOLO(
                str(
                    model_path
                )
            )

            self.model_path_var.set(
                str(
                    model_path
                )
            )

            self.detection_enabled_var.set(
                True
            )

            shared = (
                self.app_context
                .shared_data
            )

            shared[
                "wrist_obb_model_path"
            ] = str(
                model_path
            )

            shared[
                "wrist_obb_enabled"
            ] = True

            self.ai_status_var.set(
                f"OBB: loaded "
                f"{model_path.name}"
            )

            return True

        except Exception as error:
            self.model = None

            self.ai_status_var.set(
                f"OBB load error: {error}"
            )

            return False

    def on_detection_enabled(self):
        enabled = bool(
            self.detection_enabled_var.get()
        )

        if (
            enabled
            and self.model is None
        ):
            self.detection_enabled_var.set(
                False
            )

            self.ai_status_var.set(
                "OBB: load model first"
            )

            return

        self.app_context.shared_data[
            "wrist_obb_enabled"
        ] = enabled

    def apply_obb_settings(self):
        try:
            conf = max(
                0.01,
                min(
                    0.99,
                    float(
                        self.conf_var.get()
                    ),
                ),
            )

            fps = max(
                0.5,
                min(
                    30.0,
                    float(
                        self.max_fps_var.get()
                    ),
                ),
            )

            self.conf_var.set(
                conf
            )

            self.max_fps_var.set(
                fps
            )

            shared = (
                self.app_context
                .shared_data
            )

            shared[
                "wrist_obb_confidence"
            ] = conf

            shared[
                "wrist_obb_max_fps"
            ] = fps

            self.ai_status_var.set(
                f"OBB settings: "
                f"conf={conf:.2f}, "
                f"fps={fps:.1f}"
            )

        except Exception as error:
            self.ai_status_var.set(
                f"OBB settings error: {error}"
            )

    # ========================================================
    # Target / angle helpers
    # ========================================================

    @staticmethod
    def _square_axis_error(
        target_deg,
        reference_deg,
    ):
        return (
            (
                float(
                    target_deg
                )
                - float(
                    reference_deg
                )
                + 45.0
            )
            % 90.0
        ) - 45.0

    @staticmethod
    def _obb_angle(
        points,
    ):
        p0, p1, p2, p3 = (
            points
        )

        def edge(
            a,
            b,
        ):
            dx = (
                float(
                    b[0]
                )
                - float(
                    a[0]
                )
            )

            dy = (
                float(
                    b[1]
                )
                - float(
                    a[1]
                )
            )

            return (
                math.hypot(
                    dx,
                    dy,
                ),
                dx,
                dy,
            )

        e01, dx01, dy01 = (
            edge(
                p0,
                p1,
            )
        )

        e12, dx12, dy12 = (
            edge(
                p1,
                p2,
            )
        )

        if e01 >= e12:
            dx = dx01
            dy = dy01
        else:
            dx = dx12
            dy = dy12

        return (
            math.degrees(
                math.atan2(
                    dy,
                    dx,
                )
            )
            % 180.0
        )

    @staticmethod
    def _canonical_square_angle(
        angle_deg,
    ):
        # A square is equivalent every 90°.
        return (
            float(
                angle_deg
            )
            % 90.0
        )

    @staticmethod
    def _circular_square_mean(
        angles,
    ):
        """
        Mean of square orientation.
        Multiply by 4 because square orientation repeats every 90°.
        """

        if not angles:
            return None

        sx = 0.0
        sy = 0.0

        for angle in angles:
            radians = math.radians(
                4.0
                * float(
                    angle
                )
            )

            sx += math.cos(
                radians
            )

            sy += math.sin(
                radians
            )

        result = (
            math.degrees(
                math.atan2(
                    sy,
                    sx,
                )
            )
            / 4.0
        )

        return (
            result
            % 90.0
        )

    def get_requested_class(self):
        if self.follow_external_target:
            requested = str(
                self.app_context.shared_data.get(
                    "wrist_requested_class",
                    "all",
                )
            )

            if requested in (
                "all",
                "cube_red",
                "cube_green",
                "cube_blue",
                "cube_yellow",
            ):
                return requested

        return self.manual_target_class

    def choose_target(
        self,
        detections,
        width,
        height,
    ):
        requested = (
            self.get_requested_class()
        )

        candidates = [
            detection
            for detection in detections
            if str(
                detection.get(
                    "class",
                    "",
                )
            ).startswith(
                "cube_"
            )
            and (
                requested == "all"
                or detection.get(
                    "class"
                )
                == requested
            )
        ]

        if not candidates:
            return None

        # Anti-jump association:
        # if we already track one cube, prefer the same neighborhood.
        if self.smoothed_target is not None:
            old_center = (
                self.smoothed_target.get(
                    "center"
                )
            )

            if old_center is not None:
                gate = max(
                    10.0,
                    float(self.target_gate_px),
                )

                gated = [
                    detection
                    for detection
                    in candidates
                    if math.hypot(
                        float(
                            detection["center"][0]
                        )
                        - float(
                            old_center[0]
                        ),
                        float(
                            detection["center"][1]
                        )
                        - float(
                            old_center[1]
                        ),
                    )
                    <= gate
                ]

                if gated:
                    return min(
                        gated,
                        key=lambda detection:
                            math.hypot(
                                float(
                                    detection[
                                        "center"
                                    ][0]
                                )
                                - float(
                                    old_center[0]
                                ),
                                float(
                                    detection[
                                        "center"
                                    ][1]
                                )
                                - float(
                                    old_center[1]
                                ),
                            ),
                    )

        # First acquisition = closest cube to image center.
        cx = (
            width / 2.0
        )

        cy = (
            height / 2.0
        )

        return min(
            candidates,
            key=lambda detection:
                math.hypot(
                    float(
                        detection[
                            "center"
                        ][0]
                    )
                    - cx,
                    float(
                        detection[
                            "center"
                        ][1]
                    )
                    - cy,
                ),
        )

    # ========================================================
    # Smoothing / stability
    # ========================================================

    def reset_tracking_history(self):
        self.target_history.clear()
        self.smoothed_target = None

    def update_tracking_filter(
        self,
        target,
    ):
        if target is None:
            return None

        # If class changed, reset history completely.
        if (
            self.smoothed_target
            is not None
            and self.smoothed_target.get(
                "class"
            )
            != target.get(
                "class"
            )
        ):
            self.reset_tracking_history()

        sample = {
            "class":
                str(
                    target[
                        "class"
                    ]
                ),
            "confidence":
                float(
                    target[
                        "confidence"
                    ]
                ),
            "center":
                (
                    float(
                        target[
                            "center"
                        ][0]
                    ),
                    float(
                        target[
                            "center"
                        ][1]
                    ),
                ),
            "angle_deg":
                self._canonical_square_angle(
                    target[
                        "angle_deg"
                    ]
                ),
            "points":
                target[
                    "points"
                ],
            "timestamp":
                time.monotonic(),
        }

        desired_window = max(
            3,
            int(self.stability_window_var.get()),
        )

        if (
            self.target_history.maxlen
            != desired_window
        ):
            previous = list(
                self.target_history
            )[
                -desired_window:
            ]

            self.target_history = deque(
                previous,
                maxlen=
                    desired_window,
            )

        self.target_history.append(
            sample
        )

        alpha = max(
            0.05,
            min(
                1.0,
                float(self.smooth_center_alpha_var.get()),
            ),
        )

        if self.smoothed_target is None:
            smoothed_center = (
                sample[
                    "center"
                ]
            )
        else:
            old_center = (
                self.smoothed_target[
                    "center"
                ]
            )

            smoothed_center = (
                (
                    1.0 - alpha
                )
                * old_center[0]
                + alpha
                * sample[
                    "center"
                ][0],

                (
                    1.0 - alpha
                )
                * old_center[1]
                + alpha
                * sample[
                    "center"
                ][1],
            )

        angles = [
            item[
                "angle_deg"
            ]
            for item in self.target_history
        ]

        angle_mean = (
            self._circular_square_mean(
                angles
            )
        )

        confidence_mean = (
            sum(
                item[
                    "confidence"
                ]
                for item
                in self.target_history
            )
            / len(
                self.target_history
            )
        )

        self.smoothed_target = {
            "class":
                sample[
                    "class"
                ],
            "confidence":
                confidence_mean,
            "center":
                smoothed_center,
            "angle_deg":
                float(
                    angle_mean
                ),
            "points":
                sample[
                    "points"
                ],
            "timestamp":
                sample[
                    "timestamp"
                ],
        }

        return (
            self.smoothed_target
        )

    def get_stability_result(self):
        history = list(
            self.target_history
        )

        minimum_frames = max(
            3,
            int(self.stability_min_frames_var.get()),
        )

        if len(history) < minimum_frames:
            return {
                "stable": False,
                "reason":
                    f"collecting {len(history)}/{minimum_frames}",
            }

        # Use last minimum_frames to avoid very old samples blocking lock.
        samples = history[
            -minimum_frames:
        ]

        classes = {
            item[
                "class"
            ]
            for item in samples
        }

        if len(classes) != 1:
            return {
                "stable": False,
                "reason":
                    "class changed",
            }

        min_conf = min(
            item[
                "confidence"
            ]
            for item in samples
        )

        required_conf = float(self.stability_min_conf_var.get())

        if min_conf < required_conf:
            return {
                "stable": False,
                "reason":
                    f"confidence {min_conf:.2f} < {required_conf:.2f}",
            }

        xs = [
            item[
                "center"
            ][0]
            for item in samples
        ]

        ys = [
            item[
                "center"
            ][1]
            for item in samples
        ]

        center_jitter = max(
            max(xs) - min(xs),
            max(ys) - min(ys),
        )

        center_limit = float(self.stability_center_jitter_var.get())

        if center_jitter > center_limit:
            return {
                "stable": False,
                "reason":
                    f"center jitter {center_jitter:.1f}px",
                "center_jitter_px":
                    center_jitter,
            }

        mean_angle = (
            self._circular_square_mean(
                [
                    item[
                        "angle_deg"
                    ]
                    for item in samples
                ]
            )
        )

        angle_errors = [
            abs(
                self._square_axis_error(
                    item[
                        "angle_deg"
                    ],
                    mean_angle,
                )
            )
            for item in samples
        ]

        angle_jitter = max(
            angle_errors
        )

        angle_limit = float(self.stability_angle_jitter_var.get())

        if angle_jitter > angle_limit:
            return {
                "stable": False,
                "reason":
                    f"angle jitter {angle_jitter:.1f}°",
                "angle_jitter_deg":
                    angle_jitter,
            }

        cx = sum(
            xs
        ) / len(
            xs
        )

        cy = sum(
            ys
        ) / len(
            ys
        )

        return {
            "stable": True,
            "reason": "stable",
            "class":
                samples[-1][
                    "class"
                ],
            "confidence":
                sum(
                    item[
                        "confidence"
                    ]
                    for item in samples
                )
                / len(
                    samples
                ),
            "center":
                (
                    cx,
                    cy,
                ),
            "angle_deg":
                float(
                    mean_angle
                ),
            "center_jitter_px":
                float(
                    center_jitter
                ),
            "angle_jitter_deg":
                float(
                    angle_jitter
                ),
            "frames":
                len(
                    samples
                ),
            "timestamp":
                time.monotonic(),
        }

    def lock_stable_target(self):
        """
        Freeze the current STABLE visual observation.

        This is the key handoff point:
        approach first -> observe until stable -> LOCK -> only then Module 8
        may rotate wrist_roll using this frozen angle.
        """
        result = self.get_stability_result()

        if not result.get("stable", False):
            self.lock_status_var.set(
                "Lock refused: "
                + str(result.get("reason", "not stable"))
            )
            return False

        self.grasp_lock = {
            key: value
            for key, value in result.items()
            if key not in ("stable", "reason")
        }

        self.app_context.shared_data[
            "wrist_grasp_lock"
        ] = dict(self.grasp_lock)

        self.lock_status_var.set(
            f"LOCKED: {self.grasp_lock['class']} | "
            f"angle={self.grasp_lock['angle_deg']:.1f}° | "
            f"conf={self.grasp_lock['confidence']:.2f}"
        )

        return True

    def unlock_target(self):
        self.grasp_lock = None

        self.app_context.shared_data[
            "wrist_grasp_lock"
        ] = None

        self.lock_status_var.set(
            "Lock: none"
        )

        self.reset_tracking_history()

    # ========================================================
    # Detection
    # ========================================================

    def run_detection(
        self,
        clean_frame,
    ):
        if (
            self.model is None
            or not self.detection_enabled_var.get()
        ):
            self.latest_detections = []
            self.current_target = None

            self.app_context.shared_data[
                "wrist_obb_detections"
            ] = []

            self.app_context.shared_data[
                "wrist_tracking_target"
            ] = None

            return

        now = time.monotonic()

        max_fps = max(
            0.5,
            float(
                self.max_fps_var.get()
            ),
        )

        if (
            now
            - self.last_inference_time
            < 1.0
            / max_fps
        ):
            return

        self.last_inference_time = now

        try:
            prediction = (
                self.model.predict(
                    source=
                        clean_frame,
                    conf=max(
                        0.01,
                        float(
                            self.conf_var.get()
                        ),
                    ),
                    imgsz=640,
                    verbose=False,
                )[0]
            )

        except Exception as error:
            self.ai_status_var.set(
                f"OBB inference error: {error}"
            )
            return

        detections = []

        names = (
            self.model.names
        )

        if prediction.obb is not None:
            for obb in prediction.obb:
                cls_id = int(
                    obb.cls[0].item()
                )

                confidence = float(
                    obb.conf[0].item()
                )

                label = (
                    names[
                        cls_id
                    ]
                    if isinstance(
                        names,
                        dict,
                    )
                    else names[
                        cls_id
                    ]
                )

                points = (
                    obb
                    .xyxyxyxy[0]
                    .detach()
                    .cpu()
                    .tolist()
                )

                cx = (
                    sum(
                        float(
                            point[0]
                        )
                        for point
                        in points
                    )
                    / 4.0
                )

                cy = (
                    sum(
                        float(
                            point[1]
                        )
                        for point
                        in points
                    )
                    / 4.0
                )

                detections.append(
                    {
                        "class":
                            str(
                                label
                            ),
                        "confidence":
                            confidence,
                        "center":
                            (
                                cx,
                                cy,
                            ),
                        "angle_deg":
                            self._obb_angle(
                                points
                            ),
                        "points":
                            points,
                    }
                )

        self.latest_detections = detections

        h, w = (
            clean_frame.shape[:2]
        )

        self.current_target = (
            self.choose_target(
                detections,
                w,
                h,
            )
        )

        smoothed = (
            self.update_tracking_filter(
                self.current_target
            )
            if self.current_target
            is not None
            else None
        )

        stability = (
            self.get_stability_result()
            if smoothed is not None
            else {
                "stable": False,
                "reason":
                    "no target",
            }
        )

        shared = (
            self.app_context
            .shared_data
        )

        shared[
            "wrist_obb_detections"
        ] = [
            dict(
                detection
            )
            for detection
            in detections
        ]

        shared[
            "wrist_tracking_target"
        ] = (
            None
            if smoothed is None
            else dict(
                smoothed
            )
        )

        shared[
            "wrist_stable_target"
        ] = (
            dict(
                stability
            )
            if stability.get(
                "stable",
                False,
            )
            else None
        )

        shared[
            "wrist_frame_size"
        ] = (
            w,
            h,
        )

        self.ai_status_var.set(
            f"OBB: running | "
            f"detections="
            f"{len(detections)}"
        )

        if smoothed is None:
            self.target_status_var.set(
                f"Target: — | "
                f"requested="
                f"{self.get_requested_class()}"
            )

            self.stability_status_var.set(
                "Stability: no target"
            )

        else:
            dx = (
                smoothed[
                    "center"
                ][0]
                - w / 2.0
            )

            dy = (
                smoothed[
                    "center"
                ][1]
                - h / 2.0
            )

            self.target_status_var.set(
                f"Target: "
                f"{smoothed['class']} | "
                f"conf="
                f"{smoothed['confidence']:.2f} | "
                f"dx={dx:+.0f}px "
                f"dy={dy:+.0f}px | "
                f"angle="
                f"{smoothed['angle_deg']:.1f}°"
            )

            if stability.get(
                "stable",
                False,
            ):
                self.stability_status_var.set(
                    f"Stability: READY | "
                    f"{stability['frames']} frames | "
                    f"center jitter="
                    f"{stability['center_jitter_px']:.1f}px | "
                    f"angle jitter="
                    f"{stability['angle_jitter_deg']:.1f}°"
                )
            else:
                self.stability_status_var.set(
                    "Stability: "
                    + str(
                        stability.get(
                            "reason",
                            "—",
                        )
                    )
                )

    # ========================================================
    # External Module-10 requests
    # ========================================================

    def process_external_requests(
        self,
    ):
        shared = (
            self.app_context
            .shared_data
        )

        if bool(
            shared.get(
                "wrist_unlock_request",
                False,
            )
        ):
            shared[
                "wrist_unlock_request"
            ] = False

            self.unlock_target()

        if bool(
            shared.get(
                "wrist_lock_request",
                False,
            )
        ):
            shared[
                "wrist_lock_request"
            ] = False

            self.lock_stable_target()


    # ========================================================
    # Image pipeline
    # ========================================================

    def orient_frame(
        self,
        frame,
    ):
        result = frame

        if self.flip_horizontal_var.get():
            result = cv2.flip(
                result,
                1,
            )

        if self.flip_vertical_var.get():
            result = cv2.flip(
                result,
                0,
            )

        rotation = int(
            self.rotation_var.get()
        )

        if rotation == 90:
            result = cv2.rotate(
                result,
                cv2.ROTATE_90_CLOCKWISE,
            )

        elif rotation == 180:
            result = cv2.rotate(
                result,
                cv2.ROTATE_180,
            )

        elif rotation == 270:
            result = cv2.rotate(
                result,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            )

        return result

    # ========================================================
    # Overlay
    # ========================================================

    def draw_overlay(
        self,
        frame,
    ):
        class_colors = {
            "cube_red":
                (
                    0,
                    0,
                    255,
                ),
            "cube_green":
                (
                    0,
                    200,
                    0,
                ),
            "cube_blue":
                (
                    255,
                    0,
                    0,
                ),
            "cube_yellow":
                (
                    0,
                    220,
                    255,
                ),
            "hand":
                (
                    180,
                    0,
                    180,
                ),
        }

        for detection in (
            self.latest_detections
        ):
            points = [
                (
                    int(
                        round(
                            point[0]
                        )
                    ),
                    int(
                        round(
                            point[1]
                        )
                    ),
                )
                for point
                in detection[
                    "points"
                ]
            ]

            is_raw_target = (
                detection
                is self.current_target
            )

            color = class_colors.get(
                detection[
                    "class"
                ],
                (
                    255,
                    255,
                    255,
                ),
            )

            thickness = (
                3
                if is_raw_target
                else 1
            )

            for i in range(
                4
            ):
                cv2.line(
                    frame,
                    points[i],
                    points[
                        (
                            i + 1
                        )
                        % 4
                    ],
                    color,
                    thickness,
                )

            cv2.putText(
                frame,
                (
                    f"{detection['class']} "
                    f"{detection['confidence']:.2f}"
                ),
                (
                    points[0][0],
                    max(
                        20,
                        points[0][1]
                        - 7,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )

        h, w = (
            frame.shape[:2]
        )

        cv2.drawMarker(
            frame,
            (
                int(
                    w / 2
                ),
                int(
                    h / 2
                ),
            ),
            (
                255,
                255,
                255,
            ),
            cv2.MARKER_CROSS,
            24,
            1,
        )

        # Smoothed tracking target.
        if self.smoothed_target is not None:
            cx, cy = (
                self.smoothed_target[
                    "center"
                ]
            )

            cv2.circle(
                frame,
                (
                    int(
                        round(
                            cx
                        )
                    ),
                    int(
                        round(
                            cy
                        )
                    ),
                ),
                7,
                (
                    0,
                    255,
                    255,
                ),
                2,
            )

        # Frozen grasp lock.
        if self.grasp_lock is not None:
            cx, cy = (
                self.grasp_lock[
                    "center"
                ]
            )

            cv2.drawMarker(
                frame,
                (
                    int(
                        round(
                            cx
                        )
                    ),
                    int(
                        round(
                            cy
                        )
                    ),
                ),
                (
                    255,
                    0,
                    255,
                ),
                cv2.MARKER_TILTED_CROSS,
                28,
                3,
            )

            cv2.putText(
                frame,
                "LOCKED",
                (
                    int(
                        round(
                            cx
                        )
                    )
                    + 10,
                    int(
                        round(
                            cy
                        )
                    )
                    - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (
                    255,
                    0,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )


    # ========================================================
    # Preview
    # ========================================================

    def draw_to_canvas(
        self,
        frame,
    ):
        canvas_w = (
            self.video_canvas
            .winfo_width()
        )

        canvas_h = (
            self.video_canvas
            .winfo_height()
        )

        if (
            canvas_w <= 1
            or canvas_h <= 1
        ):
            return

        h, w = (
            frame.shape[:2]
        )

        scale = min(
            canvas_w / w,
            canvas_h / h,
        )

        dw = max(
            1,
            int(
                w * scale
            ),
        )

        dh = max(
            1,
            int(
                h * scale
            ),
        )

        x = (
            canvas_w
            - dw
        ) // 2

        y = (
            canvas_h
            - dh
        ) // 2

        resized = cv2.resize(
            frame,
            (
                dw,
                dh,
            ),
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB,
        )

        self.photo = (
            ImageTk.PhotoImage(
                image=
                    Image.fromarray(
                        rgb
                    )
            )
        )

        if self.canvas_image_id is None:
            self.canvas_image_id = (
                self.video_canvas
                .create_image(
                    x,
                    y,
                    image=
                        self.photo,
                    anchor="nw",
                )
            )

        else:
            self.video_canvas.coords(
                self.canvas_image_id,
                x,
                y,
            )

            self.video_canvas.itemconfigure(
                self.canvas_image_id,
                image=
                    self.photo,
            )

    # ========================================================
    # Update
    # ========================================================

    def update_loop(self):
        try:
            self.process_external_requests()

            raw = (
                self.capture
                .get_frame()
            )

            if raw is not None:
                corrected = self.apply_undistortion(
                    raw
                )

                clean = (
                    self.orient_frame(
                        corrected
                    )
                )

                self.run_detection(
                    clean
                )

                display = (
                    clean.copy()
                )

                self.draw_overlay(
                    display
                )

                self.draw_to_canvas(
                    display
                )

                self.last_display_frame = (
                    display
                )

                self.fps_var.set(
                    f"FPS: "
                    f"{self.capture.fps:.1f}"
                )


            self.update_lens_status()

            if self.capture.running:
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
                    f"frame age "
                    f"{age_ms:.0f} ms"
                )

        except Exception as error:
            self.ai_status_var.set(
                f"Module error: {error}"
            )

        self.frame.after(
            self.UI_UPDATE_MS,
            self.update_loop,
        )

    # ========================================================
    # Module API
    # ========================================================

    def get_frame(self):
        return self.frame

    def shutdown(self):
        self.capture.shutdown()

        if (
            self.calibration_window is not None
            and self.calibration_window.winfo_exists()
        ):
            try:
                self.calibration_window.destroy()
            except Exception:
                pass

        self.app_context.log(
            "Wrist Vision module shutdown."
        )


# ============================================================
# WRIST CAMERA CALIBRATION PANEL
# ============================================================

class WristCameraCalibrationPanel:
    def __init__(
        self,
        parent,
        wrist_panel,
    ):
        self.wrist = (
            wrist_panel
        )

        self.frame = ttk.Frame(
            parent,
            padding=12,
        )

        self.status_var = tk.StringVar(
            value="Ready"
        )

        self.board_status_var = tk.StringVar(
            value="Board detected: —"
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
            text="WRIST CAMERA CALIBRATION",
            font=(
                "Segoe UI",
                18,
                "bold",
            ),
        ).pack(
            anchor="w"
        )

        ttk.Label(
            self.frame,
            text="Same 3×3 ArUco board — IDs 10–18",
            font=(
                "Segoe UI",
                11,
                "bold",
            ),
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
        ).pack(
            anchor="w"
        )

        ttk.Label(
            board,
            text=(
                "Move/tilt the WRIST CAMERA relative to the board. "
                "The camera does not need to stay fixed. "
                "Capture the board in the center, near all image edges, "
                "at several distances and tilts."
            ),
            wraplength=650,
        ).pack(
            anchor="w",
            pady=(6, 0),
        )

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
            textvariable=
                self.board_status_var,
        ).pack(
            anchor="w"
        )

        ttk.Label(
            live,
            textvariable=
                self.ids_var,
        ).pack(
            anchor="w"
        )

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
            textvariable=
                self.samples_var,
        ).pack(
            anchor="w"
        )

        ttk.Label(
            samples,
            textvariable=
                self.points_var,
        ).pack(
            anchor="w"
        )

        row = ttk.Frame(
            samples
        )

        row.pack(
            fill="x",
            pady=(6, 0),
        )

        ttk.Button(
            row,
            text="Capture Sample",
            command=
                self.capture_sample,
        ).pack(
            side="left",
            padx=3,
        )

        ttk.Button(
            row,
            text="Clear Samples",
            command=
                self.clear_samples,
        ).pack(
            side="left",
            padx=3,
        )

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
            command=
                self.calculate,
        ).pack(
            fill="x"
        )

        ttk.Label(
            calculate,
            textvariable=
                self.rms_var,
        ).pack(
            anchor="w",
            pady=(7, 0),
        )

        ttk.Label(
            calculate,
            textvariable=
                self.error_var,
        ).pack(
            anchor="w"
        )

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
            textvariable=
                self.matrix_var,
            wraplength=650,
        ).pack(
            anchor="w"
        )

        ttk.Label(
            result,
            textvariable=
                self.distortion_var,
            wraplength=650,
        ).pack(
            anchor="w",
            pady=5,
        )

        ttk.Button(
            result,
            text="Delete Saved Calibration",
            command=
                self.delete_calibration,
        ).pack(
            fill="x",
            pady=(5, 0),
        )

        status = ttk.LabelFrame(
            self.frame,
            text="Status",
            padding=8,
        )

        status.pack(
            fill="x"
        )

        ttk.Label(
            status,
            textvariable=
                self.status_var,
            wraplength=650,
        ).pack(
            anchor="w"
        )

    def get_live_board_info(self):
        raw = (
            self.wrist.capture
            .get_frame()
        )

        if raw is None:
            return []

        corners, ids, _ = (
            self.wrist
            .calibration_detector
            .detectMarkers(
                raw
            )
        )

        if ids is None:
            return []

        return [
            int(
                marker_id
            )
            for marker_id
            in ids.flatten()
            if int(
                marker_id
            )
            in self.wrist.CALIBRATION_IDS
        ]

    def capture_sample(self):
        result = (
            self.wrist
            .capture_calibration_sample()
        )

        self.status_var.set(
            result[
                "message"
            ]
        )

        self.update_sample_info()

    def clear_samples(self):
        self.wrist.clear_calibration_samples()

        self.update_sample_info()

        self.status_var.set(
            "Calibration samples cleared."
        )

    def calculate(self):
        result = (
            self.wrist
            .calculate_wrist_camera_calibration()
        )

        self.status_var.set(
            result[
                "message"
            ]
        )

        if not result[
            "ok"
        ]:
            return

        self.rms_var.set(
            f"RMS: "
            f"{result['rms']:.4f} px"
        )

        self.error_var.set(
            "Mean reprojection error: "
            f"{result['mean_error']:.4f} px"
        )

        self.update_existing_calibration_info()

    def delete_calibration(self):
        result = (
            self.wrist
            .delete_wrist_camera_calibration()
        )

        self.status_var.set(
            result[
                "message"
            ]
        )

        self.update_existing_calibration_info()

    def update_sample_info(self):
        samples = len(
            self.wrist
            .calibration_image_points
        )

        total_points = sum(
            len(
                sample
            )
            for sample
            in self.wrist
            .calibration_image_points
        )

        self.samples_var.set(
            f"Samples: "
            f"{samples} / "
            f"{self.wrist.CALIB_RECOMMENDED_SAMPLES}"
        )

        self.points_var.set(
            f"Captured corners: "
            f"{total_points}"
        )

    def update_existing_calibration_info(self):
        matrix = (
            self.wrist
            .camera_matrix
        )

        dist = (
            self.wrist
            .dist_coeffs
        )

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
                    dist.reshape(
                        -1
                    ),
                    precision=5,
                    suppress_small=True,
                )
            )

        if (
            self.wrist
            .calibration_rms
            is not None
        ):
            self.rms_var.set(
                "RMS: "
                f"{self.wrist.calibration_rms:.4f} px"
            )

        if (
            self.wrist
            .calibration_mean_error
            is not None
        ):
            self.error_var.set(
                "Mean reprojection error: "
                f"{self.wrist.calibration_mean_error:.4f} px"
            )

    def update_loop(self):
        try:
            visible_ids = (
                self.get_live_board_info()
            )

            self.board_status_var.set(
                f"Board detected: "
                f"{len(visible_ids)} / 9"
            )

            self.ids_var.set(
                "Visible IDs: "
                + (
                    ", ".join(
                        str(
                            marker_id
                        )
                        for marker_id
                        in visible_ids
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
