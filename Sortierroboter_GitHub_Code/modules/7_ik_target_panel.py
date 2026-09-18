import json
import math
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from core import commands
from robotics.forward_kinematics import (
    JointAngles,
    RobotGeometry,
    forward_kinematics,
)
from robotics.ik_solver_v2 import IKSolverV2, IKGeometry


PROJECT_DIR = Path(__file__).resolve().parents[1]
IK_CALIBRATION_FILE = PROJECT_DIR / "data" / "ik_calibration.json"
IK_TARGET_CALIBRATION_FILE = PROJECT_DIR / "data" / "ik_target_calibration.json"

JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
]

JOINT_SHORT = {
    "shoulder_pan.pos": "Base",
    "shoulder_lift.pos": "Shoulder",
    "elbow_flex.pos": "Elbow",
    "wrist_flex.pos": "Wrist",
}


def normalize_angle_deg(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


def distance_2d(a, b):
    return math.hypot(
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1]),
    )


def distance_3d(a, b):
    return math.sqrt(
        sum(
            (float(a[i]) - float(b[i])) ** 2
            for i in range(3)
        )
    )


class ModulePanel:
    title = "IK Target"

    UPDATE_MS = 100

    def __init__(self, parent, app_context):
        self.app_context = app_context
        self.worker = app_context.robot_worker
        self.frame = ttk.Frame(parent, padding=10)

        self.old_calibration = self._load_json(
            IK_CALIBRATION_FILE,
            {"poses": {}, "geometry": {}},
        )
        self.calibration = self._load_json(
            IK_TARGET_CALIBRATION_FILE,
            {},
        )

        self._load_model_from_calibration()
        self._load_z_compensation()

        # --------------------------------------------------
        # IK target / follow
        # --------------------------------------------------

        self.target_source_var = tk.StringVar(
            value=str(
                self.app_context.shared_data.get("ik_target_source", "ID8")
            )
        )

        self.target_z_var = tk.DoubleVar(
            value=float(
                self.app_context.shared_data.get("ik_target_z", 50.0)
            )
        )

        self.last_solution = None

        self.follow_enabled = False
        self.follow_interval = 0.20
        self.last_follow_time = 0.0
        self.follow_deadband_mm = 5.0
        self.last_follow_target = None
        self.filtered_target = None
        self.follow_filter_alpha = 0.25
        self.follow_max_step_deg = 4.0

        # --------------------------------------------------
        # Joint limit calibration
        # --------------------------------------------------

        self.limit_calibration_active = False
        self.limit_calibration_samples = {}

        # --------------------------------------------------
        # Frame calibration
        # --------------------------------------------------

        self.frame_samples = []
        self.sample_error_window = None
        self.frame_locked_target = None

        # --------------------------------------------------
        # Z compensation calibration
        # --------------------------------------------------

        self.spacer_height_var = tk.DoubleVar(value=50.0)
        self.measured_actual_z_var = tk.DoubleVar(value=50.0)
        loaded_z_samples = list(
            self.calibration
            .get("z_compensation", {})
            .get("samples", [])
        )

        # Only keep samples from the new empirical method:
        # command the robot to a known physical target, measure the
        # REAL gripper height, then store desired - measured error.
        self.z_samples = [
            sample
            for sample in loaded_z_samples
            if isinstance(sample, dict)
            and sample.get("measurement_mode") == "go_to_measured"
        ]

        # If the saved model came from the old FK-vs-spacer method,
        # do not apply it accidentally.
        if len(self.z_samples) != len(loaded_z_samples):
            self.z_comp_enabled = False
            self.z_comp_slope = 0.0
            self.z_comp_intercept = 0.0
            self.z_comp_reference_reach = 0.0
            self.z_comp_rms = None

        # --------------------------------------------------
        # UI variables
        # --------------------------------------------------

        self.robot_var = tk.StringVar(value="Robot: —")
        self.encoder_var = tk.StringVar(value="Encoder/FK: —")
        self.target_var = tk.StringVar(value="ID8: —")
        self.local_target_var = tk.StringVar(value="Robot local: —")
        self.ik_var = tk.StringVar(value="IK: —")
        self.follow_var = tk.StringVar(value="Follow: OFF")

        self.model_var = tk.StringVar(value="Model: —")
        self.frame_var = tk.StringVar(value="Robot frame: —")
        self.limit_status_var = tk.StringVar(value="Joint limits: —")

        self.frame_samples_var = tk.StringVar(value="Frame samples: 0")
        self.frame_lock_var = tk.StringVar(value="Frame target: not locked")
        self.frame_result_var = tk.StringVar(value="Frame calibration: —")

        self.align_shoulder_var = tk.DoubleVar(
            value=float(self.offsets["shoulder_lift.pos"])
        )
        self.align_elbow_var = tk.DoubleVar(
            value=float(self.offsets["elbow_flex.pos"])
        )
        self.align_wrist_var = tk.DoubleVar(
            value=float(self.offsets["wrist_flex.pos"])
        )
        self.alignment_status_var = tk.StringVar(
            value="Model alignment: using saved mapping"
        )

        self.z_samples_var = tk.StringVar(value="Z samples: 0")
        self.z_lock_var = tk.StringVar(value="Z sample target: not locked")
        self.z_locked_target = None
        self.z_model_var = tk.StringVar(value="Z compensation: —")
        self.z_last_sample_var = tk.StringVar(value="Last Z sample: —")

        self.status_var = tk.StringVar(value="Ready")

        self.build_ui()

        self.update_joint_limit_status()
        self.update_z_status()

        self.publish_target_z()
        self.publish_model_calibration()

        self.update_loop()

    # ======================================================
    # Load / save
    # ======================================================

    @staticmethod
    def _load_json(path, default):
        path = Path(path)

        if not path.exists():
            return default

        try:
            return json.loads(
                path.read_text(encoding="utf-8")
            )
        except Exception:
            return default

    def _load_model_from_calibration(self):
        geom = self.calibration.get("geometry", {})
        mapping = self.calibration.get("model_mapping", {})
        frame_cal = self.calibration.get("robot_frame", {})

        self.l1 = float(geom.get("L1", 116.0))
        self.l2 = float(geom.get("L2", 135.0))
        self.l3 = float(geom.get("L3", 165.0))
        self.shoulder_height = float(
            geom.get("shoulder_height", 120.0)
        )
        self.base_to_shoulder = float(
            geom.get("base_to_shoulder_offset", 30.0)
        )

        self.signs = {
            "shoulder_pan.pos": float(
                mapping.get("shoulder_pan.pos", {}).get("sign", 1.0)
            ),
            "shoulder_lift.pos": float(
                mapping.get("shoulder_lift.pos", {}).get("sign", -1.0)
            ),
            "elbow_flex.pos": float(
                mapping.get("elbow_flex.pos", {}).get("sign", -1.0)
            ),
            "wrist_flex.pos": float(
                mapping.get("wrist_flex.pos", {}).get("sign", -1.0)
            ),
        }

        self.offsets = {
            "shoulder_pan.pos": float(
                mapping.get("shoulder_pan.pos", {}).get("offset", 0.0)
            ),
            "shoulder_lift.pos": float(
                mapping.get("shoulder_lift.pos", {}).get("offset", 150.0)
            ),
            "elbow_flex.pos": float(
                mapping.get("elbow_flex.pos", {}).get("offset", 210.0)
            ),
            "wrist_flex.pos": float(
                mapping.get("wrist_flex.pos", {}).get("offset", -60.0)
            ),
        }

        self.robot_frame_offset_x = float(
            frame_cal.get("offset_x", 0.0)
        )
        self.robot_frame_offset_y = float(
            frame_cal.get("offset_y", 0.0)
        )
        self.robot_frame_heading_offset = float(
            frame_cal.get("heading_offset_deg", 0.0)
        )

        self.joint_limits = self.calibration.get(
            "joint_limits",
            {},
        )

    def _load_z_compensation(self):
        data = self.calibration.get("z_compensation", {})

        self.z_comp_enabled = bool(
            data.get("enabled", False)
        )
        self.z_comp_slope = float(
            data.get("slope_mm_per_mm", 0.0)
        )
        self.z_comp_intercept = float(
            data.get("intercept_mm", 0.0)
        )
        self.z_comp_reference_reach = float(
            data.get("reference_reach_mm", 0.0)
        )
        self.z_comp_rms = data.get("rms_mm")

    def save_calibration(self):
        data = self._load_json(
            IK_TARGET_CALIBRATION_FILE,
            {},
        )

        data["geometry"] = {
            "L1": float(self.l1),
            "L2": float(self.l2),
            "L3": float(self.l3),
            "shoulder_height": float(self.shoulder_height),
            "base_to_shoulder_offset": float(
                self.base_to_shoulder
            ),
        }

        data["model_mapping"] = {
            joint: {
                "sign": float(self.signs[joint]),
                "offset": float(self.offsets[joint]),
            }
            for joint in JOINTS
        }

        data["robot_frame"] = {
            "offset_x": float(self.robot_frame_offset_x),
            "offset_y": float(self.robot_frame_offset_y),
            "heading_offset_deg": float(
                self.robot_frame_heading_offset
            ),
        }

        data["joint_limits"] = self.joint_limits

        data["z_compensation"] = {
            "enabled": bool(self.z_comp_enabled),
            "model": "linear",
            "reference_reach_mm": float(
                self.z_comp_reference_reach
            ),
            "slope_mm_per_mm": float(
                self.z_comp_slope
            ),
            "intercept_mm": float(
                self.z_comp_intercept
            ),
            "rms_mm": (
                None
                if self.z_comp_rms is None
                else float(self.z_comp_rms)
            ),
            "samples": self.z_samples,
        }

        IK_TARGET_CALIBRATION_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        IK_TARGET_CALIBRATION_FILE.write_text(
            json.dumps(data, indent=4),
            encoding="utf-8",
        )

        self.calibration = data

    def publish_model_calibration(self):
        revision = time.time()

        self.app_context.shared_data[
            "robot_model_calibration"
        ] = {
            "revision": revision,
            "geometry": {
                "L1": float(self.l1),
                "L2": float(self.l2),
                "L3": float(self.l3),
                "shoulder_height": float(
                    self.shoulder_height
                ),
                "base_to_shoulder_offset": float(
                    self.base_to_shoulder
                ),
            },
            "mapping": {
                joint: {
                    "sign": float(self.signs[joint]),
                    "offset": float(self.offsets[joint]),
                }
                for joint in JOINTS
            },
        }

        self.app_context.shared_data[
            "robot_frame_calibration"
        ] = {
            "offset_x": float(self.robot_frame_offset_x),
            "offset_y": float(self.robot_frame_offset_y),
            "heading_offset_deg": float(
                self.robot_frame_heading_offset
            ),
        }

        self.app_context.shared_data[
            "ik_z_compensation"
        ] = {
            "enabled": bool(self.z_comp_enabled),
            "reference_reach_mm": float(
                self.z_comp_reference_reach
            ),
            "slope_mm_per_mm": float(
                self.z_comp_slope
            ),
            "intercept_mm": float(
                self.z_comp_intercept
            ),
        }

    # ======================================================
    # UI
    # ======================================================

    def build_ui(self):
        canvas = tk.Canvas(
            self.frame,
            highlightthickness=0,
        )

        scrollbar = ttk.Scrollbar(
            self.frame,
            orient="vertical",
            command=canvas.yview,
        )

        content = ttk.Frame(
            canvas,
            padding=4,
        )

        window_id = canvas.create_window(
            (0, 0),
            window=content,
            anchor="nw",
        )

        content.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            ),
        )

        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(
                window_id,
                width=e.width,
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

        ttk.Label(
            content,
            text="IK TARGET",
            font=("Segoe UI", 18, "bold"),
        ).pack(
            anchor="w",
            pady=(0, 10),
        )

        # --------------------------------------------------
        # Robot
        # --------------------------------------------------

        robot = ttk.LabelFrame(
            content,
            text="Robot",
            padding=8,
        )
        robot.pack(
            fill="x",
            pady=(0, 8),
        )

        row = ttk.Frame(robot)
        row.pack(fill="x")

        ttk.Button(
            row,
            text="Free Move",
            command=self.free_move,
        ).pack(side="left", padx=3)

        ttk.Button(
            row,
            text="Hold",
            command=self.hold,
        ).pack(side="left", padx=3)

        ttk.Label(
            robot,
            textvariable=self.robot_var,
        ).pack(
            anchor="w",
            pady=(6, 0),
        )

        ttk.Label(
            robot,
            textvariable=self.encoder_var,
            wraplength=900,
        ).pack(anchor="w")

        # --------------------------------------------------
        # Target / IK
        # --------------------------------------------------

        target = ttk.LabelFrame(
            content,
            text="ID8 Target",
            padding=8,
        )
        target.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Label(
            target,
            textvariable=self.target_var,
        ).pack(anchor="w")

        ttk.Label(
            target,
            textvariable=self.local_target_var,
        ).pack(anchor="w")

        source_row = ttk.Frame(target)
        source_row.pack(fill="x", pady=(5, 2))

        ttk.Label(
            source_row,
            text="Target source:",
        ).pack(side="left")

        ttk.Radiobutton(
            source_row,
            text="ID8 marker",
            value="ID8",
            variable=self.target_source_var,
            command=self.on_target_source_changed,
        ).pack(side="left", padx=5)

        ttk.Radiobutton(
            source_row,
            text="Mouse target",
            value="Mouse",
            variable=self.target_source_var,
            command=self.on_target_source_changed,
        ).pack(side="left", padx=5)

        zrow = ttk.Frame(target)
        zrow.pack(
            fill="x",
            pady=5,
        )

        ttk.Label(
            zrow,
            text="Physical target Z [mm]:",
        ).pack(side="left")

        z_entry = ttk.Entry(
            zrow,
            textvariable=self.target_z_var,
            width=10,
        )
        z_entry.pack(
            side="left",
            padx=5,
        )
        z_entry.bind(
            "<Return>",
            lambda e: self.publish_target_z(),
        )

        ttk.Button(
            target,
            text="SOLVE IK",
            command=self.solve_only,
        ).pack(
            fill="x",
            pady=2,
        )

        ttk.Button(
            target,
            text="GO TO ID8",
            command=self.go_to_id8,
        ).pack(
            fill="x",
            pady=2,
        )

        ttk.Label(
            target,
            textvariable=self.ik_var,
            wraplength=900,
        ).pack(
            anchor="w",
            pady=(5, 0),
        )

        # --------------------------------------------------
        # Follow
        # --------------------------------------------------

        follow = ttk.LabelFrame(
            content,
            text="Follow ID8",
            padding=8,
        )
        follow.pack(
            fill="x",
            pady=(0, 8),
        )

        row = ttk.Frame(follow)
        row.pack(fill="x")

        ttk.Button(
            row,
            text="START FOLLOW",
            command=self.start_follow,
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=2,
        )

        ttk.Button(
            row,
            text="STOP FOLLOW",
            command=self.stop_follow,
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=2,
        )

        ttk.Label(
            follow,
            textvariable=self.follow_var,
        ).pack(
            anchor="w",
            pady=(5, 0),
        )

        # --------------------------------------------------
        # Z compensation
        # --------------------------------------------------

        zbox = ttk.LabelFrame(
            content,
            text="Z Compensation Calibration",
            padding=8,
        )
        zbox.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Label(
            zbox,
            text=(
                "Empirical calibration of the REAL robot. "
                "1) Put ID8 at a test position and LOCK it. "
                "2) Command the robot to the locked target WITHOUT Z compensation. "
                "3) Measure the real physical gripper-tip height with a ruler. "
                "4) Enter the measured height and Capture Error Sample. "
                "Repeat at different reaches."
            ),
            wraplength=900,
        ).pack(anchor="w")

        lock_row = ttk.Frame(zbox)
        lock_row.pack(
            fill="x",
            pady=(7, 3),
        )

        ttk.Button(
            lock_row,
            text="1. LOCK ID8",
            command=self.lock_id8_for_z_sample,
        ).pack(
            side="left",
            padx=(0, 5),
        )

        ttk.Button(
            lock_row,
            text="Clear Locked ID8",
            command=self.clear_locked_z_target,
        ).pack(
            side="left",
            padx=3,
        )

        ttk.Label(
            zbox,
            textvariable=self.z_lock_var,
            wraplength=900,
        ).pack(anchor="w", pady=(0, 5))

        desired_row = ttk.Frame(zbox)
        desired_row.pack(
            fill="x",
            pady=3,
        )

        ttk.Label(
            desired_row,
            text="Desired physical Z [mm]:",
        ).pack(side="left")

        ttk.Entry(
            desired_row,
            textvariable=self.spacer_height_var,
            width=10,
        ).pack(
            side="left",
            padx=5,
        )

        ttk.Button(
            desired_row,
            text="2. GO TO LOCKED TARGET (NO Z COMP)",
            command=self.go_to_locked_z_target,
        ).pack(
            side="left",
            padx=8,
        )

        measured_row = ttk.Frame(zbox)
        measured_row.pack(
            fill="x",
            pady=3,
        )

        ttk.Label(
            measured_row,
            text="Measured REAL gripper Z [mm]:",
        ).pack(side="left")

        ttk.Entry(
            measured_row,
            textvariable=self.measured_actual_z_var,
            width=10,
        ).pack(
            side="left",
            padx=5,
        )

        ttk.Button(
            measured_row,
            text="3. Capture Error Sample",
            command=self.capture_z_sample,
        ).pack(
            side="left",
            padx=8,
        )

        ttk.Button(
            measured_row,
            text="Clear Z Samples",
            command=self.clear_z_samples,
        ).pack(
            side="left",
            padx=3,
        )

        ttk.Label(
            zbox,
            textvariable=self.z_samples_var,
        ).pack(anchor="w", pady=(4, 0))

        ttk.Label(
            zbox,
            textvariable=self.z_last_sample_var,
            wraplength=900,
        ).pack(anchor="w")

        fit_row = ttk.Frame(zbox)
        fit_row.pack(
            fill="x",
            pady=(6, 3),
        )

        ttk.Button(
            fit_row,
            text="FIT LINEAR Z COMPENSATION",
            command=self.fit_z_compensation,
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 3),
        )

        ttk.Button(
            fit_row,
            text="Enable",
            command=self.enable_z_compensation,
        ).pack(side="left", padx=3)

        ttk.Button(
            fit_row,
            text="Disable",
            command=self.disable_z_compensation,
        ).pack(side="left", padx=3)

        ttk.Label(
            zbox,
            textvariable=self.z_model_var,
            wraplength=900,
        ).pack(
            anchor="w",
            pady=(4, 0),
        )

        # --------------------------------------------------
        # Manual model alignment
        # --------------------------------------------------

        alignment = ttk.LabelFrame(
            content,
            text="Manual Model Alignment (Encoder → FK)",
            padding=8,
        )
        alignment.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Label(
            alignment,
            text=(
                "Use this ONLY to align the encoder reconstruction with the real arm. "
                "Free Move the robot into a simple known pose (ideally L1/L2/L3 nearly "
                "straight). Adjust Shoulder / Elbow / Wrist offsets until the encoder "
                "overlay in the 3D Control Center visually matches the physical robot. "
                "This changes motor→model angle mapping, not robot-frame X/Y."
            ),
            wraplength=900,
        ).pack(anchor="w")

        ttk.Label(
            alignment,
            textvariable=self.alignment_status_var,
            wraplength=900,
        ).pack(
            anchor="w",
            pady=(6, 5),
        )

        self._build_alignment_row(
            alignment,
            "Shoulder offset",
            "shoulder_lift.pos",
            self.align_shoulder_var,
        )

        self._build_alignment_row(
            alignment,
            "Elbow offset",
            "elbow_flex.pos",
            self.align_elbow_var,
        )

        self._build_alignment_row(
            alignment,
            "Wrist offset",
            "wrist_flex.pos",
            self.align_wrist_var,
        )

        button_row = ttk.Frame(alignment)
        button_row.pack(
            fill="x",
            pady=(7, 0),
        )

        ttk.Button(
            button_row,
            text="Apply Live",
            command=self.apply_manual_alignment,
        ).pack(
            side="left",
            padx=2,
        )

        ttk.Button(
            button_row,
            text="Save Mapping",
            command=self.save_manual_alignment,
        ).pack(
            side="left",
            padx=2,
        )

        ttk.Button(
            button_row,
            text="Reset to Saved",
            command=self.reset_manual_alignment,
        ).pack(
            side="left",
            padx=2,
        )

        # --------------------------------------------------
        # Joint limits
        # --------------------------------------------------

        limits = ttk.LabelFrame(
            content,
            text="Joint Limit Calibration",
            padding=8,
        )
        limits.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Label(
            limits,
            text=(
                "Start → arm enters Free Move → slowly move Base / Shoulder / "
                "Elbow / Wrist through the SAFE physical range → Finish + Save."
            ),
            wraplength=900,
        ).pack(anchor="w")

        ttk.Label(
            limits,
            textvariable=self.limit_status_var,
            wraplength=900,
        ).pack(
            anchor="w",
            pady=(6, 4),
        )

        row = ttk.Frame(limits)
        row.pack(fill="x")

        ttk.Button(
            row,
            text="Start Limit Calibration",
            command=self.start_limit_calibration,
        ).pack(side="left", padx=2)

        ttk.Button(
            row,
            text="Finish + Save",
            command=self.finish_limit_calibration,
        ).pack(side="left", padx=2)

        ttk.Button(
            row,
            text="Clear Limits",
            command=self.clear_joint_limits,
        ).pack(side="left", padx=2)

        # --------------------------------------------------
        # Frame calibration
        # --------------------------------------------------

        frame_box = ttk.LabelFrame(
            content,
            text="Robot Frame Calibration (XY + Heading)",
            padding=8,
        )
        frame_box.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Label(
            frame_box,
            text=(
                "1) Put ID8 on the table while ID8 and ID4 are visible → LOCK ID8. "
                "2) Free Move → place the SAME physical gripper tip exactly in the "
                "locked ID8 center. ID8 may now be completely hidden. "
                "3) Capture Frame Sample. Repeat at several positions. "
                "This calibration adjusts ONLY robot-frame X/Y/heading; "
                "geometry is not modified."
            ),
            wraplength=900,
        ).pack(anchor="w")

        lock_row = ttk.Frame(frame_box)
        lock_row.pack(
            fill="x",
            pady=(7, 3),
        )

        ttk.Button(
            lock_row,
            text="1. LOCK ID8 FOR FRAME SAMPLE",
            command=self.lock_id8_for_frame_sample,
        ).pack(
            side="left",
            padx=(0, 5),
        )

        ttk.Button(
            lock_row,
            text="Clear Locked ID8",
            command=self.clear_locked_frame_target,
        ).pack(
            side="left",
            padx=3,
        )

        ttk.Label(
            frame_box,
            textvariable=self.frame_lock_var,
            wraplength=900,
        ).pack(
            anchor="w",
            pady=(0, 5),
        )

        ttk.Label(
            frame_box,
            textvariable=self.frame_samples_var,
        ).pack(
            anchor="w",
            pady=(3, 0),
        )

        row = ttk.Frame(frame_box)
        row.pack(
            fill="x",
            pady=5,
        )

        ttk.Button(
            row,
            text="2. Capture Frame Sample",
            command=self.capture_frame_sample,
        ).pack(side="left", padx=2)

        ttk.Button(
            row,
            text="Clear Samples",
            command=self.clear_frame_samples,
        ).pack(side="left", padx=2)

        ttk.Button(
            row,
            text="Show Sample Errors",
            command=self.show_sample_errors,
        ).pack(side="left", padx=8)

        ttk.Button(
            frame_box,
            text="CALCULATE FRAME CALIBRATION",
            command=self.calculate_frame_calibration,
        ).pack(
            fill="x",
            pady=2,
        )

        ttk.Label(
            frame_box,
            textvariable=self.frame_result_var,
            wraplength=900,
        ).pack(
            anchor="w",
            pady=(5, 0),
        )

        # --------------------------------------------------
        # Model info
        # --------------------------------------------------

        model = ttk.LabelFrame(
            content,
            text="Active Model",
            padding=8,
        )
        model.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Label(
            model,
            textvariable=self.model_var,
            wraplength=900,
        ).pack(anchor="w")

        ttk.Label(
            model,
            textvariable=self.frame_var,
            wraplength=900,
        ).pack(anchor="w")

        # --------------------------------------------------
        # Status
        # --------------------------------------------------

        status = ttk.LabelFrame(
            content,
            text="Status",
            padding=8,
        )
        status.pack(fill="x")

        ttk.Label(
            status,
            textvariable=self.status_var,
            wraplength=900,
        ).pack(anchor="w")

    # ======================================================
    # Geometry / mapping
    # ======================================================

    def get_geometry(self):
        return RobotGeometry(
            l1=self.l1,
            l2=self.l2,
            l3=self.l3,
            base_height=self.shoulder_height,
            base_to_shoulder_offset=self.base_to_shoulder,
        )

    def get_ik_geometry(self):
        return IKGeometry(
            l1=self.l1,
            l2=self.l2,
            l3=self.l3,
            shoulder_height=self.shoulder_height,
            base_to_shoulder_offset=self.base_to_shoulder,
        )

    def get_neutral_pose(self):
        return (
            self.old_calibration
            .get("poses", {})
            .get("neutral", {})
        )

    def motor_to_model(self, joint, motor):
        neutral = float(
            self.get_neutral_pose().get(joint, 0.0)
        )

        delta = normalize_angle_deg(
            float(motor) - neutral
        )

        return normalize_angle_deg(
            self.signs[joint] * delta
            + self.offsets[joint]
        )

    def positions_to_angles(self, positions):
        return JointAngles(
            base=self.motor_to_model(
                "shoulder_pan.pos",
                positions["shoulder_pan.pos"],
            ),
            shoulder=self.motor_to_model(
                "shoulder_lift.pos",
                positions["shoulder_lift.pos"],
            ),
            elbow=self.motor_to_model(
                "elbow_flex.pos",
                positions["elbow_flex.pos"],
            ),
            wrist=self.motor_to_model(
                "wrist_flex.pos",
                positions["wrist_flex.pos"],
            ),
        )

    def get_current_angles(self):
        state = self.worker.get_state_snapshot()
        positions = state.get("positions", {})

        if not all(
            joint in positions
            for joint in JOINTS
        ):
            raise RuntimeError(
                "Motor positions incomplete."
            )

        return self.positions_to_angles(
            positions
        )

    # ======================================================
    # Robust model -> motor conversion
    # ======================================================

    def _choose_equivalent_motor_value(
        self,
        joint,
        value,
        current_motor,
    ):
        """
        Motor values are angular and may have equivalent +/-360
        representations. Pick the equivalent closest to current
        position, and when calibrated limits exist, prefer one
        inside those limits.
        """

        candidates = [
            float(value) + 360.0 * k
            for k in (-2, -1, 0, 1, 2)
        ]

        if joint in self.joint_limits:
            item = self.joint_limits[joint]

            lo = float(item["min_unwrapped"])
            hi = float(item["max_unwrapped"])

            if lo > hi:
                lo, hi = hi, lo

            inside = [
                candidate
                for candidate in candidates
                if lo <= candidate <= hi
            ]

            if inside:
                return min(
                    inside,
                    key=lambda candidate:
                        abs(candidate - current_motor),
                )

            return None

        return min(
            candidates,
            key=lambda candidate:
                abs(candidate - current_motor),
        )

    def angles_to_motor_positions(
        self,
        target_angles,
    ):
        state = self.worker.get_state_snapshot()
        current_pos = state.get("positions", {})

        if not all(
            joint in current_pos
            for joint in JOINTS
        ):
            return None

        current_angles = self.get_current_angles()

        pairs = {
            "shoulder_pan.pos": (
                current_angles.base,
                target_angles.base,
            ),
            "shoulder_lift.pos": (
                current_angles.shoulder,
                target_angles.shoulder,
            ),
            "elbow_flex.pos": (
                current_angles.elbow,
                target_angles.elbow,
            ),
            "wrist_flex.pos": (
                current_angles.wrist,
                target_angles.wrist,
            ),
        }

        result = {}

        for joint, (
            current_model,
            target_model,
        ) in pairs.items():

            current_motor = float(
                current_pos[joint]
            )

            model_delta = normalize_angle_deg(
                float(target_model)
                - float(current_model)
            )

            raw_target = (
                current_motor
                + model_delta
                / float(self.signs[joint])
            )

            safe_target = (
                self._choose_equivalent_motor_value(
                    joint,
                    raw_target,
                    current_motor,
                )
            )

            if safe_target is None:
                return None

            result[joint] = float(
                safe_target
            )

        return result

    def on_target_source_changed(self):
        source = self.target_source_var.get()
        if source not in ("ID8", "Mouse"):
            source = "ID8"
            self.target_source_var.set(source)

        self.app_context.shared_data["ik_target_source"] = source

    # ======================================================
    # Coordinates
    # ======================================================

    def publish_target_z(self):
        self.app_context.shared_data[
            "ik_target_z"
        ] = float(
            self.target_z_var.get()
        )

    def get_marker_target_world(self):
        target = self.app_context.shared_data.get(
            "marker_target_world"
        )

        if target is None:
            return None

        # Established convention:
        # scene/world = x, -y
        return (
            float(target[0]),
            -float(target[1]),
        )

    def get_selected_target_world_xy(self):
        source = self.target_source_var.get()

        if source == "Mouse":
            target = self.app_context.shared_data.get(
                "target_world"
            )
            if target is None:
                return None
            return (
                float(target[0]),
                -float(target[1]),
            )

        return self.get_marker_target_world()

    def get_target_world(self):
        self.publish_target_z()

        target = self.get_selected_target_world_xy()

        if target is None:
            return None

        return (
            target[0],
            target[1],
            float(self.target_z_var.get()),
        )

    def get_base_world(self):
        base = self.app_context.shared_data.get(
            "robot_base_world"
        )

        if base is None:
            return None

        return (
            float(base[0]),
            -float(base[1]),
        )

    def get_heading(self, heading_extra=None):
        angle = self.app_context.shared_data.get(
            "robot_base_marker_angle_deg"
        )

        extra = (
            self.robot_frame_heading_offset
            if heading_extra is None
            else float(heading_extra)
        )

        if angle is None:
            return extra

        return normalize_angle_deg(
            -float(angle) + extra
        )

    @staticmethod
    def world_to_robot_local_raw(
        wx,
        wy,
        base,
        heading_deg,
    ):
        bx, by = base

        dx = float(wx) - float(bx)
        dy = float(wy) - float(by)

        h = math.radians(
            heading_deg
        )

        fx = math.cos(h)
        fy = math.sin(h)

        rx = math.sin(h)
        ry = -math.cos(h)

        local_y = (
            dx * fx
            + dy * fy
        )

        local_x = (
            dx * rx
            + dy * ry
        )

        return (
            local_x,
            local_y,
        )

    def world_to_robot_local(
        self,
        wx,
        wy,
        base,
        heading_deg,
        frame_offset_x=None,
        frame_offset_y=None,
    ):
        x, y = self.world_to_robot_local_raw(
            wx,
            wy,
            base,
            heading_deg,
        )

        ox = (
            self.robot_frame_offset_x
            if frame_offset_x is None
            else float(frame_offset_x)
        )

        oy = (
            self.robot_frame_offset_y
            if frame_offset_y is None
            else float(frame_offset_y)
        )

        # Same established convention as the old working IK:
        # positive calibration offset means camera-derived
        # coordinate was too large.
        return (
            x - ox,
            y - oy,
        )

    def get_target_local_xy(self):
        target = self.get_selected_target_world_xy()
        base = self.get_base_world()

        if target is None or base is None:
            return None

        heading = self.get_heading()

        return self.world_to_robot_local(
            target[0],
            target[1],
            base,
            heading,
        )

    # ======================================================
    # Z compensation
    # ======================================================

    def get_z_correction(self, reach):
        if not self.z_comp_enabled:
            return 0.0

        return (
            self.z_comp_intercept
            + self.z_comp_slope
            * (
                float(reach)
                - self.z_comp_reference_reach
            )
        )

    def lock_id8_for_z_sample(self):
        """
        Freeze the camera-derived target geometry for one empirical
        Z-error measurement. After this, ID8 may be hidden.
        """
        id8 = self.get_marker_target_world()
        base = self.get_base_world()

        raw_heading = self.app_context.shared_data.get(
            "robot_base_marker_angle_deg"
        )

        if id8 is None:
            self.status_var.set(
                "Cannot lock Z target: ID8 is not visible."
            )
            return

        if base is None or raw_heading is None:
            self.status_var.set(
                "Cannot lock Z target: need ID4 base + heading."
            )
            return

        heading = self.get_heading()

        local = self.world_to_robot_local(
            id8[0],
            id8[1],
            base,
            heading,
        )

        reach = math.hypot(
            local[0],
            local[1],
        )

        self.z_locked_target = {
            "id8_world": (
                float(id8[0]),
                float(id8[1]),
            ),
            "base_world": (
                float(base[0]),
                float(base[1]),
            ),
            "heading_deg": float(heading),
            "local_x_mm": float(local[0]),
            "local_y_mm": float(local[1]),
            "reach_mm": float(reach),
            "locked_at": float(time.time()),
        }

        self.z_lock_var.set(
            "Z target LOCKED | "
            f"reach={reach:.1f} mm | "
            f"local X={local[0]:.1f}, Y={local[1]:.1f} mm"
        )

        self.status_var.set(
            "ID8 locked. Set Desired physical Z, then GO TO LOCKED TARGET."
        )

    def clear_locked_z_target(self):
        self.z_locked_target = None

        self.z_lock_var.set(
            "Z sample target: not locked"
        )

    def solve_locked_z_target_no_compensation(self):
        """
        Solve the locked local XY at the requested physical Z while
        deliberately bypassing any existing Z compensation.

        This gives us a clean baseline from which the real-world
        vertical error can be measured.
        """
        locked = self.z_locked_target

        if locked is None:
            return {
                "ok": False,
                "message": "Lock ID8 first.",
            }

        try:
            desired_z = float(
                self.spacer_height_var.get()
            )
        except Exception:
            return {
                "ok": False,
                "message": "Invalid Desired physical Z.",
            }

        local_x = float(
            locked["local_x_mm"]
        )
        local_y = float(
            locked["local_y_mm"]
        )

        solver = IKSolverV2(
            self.get_ik_geometry()
        )

        solver.table_z = 0.0
        solver.minimum_link_z = 0.0
        solver.minimum_gripper_z = 0.0

        try:
            current_reference = self.get_current_angles()
        except Exception:
            current_reference = None

        result = solver.solve(
            x=local_x,
            y=local_y,
            z=desired_z,
            current_angles=current_reference,
        )

        if not result.reachable:
            return {
                "ok": False,
                "message": result.error_message,
            }

        rejected_limits = 0
        rejected_floor = 0

        for candidate in result.candidates:
            angles = candidate.angles

            if not self.validate_pose(angles):
                rejected_floor += 1
                continue

            motors = self.angles_to_motor_positions(
                angles
            )

            if motors is None:
                rejected_limits += 1
                continue

            self.app_context.shared_data[
                "ik_solution_angles"
            ] = {
                "base": float(angles.base),
                "shoulder": float(angles.shoulder),
                "elbow": float(angles.elbow),
                "wrist": float(angles.wrist),
            }

            return {
                "ok": True,
                "angles": angles,
                "candidate": candidate,
                "motors": motors,
                "desired_z": float(desired_z),
                "reach": float(
                    locked["reach_mm"]
                ),
            }

        return {
            "ok": False,
            "message": (
                "Locked target has IK candidates, but none pass "
                f"limits/safety. limit rejects={rejected_limits}, "
                f"floor rejects={rejected_floor}."
            ),
        }

    def go_to_locked_z_target(self):
        result = self.solve_locked_z_target_no_compensation()

        if not result["ok"]:
            self.status_var.set(
                result["message"]
            )
            return

        self.worker.send(
            commands.move_to_positions(
                result["motors"]
            )
        )

        self.last_solution = result["angles"]

        # Initialize the measurement field with the desired value,
        # so the user only needs to replace it with the ruler result.
        self.measured_actual_z_var.set(
            float(result["desired_z"])
        )

        self.status_var.set(
            "Robot commanded WITHOUT Z compensation. "
            "Measure the REAL gripper-tip height, enter it, then Capture Error Sample."
        )

    def capture_z_sample(self):
        locked = self.z_locked_target

        if locked is None:
            self.status_var.set(
                "Lock ID8 first."
            )
            return

        try:
            desired_z = float(
                self.spacer_height_var.get()
            )

            measured_z = float(
                self.measured_actual_z_var.get()
            )
        except Exception:
            self.status_var.set(
                "Invalid desired/measured Z value."
            )
            return

        reach = float(
            locked["reach_mm"]
        )

        # Empirical physical error:
        # desired 50, real 30 -> correction +20.
        correction = (
            desired_z - measured_z
        )

        state = self.worker.get_state_snapshot()
        positions = state.get("positions", {})

        sample = {
            "measurement_mode": "go_to_measured",
            "reach_mm": float(reach),
            "desired_z_mm": float(desired_z),
            "measured_real_z_mm": float(measured_z),
            "correction_mm": float(correction),
            "local_x_mm": float(
                locked["local_x_mm"]
            ),
            "local_y_mm": float(
                locked["local_y_mm"]
            ),
            "locked_id8_world": list(
                locked["id8_world"]
            ),
            "locked_base_world": list(
                locked["base_world"]
            ),
            "locked_heading_deg": float(
                locked["heading_deg"]
            ),
            "motors_after_go_to": {
                joint: float(positions[joint])
                for joint in JOINTS
                if joint in positions
            },
        }

        self.z_samples.append(sample)

        self.save_calibration()

        self.z_last_sample_var.set(
            "Last Z error sample | "
            f"reach={reach:.1f} mm | "
            f"desired={desired_z:.1f} mm | "
            f"measured REAL={measured_z:.1f} mm | "
            f"needed correction={correction:+.1f} mm"
        )

        self.update_z_status()

        # One lock = one empirical measurement.
        self.clear_locked_z_target()

        self.status_var.set(
            "Z error sample captured. Move ID8, LOCK again, and repeat."
        )

    def clear_z_samples(self):
        self.z_samples = []
        self.clear_locked_z_target()

        self.z_comp_enabled = False
        self.z_comp_slope = 0.0
        self.z_comp_intercept = 0.0
        self.z_comp_reference_reach = 0.0
        self.z_comp_rms = None

        self.save_calibration()
        self.publish_model_calibration()
        self.update_z_status()

        self.status_var.set(
            "Z samples and Z compensation cleared."
        )

    def fit_z_compensation(self):
        if len(self.z_samples) < 2:
            self.status_var.set(
                "Capture at least 2 Z samples at different reaches."
            )
            return

        reaches = [
            float(sample["reach_mm"])
            for sample in self.z_samples
        ]

        corrections = [
            float(sample["correction_mm"])
            for sample in self.z_samples
        ]

        min_reach = min(reaches)
        max_reach = max(reaches)

        if max_reach - min_reach < 20.0:
            self.status_var.set(
                "Reach spread is too small. "
                "Capture samples at clearly different distances."
            )
            return

        reference = (
            sum(reaches)
            / len(reaches)
        )

        xs = [
            reach - reference
            for reach in reaches
        ]

        ys = corrections

        x_mean = (
            sum(xs)
            / len(xs)
        )

        y_mean = (
            sum(ys)
            / len(ys)
        )

        denominator = sum(
            (x - x_mean) ** 2
            for x in xs
        )

        if abs(denominator) < 1e-12:
            self.status_var.set(
                "Cannot fit Z compensation."
            )
            return

        slope = (
            sum(
                (x - x_mean)
                * (y - y_mean)
                for x, y
                in zip(xs, ys)
            )
            / denominator
        )

        intercept = (
            y_mean
            - slope * x_mean
        )

        errors = []

        for x, y in zip(xs, ys):
            predicted = (
                intercept
                + slope * x
            )

            errors.append(
                y - predicted
            )

        rms = math.sqrt(
            sum(error * error for error in errors)
            / len(errors)
        )

        self.z_comp_reference_reach = float(reference)
        self.z_comp_slope = float(slope)
        self.z_comp_intercept = float(intercept)
        self.z_comp_rms = float(rms)
        self.z_comp_enabled = True

        self.save_calibration()
        self.publish_model_calibration()
        self.update_z_status()

        self.status_var.set(
            "Linear Z compensation fitted, enabled and saved."
        )

    def enable_z_compensation(self):
        self.z_comp_enabled = True
        self.save_calibration()
        self.publish_model_calibration()
        self.update_z_status()

    def disable_z_compensation(self):
        self.z_comp_enabled = False
        self.save_calibration()
        self.publish_model_calibration()
        self.update_z_status()

    def update_z_status(self):
        self.z_samples_var.set(
            f"Z samples: {len(self.z_samples)}"
        )

        state = (
            "ON"
            if self.z_comp_enabled
            else "OFF"
        )

        if (
            abs(self.z_comp_slope) < 1e-12
            and abs(self.z_comp_intercept) < 1e-12
            and self.z_comp_rms is None
        ):
            self.z_model_var.set(
                f"Z compensation: {state} | not fitted"
            )
            return

        rms_text = (
            "—"
            if self.z_comp_rms is None
            else f"{float(self.z_comp_rms):.2f} mm"
        )

        self.z_model_var.set(
            "Z compensation: "
            f"{state} | empirical correction = "
            f"{self.z_comp_intercept:+.2f} "
            f"{self.z_comp_slope:+.5f} × "
            f"(reach - {self.z_comp_reference_reach:.1f}) mm "
            f"| fit RMS={rms_text}"
        )

    # ======================================================
    # IK
    # ======================================================

    def validate_pose(self, angles):
        fk = forward_kinematics(
            angles,
            self.get_geometry(),
        )

        return all(
            float(fk[name][2]) >= -0.5
            for name in [
                "shoulder",
                "elbow",
                "wrist",
                "gripper",
            ]
        )

    def solve_target(
        self,
        target_override=None,
        current_reference=None,
    ):
        target = (
            target_override
            if target_override is not None
            else self.get_target_world()
        )

        if target is None:
            return {
                "ok": False,
                "message": "ID8 not visible.",
            }

        base = self.get_base_world()

        if base is None:
            return {
                "ok": False,
                "message": "ID4 base missing.",
            }

        heading = self.get_heading()

        tx, ty, physical_z = target

        local_x, local_y = (
            self.world_to_robot_local(
                tx,
                ty,
                base,
                heading,
            )
        )

        reach = math.hypot(
            local_x,
            local_y,
        )

        z_correction = self.get_z_correction(
            reach
        )

        solver_z = (
            float(physical_z)
            + z_correction
        )

        solver = IKSolverV2(
            self.get_ik_geometry()
        )

        solver.table_z = 0.0
        solver.minimum_link_z = 0.0

        # If Z compensation requires a mathematically negative
        # model target for a physical Z near the table, allow
        # the solver to represent it. Physical floor safety is
        # handled by the calibrated target interpretation.
        solver.minimum_gripper_z = min(
            0.0,
            solver_z,
        )

        if current_reference is None:
            try:
                current_reference = self.get_current_angles()
            except Exception:
                current_reference = None

        result = solver.solve(
            x=local_x,
            y=local_y,
            z=solver_z,
            current_angles=current_reference,
        )

        if not result.reachable:
            return {
                "ok": False,
                "message": result.error_message,
            }

        rejected_limits = 0
        rejected_floor = 0

        for candidate in result.candidates:
            angles = candidate.angles

            if not self.validate_pose(
                angles
            ):
                rejected_floor += 1
                continue

            motors = self.angles_to_motor_positions(
                angles
            )

            if motors is None:
                rejected_limits += 1
                continue

            self.app_context.shared_data[
                "ik_solution_angles"
            ] = {
                "base": float(angles.base),
                "shoulder": float(angles.shoulder),
                "elbow": float(angles.elbow),
                "wrist": float(angles.wrist),
            }

            self.app_context.shared_data[
                "ik_last_solution"
            ] = {
                "physical_target": (
                    float(tx),
                    float(ty),
                    float(physical_z),
                ),
                "robot_local_target": (
                    float(local_x),
                    float(local_y),
                    float(physical_z),
                ),
                "reach_mm": float(reach),
                "z_correction_mm": float(z_correction),
                "solver_z_mm": float(solver_z),
            }

            return {
                "ok": True,
                "angles": angles,
                "candidate": candidate,
                "motors": motors,
                "target": target,
                "target_local": (
                    local_x,
                    local_y,
                    physical_z,
                ),
                "reach": reach,
                "z_correction": z_correction,
                "solver_z": solver_z,
                "message": "IK OK",
            }

        return {
            "ok": False,
            "message": (
                "IK geometry has candidates, but none pass "
                f"robot limits/safety. "
                f"limit rejects={rejected_limits}, "
                f"floor rejects={rejected_floor}, "
                f"solver candidates={len(result.candidates)}"
            ),
        }

    def solve_only(self):
        result = self.solve_target()

        if not result["ok"]:
            self.ik_var.set(
                "IK REJECTED | "
                + result["message"]
            )
            self.status_var.set(
                result["message"]
            )
            return

        angles = result["angles"]
        self.last_solution = angles

        self.ik_var.set(
            "IK OK | "
            f"B={angles.base:.1f}° | "
            f"S={angles.shoulder:.1f}° | "
            f"E={angles.elbow:.1f}° | "
            f"W={angles.wrist:.1f}° | "
            f"tool={result['candidate'].tool_angle_deg:.1f}° | "
            f"reach={result['reach']:.1f} mm | "
            f"Zcorr={result['z_correction']:+.1f} mm | "
            f"solver Z={result['solver_z']:.1f} mm"
        )

        self.status_var.set(
            "IK solved and published to 3D."
        )

    def go_to_id8(self):
        result = self.solve_target()

        if not result["ok"]:
            self.status_var.set(
                result["message"]
            )
            return

        self.worker.send(
            commands.move_to_positions(
                result["motors"]
            )
        )

        self.last_solution = result["angles"]

        self.status_var.set(
            "GO TO ID8 command sent."
        )

    # ======================================================
    # Follow
    # ======================================================

    def start_follow(self):
        self.follow_enabled = True
        self.last_follow_time = 0.0
        self.last_follow_target = None
        self.filtered_target = None

        try:
            self.last_solution = self.get_current_angles()
        except Exception:
            self.last_solution = None

        self.follow_var.set(
            "Follow: ON"
        )

    def stop_follow(self):
        self.follow_enabled = False
        self.filtered_target = None
        self.last_follow_target = None

        self.follow_var.set(
            "Follow: OFF"
        )

    def update_follow(self):
        if not self.follow_enabled:
            return

        now = time.time()

        if (
            now - self.last_follow_time
            < self.follow_interval
        ):
            return

        self.last_follow_time = now

        raw = self.get_target_world()

        if raw is None:
            return

        if self.filtered_target is None:
            self.filtered_target = raw
        else:
            alpha = self.follow_filter_alpha

            self.filtered_target = (
                alpha * raw[0]
                + (1.0 - alpha)
                * self.filtered_target[0],

                alpha * raw[1]
                + (1.0 - alpha)
                * self.filtered_target[1],

                raw[2],
            )

        target = self.filtered_target

        if self.last_follow_target is not None:
            if (
                distance_2d(
                    target,
                    self.last_follow_target,
                )
                < self.follow_deadband_mm
            ):
                return

        result = self.solve_target(
            target_override=target,
            current_reference=self.last_solution,
        )

        if not result["ok"]:
            self.follow_var.set(
                "Follow: IK rejected"
            )
            return

        desired = result["motors"]

        state = self.worker.get_state_snapshot()

        commanded = state.get(
            "commanded_positions",
            {},
        )

        current = state.get(
            "positions",
            {},
        )

        safe = {}

        for joint, desired_value in desired.items():
            reference = commanded.get(
                joint,
                current.get(joint),
            )

            if reference is None:
                continue

            delta = (
                float(desired_value)
                - float(reference)
            )

            delta = max(
                -self.follow_max_step_deg,
                min(
                    self.follow_max_step_deg,
                    delta,
                ),
            )

            safe[joint] = (
                float(reference)
                + delta
            )

        if safe:
            self.worker.send(
                commands.move_to_positions(
                    safe
                )
            )

        self.last_solution = result["angles"]
        self.last_follow_target = target

    # ======================================================
    # Manual model alignment
    # ======================================================

    def _build_alignment_row(
        self,
        parent,
        label,
        joint,
        variable,
    ):
        row = ttk.Frame(parent)
        row.pack(
            fill="x",
            pady=3,
        )

        ttk.Label(
            row,
            text=label,
            width=18,
        ).pack(side="left")

        ttk.Button(
            row,
            text="-5",
            width=4,
            command=lambda v=variable:
                self.change_alignment_value(v, -5.0),
        ).pack(side="left", padx=1)

        ttk.Button(
            row,
            text="-1",
            width=4,
            command=lambda v=variable:
                self.change_alignment_value(v, -1.0),
        ).pack(side="left", padx=1)

        ttk.Entry(
            row,
            textvariable=variable,
            width=10,
        ).pack(
            side="left",
            padx=5,
        )

        ttk.Button(
            row,
            text="+1",
            width=4,
            command=lambda v=variable:
                self.change_alignment_value(v, 1.0),
        ).pack(side="left", padx=1)

        ttk.Button(
            row,
            text="+5",
            width=4,
            command=lambda v=variable:
                self.change_alignment_value(v, 5.0),
        ).pack(side="left", padx=1)

        ttk.Label(
            row,
            text="deg",
        ).pack(
            side="left",
            padx=(5, 0),
        )

        # Enter applies immediately as well.
        entry_widgets = [
            child
            for child in row.winfo_children()
            if isinstance(child, ttk.Entry)
        ]

        if entry_widgets:
            entry_widgets[-1].bind(
                "<Return>",
                lambda event:
                    self.apply_manual_alignment(),
            )

    def change_alignment_value(
        self,
        variable,
        delta,
    ):
        try:
            variable.set(
                float(variable.get())
                + float(delta)
            )

            self.apply_manual_alignment()

        except Exception as error:
            self.status_var.set(
                f"Alignment value error: {error}"
            )

    def apply_manual_alignment(self):
        """
        Apply mapping offsets in memory and publish them immediately.

        No file is written here, so the user can experiment freely.
        The 3D Control Center receives the new mapping through
        shared_data["robot_model_calibration"].
        """
        try:
            self.offsets[
                "shoulder_lift.pos"
            ] = float(
                self.align_shoulder_var.get()
            )

            self.offsets[
                "elbow_flex.pos"
            ] = float(
                self.align_elbow_var.get()
            )

            self.offsets[
                "wrist_flex.pos"
            ] = float(
                self.align_wrist_var.get()
            )

        except Exception as error:
            self.status_var.set(
                f"Invalid alignment value: {error}"
            )
            return

        self.publish_model_calibration()

        self.alignment_status_var.set(
            "Model alignment LIVE | "
            f"Shoulder={self.offsets['shoulder_lift.pos']:.1f}° | "
            f"Elbow={self.offsets['elbow_flex.pos']:.1f}° | "
            f"Wrist={self.offsets['wrist_flex.pos']:.1f}°"
        )

        self.status_var.set(
            "Manual model alignment applied live to encoder/FK reconstruction."
        )

    def save_manual_alignment(self):
        self.apply_manual_alignment()

        self.save_calibration()
        self.publish_model_calibration()

        self.alignment_status_var.set(
            "Model alignment SAVED | "
            f"Shoulder={self.offsets['shoulder_lift.pos']:.1f}° | "
            f"Elbow={self.offsets['elbow_flex.pos']:.1f}° | "
            f"Wrist={self.offsets['wrist_flex.pos']:.1f}°"
        )

        self.status_var.set(
            "Motor→model mapping saved."
        )

    def reset_manual_alignment(self):
        saved = self._load_json(
            IK_TARGET_CALIBRATION_FILE,
            {},
        )

        mapping = saved.get(
            "model_mapping",
            {},
        )

        defaults = {
            "shoulder_lift.pos": 150.0,
            "elbow_flex.pos": 210.0,
            "wrist_flex.pos": -60.0,
        }

        for joint, variable in [
            (
                "shoulder_lift.pos",
                self.align_shoulder_var,
            ),
            (
                "elbow_flex.pos",
                self.align_elbow_var,
            ),
            (
                "wrist_flex.pos",
                self.align_wrist_var,
            ),
        ]:
            value = float(
                mapping.get(
                    joint,
                    {},
                ).get(
                    "offset",
                    defaults[joint],
                )
            )

            variable.set(value)
            self.offsets[joint] = value

        self.publish_model_calibration()

        self.alignment_status_var.set(
            "Model alignment: reset to saved mapping"
        )

        self.status_var.set(
            "Manual alignment reset to saved values."
        )

    # ======================================================
    # Joint limits
    # ======================================================

    def start_limit_calibration(self):
        state = self.worker.get_state_snapshot()

        if not state.get("connected", False):
            self.status_var.set(
                "Robot not connected."
            )
            return

        self.limit_calibration_active = True

        self.limit_calibration_samples = {
            joint: {
                "min_unwrapped": None,
                "max_unwrapped": None,
                "last_raw": None,
                "unwrapped": None,
                "samples": 0,
            }
            for joint in JOINTS
        }

        self.worker.send(
            commands.free_move()
        )

        self.limit_status_var.set(
            "Joint limits: CALIBRATING"
        )

        self.status_var.set(
            "Move every joint slowly through its safe range."
        )

    def update_limit_calibration(self):
        if not self.limit_calibration_active:
            return

        positions = (
            self.worker
            .get_state_snapshot()
            .get("positions", {})
        )

        for joint in JOINTS:
            if joint not in positions:
                continue

            raw = float(
                positions[joint]
            )

            data = self.limit_calibration_samples[
                joint
            ]

            if data["last_raw"] is None:
                data["last_raw"] = raw
                data["unwrapped"] = raw
            else:
                delta = normalize_angle_deg(
                    raw - data["last_raw"]
                )

                data["unwrapped"] += delta
                data["last_raw"] = raw

            value = float(
                data["unwrapped"]
            )

            if data["min_unwrapped"] is None:
                data["min_unwrapped"] = value
                data["max_unwrapped"] = value
            else:
                data["min_unwrapped"] = min(
                    data["min_unwrapped"],
                    value,
                )

                data["max_unwrapped"] = max(
                    data["max_unwrapped"],
                    value,
                )

            data["samples"] += 1

        parts = []

        for joint in JOINTS:
            data = self.limit_calibration_samples[
                joint
            ]

            if data["min_unwrapped"] is None:
                continue

            parts.append(
                f"{JOINT_SHORT[joint]} "
                f"{data['min_unwrapped']:.1f}..."
                f"{data['max_unwrapped']:.1f}"
            )

        if parts:
            self.limit_status_var.set(
                " | ".join(parts)
            )

    def finish_limit_calibration(self):
        if not self.limit_calibration_active:
            self.status_var.set(
                "Limit calibration is not active."
            )
            return

        self.limit_calibration_active = False

        valid = {}

        for joint, data in (
            self.limit_calibration_samples.items()
        ):
            if (
                data["min_unwrapped"] is None
                or data["max_unwrapped"] is None
            ):
                continue

            valid[joint] = {
                "min_unwrapped": float(
                    data["min_unwrapped"]
                ),
                "max_unwrapped": float(
                    data["max_unwrapped"]
                ),
                "samples": int(
                    data["samples"]
                ),
            }

        self.joint_limits = valid

        self.save_calibration()

        self.worker.send(
            commands.hold_current()
        )

        self.update_joint_limit_status()

        self.status_var.set(
            f"Saved limits for {len(valid)} joints."
        )

    def clear_joint_limits(self):
        self.limit_calibration_active = False
        self.joint_limits = {}

        self.save_calibration()
        self.update_joint_limit_status()

        self.status_var.set(
            "IK joint limits cleared."
        )

    def update_joint_limit_status(self):
        if not self.joint_limits:
            self.limit_status_var.set(
                "Joint limits: not calibrated"
            )
            return

        parts = []

        for joint in JOINTS:
            if joint not in self.joint_limits:
                continue

            data = self.joint_limits[
                joint
            ]

            parts.append(
                f"{JOINT_SHORT[joint]} "
                f"{float(data['min_unwrapped']):.1f}..."
                f"{float(data['max_unwrapped']):.1f}"
            )

        self.limit_status_var.set(
            "Limits | "
            + " | ".join(parts)
        )

    # ======================================================
    # Frame calibration
    # ======================================================

    def get_current_motor_sample(self):
        positions = (
            self.worker
            .get_state_snapshot()
            .get("positions", {})
        )

        if not all(
            joint in positions
            for joint in JOINTS
        ):
            return None

        return {
            joint: float(positions[joint])
            for joint in JOINTS
        }

    def lock_id8_for_frame_sample(self):
        """
        Freeze camera-derived ID8/base/heading for one frame sample.

        After locking, ID8 and ID4 may be occluded. Capture later uses
        the locked camera geometry plus the CURRENT encoder positions.
        """
        id8 = self.get_marker_target_world()
        base = self.get_base_world()

        marker_heading = self.app_context.shared_data.get(
            "robot_base_marker_angle_deg"
        )

        if id8 is None:
            self.status_var.set(
                "Cannot lock frame target: ID8 is not visible."
            )
            return

        if base is None or marker_heading is None:
            self.status_var.set(
                "Cannot lock frame target: need visible ID4 base + heading."
            )
            return

        corrected_heading = self.get_heading()

        current_local = self.world_to_robot_local(
            id8[0],
            id8[1],
            base,
            corrected_heading,
        )

        self.frame_locked_target = {
            "world_x": float(id8[0]),
            "world_y": float(id8[1]),
            "base_x": float(base[0]),
            "base_y": float(base[1]),
            "marker_heading_deg": float(marker_heading),
            "preview_local_x_mm": float(current_local[0]),
            "preview_local_y_mm": float(current_local[1]),
            "locked_at": float(time.time()),
        }

        self.frame_lock_var.set(
            "Frame target LOCKED | "
            f"local≈({current_local[0]:.1f}, "
            f"{current_local[1]:.1f}) mm | "
            "ID8 may now be covered"
        )

        self.status_var.set(
            "Frame target locked. Free Move, place the gripper tip "
            "in the locked ID8 center, then Capture."
        )

    def clear_locked_frame_target(self):
        self.frame_locked_target = None

        self.frame_lock_var.set(
            "Frame target: not locked"
        )

    def capture_frame_sample(self):
        motors = self.get_current_motor_sample()

        if motors is None:
            self.status_var.set(
                "Robot position unavailable."
            )
            return

        locked = self.frame_locked_target

        if locked is None:
            self.status_var.set(
                "Lock ID8 for Frame Sample first. "
                "ID8 does not need to remain visible after locking."
            )
            return

        sample = {
            "motors": motors,
            "world": {
                "x": float(locked["world_x"]),
                "y": float(locked["world_y"]),
                "z": 0.0,
            },
            "base_world": {
                "x": float(locked["base_x"]),
                "y": float(locked["base_y"]),
            },
            "marker_heading_deg": float(
                locked["marker_heading_deg"]
            ),
        }

        self.frame_samples.append(sample)

        self.frame_samples_var.set(
            f"Frame samples: {len(self.frame_samples)}"
        )

        expected = self.get_expected_local_for_sample(
            sample
        )

        actual = self.calculate_fk_for_sample(
            sample
        )

        error = distance_3d(
            actual,
            expected,
        )

        # One lock = one sample.
        self.clear_locked_frame_target()

        self.status_var.set(
            "Frame sample captured | "
            f"error={error:.1f} mm | "
            "Move ID8 and LOCK again for the next sample."
        )

    def clear_frame_samples(self):
        self.frame_samples = []
        self.clear_locked_frame_target()

        self.frame_samples_var.set(
            "Frame samples: 0"
        )

        self.frame_result_var.set(
            "Frame calibration: —"
        )

    def calculate_fk_for_sample(
        self,
        sample,
    ):
        angles = self.positions_to_angles(
            sample["motors"]
        )

        fk = forward_kinematics(
            angles,
            self.get_geometry(),
        )

        return tuple(
            float(value)
            for value in fk["gripper"]
        )

    def get_expected_local_for_sample(
        self,
        sample,
        frame_offset_x=None,
        frame_offset_y=None,
        heading_offset=None,
    ):
        world = sample["world"]
        base_data = sample["base_world"]

        base = (
            float(base_data["x"]),
            float(base_data["y"]),
        )

        marker_heading = float(
            sample.get(
                "marker_heading_deg",
                0.0,
            )
        )

        h_extra = (
            self.robot_frame_heading_offset
            if heading_offset is None
            else float(heading_offset)
        )

        heading = normalize_angle_deg(
            -marker_heading
            + h_extra
        )

        x, y = self.world_to_robot_local(
            float(world["x"]),
            float(world["y"]),
            base,
            heading,
            frame_offset_x=frame_offset_x,
            frame_offset_y=frame_offset_y,
        )

        return (
            x,
            y,
            0.0,
        )

    def build_sample_error_rows(self):
        rows = []

        for index, sample in enumerate(
            self.frame_samples,
            start=1,
        ):
            expected = (
                self.get_expected_local_for_sample(
                    sample
                )
            )

            actual = (
                self.calculate_fk_for_sample(
                    sample
                )
            )

            dx = actual[0] - expected[0]
            dy = actual[1] - expected[1]
            dz = actual[2] - expected[2]

            error = math.sqrt(
                dx * dx
                + dy * dy
                + dz * dz
            )

            reach = math.hypot(
                expected[0],
                expected[1],
            )

            rows.append(
                {
                    "index": index,
                    "expected": expected,
                    "actual": actual,
                    "dx": dx,
                    "dy": dy,
                    "dz": dz,
                    "error": error,
                    "reach": reach,
                }
            )

        return rows

    def show_sample_errors(self):
        if (
            self.sample_error_window is not None
            and self.sample_error_window.winfo_exists()
        ):
            self.sample_error_window.destroy()

        self.sample_error_window = tk.Toplevel(
            self.frame
        )

        self.sample_error_window.title(
            "Frame Calibration Sample Errors"
        )

        self.sample_error_window.geometry(
            "1100x620"
        )

        outer = ttk.Frame(
            self.sample_error_window,
            padding=10,
        )

        outer.pack(
            fill="both",
            expand=True,
        )

        rows = self.build_sample_error_rows()

        if not rows:
            ttk.Label(
                outer,
                text="No frame samples.",
            ).pack(anchor="w")
            return

        columns = (
            "n",
            "expected",
            "fk",
            "dx",
            "dy",
            "dz",
            "error",
            "reach",
        )

        tree = ttk.Treeview(
            outer,
            columns=columns,
            show="headings",
        )

        headings = {
            "n": "#",
            "expected": "Expected XYZ",
            "fk": "FK XYZ",
            "dx": "dX",
            "dy": "dY",
            "dz": "dZ",
            "error": "3D Error",
            "reach": "Reach",
        }

        for column, title in headings.items():
            tree.heading(
                column,
                text=title,
            )

        widths = {
            "n": 45,
            "expected": 220,
            "fk": 220,
            "dx": 85,
            "dy": 85,
            "dz": 85,
            "error": 100,
            "reach": 100,
        }

        for column, width in widths.items():
            tree.column(
                column,
                width=width,
                anchor=(
                    "center"
                    if column == "n"
                    else "e"
                ),
            )

        scrollbar = ttk.Scrollbar(
            outer,
            orient="vertical",
            command=tree.yview,
        )

        tree.configure(
            yscrollcommand=scrollbar.set
        )

        tree.pack(
            side="left",
            fill="both",
            expand=True,
        )

        scrollbar.pack(
            side="right",
            fill="y",
        )

        for row in rows:
            expected = row["expected"]
            actual = row["actual"]

            tree.insert(
                "",
                "end",
                values=(
                    row["index"],
                    (
                        f"{expected[0]:.1f}, "
                        f"{expected[1]:.1f}, "
                        f"{expected[2]:.1f}"
                    ),
                    (
                        f"{actual[0]:.1f}, "
                        f"{actual[1]:.1f}, "
                        f"{actual[2]:.1f}"
                    ),
                    f"{row['dx']:+.1f}",
                    f"{row['dy']:+.1f}",
                    f"{row['dz']:+.1f}",
                    f"{row['error']:.1f}",
                    f"{row['reach']:.1f}",
                ),
            )

    def evaluate_frame_calibration(
        self,
        offset_x,
        offset_y,
        heading_offset,
    ):
        errors_sq = []

        for sample in self.frame_samples:
            actual = (
                self.calculate_fk_for_sample(
                    sample
                )
            )

            expected = (
                self.get_expected_local_for_sample(
                    sample,
                    frame_offset_x=offset_x,
                    frame_offset_y=offset_y,
                    heading_offset=heading_offset,
                )
            )

            dx = actual[0] - expected[0]
            dy = actual[1] - expected[1]
            dz = actual[2] - expected[2]

            errors_sq.append(
                dx * dx
                + dy * dy
                + dz * dz
            )

        if not errors_sq:
            return float("inf")

        return math.sqrt(
            sum(errors_sq)
            / len(errors_sq)
        )

    def calculate_frame_calibration(self):
        if len(self.frame_samples) < 5:
            self.status_var.set(
                "Capture at least 5 frame samples."
            )
            return

        x = float(
            self.robot_frame_offset_x
        )

        y = float(
            self.robot_frame_offset_y
        )

        heading = float(
            self.robot_frame_heading_offset
        )

        best_rms = (
            self.evaluate_frame_calibration(
                x,
                y,
                heading,
            )
        )

        schedules = [
            (10.0, 3.0),
            (5.0, 1.5),
            (2.0, 0.7),
            (1.0, 0.3),
            (0.5, 0.1),
            (0.2, 0.05),
        ]

        for xy_step, h_step in schedules:
            improved = True

            while improved:
                improved = False

                for name, step in [
                    ("x", xy_step),
                    ("y", xy_step),
                    ("heading", h_step),
                ]:
                    for direction in (
                        -1.0,
                        1.0,
                    ):
                        cx = x
                        cy = y
                        ch = heading

                        if name == "x":
                            cx += direction * step
                        elif name == "y":
                            cy += direction * step
                        else:
                            ch += direction * step

                        cx = max(
                            -80.0,
                            min(80.0, cx),
                        )

                        cy = max(
                            -80.0,
                            min(80.0, cy),
                        )

                        ch = max(
                            -20.0,
                            min(20.0, ch),
                        )

                        rms = (
                            self.evaluate_frame_calibration(
                                cx,
                                cy,
                                ch,
                            )
                        )

                        if rms < best_rms - 1e-6:
                            x = cx
                            y = cy
                            heading = ch
                            best_rms = rms
                            improved = True
                            break

                    if improved:
                        break

        self.robot_frame_offset_x = x
        self.robot_frame_offset_y = y
        self.robot_frame_heading_offset = heading

        self.save_calibration()
        self.publish_model_calibration()

        self.frame_result_var.set(
            "Frame calibration | "
            f"X={x:.2f} mm | "
            f"Y={y:.2f} mm | "
            f"Heading={heading:.2f}° | "
            f"RMS XYZ={best_rms:.2f} mm"
        )

        self.status_var.set(
            "Frame calibration saved and published."
        )

    # ======================================================
    # Robot commands
    # ======================================================

    def free_move(self):
        self.stop_follow()
        self.worker.send(
            commands.free_move()
        )

    def hold(self):
        self.worker.send(
            commands.hold_current()
        )

    # ======================================================
    # Update loop
    # ======================================================

    def update_loop(self):
        try:
            shared_source = str(
                self.app_context.shared_data.get(
                    "ik_target_source",
                    self.target_source_var.get(),
                )
            )

            if (
                shared_source in ("ID8", "Mouse")
                and shared_source != self.target_source_var.get()
            ):
                self.target_source_var.set(shared_source)

            state = self.worker.get_state_snapshot()

            connected = bool(
                state.get("connected", False)
            )

            torque = bool(
                state.get("torque_enabled", False)
            )

            if not connected:
                self.robot_var.set(
                    "Robot: offline"
                )
            else:
                self.robot_var.set(
                    "Robot: connected | "
                    + (
                        "Torque ON"
                        if torque
                        else "FREE MOVE"
                    )
                )

            if self.limit_calibration_active:
                self.update_limit_calibration()

            if connected:
                try:
                    angles = self.get_current_angles()

                    fk = forward_kinematics(
                        angles,
                        self.get_geometry(),
                    )

                    gripper = fk["gripper"]

                    self.encoder_var.set(
                        "Encoder/FK | "
                        f"B={angles.base:.1f}° "
                        f"S={angles.shoulder:.1f}° "
                        f"E={angles.elbow:.1f}° "
                        f"W={angles.wrist:.1f}° | "
                        f"Gripper local="
                        f"({gripper[0]:.1f}, "
                        f"{gripper[1]:.1f}, "
                        f"{gripper[2]:.1f})"
                    )
                except Exception:
                    pass
            else:
                self.encoder_var.set(
                    "Encoder/FK: —"
                )

            target = self.get_target_world()

            source_name = (
                "Mouse"
                if self.target_source_var.get() == "Mouse"
                else "ID8"
            )

            if target is None:
                self.target_var.set(
                    f"{source_name}: not available"
                )
                self.local_target_var.set(
                    "Robot local: —"
                )
            else:
                self.target_var.set(
                    f"{source_name} world | "
                    f"X={target[0]:.1f}, "
                    f"Y={target[1]:.1f}, "
                    f"physical Z={target[2]:.1f} mm"
                )

                local = self.get_target_local_xy()

                if local is None:
                    self.local_target_var.set(
                        "Robot local: —"
                    )
                else:
                    reach = math.hypot(
                        local[0],
                        local[1],
                    )

                    correction = (
                        self.get_z_correction(
                            reach
                        )
                    )

                    self.local_target_var.set(
                        "Robot local | "
                        f"X={local[0]:.1f}, "
                        f"Y={local[1]:.1f}, "
                        f"reach={reach:.1f} mm | "
                        f"Z correction={correction:+.1f} mm"
                    )

            self.model_var.set(
                "Geometry | "
                f"L1={self.l1:.2f} | "
                f"L2={self.l2:.2f} | "
                f"L3={self.l3:.2f} | "
                f"Shoulder Z={self.shoulder_height:.2f} | "
                f"Base offset={self.base_to_shoulder:.2f} | "
                f"Offsets S={self.offsets['shoulder_lift.pos']:.2f}° "
                f"E={self.offsets['elbow_flex.pos']:.2f}° "
                f"W={self.offsets['wrist_flex.pos']:.2f}°"
            )

            self.frame_var.set(
                "Robot frame | "
                f"X={self.robot_frame_offset_x:.2f} mm | "
                f"Y={self.robot_frame_offset_y:.2f} mm | "
                f"Heading={self.robot_frame_heading_offset:.2f}°"
            )

            if (
                self.follow_enabled
                and connected
                and torque
            ):
                self.update_follow()

        except Exception as error:
            self.status_var.set(
                f"IK loop error: {error}"
            )

        self.frame.after(
            self.UPDATE_MS,
            self.update_loop,
        )

    # ======================================================
    # Module API
    # ======================================================

    def get_frame(self):
        return self.frame

    def shutdown(self):
        self.stop_follow()

        if (
            self.sample_error_window is not None
            and self.sample_error_window.winfo_exists()
        ):
            try:
                self.sample_error_window.destroy()
            except Exception:
                pass
