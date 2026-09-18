from __future__ import annotations

import json
import math
import time
import tkinter as tk
import numpy as np
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
CUBE_TARGET_CALIBRATION_FILE = PROJECT_DIR / "data" / "cube_target_calibration.json"

WRIST_ROLL_JOINT = "wrist_roll.pos"

JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
]


def normalize_angle_deg(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


class BaseModulePanel:
    """
    Cube target diagnostic module.

    IMPORTANT:
    - Does NOT modify Module 7.
    - Uses the same saved geometry / model mapping / frame calibration / Z compensation.
    - Reads cube_detections from Module 3.
    - Locks one cube so YOLO jitter/occlusion cannot move the target during a test.
    - Adds a TEMPORARY cube-only X/Y correction for diagnosis.
    - Does not save those corrections into ik_target_calibration.json.
    """

    title = "Cube Pick Test"
    UPDATE_MS = 120

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

        self.locked_cube = None
        self.selected_index = 0

        self.target_z_var = tk.DoubleVar(value=40.0)

        # Manual trim is intentionally temporary. The learned calibration
        # below supplies the persistent position-dependent correction.
        self.correction_x_var = tk.DoubleVar(value=0.0)
        self.correction_y_var = tk.DoubleVar(value=0.0)

        # --------------------------------------------------
        # Cube-specific XYZ calibration
        # --------------------------------------------------
        # One sample now contains:
        #   - current raw cube X/Y
        #   - final manual X/Y correction after live trimming
        #   - commanded Z
        #   - measured REAL Z
        #
        # FIT SMART XYZ then learns:
        #   dX = f(X,Y)
        #   dY = f(X,Y)
        #   dZ = f(reach)
        #
        # This keeps the working Module-7 calibration untouched and adds
        # a cube-specific residual correction on top.
        self.cube_calibration = self._load_json(
            CUBE_TARGET_CALIBRATION_FILE,
            {},
        )

        xyz_data = self.cube_calibration.get(
            "xyz_model",
            self.cube_calibration.get("xy_model", {}),
        )

        self.cube_xyz_enabled = bool(
            xyz_data.get(
                "enabled",
                self.cube_calibration.get("xy_model", {}).get(
                    "enabled",
                    False,
                ),
            )
        )

        self.cube_xyz_coeff_x = [
            float(v)
            for v in xyz_data.get(
                "coeff_x",
                self.cube_calibration.get(
                    "xy_model",
                    {},
                ).get(
                    "coeff_x",
                    [0.0, 0.0, 0.0],
                ),
            )
        ]

        self.cube_xyz_coeff_y = [
            float(v)
            for v in xyz_data.get(
                "coeff_y",
                self.cube_calibration.get(
                    "xy_model",
                    {},
                ).get(
                    "coeff_y",
                    [0.0, 0.0, 0.0],
                ),
            )
        ]

        if len(self.cube_xyz_coeff_x) != 3:
            self.cube_xyz_coeff_x = [0.0, 0.0, 0.0]

        if len(self.cube_xyz_coeff_y) != 3:
            self.cube_xyz_coeff_y = [0.0, 0.0, 0.0]

        self.cube_z_slope = float(
            xyz_data.get("z_slope_mm_per_mm", 0.0)
        )
        self.cube_z_intercept = float(
            xyz_data.get("z_intercept_mm", 0.0)
        )
        self.cube_z_reference_reach = float(
            xyz_data.get("z_reference_reach_mm", 0.0)
        )

        self.cube_xyz_samples = list(
            self.cube_calibration.get(
                "xyz_samples",
                [],
            )
        )

        # Backward compatibility: old XY samples can still be displayed,
        # but FIT SMART XYZ needs real measured Z, so only xyz_samples are fitted.
        self.cube_xyz_rms_xy = xyz_data.get("rms_xy_mm")
        self.cube_xyz_rms_z = xyz_data.get("rms_z_mm")

        self.cube_xyz_enabled_var = tk.BooleanVar(
            value=self.cube_xyz_enabled
        )

        self.measured_z_var = tk.DoubleVar(
            value=float(self.target_z_var.get())
        )

        self.cube_cal_var = tk.StringVar(
            value="Cube XYZ calibration: —"
        )

        # --------------------------------------------------
        # Safe HOME return
        # --------------------------------------------------
        self.home_sequence_active = False
        self.home_sequence_phase = None
        self.home_sequence_target = None

        # HOME motion remains deliberately staged for safe return.
        self.home_step_shoulder_deg = 10.0
        self.home_step_elbow_deg = 10.0
        self.home_step_wrist_deg = 8.0
        self.home_step_base_deg = 7.0
        self.home_arrival_deg = 2.5
        self.home_phase_timeout_s = 5.0
        self.home_phase_started_at = None
        self.home_last_position = None
        self.home_last_motion_at = None

        # --------------------------------------------------
        # Gripper / one-cube pick-and-place
        # --------------------------------------------------
        gripper_data = self.cube_calibration.get("gripper", {})

        # Physical travel limits captured with Free Move:
        # fully open and fully closed with NO object.
        self.gripper_open_var = tk.DoubleVar(
            value=float(gripper_data.get("open_limit", gripper_data.get("open", 15.0)))
        )
        self.gripper_close_limit_var = tk.DoubleVar(
            value=float(gripper_data.get("close_limit", gripper_data.get("grip", 35.0)))
        )

        # RobotWorker currently exposes position feedback, but not servo load/current.
        # Therefore "force" is a resistance proxy:
        # commanded position gets ahead of measured position when an object resists.
        self.grip_strength_percent_var = tk.DoubleVar(
            value=float(gripper_data.get("strength_percent", 50.0))
        )
        self.grip_resistance_100_var = tk.DoubleVar(
            value=float(gripper_data.get("resistance_100_deg", 5.0))
        )
        self.grip_required_hits_var = tk.IntVar(
            value=int(gripper_data.get("required_hits", 3))
        )

        self.gripper_step_deg = 1.5
        self.gripper_step_interval_s = 0.16
        self.gripper_arrival_deg = 1.0

        self.gripper_close_started_at = None
        self.gripper_last_command_at = 0.0
        self.gripper_resistance_hits = 0
        self.grip_proxy_percent = 0.0
        self.grip_force_var = tk.StringVar(
            value="Grip force proxy: 0%"
        )

        self.travel_z_var = tk.DoubleVar(value=50.0)

        # --------------------------------------------------
        # Independent GRASP calibration layer
        # --------------------------------------------------
        grasp_data = self.cube_calibration.get("grasp_model", {})

        self.grasp_layer_enabled = bool(
            grasp_data.get("enabled", False)
        )

        self.grasp_coeff_x = [
            float(v)
            for v in grasp_data.get("coeff_x", [0.0, 0.0, 0.0])
        ]
        self.grasp_coeff_y = [
            float(v)
            for v in grasp_data.get("coeff_y", [0.0, 0.0, 0.0])
        ]

        if len(self.grasp_coeff_x) != 3:
            self.grasp_coeff_x = [0.0, 0.0, 0.0]
        if len(self.grasp_coeff_y) != 3:
            self.grasp_coeff_y = [0.0, 0.0, 0.0]

        # Grasp Z is learned as an ABSOLUTE physical target versus reach.
        # Therefore Pick & Place no longer needs a manual Grasp Z field.
        self.grasp_z_slope = float(
            grasp_data.get("z_slope_mm_per_mm", 0.0)
        )
        self.grasp_z_intercept = float(
            grasp_data.get("z_intercept_mm", 18.0)
        )
        self.grasp_z_reference_reach = float(
            grasp_data.get("z_reference_reach_mm", 0.0)
        )

        self.grasp_samples = list(
            self.cube_calibration.get("grasp_samples", [])
        )
        self.grasp_rms_xy = grasp_data.get("rms_xy_mm")
        self.grasp_rms_z = grasp_data.get("rms_z_mm")

        self.grasp_layer_enabled_var = tk.BooleanVar(
            value=self.grasp_layer_enabled
        )

        # Manual live trim for GRASP layer only.
        self.grasp_trim_x_var = tk.DoubleVar(value=0.0)
        self.grasp_trim_y_var = tk.DoubleVar(value=0.0)
        self.grasp_trim_z_var = tk.DoubleVar(value=-22.0)

        self.grasp_cal_var = tk.StringVar(
            value="Grasp calibration: —"
        )

        # Dedicated vertical clearance after gripping.
        # The robot MUST complete this lift before any horizontal move.
        self.lift_clear_z_var = tk.DoubleVar(value=55.0)

        self.container_travel_z_var = tk.DoubleVar(value=90.0)
        self.drop_z_var = tk.DoubleVar(value=90.0)

        # --------------------------------------------------
        # Position-dependent CONTAINER XYZ calibration
        # --------------------------------------------------
        # This is NOT tied to a particular marker ID.  The learned correction
        # depends on the container position in robot-local X/Y, so containers
        # can be moved anywhere in the workspace between demo runs.
        container_cal = self.cube_calibration.get("container_model", {})
        self.container_cal_enabled = bool(container_cal.get("enabled", False))
        self.container_coeff_x = [float(v) for v in container_cal.get("coeff_x", [0.0, 0.0, 0.0])]
        self.container_coeff_y = [float(v) for v in container_cal.get("coeff_y", [0.0, 0.0, 0.0])]
        self.container_coeff_z = [float(v) for v in container_cal.get("coeff_z", [0.0, 0.0, 0.0])]
        if len(self.container_coeff_x) != 3:
            self.container_coeff_x = [0.0, 0.0, 0.0]
        if len(self.container_coeff_y) != 3:
            self.container_coeff_y = [0.0, 0.0, 0.0]
        if len(self.container_coeff_z) != 3:
            self.container_coeff_z = [0.0, 0.0, 0.0]

        self.container_cal_samples = list(
            self.cube_calibration.get("container_samples", [])
        )
        self.container_cal_enabled_var = tk.BooleanVar(
            value=self.container_cal_enabled
        )
        self.container_trim_x_var = tk.DoubleVar(value=0.0)
        self.container_trim_y_var = tk.DoubleVar(value=0.0)
        self.container_trim_z_var = tk.DoubleVar(value=0.0)
        self.container_cal_var = tk.StringVar(value="Container XYZ calibration: —")
        self.container_cal_marker_var = tk.IntVar(value=5)

        self.container_map = {
            "cube_red": 5,
            "cube_yellow": 6,
            "cube_green": 7,
            "cube_blue": 19,
        }

        self.pick_place_active = False
        self.pick_place_state = "IDLE"
        self.pick_place_state_started_at = None
        self.pick_phase_result = None
        self.lift_progress_z = None
        self.lift_waypoints_z = []
        self.container_travel_waypoints = []
        self.container_waypoint_index = 0

        # Short vertical safety lift after GRIP.
        # Only 3 Z targets, accepted by the existing stall-driven motion logic.
        self.safe_lift_targets = []
        self.safe_lift_index = 0

        # Each motion phase solves its IK target ONCE on entry.
        # The state machine then drives toward that frozen motor target instead
        # of re-solving every update tick.
        self.pick_phase_result = None

        # After gripping, raise Z in small slices instead of asking IK for one
        # large jump directly from grasp height to lift-clear height.
        self.lift_progress_z = None

        # Faster vertical lift profile after gripping.
        # We use a few meaningful Z waypoints instead of many tiny 10 mm slices.
        self.lift_waypoints_z = []

        # Intermediate lift waypoints are only trajectory guides:
        # once every active joint is "close enough", immediately continue.
        # This avoids oscillating around a waypoint and shaking the cube.
        self.lift_intermediate_accept_deg = 7.0
        self.lift_final_accept_deg = 5.0

        # Do not keep re-commanding joints that are already close enough.
        self.lift_joint_deadband_deg = 3.0

        # Container travel is also segmented through intermediate XY waypoints.
        self.container_travel_waypoints = []
        self.container_waypoint_index = 0

        self.pick_place_timeout_s = 18.0
        self.pick_arrival_deg = 3.0
        self.pick_stall_accept_deg = 6.0

        # GUI-adjustable phase acceptance values.
        # These are intentionally exposed so real robot residuals can be tuned
        # without editing the script after every test.
        self.approach_accept_var = tk.DoubleVar(value=18.0)
        self.grasp_accept_var = tk.DoubleVar(value=15.0)
        self.grasp_final_snap_var = tk.DoubleVar(value=8.0)
        self.grasp_snap_accept_var = tk.DoubleVar(value=4.5)
        self.lift_accept_var = tk.DoubleVar(value=12.0)
        self.container_accept_var = tk.DoubleVar(value=12.0)
        self.drop_accept_var = tk.DoubleVar(value=12.0)
        self.motion_timeout_var = tk.DoubleVar(value=18.0)

        # Normal task phases are intentionally stall-driven:
        # once the real encoders have stopped changing for this long,
        # the phase is considered complete. Precision numbers remain diagnostic.
        self.normal_phase_stall_s = 1.5

        # Final descent is allowed to accept a small residual if the arm has
        # physically stopped close to the grasp pose. This prevents the state
        # machine from waiting 18 s for one servo to dither around a few degrees.
        self.descend_stall_accept_deg = 15.0
        self.descend_stall_accept_after_s = 2.0

        self.pick_phase_last_positions = None
        self.pick_phase_last_motion_at = None
        self.grasp_snap_started_at = None

        self.pick_place_var = tk.StringVar(value="Pick & Place: IDLE")
        self.manual_grip_active = False

        # Wrist alignment is intentionally staged:
        # APPROACH -> OBSERVE -> FREEZE ORIENTATION -> ROTATE.
        self.wrist_locked_angle_deg = None
        self.wrist_locked_class = None
        self.wrist_locked_roll_target = None
        self.wrist_observation_active = False

        self.wrist_deadband_deg = 3.0
        self.wrist_max_step_deg = 8.0
        self.wrist_servo_sign = -1.0

        self.wrist_var = tk.StringVar(
            value="Wrist: approach cube first; orientation not locked"
        )

        self.cube_var = tk.StringVar(value="Cube: —")
        self.raw_var = tk.StringVar(value="Raw: —")
        self.corrected_var = tk.StringVar(value="Corrected: —")
        self.ik_var = tk.StringVar(value="IK: —")
        self.robot_var = tk.StringVar(value="Robot: —")
        self.status_var = tk.StringVar(
            value="Lock a cube, approach it, then use wrist observation/alignment."
        )

        self._build_ui()
        self.update_cube_calibration_status()
        self.update_grasp_calibration_status()
        self.update_loop()

    @staticmethod
    def _load_json(path, default):
        path = Path(path)
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _load_model_from_calibration(self):
        geom = self.calibration.get("geometry", {})
        mapping = self.calibration.get("model_mapping", {})
        frame_cal = self.calibration.get("robot_frame", {})

        self.l1 = float(geom.get("L1", 116.0))
        self.l2 = float(geom.get("L2", 135.0))
        self.l3 = float(geom.get("L3", 165.0))
        self.shoulder_height = float(geom.get("shoulder_height", 120.0))
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
        self.z_comp_enabled = bool(data.get("enabled", False))
        self.z_comp_slope = float(data.get("slope_mm_per_mm", 0.0))
        self.z_comp_intercept = float(data.get("intercept_mm", 0.0))
        self.z_comp_reference_reach = float(
            data.get("reference_reach_mm", 0.0)
        )

    # ------------------------------------------------------
    # UI
    # ------------------------------------------------------

    def _build_ui(self):
        # Entire Module 8 is scrollable because it contains calibration,
        # targeting, wrist, gripper and pick/place controls.
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
            text="CUBE PICK TEST",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(
            content,
            text=(
                "Module 7 calibration is used as-is. "
                "This module handles cube targeting, cube-only XY correction, "
                "wrist orientation, gripper and one-cube pick & place."
            ),
            wraplength=950,
        ).pack(anchor="w", pady=(0, 8))

        # ==================================================
        # 1. ROBOT / HOME
        # ==================================================
        robot_box = ttk.LabelFrame(
            content,
            text="1. Robot / Home",
            padding=8,
        )
        robot_box.pack(fill="x", pady=(0, 8))

        row = ttk.Frame(robot_box)
        row.pack(fill="x")

        ttk.Button(
            row,
            text="Free Move",
            command=lambda: self.worker.send(
                commands.free_move()
            ),
        ).pack(side="left", padx=2)

        ttk.Button(
            row,
            text="Hold",
            command=lambda: self.worker.send(
                commands.hold_current()
            ),
        ).pack(side="left", padx=2)

        ttk.Button(
            row,
            text="CAPTURE HOME",
            command=self.capture_home,
        ).pack(side="left", padx=(12, 2))

        ttk.Button(
            row,
            text="SAFE HOME",
            command=self.go_home,
        ).pack(side="left", padx=2)

        ttk.Label(
            robot_box,
            textvariable=self.robot_var,
        ).pack(anchor="w", pady=(5, 0))

        # ==================================================
        # 2. CUBE TARGET + LIVE XY
        # ==================================================
        cube_box = ttk.LabelFrame(
            content,
            text="2. Cube Target",
            padding=8,
        )
        cube_box.pack(fill="x", pady=(0, 8))

        row = ttk.Frame(cube_box)
        row.pack(fill="x")

        ttk.Button(
            row,
            text="LOCK CUBE",
            command=self.lock_best_cube,
        ).pack(side="left", fill="x", expand=True, padx=(0, 2))

        ttk.Button(
            row,
            text="CLEAR CUBE",
            command=self.clear_lock,
        ).pack(side="left", fill="x", expand=True, padx=(2, 0))

        ttk.Label(
            cube_box,
            textvariable=self.cube_var,
            wraplength=950,
        ).pack(anchor="w", pady=(5, 0))

        ttk.Label(
            cube_box,
            textvariable=self.raw_var,
            wraplength=950,
        ).pack(anchor="w")

        # ==================================================
        # 3. XYZ CALIBRATION / LIVE TRIM
        # ==================================================
        correction = ttk.LabelFrame(
            content,
            text="3. Cube XYZ Calibration / Live Trim",
            padding=8,
        )
        correction.pack(fill="x", pady=(0, 8))

        ttk.Label(
            correction,
            text=(
                "Workflow: LOCK CUBE -> GO ABOVE CUBE at a known Z (for example 40 mm). "
                "Use X/Y buttons until the gripper is centered. Measure the REAL TCP height, "
                "enter it below, then CAPTURE XYZ SET. One sample stores X, Y and Z together. "
                "Repeat in several table positions, then FIT SMART XYZ."
            ),
            wraplength=950,
        ).pack(anchor="w", pady=(0, 6))

        self._correction_row(
            correction,
            "Local X trim [mm]",
            self.correction_x_var,
        )

        self._correction_row(
            correction,
            "Local Y trim [mm]",
            self.correction_y_var,
        )

        zrow = ttk.Frame(correction)
        zrow.pack(fill="x", pady=(5, 0))

        ttk.Label(
            zrow,
            text="Commanded Z [mm]",
            width=18,
        ).pack(side="left")

        ttk.Label(
            zrow,
            textvariable=self.target_z_var,
            width=9,
        ).pack(side="left", padx=(0, 12))

        ttk.Label(
            zrow,
            text="Measured REAL Z [mm]",
        ).pack(side="left")

        ttk.Entry(
            zrow,
            textvariable=self.measured_z_var,
            width=9,
        ).pack(side="left", padx=5)

        row = ttk.Frame(correction)
        row.pack(fill="x", pady=(7, 0))

        ttk.Button(
            row,
            text="RESET MANUAL XY",
            command=self.reset_correction,
        ).pack(side="left", fill="x", expand=True, padx=(0, 2))

        ttk.Button(
            row,
            text="CAPTURE XYZ SET",
            command=self.capture_cube_xyz_set,
        ).pack(side="left", fill="x", expand=True, padx=2)

        ttk.Button(
            row,
            text="FIT SMART XYZ",
            command=self.fit_cube_xyz_model,
        ).pack(side="left", fill="x", expand=True, padx=2)

        ttk.Button(
            row,
            text="CLEAR XYZ CALIBRATION",
            command=self.clear_cube_xyz_calibration,
        ).pack(side="left", fill="x", expand=True, padx=(2, 0))

        ttk.Checkbutton(
            correction,
            text="Use learned cube XYZ correction",
            variable=self.cube_xyz_enabled_var,
            command=self.on_cube_xyz_enabled_changed,
        ).pack(anchor="w", pady=(6, 0))

        ttk.Label(
            correction,
            textvariable=self.cube_cal_var,
            wraplength=950,
        ).pack(anchor="w", pady=(4, 0))

        ttk.Label(
            correction,
            textvariable=self.corrected_var,
            wraplength=950,
        ).pack(anchor="w", pady=(4, 0))

        grasp_cal = ttk.LabelFrame(
            content,
            text="3B. Grasp XYZ Calibration",
            padding=8,
        )
        grasp_cal.pack(fill="x", pady=(0, 8))

        ttk.Label(
            grasp_cal,
            text=(
                "Second calibration layer for the FINAL pickup pose. "
                "First use GO ABOVE CUBE. Then press GO TO GRASP POSE and use "
                "the X/Y/Z live trim below until the fingers are exactly where "
                "the cube should be grabbed. CAPTURE GRASP SET stores all 3 directions."
            ),
            wraplength=950,
        ).pack(anchor="w", pady=(0, 6))

        self._correction_row(
            grasp_cal,
            "Grasp X trim [mm]",
            self.grasp_trim_x_var,
        )
        self._correction_row(
            grasp_cal,
            "Grasp Y trim [mm]",
            self.grasp_trim_y_var,
        )
        self._correction_row(
            grasp_cal,
            "Grasp Z trim [mm]",
            self.grasp_trim_z_var,
        )

        row = ttk.Frame(grasp_cal)
        row.pack(fill="x", pady=(7, 0))

        ttk.Button(
            row,
            text="GO TO GRASP POSE",
            command=self.go_to_grasp_pose,
        ).pack(side="left", fill="x", expand=True, padx=(0, 2))

        ttk.Button(
            row,
            text="CAPTURE GRASP SET",
            command=self.capture_grasp_set,
        ).pack(side="left", fill="x", expand=True, padx=2)

        ttk.Button(
            row,
            text="FIT GRASP XYZ",
            command=self.fit_grasp_model,
        ).pack(side="left", fill="x", expand=True, padx=2)

        ttk.Button(
            row,
            text="CLEAR GRASP",
            command=self.clear_grasp_calibration,
        ).pack(side="left", fill="x", expand=True, padx=(2, 0))

        ttk.Checkbutton(
            grasp_cal,
            text="Use learned grasp XYZ layer",
            variable=self.grasp_layer_enabled_var,
            command=self.on_grasp_layer_enabled_changed,
        ).pack(anchor="w", pady=(6, 0))

        ttk.Label(
            grasp_cal,
            textvariable=self.grasp_cal_var,
            wraplength=950,
        ).pack(anchor="w", pady=(4, 0))

        # ==================================================
        # 4. APPROACH
        # ==================================================
        motion = ttk.LabelFrame(
            content,
            text="4. Approach",
            padding=8,
        )
        motion.pack(fill="x", pady=(0, 8))

        zrow = ttk.Frame(motion)
        zrow.pack(fill="x", pady=(0, 5))

        ttk.Label(
            zrow,
            text="Approach Z [mm]",
        ).pack(side="left")

        ttk.Entry(
            zrow,
            textvariable=self.target_z_var,
            width=10,
        ).pack(side="left", padx=5)

        ttk.Button(
            motion,
            text="CHECK TARGET",
            command=self.solve_only,
        ).pack(fill="x", pady=2)

        ttk.Button(
            motion,
            text="GO ABOVE CUBE",
            command=self.go_to_cube,
        ).pack(fill="x", pady=2)

        ttk.Label(
            motion,
            textvariable=self.ik_var,
            wraplength=950,
        ).pack(anchor="w", pady=(5, 0))

        # ==================================================
        # 5. WRIST ORIENTATION
        # ==================================================
        wrist = ttk.LabelFrame(
            content,
            text="5. Wrist Orientation",
            padding=8,
        )
        wrist.pack(fill="x", pady=(0, 8))

        ttk.Label(
            wrist,
            text=(
                "Approach first. OBSERVE keeps the wrist still and automatically "
                "freezes a stable cube angle. Only after the angle is frozen do we rotate."
            ),
            wraplength=950,
        ).pack(anchor="w")

        row = ttk.Frame(wrist)
        row.pack(fill="x", pady=(6, 0))

        ttk.Button(
            row,
            text="OBSERVE",
            command=self.start_wrist_observation,
        ).pack(side="left", fill="x", expand=True, padx=(0, 2))

        ttk.Button(
            row,
            text="FREEZE ANGLE",
            command=self.freeze_wrist_orientation,
        ).pack(side="left", fill="x", expand=True, padx=2)

        ttk.Button(
            row,
            text="ALIGN WRIST",
            command=self.align_wrist_once,
        ).pack(side="left", fill="x", expand=True, padx=(2, 0))

        ttk.Button(
            wrist,
            text="CLEAR WRIST LOCK",
            command=self.clear_wrist_orientation,
        ).pack(fill="x", pady=(5, 0))

        ttk.Label(
            wrist,
            textvariable=self.wrist_var,
            wraplength=950,
        ).pack(anchor="w", pady=(6, 0))

        # ==================================================
        # 6. GRIPPER
        # ==================================================
        gripper_box = ttk.LabelFrame(
            content,
            text="6. Gripper Limits / Grip Force",
            padding=8,
        )
        gripper_box.pack(fill="x", pady=(0, 8))

        ttk.Label(
            gripper_box,
            text=(
                "Free Move: put the gripper at the FULLY OPEN extreme and capture it, "
                "then at the FULLY CLOSED extreme and capture it. During pickup the "
                "gripper closes gradually and stops when resistance reaches the selected "
                "force level."
            ),
            wraplength=950,
        ).pack(anchor="w", pady=(0, 6))

        row = ttk.Frame(gripper_box)
        row.pack(fill="x")

        ttk.Button(
            row,
            text="CAPTURE FULL OPEN",
            command=self.capture_gripper_open,
        ).pack(side="left", fill="x", expand=True, padx=(0, 2))

        ttk.Button(
            row,
            text="CAPTURE FULL CLOSED",
            command=self.capture_gripper_close_limit,
        ).pack(side="left", fill="x", expand=True, padx=2)

        ttk.Button(
            row,
            text="OPEN",
            command=lambda: self.command_gripper(
                float(self.gripper_open_var.get())
            ),
        ).pack(side="left", fill="x", expand=True, padx=2)

        ttk.Button(
            row,
            text="CLOSE TO LIMIT",
            command=lambda: self.command_gripper(
                float(self.gripper_close_limit_var.get())
            ),
        ).pack(side="left", fill="x", expand=True, padx=(2, 0))

        grid = ttk.Frame(gripper_box)
        grid.pack(fill="x", pady=(7, 0))

        fields = [
            ("Open limit", self.gripper_open_var),
            ("Closed limit", self.gripper_close_limit_var),
            ("Grip strength [%]", self.grip_strength_percent_var),
            ("100% resistance [deg]", self.grip_resistance_100_var),
            ("Required hits", self.grip_required_hits_var),
        ]

        for index, (label, variable) in enumerate(fields):
            row_index = index // 3
            col = (index % 3) * 2

            ttk.Label(
                grid,
                text=label,
            ).grid(
                row=row_index,
                column=col,
                sticky="w",
                padx=(0, 4),
                pady=2,
            )

            ttk.Entry(
                grid,
                textvariable=variable,
                width=9,
            ).grid(
                row=row_index,
                column=col + 1,
                sticky="w",
                padx=(0, 16),
                pady=2,
            )

        self.grip_force_bar = ttk.Progressbar(
            gripper_box,
            orient="horizontal",
            maximum=100.0,
            mode="determinate",
        )
        self.grip_force_bar.pack(fill="x", pady=(7, 2))

        ttk.Label(
            gripper_box,
            textvariable=self.grip_force_var,
        ).pack(anchor="w")

        ttk.Label(
            gripper_box,
            text=(
                "Note: this is currently a resistance proxy based on "
                "commanded-vs-measured gripper position, not a direct servo load reading."
            ),
            wraplength=950,
        ).pack(anchor="w", pady=(4, 0))

        # ==================================================
        # 7. CONTAINER XYZ CALIBRATION
        # ==================================================
        container_cal_box = ttk.LabelFrame(
            content,
            text="7. Container XYZ Calibration (position-dependent)",
            padding=8,
        )
        container_cal_box.pack(fill="x", pady=(0, 8))

        ttk.Label(
            container_cal_box,
            text=(
                "Move any container marker to different workspace positions. Select its ID, "
                "GO CONTAINER CAL, then use X/Y/Z trim until the TCP is centered safely above "
                "the container. CAPTURE SET. Repeat at several positions and FIT CONTAINER XYZ. "
                "The model depends on X/Y position, NOT on marker ID."
            ),
            wraplength=950,
        ).pack(anchor="w", pady=(0, 6))

        top = ttk.Frame(container_cal_box)
        top.pack(fill="x", pady=(0, 5))
        ttk.Label(top, text="Marker ID").pack(side="left")
        ttk.Combobox(
            top,
            textvariable=self.container_cal_marker_var,
            values=(5, 6, 7, 19),
            width=5,
            state="readonly",
        ).pack(side="left", padx=(5, 10))
        ttk.Button(
            top,
            text="GO CONTAINER CAL",
            command=self.go_container_calibration,
        ).pack(side="left", padx=2)
        ttk.Button(
            top,
            text="CAPTURE SET",
            command=self.capture_container_calibration,
        ).pack(side="left", padx=2)
        ttk.Button(
            top,
            text="FIT CONTAINER XYZ",
            command=self.fit_container_calibration,
        ).pack(side="left", padx=2)
        ttk.Button(
            top,
            text="CLEAR",
            command=self.clear_container_calibration,
        ).pack(side="left", padx=2)

        self._container_correction_row(
            container_cal_box, "Container X trim [mm]", self.container_trim_x_var
        )
        self._container_correction_row(
            container_cal_box, "Container Y trim [mm]", self.container_trim_y_var
        )
        self._container_correction_row(
            container_cal_box, "Container Z trim [mm]", self.container_trim_z_var
        )

        ttk.Checkbutton(
            container_cal_box,
            text="Use learned Container XYZ correction",
            variable=self.container_cal_enabled_var,
            command=self._container_cal_toggle,
        ).pack(anchor="w", pady=(5, 0))

        ttk.Label(
            container_cal_box,
            textvariable=self.container_cal_var,
            wraplength=950,
        ).pack(anchor="w", pady=(4, 0))

        self.update_container_calibration_status()

        # ==================================================
        # 8. ONE CUBE PICK + PLACE
        # ==================================================
        pick_box = ttk.LabelFrame(
            content,
            text="8. One Cube Pick & Place",
            padding=8,
        )
        pick_box.pack(fill="x", pady=(0, 8))

        settings = ttk.Frame(pick_box)
        settings.pack(fill="x")

        for label, variable in (
            ("Travel Z", self.travel_z_var),
            ("Lift clear Z (min 70)", self.lift_clear_z_var),
            ("Container travel Z", self.container_travel_z_var),
            ("Drop Z", self.drop_z_var),
        ):
            ttk.Label(
                settings,
                text=label,
            ).pack(side="left", padx=(0, 3))

            ttk.Entry(
                settings,
                textvariable=variable,
                width=7,
            ).pack(side="left", padx=(0, 10))

        test_row = ttk.Frame(pick_box)
        test_row.pack(fill="x", pady=(7, 0))

        ttk.Button(
            test_row,
            text="CLOSE GRIPPER",
            command=self.start_manual_grip_close,
        ).pack(side="left", fill="x", expand=True, padx=(0, 2))

        ttk.Button(
            test_row,
            text="TEST LIFT",
            command=self.test_lift,
        ).pack(side="left", fill="x", expand=True, padx=2)

        ttk.Button(
            test_row,
            text="GO CONTAINER",
            command=self.go_container_manual,
        ).pack(side="left", fill="x", expand=True, padx=2)

        ttk.Button(
            test_row,
            text="RELEASE",
            command=self.release_manual,
        ).pack(side="left", fill="x", expand=True, padx=(2, 0))

        ttk.Label(
            pick_box,
            text=(
                "Manual test order: GO ABOVE CUBE -> GO TO GRASP POSE -> "
                "CLOSE GRIPPER -> TEST LIFT -> GO CONTAINER -> RELEASE."
            ),
            wraplength=950,
        ).pack(anchor="w", pady=(5, 0))

        tolerance_box = ttk.LabelFrame(
            pick_box,
            text="Motion acceptance / timeout",
            padding=6,
        )
        tolerance_box.pack(fill="x", pady=(7, 0))

        tolerance_fields = (
            ("Approach [deg]", self.approach_accept_var),
            ("Grasp [deg]", self.grasp_accept_var),
            ("Grasp snap [deg]", self.grasp_final_snap_var),
            ("Snap accept [deg]", self.grasp_snap_accept_var),
            ("Lift [deg]", self.lift_accept_var),
            ("Container [deg]", self.container_accept_var),
            ("Drop [deg]", self.drop_accept_var),
            ("Timeout [s]", self.motion_timeout_var),
        )

        for label, variable in tolerance_fields:
            ttk.Label(
                tolerance_box,
                text=label,
            ).pack(side="left", padx=(0, 3))

            ttk.Entry(
                tolerance_box,
                textvariable=variable,
                width=6,
            ).pack(side="left", padx=(0, 10))

        ttk.Label(
            pick_box,
            text=(
                "Normal motion: phase completes after encoder motion stops for 1.5 s. "
                "Degree values are diagnostic; GRASP Final Snap remains active."
            ),
            wraplength=950,
        ).pack(anchor="w", pady=(5, 0))

        row = ttk.Frame(pick_box)
        row.pack(fill="x", pady=(7, 0))

        ttk.Button(
            row,
            text="START PICK + PLACE",
            command=self.start_pick_place,
        ).pack(side="left", fill="x", expand=True, padx=(0, 2))

        ttk.Button(
            row,
            text="STOP",
            command=self.stop_pick_place,
        ).pack(side="left", fill="x", expand=True, padx=(2, 0))

        ttk.Label(
            pick_box,
            text=(
                "Color mapping: red->ID5, yellow->ID6, green->ID7, blue->ID19."
            ),
            wraplength=950,
        ).pack(anchor="w", pady=(5, 0))

        ttk.Label(
            pick_box,
            textvariable=self.pick_place_var,
            wraplength=950,
        ).pack(anchor="w", pady=(4, 0))

        # ==================================================
        # STATUS
        # ==================================================
        status = ttk.LabelFrame(
            content,
            text="Status",
            padding=8,
        )
        status.pack(fill="x", pady=(0, 8))

        ttk.Label(
            status,
            textvariable=self.status_var,
            wraplength=950,
        ).pack(anchor="w")

    def _correction_row(self, parent, label, variable):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)

        ttk.Label(row, text=label, width=16).pack(side="left")

        for text, delta in (
            ("-5", -5.0),
            ("-1", -1.0),
        ):
            ttk.Button(
                row, text=text, width=4,
                command=lambda d=delta, v=variable: self.bump(v, d),
            ).pack(side="left", padx=1)

        ttk.Entry(
            row, textvariable=variable, width=9
        ).pack(side="left", padx=5)

        for text, delta in (
            ("+1", 1.0),
            ("+5", 5.0),
        ):
            ttk.Button(
                row, text=text, width=4,
                command=lambda d=delta, v=variable: self.bump(v, d),
            ).pack(side="left", padx=1)

    def bump(self, variable, delta):
        variable.set(
            float(variable.get()) + float(delta)
        )
        self.update_corrected_info()

        # Live trim:
        # approach trim updates the approach target;
        # grasp trim updates the independent grasp target.
        if self.locked_cube is None:
            return

        state = self.worker.get_state_snapshot()

        if not state.get("connected", False):
            return

        grasp_vars = {
            id(self.grasp_trim_x_var),
            id(self.grasp_trim_y_var),
            id(self.grasp_trim_z_var),
        }

        try:
            if id(variable) in grasp_vars:
                result = self.solve_grasp_target(
                    include_manual_trim=True
                )
                label = "Live GRASP XYZ trim"
            else:
                result = self.solve_cube(
                    physical_z=float(self.target_z_var.get())
                )
                label = "Live APPROACH XY trim"
        except Exception as exc:
            self.status_var.set(
                f"{label} failed: {exc}"
            )
            return

        if not result.get("ok", False):
            self.status_var.set(
                f"{label} IK rejected: "
                + result.get("message", "unknown")
            )
            return

        self.worker.send(
            commands.move_to_positions(
                result["motors"]
            )
        )

        if id(variable) in grasp_vars:
            self.status_var.set(
                "Live GRASP XYZ trim applied | "
                f"dX={float(self.grasp_trim_x_var.get()):+.1f} | "
                f"dY={float(self.grasp_trim_y_var.get()):+.1f} | "
                f"dZ={float(self.grasp_trim_z_var.get()):+.1f} mm"
            )
        else:
            self.status_var.set(
                "Live APPROACH XY trim applied | "
                f"dX={float(self.correction_x_var.get()):+.1f} mm | "
                f"dY={float(self.correction_y_var.get()):+.1f} mm"
            )

    def reset_correction(self):
        self.correction_x_var.set(0.0)
        self.correction_y_var.set(0.0)
        self.update_corrected_info()

        if self.locked_cube is None:
            return

        state = self.worker.get_state_snapshot()
        if not state.get("connected", False):
            return

        try:
            result = self.solve_cube(
                physical_z=float(self.target_z_var.get())
            )
        except Exception:
            return

        if result.get("ok", False):
            self.worker.send(
                commands.move_to_positions(
                    result["motors"]
                )
            )

    # ------------------------------------------------------
    # Smart cube XYZ calibration
    # ------------------------------------------------------

    def _save_cube_calibration(self):
        data = self._load_json(
            CUBE_TARGET_CALIBRATION_FILE,
            {},
        )

        if not isinstance(data, dict):
            data = {}

        data["xyz_samples"] = self.cube_xyz_samples

        data["xyz_model"] = {
            "enabled": bool(
                self.cube_xyz_enabled_var.get()
            ),
            "coeff_x": [
                float(v)
                for v in self.cube_xyz_coeff_x
            ],
            "coeff_y": [
                float(v)
                for v in self.cube_xyz_coeff_y
            ],
            "z_slope_mm_per_mm": float(
                self.cube_z_slope
            ),
            "z_intercept_mm": float(
                self.cube_z_intercept
            ),
            "z_reference_reach_mm": float(
                self.cube_z_reference_reach
            ),
            "rms_xy_mm": (
                None
                if self.cube_xyz_rms_xy is None
                else float(self.cube_xyz_rms_xy)
            ),
            "rms_z_mm": (
                None
                if self.cube_xyz_rms_z is None
                else float(self.cube_xyz_rms_z)
            ),
        }

        data["grasp_samples"] = self.grasp_samples
        data["grasp_model"] = {
            "enabled": bool(
                self.grasp_layer_enabled_var.get()
            ),
            "coeff_x": [
                float(v)
                for v in self.grasp_coeff_x
            ],
            "coeff_y": [
                float(v)
                for v in self.grasp_coeff_y
            ],
            "z_slope_mm_per_mm": float(
                self.grasp_z_slope
            ),
            "z_intercept_mm": float(
                self.grasp_z_intercept
            ),
            "z_reference_reach_mm": float(
                self.grasp_z_reference_reach
            ),
            "rms_xy_mm": (
                None
                if self.grasp_rms_xy is None
                else float(self.grasp_rms_xy)
            ),
            "rms_z_mm": (
                None
                if self.grasp_rms_z is None
                else float(self.grasp_rms_z)
            ),
        }

        data["container_samples"] = self.container_cal_samples
        data["container_model"] = {
            "enabled": bool(self.container_cal_enabled_var.get()),
            "coeff_x": [float(v) for v in self.container_coeff_x],
            "coeff_y": [float(v) for v in self.container_coeff_y],
            "coeff_z": [float(v) for v in self.container_coeff_z],
        }

        data["gripper"] = {
            "open_limit": float(
                self.gripper_open_var.get()
            ),
            "close_limit": float(
                self.gripper_close_limit_var.get()
            ),
            "strength_percent": float(
                self.grip_strength_percent_var.get()
            ),
            "resistance_100_deg": float(
                self.grip_resistance_100_var.get()
            ),
            "required_hits": int(
                self.grip_required_hits_var.get()
            ),
        }

        CUBE_TARGET_CALIBRATION_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        CUBE_TARGET_CALIBRATION_FILE.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self.cube_calibration = data

    def get_learned_cube_xy_correction(
        self,
        raw_x,
        raw_y,
    ):
        if not bool(
            self.cube_xyz_enabled_var.get()
        ):
            return 0.0, 0.0

        x = float(raw_x)
        y = float(raw_y)

        ax = self.cube_xyz_coeff_x
        ay = self.cube_xyz_coeff_y

        dx = (
            ax[0]
            + ax[1] * x
            + ax[2] * y
        )

        dy = (
            ay[0]
            + ay[1] * x
            + ay[2] * y
        )

        return float(dx), float(dy)

    def get_learned_cube_z_correction(
        self,
        reach,
    ):
        if not bool(
            self.cube_xyz_enabled_var.get()
        ):
            return 0.0

        return (
            float(self.cube_z_intercept)
            + float(self.cube_z_slope)
            * (
                float(reach)
                - float(
                    self.cube_z_reference_reach
                )
            )
        )

    def update_cube_calibration_status(self):
        state = (
            "ON"
            if self.cube_xyz_enabled_var.get()
            else "OFF"
        )

        rms_xy = (
            "—"
            if self.cube_xyz_rms_xy is None
            else f"{float(self.cube_xyz_rms_xy):.2f} mm"
        )

        rms_z = (
            "—"
            if self.cube_xyz_rms_z is None
            else f"{float(self.cube_xyz_rms_z):.2f} mm"
        )

        self.cube_cal_var.set(
            f"Cube XYZ: {state} | "
            f"sets={len(self.cube_xyz_samples)} | "
            f"RMS XY={rms_xy} | RMS Z={rms_z}"
        )

    def on_cube_xyz_enabled_changed(self):
        self.cube_xyz_enabled = bool(
            self.cube_xyz_enabled_var.get()
        )

        self._save_cube_calibration()
        self.update_cube_calibration_status()
        self.update_corrected_info()

    def capture_cube_xyz_set(self):
        """
        Capture ONE combined XYZ sample.

        X/Y:
            The cube is locked at raw camera-derived coordinates.
            The user live-trims X/Y until the real gripper is centered.

        Z:
            target_z_var = commanded physical height.
            measured_z_var = manually measured real TCP height.

        If an XYZ model is already enabled, the sample stores the TOTAL
        correction that would be needed, not only the newest manual residual.
        """
        local = self.get_locked_target_local()

        if local is None:
            self.status_var.set(
                "Cannot capture XYZ set: LOCK CUBE first."
            )
            return

        raw_x, raw_y, _, _ = local

        learned_dx, learned_dy = (
            self.get_learned_cube_xy_correction(
                raw_x,
                raw_y,
            )
        )

        total_dx = (
            learned_dx
            + float(
                self.correction_x_var.get()
            )
        )

        total_dy = (
            learned_dy
            + float(
                self.correction_y_var.get()
            )
        )

        reach = math.hypot(
            float(raw_x),
            float(raw_y),
        )

        commanded_z = float(
            self.target_z_var.get()
        )

        measured_z = float(
            self.measured_z_var.get()
        )

        current_learned_dz = (
            self.get_learned_cube_z_correction(
                reach
            )
        )

        # Example:
        # commanded 40, measured 34 => need +6 mm more.
        total_dz = (
            current_learned_dz
            + commanded_z
            - measured_z
        )

        self.cube_xyz_samples.append(
            {
                "raw_x_mm": float(raw_x),
                "raw_y_mm": float(raw_y),
                "reach_mm": float(reach),
                "correction_x_mm": float(
                    total_dx
                ),
                "correction_y_mm": float(
                    total_dy
                ),
                "commanded_z_mm": float(
                    commanded_z
                ),
                "measured_z_mm": float(
                    measured_z
                ),
                "correction_z_mm": float(
                    total_dz
                ),
                "cube_class": (
                    None
                    if self.locked_cube is None
                    else str(
                        self.locked_cube.get(
                            "class",
                            "cube",
                        )
                    )
                ),
            }
        )

        self._save_cube_calibration()
        self.update_cube_calibration_status()

        self.status_var.set(
            "XYZ set captured | "
            f"raw=({raw_x:.1f},{raw_y:.1f}) mm | "
            f"dXY=({total_dx:+.1f},{total_dy:+.1f}) mm | "
            f"Z cmd={commanded_z:.1f}, real={measured_z:.1f} mm | "
            f"dZ={total_dz:+.1f} mm"
        )

    @staticmethod
    def _solve_3x3(
        matrix,
        vector,
    ):
        a = [
            [
                float(matrix[r][c])
                for c in range(3)
            ]
            + [float(vector[r])]
            for r in range(3)
        ]

        for col in range(3):
            pivot = max(
                range(col, 3),
                key=lambda r: abs(
                    a[r][col]
                ),
            )

            if abs(a[pivot][col]) < 1e-9:
                raise RuntimeError(
                    "Calibration points are geometrically degenerate."
                )

            a[col], a[pivot] = (
                a[pivot],
                a[col],
            )

            divisor = a[col][col]

            a[col] = [
                value / divisor
                for value in a[col]
            ]

            for row in range(3):
                if row == col:
                    continue

                factor = a[row][col]

                a[row] = [
                    a[row][i]
                    - factor * a[col][i]
                    for i in range(4)
                ]

        return [
            a[i][3]
            for i in range(3)
        ]

    @classmethod
    def _fit_affine(
        cls,
        rows,
        values,
    ):
        ata = [
            [0.0] * 3
            for _ in range(3)
        ]

        atb = [0.0] * 3

        for row, value in zip(
            rows,
            values,
        ):
            for i in range(3):
                atb[i] += (
                    row[i]
                    * value
                )

                for j in range(3):
                    ata[i][j] += (
                        row[i]
                        * row[j]
                    )

        return cls._solve_3x3(
            ata,
            atb,
        )

    def fit_cube_xyz_model(self):
        """
        Fit all 3 correction directions from the SAME sample set.

        XY:
            affine in raw X/Y.

        Z:
            linear versus horizontal reach, same idea as Module-7 Z
            compensation.
        """
        samples = self.cube_xyz_samples

        if len(samples) < 3:
            self.status_var.set(
                "Need at least 3 XYZ sets; 5-8 spread across the table is better."
            )
            return

        rows = []
        values_x = []
        values_y = []

        reaches = []
        values_z = []

        for sample in samples:
            x = float(
                sample["raw_x_mm"]
            )
            y = float(
                sample["raw_y_mm"]
            )

            rows.append(
                [1.0, x, y]
            )

            values_x.append(
                float(
                    sample[
                        "correction_x_mm"
                    ]
                )
            )

            values_y.append(
                float(
                    sample[
                        "correction_y_mm"
                    ]
                )
            )

            reaches.append(
                float(
                    sample.get(
                        "reach_mm",
                        math.hypot(x, y),
                    )
                )
            )

            values_z.append(
                float(
                    sample[
                        "correction_z_mm"
                    ]
                )
            )

        try:
            coeff_x = self._fit_affine(
                rows,
                values_x,
            )

            coeff_y = self._fit_affine(
                rows,
                values_y,
            )

        except Exception as exc:
            self.status_var.set(
                f"Cannot fit XYZ calibration: {exc}"
            )
            return

        xy_errors_sq = []

        for row, dx, dy in zip(
            rows,
            values_x,
            values_y,
        ):
            pred_x = sum(
                coeff_x[i] * row[i]
                for i in range(3)
            )

            pred_y = sum(
                coeff_y[i] * row[i]
                for i in range(3)
            )

            xy_errors_sq.append(
                (dx - pred_x) ** 2
                + (dy - pred_y) ** 2
            )

        reference_reach = (
            sum(reaches)
            / len(reaches)
        )

        xs = [
            reach - reference_reach
            for reach in reaches
        ]

        mean_z = (
            sum(values_z)
            / len(values_z)
        )

        denominator = sum(
            x * x
            for x in xs
        )

        if denominator < 1e-9:
            z_slope = 0.0
            z_intercept = mean_z
        else:
            z_slope = (
                sum(
                    x * (
                        value
                        - mean_z
                    )
                    for x, value
                    in zip(
                        xs,
                        values_z,
                    )
                )
                / denominator
            )

            z_intercept = mean_z

        z_errors_sq = []

        for reach, actual in zip(
            reaches,
            values_z,
        ):
            predicted = (
                z_intercept
                + z_slope
                * (
                    reach
                    - reference_reach
                )
            )

            z_errors_sq.append(
                (
                    actual
                    - predicted
                ) ** 2
            )

        self.cube_xyz_coeff_x = [
            float(v)
            for v in coeff_x
        ]

        self.cube_xyz_coeff_y = [
            float(v)
            for v in coeff_y
        ]

        self.cube_z_reference_reach = float(
            reference_reach
        )

        self.cube_z_slope = float(
            z_slope
        )

        self.cube_z_intercept = float(
            z_intercept
        )

        self.cube_xyz_rms_xy = math.sqrt(
            sum(xy_errors_sq)
            / len(xy_errors_sq)
        )

        self.cube_xyz_rms_z = math.sqrt(
            sum(z_errors_sq)
            / len(z_errors_sq)
        )

        self.cube_xyz_enabled = True
        self.cube_xyz_enabled_var.set(
            True
        )

        self._save_cube_calibration()
        self.update_cube_calibration_status()

        self.status_var.set(
            "Smart cube XYZ calibration fitted and enabled | "
            f"RMS XY={self.cube_xyz_rms_xy:.2f} mm | "
            f"RMS Z={self.cube_xyz_rms_z:.2f} mm"
        )

    def clear_cube_xyz_calibration(self):
        self.cube_xyz_samples = []

        self.cube_xyz_coeff_x = [
            0.0,
            0.0,
            0.0,
        ]

        self.cube_xyz_coeff_y = [
            0.0,
            0.0,
            0.0,
        ]

        self.cube_z_slope = 0.0
        self.cube_z_intercept = 0.0
        self.cube_z_reference_reach = 0.0

        self.cube_xyz_rms_xy = None
        self.cube_xyz_rms_z = None

        self.cube_xyz_enabled = False
        self.cube_xyz_enabled_var.set(
            False
        )

        self.reset_correction()

        self._save_cube_calibration()
        self.update_cube_calibration_status()

        self.status_var.set(
            "Cube XYZ calibration cleared. Module 7 calibration untouched."
        )

    # ------------------------------------------------------
    # Independent GRASP XYZ calibration layer
    # ------------------------------------------------------

    def get_grasp_learned_xy(self, raw_x, raw_y):
        if not bool(
            self.grasp_layer_enabled_var.get()
        ):
            return 0.0, 0.0

        x = float(raw_x)
        y = float(raw_y)

        ax = self.grasp_coeff_x
        ay = self.grasp_coeff_y

        return (
            float(
                ax[0]
                + ax[1] * x
                + ax[2] * y
            ),
            float(
                ay[0]
                + ay[1] * x
                + ay[2] * y
            ),
        )

    def get_grasp_learned_z(self, reach):
        # With no fitted grasp model yet, start from Approach Z plus manual Z trim.
        if not bool(
            self.grasp_layer_enabled_var.get()
        ):
            return (
                float(self.target_z_var.get())
                + float(self.grasp_trim_z_var.get())
            )

        return (
            float(self.grasp_z_intercept)
            + float(self.grasp_z_slope)
            * (
                float(reach)
                - float(
                    self.grasp_z_reference_reach
                )
            )
        )

    def get_grasp_target_local(
        self,
        include_manual_trim=False,
    ):
        if self.locked_cube is None:
            return None

        scene_x = float(
            self.locked_cube["camera_world_x"]
        )
        scene_y = -float(
            self.locked_cube["camera_world_y"]
        )

        raw = self.world_to_robot_local(
            scene_x,
            scene_y,
        )

        if raw is None:
            return None

        raw_x = float(raw[0])
        raw_y = float(raw[1])

        learned_dx, learned_dy = (
            self.get_grasp_learned_xy(
                raw_x,
                raw_y,
            )
        )

        manual_dx = (
            float(self.grasp_trim_x_var.get())
            if include_manual_trim
            else 0.0
        )
        manual_dy = (
            float(self.grasp_trim_y_var.get())
            if include_manual_trim
            else 0.0
        )

        reach = math.hypot(
            raw_x,
            raw_y,
        )

        grasp_z = self.get_grasp_learned_z(
            reach
        )

        if (
            include_manual_trim
            and bool(
                self.grasp_layer_enabled_var.get()
            )
        ):
            # Once a learned model exists, Z trim is an additional residual.
            grasp_z += float(
                self.grasp_trim_z_var.get()
            )

        return (
            raw_x,
            raw_y,
            raw_x + learned_dx + manual_dx,
            raw_y + learned_dy + manual_dy,
            float(grasp_z),
        )

    def solve_grasp_target(
        self,
        include_manual_trim=False,
    ):
        target = self.get_grasp_target_local(
            include_manual_trim=include_manual_trim
        )

        if target is None:
            return {
                "ok": False,
                "message": "Lock a cube first / robot base unavailable.",
            }

        _, _, x, y, z = target

        return self.solve_local_target(
            x,
            y,
            z,
            prefer_vertical=True,
        )

    def go_to_grasp_pose(self):
        if self.locked_cube is None:
            self.status_var.set(
                "Lock a cube first."
            )
            return

        result = self.solve_grasp_target(
            include_manual_trim=True
        )

        if not result.get("ok", False):
            self.status_var.set(
                "Grasp pose IK failed: "
                + str(
                    result.get(
                        "message",
                        "unknown",
                    )
                )
            )
            return

        self.worker.send(
            commands.move_to_positions(
                result["motors"]
            )
        )

        self.status_var.set(
            "GO TO GRASP POSE sent | "
            f"X trim={float(self.grasp_trim_x_var.get()):+.1f} | "
            f"Y trim={float(self.grasp_trim_y_var.get()):+.1f} | "
            f"Z trim={float(self.grasp_trim_z_var.get()):+.1f} mm"
        )

    def capture_grasp_set(self):
        target = self.get_grasp_target_local(
            include_manual_trim=True
        )

        if target is None:
            self.status_var.set(
                "Cannot capture GRASP set: LOCK CUBE first."
            )
            return

        raw_x, raw_y, x, y, z = target

        reach = math.hypot(
            raw_x,
            raw_y,
        )

        self.grasp_samples.append(
            {
                "raw_x_mm": float(raw_x),
                "raw_y_mm": float(raw_y),
                "reach_mm": float(reach),
                "target_x_mm": float(x),
                "target_y_mm": float(y),
                "target_z_mm": float(z),
                "correction_x_mm": float(x - raw_x),
                "correction_y_mm": float(y - raw_y),
                "cube_class": (
                    None
                    if self.locked_cube is None
                    else str(
                        self.locked_cube.get(
                            "class",
                            "cube",
                        )
                    )
                ),
            }
        )

        self._save_cube_calibration()
        self.update_grasp_calibration_status()

        self.status_var.set(
            "GRASP XYZ set captured | "
            f"raw=({raw_x:.1f},{raw_y:.1f}) | "
            f"target=({x:.1f},{y:.1f},{z:.1f}) mm"
        )

    def fit_grasp_model(self):
        samples = self.grasp_samples

        if len(samples) < 3:
            self.status_var.set(
                "Need at least 3 GRASP sets; 5-8 across the table is better."
            )
            return

        rows = []
        values_x = []
        values_y = []
        reaches = []
        target_z_values = []

        for sample in samples:
            x = float(
                sample["raw_x_mm"]
            )
            y = float(
                sample["raw_y_mm"]
            )

            rows.append(
                [1.0, x, y]
            )

            values_x.append(
                float(
                    sample[
                        "correction_x_mm"
                    ]
                )
            )
            values_y.append(
                float(
                    sample[
                        "correction_y_mm"
                    ]
                )
            )

            reaches.append(
                float(
                    sample.get(
                        "reach_mm",
                        math.hypot(x, y),
                    )
                )
            )

            target_z_values.append(
                float(
                    sample["target_z_mm"]
                )
            )

        try:
            coeff_x = self._fit_affine(
                rows,
                values_x,
            )
            coeff_y = self._fit_affine(
                rows,
                values_y,
            )
        except Exception as exc:
            self.status_var.set(
                f"Cannot fit GRASP XYZ: {exc}"
            )
            return

        xy_errors_sq = []

        for row, dx, dy in zip(
            rows,
            values_x,
            values_y,
        ):
            pred_x = sum(
                coeff_x[i] * row[i]
                for i in range(3)
            )
            pred_y = sum(
                coeff_y[i] * row[i]
                for i in range(3)
            )

            xy_errors_sq.append(
                (dx - pred_x) ** 2
                + (dy - pred_y) ** 2
            )

        ref = (
            sum(reaches)
            / len(reaches)
        )

        xs = [
            reach - ref
            for reach in reaches
        ]

        mean_z = (
            sum(target_z_values)
            / len(target_z_values)
        )

        denominator = sum(
            x * x
            for x in xs
        )

        if denominator < 1e-9:
            z_slope = 0.0
            z_intercept = mean_z
        else:
            z_slope = (
                sum(
                    x * (z - mean_z)
                    for x, z in zip(
                        xs,
                        target_z_values,
                    )
                )
                / denominator
            )
            z_intercept = mean_z

        z_errors_sq = []

        for reach, actual in zip(
            reaches,
            target_z_values,
        ):
            predicted = (
                z_intercept
                + z_slope
                * (
                    reach
                    - ref
                )
            )
            z_errors_sq.append(
                (
                    actual
                    - predicted
                ) ** 2
            )

        self.grasp_coeff_x = [
            float(v)
            for v in coeff_x
        ]
        self.grasp_coeff_y = [
            float(v)
            for v in coeff_y
        ]

        self.grasp_z_reference_reach = float(
            ref
        )
        self.grasp_z_slope = float(
            z_slope
        )
        self.grasp_z_intercept = float(
            z_intercept
        )

        self.grasp_rms_xy = math.sqrt(
            sum(xy_errors_sq)
            / len(xy_errors_sq)
        )
        self.grasp_rms_z = math.sqrt(
            sum(z_errors_sq)
            / len(z_errors_sq)
        )

        self.grasp_layer_enabled = True
        self.grasp_layer_enabled_var.set(
            True
        )

        # Learned model now owns the pose; reset manual residuals.
        self.grasp_trim_x_var.set(0.0)
        self.grasp_trim_y_var.set(0.0)
        self.grasp_trim_z_var.set(0.0)

        self._save_cube_calibration()
        self.update_grasp_calibration_status()

        self.status_var.set(
            "GRASP XYZ fitted and enabled | "
            f"RMS XY={self.grasp_rms_xy:.2f} mm | "
            f"RMS Z={self.grasp_rms_z:.2f} mm"
        )

    def update_grasp_calibration_status(self):
        state = (
            "ON"
            if self.grasp_layer_enabled_var.get()
            else "OFF"
        )

        rms_xy = (
            "—"
            if self.grasp_rms_xy is None
            else f"{float(self.grasp_rms_xy):.2f} mm"
        )
        rms_z = (
            "—"
            if self.grasp_rms_z is None
            else f"{float(self.grasp_rms_z):.2f} mm"
        )

        self.grasp_cal_var.set(
            f"Grasp XYZ: {state} | "
            f"sets={len(self.grasp_samples)} | "
            f"RMS XY={rms_xy} | RMS Z={rms_z}"
        )

    def on_grasp_layer_enabled_changed(self):
        self.grasp_layer_enabled = bool(
            self.grasp_layer_enabled_var.get()
        )
        self._save_cube_calibration()
        self.update_grasp_calibration_status()

    def clear_grasp_calibration(self):
        self.grasp_samples = []
        self.grasp_coeff_x = [
            0.0,
            0.0,
            0.0,
        ]
        self.grasp_coeff_y = [
            0.0,
            0.0,
            0.0,
        ]

        self.grasp_z_slope = 0.0
        self.grasp_z_intercept = 18.0
        self.grasp_z_reference_reach = 0.0

        self.grasp_rms_xy = None
        self.grasp_rms_z = None

        self.grasp_layer_enabled = False
        self.grasp_layer_enabled_var.set(
            False
        )

        self.grasp_trim_x_var.set(0.0)
        self.grasp_trim_y_var.set(0.0)
        self.grasp_trim_z_var.set(-22.0)

        self._save_cube_calibration()
        self.update_grasp_calibration_status()

        self.status_var.set(
            "Grasp XYZ calibration cleared."
        )

    # ------------------------------------------------------
    # Home position
    # ------------------------------------------------------

    def get_home_pose(self):
        """
        Use a dedicated saved HOME pose from ik_calibration.json.
        """
        poses = self.old_calibration.get("poses", {})

        home = poses.get("home")
        if isinstance(home, dict) and home:
            return home

        return None

    def capture_home(self):
        """
        Capture the robot's CURRENT measured motor positions as HOME
        and save them into data/ik_calibration.json -> poses -> home.

        Recommended workflow:
        1. Free Move
        2. Put the arm into the desired safe home pose by hand
        3. CAPTURE HOME
        4. HOME POSITION can then return to exactly that saved pose
        """
        state = self.worker.get_state_snapshot()

        if not state.get("connected", False):
            self.status_var.set(
                "Cannot capture HOME: robot is not connected."
            )
            return

        positions = state.get("positions", {})

        required = list(JOINTS)

        missing = [
            joint for joint in required
            if joint not in positions
        ]

        if missing:
            self.status_var.set(
                "Cannot capture HOME: missing measured positions: "
                + ", ".join(missing)
            )
            return

        home = {
            joint: float(positions[joint])
            for joint in required
        }

        # Save wrist roll and gripper too when the Worker exposes them.
        if WRIST_ROLL_JOINT in positions:
            home[WRIST_ROLL_JOINT] = float(
                positions[WRIST_ROLL_JOINT]
            )

        if "gripper.pos" in positions:
            home["gripper.pos"] = float(
                positions["gripper.pos"]
            )

        # Reload the file immediately before writing so Module 8 only
        # changes poses.home and preserves all other calibration data.
        data = self._load_json(
            IK_CALIBRATION_FILE,
            {"poses": {}, "geometry": {}},
        )

        if not isinstance(data, dict):
            data = {}

        poses = data.get("poses")
        if not isinstance(poses, dict):
            poses = {}

        poses["home"] = home
        data["poses"] = poses

        IK_CALIBRATION_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        IK_CALIBRATION_FILE.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # Keep this module's in-memory copy synchronized.
        self.old_calibration = data

        self.status_var.set(
            "HOME captured and saved | "
            + " | ".join(
                f"{joint.replace('.pos', '')}={value:.1f}"
                for joint, value in home.items()
            )
        )

    def go_home(self):
        """
        Safe staged HOME:
        1) fold shoulder / elbow / wrist / wrist_roll / gripper first,
           while BASE stays where it currently is;
        2) only after the upper arm is near HOME, rotate shoulder_pan.

        Movement is incremental so the base cannot suddenly sweep the arm
        through the top camera / calibration setup.
        """
        state = self.worker.get_state_snapshot()

        if not state.get("connected", False):
            self.status_var.set(
                "Cannot go HOME: robot is not connected."
            )
            return

        home = self.get_home_pose()
        if home is None:
            self.status_var.set(
                "No HOME saved yet. Use Free Move -> pose arm -> CAPTURE HOME."
            )
            return

        self.pick_place_active = False
        self.home_sequence_active = True
        self.home_sequence_phase = "SHOULDER"
        self.home_phase_started_at = time.monotonic()
        self.home_last_position = None
        self.home_last_motion_at = time.monotonic()
        self.home_sequence_target = {
            key: float(value)
            for key, value in home.items()
        }

        self.status_var.set(
            "SAFE HOME: shoulder first -> elbow -> wrist -> base."
        )

    @staticmethod
    def _step_toward(current, target, max_step):
        delta = float(target) - float(current)
        delta = max(
            -abs(float(max_step)),
            min(abs(float(max_step)), delta),
        )
        return float(current) + delta

    def _send_full_hold_frame(self, positions, overrides):
        """
        Send a complete motor-position frame.

        Some low-level paths are much more reliable when all currently known
        joints are present in the command. This also explains why an unrelated
        XY nudge could previously "wake up" a stalled HOME phase.
        """
        targets = {}

        for joint in (
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            WRIST_ROLL_JOINT,
            "gripper.pos",
        ):
            if joint in positions:
                targets[joint] = float(positions[joint])

        for joint, value in overrides.items():
            targets[joint] = float(value)

        if targets:
            self.worker.send(
                commands.move_to_positions(targets)
            )

    def _advance_home_phase(self, next_phase, message):
        self.home_sequence_phase = next_phase
        self.home_phase_started_at = time.monotonic()
        self.home_last_position = None
        self.home_last_motion_at = time.monotonic()
        self.status_var.set(message)

    def _update_home_sequence(self):
        if not self.home_sequence_active:
            return

        state = self.worker.get_state_snapshot()
        positions = state.get("positions", {})

        if not state.get("connected", False):
            self.home_sequence_active = False
            self.status_var.set("SAFE HOME stopped: robot disconnected.")
            return

        home = self.home_sequence_target or {}
        now = time.monotonic()

        def move_one_joint(joint, max_step, next_phase, label):
            if joint not in home or joint not in positions:
                self._advance_home_phase(
                    next_phase,
                    f"SAFE HOME: {joint} unavailable -> continuing."
                )
                return

            current = float(positions[joint])
            target = float(home[joint])
            error = target - current

            # Track whether the servo is still physically changing.
            if self.home_last_position is None:
                self.home_last_position = current
                self.home_last_motion_at = now
            elif abs(current - self.home_last_position) >= 0.20:
                self.home_last_position = current
                self.home_last_motion_at = now

            if abs(error) <= self.home_arrival_deg:
                self._advance_home_phase(next_phase, f"SAFE HOME: {label}")
                return

            # Do not let HOME hang forever on a servo that stops a few degrees
            # before the saved value. Continue only when it is reasonably close.
            phase_age = now - (self.home_phase_started_at or now)
            stalled_for = now - (self.home_last_motion_at or now)

            if (
                (phase_age >= self.home_phase_timeout_s or stalled_for >= 1.5)
                and abs(error) <= 7.0
            ):
                self._advance_home_phase(
                    next_phase,
                    f"SAFE HOME: {label} (accepted {error:+.1f}° residual)."
                )
                return

            if stalled_for >= 1.5 and abs(error) > 7.0:
                self.status_var.set(
                    f"SAFE HOME [{self.home_sequence_phase}] STALLED | "
                    f"{current:.1f}° -> {target:.1f}° | error={error:+.1f}° | "
                    "resending full hold-frame"
                )

            self._send_full_hold_frame(
                positions,
                {
                    joint: self._step_toward(
                        current,
                        target,
                        max_step,
                    )
                },
            )

            self.status_var.set(
                f"SAFE HOME [{self.home_sequence_phase}] "
                f"{current:.1f}° -> {target:.1f}° | error={error:+.1f}°"
            )

        if self.home_sequence_phase == "SHOULDER":
            move_one_joint(
                "shoulder_lift.pos",
                self.home_step_shoulder_deg,
                "ELBOW",
                "shoulder done -> elbow",
            )
            return

        if self.home_sequence_phase == "ELBOW":
            move_one_joint(
                "elbow_flex.pos",
                self.home_step_elbow_deg,
                "WRIST",
                "elbow done -> wrist",
            )
            return

        if self.home_sequence_phase == "WRIST":
            joints = ["wrist_flex.pos", WRIST_ROLL_JOINT, "gripper.pos"]
            command = {}
            errors = []

            for joint in joints:
                if joint not in home or joint not in positions:
                    continue
                current = float(positions[joint])
                target = float(home[joint])
                error = target - current
                errors.append(abs(error))
                if abs(error) > self.home_arrival_deg:
                    command[joint] = self._step_toward(
                        current, target, self.home_step_wrist_deg
                    )

            if not errors or max(errors) <= self.home_arrival_deg:
                self._advance_home_phase(
                    "BASE",
                    "SAFE HOME: wrist done -> rotating base last."
                )
                return

            phase_age = now - (self.home_phase_started_at or now)
            if phase_age >= self.home_phase_timeout_s and max(errors) <= 7.0:
                self._advance_home_phase(
                    "BASE",
                    "SAFE HOME: wrist close enough -> rotating base last."
                )
                return

            if command:
                self._send_full_hold_frame(
                    positions,
                    command,
                )

            self.status_var.set(
                f"SAFE HOME [WRIST] max error={max(errors):.1f}°"
            )
            return

        if self.home_sequence_phase == "BASE":
            joint = "shoulder_pan.pos"
            if joint not in home or joint not in positions:
                self.home_sequence_active = False
                self.home_sequence_phase = None
                self.status_var.set("SAFE HOME complete.")
                return

            current = float(positions[joint])
            target = float(home[joint])
            error = target - current

            if abs(error) <= self.home_arrival_deg:
                self.home_sequence_active = False
                self.home_sequence_phase = None
                self.status_var.set("SAFE HOME complete.")
                return

            self._send_full_hold_frame(
                positions,
                {
                    joint: self._step_toward(
                        current,
                        target,
                        self.home_step_base_deg,
                    )
                },
            )
            self.status_var.set(
                f"SAFE HOME [BASE] {current:.1f}° -> {target:.1f}° | "
                f"error={error:+.1f}°"
            )

    # ------------------------------------------------------
    # Cube input from Module 3
    # ------------------------------------------------------

    def get_cube_detections(self):
        data = self.app_context.shared_data.get(
            "cube_detections", []
        )

        if isinstance(data, dict):
            data = list(data.values())

        if not isinstance(data, (list, tuple)):
            return []

        valid = []
        for item in data:
            if not isinstance(item, dict):
                continue

            cls = str(
                item.get("class", item.get("class_name", ""))
            )
            if not cls.startswith("cube_"):
                continue

            world = item.get("world")
            if (
                not isinstance(world, (list, tuple))
                or len(world) < 2
            ):
                continue

            valid.append(item)

        return valid

    def lock_best_cube(self):
        detections = self.get_cube_detections()

        if not detections:
            self.status_var.set(
                "No cube_detections with world X/Y from Module 3."
            )
            return

        # Highest confidence is deterministic and easy to verify.
        cube = max(
            detections,
            key=lambda item: float(
                item.get("confidence", item.get("conf", 0.0))
            ),
        )

        world = cube["world"]

        self.locked_cube = {
            "class": str(cube.get("class", "cube")),
            "confidence": float(
                cube.get("confidence", cube.get("conf", 0.0))
            ),
            "camera_world_x": float(world[0]),
            "camera_world_y": float(world[1]),
        }

        self.correction_x_var.set(0.0)
        self.correction_y_var.set(0.0)
        self.update_corrected_info()
        self.clear_wrist_orientation()

        self.app_context.shared_data[
            "wrist_requested_class"
        ] = str(self.locked_cube["class"])

        self.status_var.set(
            "Cube locked. It stays fixed even if YOLO jitters or the arm occludes it."
        )

    def clear_lock(self):
        self.locked_cube = None
        self.cube_var.set("Cube: —")
        self.raw_var.set("Raw: —")
        self.corrected_var.set("Corrected: —")
        self.clear_wrist_orientation()

    # ------------------------------------------------------
    # Working Module-7 coordinate/model logic
    # ------------------------------------------------------

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
                "shoulder_pan.pos", positions["shoulder_pan.pos"]
            ),
            shoulder=self.motor_to_model(
                "shoulder_lift.pos", positions["shoulder_lift.pos"]
            ),
            elbow=self.motor_to_model(
                "elbow_flex.pos", positions["elbow_flex.pos"]
            ),
            wrist=self.motor_to_model(
                "wrist_flex.pos", positions["wrist_flex.pos"]
            ),
        )

    def get_current_angles(self):
        state = self.worker.get_state_snapshot()
        positions = state.get("positions", {})

        if not all(joint in positions for joint in JOINTS):
            raise RuntimeError("Motor positions incomplete.")

        return self.positions_to_angles(positions)

    def _choose_equivalent_motor_value(
        self, joint, value, current_motor
    ):
        candidates = [
            float(value) + 360.0 * k
            for k in (-2, -1, 0, 1, 2)
        ]

        if joint in self.joint_limits:
            item = self.joint_limits[joint]

            lo = float(
                item.get(
                    "min_unwrapped",
                    item.get("min", -1e9),
                )
            )
            hi = float(
                item.get(
                    "max_unwrapped",
                    item.get("max", 1e9),
                )
            )

            if lo > hi:
                lo, hi = hi, lo

            inside = [
                c for c in candidates
                if lo <= c <= hi
            ]

            if inside:
                return min(
                    inside,
                    key=lambda c: abs(c - current_motor),
                )

            return None

        return min(
            candidates,
            key=lambda c: abs(c - current_motor),
        )

    def angles_to_motor_positions(self, target_angles):
        state = self.worker.get_state_snapshot()
        current_pos = state.get("positions", {})

        if not all(joint in current_pos for joint in JOINTS):
            return None

        current_angles = self.get_current_angles()

        pairs = {
            "shoulder_pan.pos": (
                current_angles.base, target_angles.base
            ),
            "shoulder_lift.pos": (
                current_angles.shoulder, target_angles.shoulder
            ),
            "elbow_flex.pos": (
                current_angles.elbow, target_angles.elbow
            ),
            "wrist_flex.pos": (
                current_angles.wrist, target_angles.wrist
            ),
        }

        result = {}

        for joint, (current_model, target_model) in pairs.items():
            current_motor = float(current_pos[joint])

            model_delta = normalize_angle_deg(
                float(target_model) - float(current_model)
            )

            raw_target = (
                current_motor
                + model_delta / float(self.signs[joint])
            )

            safe = self._choose_equivalent_motor_value(
                joint, raw_target, current_motor
            )

            if safe is None:
                return None

            result[joint] = float(safe)

        return result

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

    def get_heading(self):
        angle = self.app_context.shared_data.get(
            "robot_base_marker_angle_deg"
        )

        if angle is None:
            return self.robot_frame_heading_offset

        return normalize_angle_deg(
            -float(angle)
            + self.robot_frame_heading_offset
        )

    @staticmethod
    def world_to_robot_local_raw(
        wx, wy, base, heading_deg
    ):
        bx, by = base

        dx = float(wx) - float(bx)
        dy = float(wy) - float(by)

        h = math.radians(heading_deg)

        fx = math.cos(h)
        fy = math.sin(h)

        rx = math.sin(h)
        ry = -math.cos(h)

        local_y = dx * fx + dy * fy
        local_x = dx * rx + dy * ry

        return local_x, local_y

    def world_to_robot_local(self, wx, wy):
        base = self.get_base_world()
        if base is None:
            return None

        x, y = self.world_to_robot_local_raw(
            wx, wy, base, self.get_heading()
        )

        # Module-7 frame convention:
        # positive saved frame offset means camera coordinate was too large.
        x -= self.robot_frame_offset_x
        y -= self.robot_frame_offset_y

        return x, y

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

    def get_locked_target_local(self):
        if self.locked_cube is None:
            return None

        # Module 3 world Y -> Module 7 scene Y.
        scene_x = float(
            self.locked_cube["camera_world_x"]
        )
        scene_y = -float(
            self.locked_cube["camera_world_y"]
        )

        raw = self.world_to_robot_local(
            scene_x, scene_y
        )

        if raw is None:
            return None

        learned_dx, learned_dy = (
            self.get_learned_cube_xy_correction(
                raw[0],
                raw[1],
            )
        )

        manual_dx = float(self.correction_x_var.get())
        manual_dy = float(self.correction_y_var.get())

        return (
            raw[0],
            raw[1],
            raw[0] + learned_dx + manual_dx,
            raw[1] + learned_dy + manual_dy,
        )

    def update_corrected_info(self):
        if self.locked_cube is None:
            return

        local = self.get_locked_target_local()

        self.cube_var.set(
            "Cube | "
            f"{self.locked_cube['class']} | "
            f"conf={self.locked_cube['confidence']:.2f}"
        )

        if local is None:
            self.raw_var.set("Raw: robot base unavailable")
            return

        rx, ry, cx, cy = local

        self.raw_var.set(
            f"Module-7 local from YOLO | X={rx:.1f}, Y={ry:.1f} mm"
        )

        learned_dx, learned_dy = (
            self.get_learned_cube_xy_correction(rx, ry)
        )

        self.corrected_var.set(
            f"Corrected target | X={cx:.1f}, Y={cy:.1f} mm | "
            f"learned=({learned_dx:+.1f},{learned_dy:+.1f}) | "
            f"manual=({float(self.correction_x_var.get()):+.1f},"
            f"{float(self.correction_y_var.get()):+.1f})"
        )

    # ------------------------------------------------------
    # IK
    # ------------------------------------------------------

    def solve_local_target(
        self,
        local_x,
        local_y,
        physical_z,
        prefer_vertical=True,
    ):
        local_x = float(local_x)
        local_y = float(local_y)
        physical_z = float(physical_z)

        reach = math.hypot(local_x, local_y)

        base_z_correction = self.get_z_correction(
            reach
        )

        cube_z_correction = (
            self.get_learned_cube_z_correction(
                reach
            )
        )

        z_correction = (
            base_z_correction
            + cube_z_correction
        )

        solver_z = (
            physical_z
            + z_correction
        )

        solver = IKSolverV2(self.get_ik_geometry())
        solver.table_z = 0.0
        solver.minimum_link_z = 0.0
        solver.minimum_gripper_z = min(0.0, solver_z)

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

        if prefer_vertical:
            candidates = sorted(
                result.candidates,
                key=lambda candidate: (
                    abs(float(candidate.tool_angle_deg) + 90.0),
                    float(candidate.score),
                ),
            )
        else:
            candidates = list(result.candidates)

        for candidate in candidates:
            angles = candidate.angles

            fk = forward_kinematics(
                angles,
                self.get_geometry(),
            )

            if any(
                float(fk[name][2]) < -0.5
                for name in (
                    "shoulder",
                    "elbow",
                    "wrist",
                    "gripper",
                )
            ):
                continue

            motors = self.angles_to_motor_positions(angles)
            if motors is None:
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
                "motors": motors,
                "angles": angles,
                "local_x": local_x,
                "local_y": local_y,
                "physical_z": physical_z,
                "reach": reach,
                "z_correction": z_correction,
                "base_z_correction": base_z_correction,
                "cube_z_correction": cube_z_correction,
                "solver_z": solver_z,
                "tool_angle_deg": float(candidate.tool_angle_deg),
            }

        return {
            "ok": False,
            "message": "No safe IK candidate.",
        }

    def solve_cube(self, physical_z=None):
        local = self.get_locked_target_local()

        if local is None:
            return {
                "ok": False,
                "message": "Lock a cube first / ID4 base unavailable.",
            }

        _, _, local_x, local_y = local

        if physical_z is None:
            physical_z = float(self.target_z_var.get())

        return self.solve_local_target(
            local_x,
            local_y,
            float(physical_z),
            prefer_vertical=True,
        )

    def solve_only(self):
        result = self.solve_cube()

        if not result["ok"]:
            self.ik_var.set(
                "IK REJECTED | " + result["message"]
            )
            return

        a = result["angles"]

        self.ik_var.set(
            "IK OK | "
            f"local=({result['local_x']:.1f}, "
            f"{result['local_y']:.1f}, "
            f"{result['physical_z']:.1f}) mm | "
            f"B={a.base:.1f} S={a.shoulder:.1f} "
            f"E={a.elbow:.1f} W={a.wrist:.1f} | "
            f"tool={result['tool_angle_deg']:.1f}° "
            f"(vertical=-90°) | "
            f"Zcorr={result['z_correction']:+.1f} "
            f"(base={result.get('base_z_correction', 0.0):+.1f}, "
            f"cube={result.get('cube_z_correction', 0.0):+.1f})"
        )

    def go_to_cube(self):
        result = self.solve_cube()

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

        if self.locked_cube is not None:
            self.app_context.shared_data[
                "wrist_requested_class"
            ] = str(self.locked_cube["class"])

        self.status_var.set(
            "Approach command sent. When the arm is near the cube, press OBSERVE."
        )

    # ------------------------------------------------------
    # Wrist-camera staged alignment
    # ------------------------------------------------------

    @staticmethod
    def _square_axis_error(target_deg, reference_deg):
        """
        Square orientation repeats every 90 degrees.
        Return the shortest equivalent angular error in [-45, +45).
        """
        return (
            (
                float(target_deg)
                - float(reference_deg)
                + 45.0
            )
            % 90.0
        ) - 45.0

    def start_wrist_observation(self):
        """
        Begin visual observation only.
        NO wrist movement is commanded here.
        """
        if self.locked_cube is None:
            self.status_var.set(
                "Lock a top-camera cube first."
            )
            return

        shared = self.app_context.shared_data

        shared["wrist_requested_class"] = str(
            self.locked_cube["class"]
        )

        # Make sure any previous frozen orientation is cleared.
        shared["wrist_unlock_request"] = True
        self.wrist_locked_angle_deg = None
        self.wrist_locked_class = None
        self.wrist_observation_active = True

        self.wrist_var.set(
            "Wrist: OBSERVING — hold still; angle will freeze automatically when stable."
        )
        self.status_var.set(
            "Wrist observation started. Keep the arm still near the cube."
        )

    def freeze_wrist_orientation(self):
        """
        Freeze a stable cube orientation from Module 9.
        We do NOT rotate in the same step.
        """
        stable = self.app_context.shared_data.get(
            "wrist_stable_target"
        )

        if not isinstance(stable, dict):
            self.wrist_var.set(
                "Wrist: not stable yet — keep observing without rotating."
            )
            return

        angle = stable.get("angle_deg")
        if angle is None:
            self.wrist_var.set(
                "Wrist: stable target has no angle."
            )
            return

        self.wrist_locked_angle_deg = float(angle)
        self.wrist_locked_class = str(
            stable.get("class", "cube")
        )
        self.wrist_observation_active = False

        # Freeze an ABSOLUTE wrist-roll target before any rotation starts.
        # Live camera angle must not be re-used while the wrist camera itself rotates.
        state = self.worker.get_state_snapshot()
        positions = state.get("positions", {})
        commanded = state.get("commanded_positions", {})

        reference = commanded.get(
            WRIST_ROLL_JOINT,
            positions.get(WRIST_ROLL_JOINT),
        )

        if reference is not None:
            frozen_error = self._square_axis_error(
                self.wrist_locked_angle_deg,
                90.0,
            )
            self.wrist_locked_roll_target = (
                float(reference)
                + self.wrist_servo_sign * float(frozen_error)
            )
        else:
            self.wrist_locked_roll_target = None

        # Also ask Module 9 to freeze its full visual observation.
        self.app_context.shared_data[
            "wrist_lock_request"
        ] = True

        self.wrist_var.set(
            f"Wrist: ORIENTATION FROZEN | "
            f"{self.wrist_locked_class} | "
            f"camera angle={self.wrist_locked_angle_deg:.1f}° | "
            "robot has NOT rotated yet"
        )

        self.status_var.set(
            "Cube orientation frozen. Now alignment can be applied."
        )

    def clear_wrist_orientation(self):
        self.wrist_locked_angle_deg = None
        self.wrist_locked_class = None
        self.wrist_locked_roll_target = None
        self.wrist_observation_active = False

        self.app_context.shared_data[
            "wrist_unlock_request"
        ] = True

        self.wrist_var.set(
            "Wrist: orientation lock cleared"
        )

    def align_wrist_once(self):
        """
        Move toward the ABSOLUTE wrist-roll target captured at FREEZE time.
        This makes repeated calls converge instead of applying the same
        frozen image-space error again and again.
        """
        if (
            self.wrist_locked_angle_deg is None
            or self.wrist_locked_roll_target is None
        ):
            self.wrist_var.set(
                "Wrist: freeze cube orientation first."
            )
            return False

        state = self.worker.get_state_snapshot()

        if not state.get("connected", False):
            self.wrist_var.set("Wrist: robot offline")
            return False

        if not state.get("torque_enabled", False):
            self.wrist_var.set("Wrist: torque disabled")
            return False

        positions = state.get("positions", {})
        current = positions.get(WRIST_ROLL_JOINT)

        if current is None:
            self.wrist_var.set(
                "Wrist: wrist_roll position unavailable"
            )
            return False

        remaining = (
            float(self.wrist_locked_roll_target)
            - float(current)
        )

        if abs(remaining) <= self.wrist_deadband_deg:
            self.wrist_var.set(
                "Wrist: ALIGNED | "
                f"target={self.wrist_locked_roll_target:.1f}° | "
                f"remaining={remaining:+.1f}°"
            )
            return True

        delta = max(
            -self.wrist_max_step_deg,
            min(
                self.wrist_max_step_deg,
                remaining,
            ),
        )

        self.worker.send(
            commands.move_to_positions(
                {
                    WRIST_ROLL_JOINT:
                        float(current) + float(delta)
                }
            )
        )

        self.wrist_var.set(
            f"Wrist: aligning | step={delta:+.1f}° | "
            f"remaining={remaining:+.1f}° | "
            f"frozen angle={self.wrist_locked_angle_deg:.1f}°"
        )

        return False

    # ------------------------------------------------------
    # Gripper + one-cube pick/place
    # ------------------------------------------------------

    def _capture_gripper_position(self, key):
        state = self.worker.get_state_snapshot()
        positions = state.get("positions", {})

        if "gripper.pos" not in positions:
            self.status_var.set(
                "Gripper position unavailable."
            )
            return

        value = float(positions["gripper.pos"])

        if key == "open":
            self.gripper_open_var.set(value)
            label = "FULL OPEN"
        else:
            self.gripper_close_limit_var.set(value)
            label = "FULL CLOSED"

        self._save_cube_calibration()

        self.status_var.set(
            f"Gripper {label} captured: {value:.1f}°"
        )

    def capture_gripper_open(self):
        self._capture_gripper_position("open")

    def capture_gripper_close_limit(self):
        self._capture_gripper_position("closed")

    def command_gripper(self, target):
        state = self.worker.get_state_snapshot()
        positions = state.get("positions", {})

        if not state.get("connected", False):
            self.status_var.set("Robot offline.")
            return

        current = positions.get("gripper.pos")

        if current is None:
            self.status_var.set(
                "Gripper position unavailable."
            )
            return

        self.worker.send(
            commands.move_to_positions(
                {"gripper.pos": float(target)}
            )
        )

    def _step_gripper_toward(self, target):
        """
        Simple position move used for OPEN / RELEASE.
        """
        state = self.worker.get_state_snapshot()
        positions = state.get("positions", {})
        current = positions.get("gripper.pos")

        if current is None:
            return False, "gripper position unavailable"

        current = float(current)
        error = float(target) - current

        if abs(error) <= max(
            self.gripper_arrival_deg,
            2.0,
        ):
            return (
                True,
                f"{current:.1f}° -> {float(target):.1f}°",
            )

        now = time.monotonic()

        if (
            now - self.gripper_last_command_at
            >= self.gripper_step_interval_s
        ):
            next_value = self._step_toward(
                current,
                target,
                self.gripper_step_deg,
            )

            self.worker.send(
                commands.move_to_positions(
                    {"gripper.pos": next_value}
                )
            )

            self.gripper_last_command_at = now

        return (
            False,
            f"{current:.1f}° -> {float(target):.1f}° | "
            f"err={error:+.1f}°",
        )

    def _reset_grip_close_state(self):
        self.gripper_close_started_at = None
        self.gripper_resistance_hits = 0
        self.grip_proxy_percent = 0.0

        try:
            self.grip_force_bar["value"] = 0.0
        except Exception:
            pass

        self.grip_force_var.set(
            "Grip force proxy: 0%"
        )

    def _close_gripper_until_strength(self):
        """
        Close gradually from the current position toward the captured
        FULL CLOSED limit.

        Because RobotWorker currently exposes no servo load/current,
        resistance = |commanded - measured| is used as a contact proxy.

        The selected Grip strength [%] is the stop threshold:
        50% means stop when the resistance proxy reaches half of the
        configured '100% resistance' value.
        """
        state = self.worker.get_state_snapshot()
        positions = state.get("positions", {})
        commanded_positions = state.get(
            "commanded_positions",
            {},
        )

        current = positions.get("gripper.pos")
        if current is None:
            return False, False, "gripper position unavailable"

        current = float(current)
        commanded = float(
            commanded_positions.get(
                "gripper.pos",
                current,
            )
        )

        if self.gripper_close_started_at is None:
            self.gripper_close_started_at = time.monotonic()
            self.gripper_resistance_hits = 0
            self.grip_proxy_percent = 0.0

        close_limit = float(
            self.gripper_close_limit_var.get()
        )

        full_resistance = max(
            0.5,
            abs(float(self.grip_resistance_100_var.get())),
        )

        requested_strength = max(
            5.0,
            min(
                100.0,
                float(self.grip_strength_percent_var.get()),
            ),
        )

        stop_resistance = (
            full_resistance
            * requested_strength
            / 100.0
        )

        resistance = abs(commanded - current)

        self.grip_proxy_percent = max(
            0.0,
            min(
                100.0,
                100.0 * resistance / full_resistance,
            ),
        )

        try:
            self.grip_force_bar["value"] = self.grip_proxy_percent
        except Exception:
            pass

        self.grip_force_var.set(
            f"Grip force proxy: {self.grip_proxy_percent:.0f}% | "
            f"resistance={resistance:.2f}° | "
            f"stop at {requested_strength:.0f}%"
        )

        if resistance >= stop_resistance:
            self.gripper_resistance_hits += 1
        else:
            self.gripper_resistance_hits = 0

        required_hits = max(
            1,
            int(self.grip_required_hits_var.get()),
        )

        if self.gripper_resistance_hits >= required_hits:
            # Do not keep pushing farther into the cube.
            # Hold the measured contact position.
            self.worker.send(
                commands.move_to_positions(
                    {"gripper.pos": current}
                )
            )

            return (
                True,
                True,
                f"object contact accepted | "
                f"force proxy={self.grip_proxy_percent:.0f}%",
            )

        # If we reached the captured no-object closed limit,
        # there was probably no cube between the fingers.
        if abs(current - close_limit) <= 2.0:
            return (
                False,
                False,
                "FULL CLOSED limit reached without enough resistance",
            )

        # Continue toward the physical close limit in small steps.
        now = time.monotonic()

        if (
            now - self.gripper_last_command_at
            >= self.gripper_step_interval_s
        ):
            next_value = self._step_toward(
                commanded,
                close_limit,
                self.gripper_step_deg,
            )

            self.worker.send(
                commands.move_to_positions(
                    {"gripper.pos": next_value}
                )
            )

            self.gripper_last_command_at = now

        return (
            True,
            False,
            f"closing | current={current:.1f}° | "
            f"cmd={commanded:.1f}° | "
            f"force={self.grip_proxy_percent:.0f}%",
        )

    def _move_arm_step_to_result(
        self,
        result,
        max_step_deg=4.0,
        arrival_deg=None,
        stall_accept_deg=None,
        final_snap_deg=None,
    ):
        """
        Drive toward one FROZEN motor target.

        For coarse APPROACH we may intentionally accept a larger residual
        after the arm has physically stopped, because the wrist camera and
        later descent stages provide the final refinement.
        """
        if not result.get("ok", False):
            return False, result.get("message", "IK failed")

        if arrival_deg is None:
            arrival_deg = self.pick_arrival_deg

        if stall_accept_deg is None:
            stall_accept_deg = self.pick_stall_accept_deg

        state = self.worker.get_state_snapshot()
        positions = state.get("positions", {})

        now = time.monotonic()

        command = {}
        errors = {}
        current_snapshot = {}
        all_arrived = True

        for joint, target in result["motors"].items():
            if joint not in positions:
                return False, f"Missing {joint}"

            current = float(positions[joint])
            target = float(target)
            current_snapshot[joint] = current

            error = target - current
            errors[joint] = error

            if abs(error) > arrival_deg:
                all_arrived = False
                command[joint] = self._step_toward(
                    current,
                    target,
                    max_step_deg,
                )

        worst_joint = None
        worst_error = 0.0

        if errors:
            worst_joint = max(
                errors,
                key=lambda joint: abs(errors[joint]),
            )
            worst_error = float(errors[worst_joint])

        max_error = abs(worst_error)

        if self.pick_phase_last_positions is None:
            self.pick_phase_last_positions = dict(current_snapshot)
            self.pick_phase_last_motion_at = now
        else:
            max_motion = 0.0

            for joint, current in current_snapshot.items():
                previous = self.pick_phase_last_positions.get(
                    joint,
                    current,
                )
                max_motion = max(
                    max_motion,
                    abs(current - previous),
                )

            if max_motion >= 0.20:
                self.pick_phase_last_positions = dict(current_snapshot)
                self.pick_phase_last_motion_at = now

        stalled_for = now - (
            self.pick_phase_last_motion_at
            if self.pick_phase_last_motion_at is not None
            else now
        )

        # FINAL SNAP:
        # Small incremental commands can be ignored by a loaded servo near the
        # final grasp pose.  When all remaining joint errors are already within
        # the configured snap window, send the exact frozen IK motor targets
        # instead of another current+step command. Wrist-roll and gripper are
        # held at their current values by _send_full_hold_frame().
        if (
            final_snap_deg is not None
            and max_error <= max(0.1, float(final_snap_deg))
            and max_error > float(arrival_deg)
        ):
            self._send_full_hold_frame(
                positions,
                {
                    joint: float(target)
                    for joint, target in result["motors"].items()
                    if joint in positions
                },
            )
            return (
                False,
                f"FINAL SNAP | max={max_error:.1f}° "
                f"({worst_joint}) | target exact",
            )

        if stalled_for >= self.normal_phase_stall_s:
            return (
                True,
                f"stopped | max={max_error:.1f}° "
                f"({worst_joint}) | stall={stalled_for:.1f}s",
            )

        if command:
            # Same reliability trick as SAFE HOME:
            # send all currently known joints, overriding only the joints
            # that need to move. This avoids a partial-command stall where
            # one joint (often shoulder_lift) stops responding until another
            # unrelated command is sent.
            self._send_full_hold_frame(
                positions,
                command,
            )

        return (
            False,
            f"moving | joint diagnostic={max_error:.1f}° "
            f"({worst_joint}) | stall={stalled_for:.1f}s",
        )

    def _move_lift_waypoint(
        self,
        result,
        accept_deg,
        max_step_deg,
    ):
        """
        Special controller for vertical post-grip lift.

        Unlike generic trajectory phases, lift waypoints do not wait for
        stall detection. If the whole arm is already close enough to the
        frozen waypoint, advance immediately.

        Joints within lift_joint_deadband_deg are not commanded again, which
        prevents the small back-and-forth corrections that were shaking the cube.
        """
        if not result.get("ok", False):
            return False, result.get("message", "IK failed")

        state = self.worker.get_state_snapshot()
        positions = state.get("positions", {})

        errors = {}
        command = {}

        for joint, target in result["motors"].items():
            if joint not in positions:
                return False, f"Missing {joint}"

            current = float(positions[joint])
            target = float(target)
            error = target - current
            errors[joint] = error

            # Deadband: once a joint is already close, stop touching it.
            if abs(error) <= self.lift_joint_deadband_deg:
                continue

            command[joint] = self._step_toward(
                current,
                target,
                max_step_deg,
            )

        worst_joint = None
        worst_error = 0.0

        if errors:
            worst_joint = max(
                errors,
                key=lambda joint: abs(errors[joint]),
            )
            worst_error = float(errors[worst_joint])

        max_error = abs(worst_error)

        # Main rule for lift:
        # close enough -> NEXT WAYPOINT immediately.
        if max_error <= float(accept_deg):
            return (
                True,
                f"close enough | max={max_error:.1f}° "
                f"({worst_joint or '—'})",
            )

        if command:
            self._send_full_hold_frame(
                positions,
                command,
            )

        return (
            False,
            f"lifting | max={max_error:.1f}° "
            f"({worst_joint})",
        )

    def _container_correction_row(self, parent, label, variable):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=22).pack(side="left")
        for button_text, delta in (("-5", -5.0), ("-1", -1.0)):
            ttk.Button(
                row, text=button_text, width=4,
                command=lambda d=delta, v=variable: self.bump_container_trim(v, d),
            ).pack(side="left", padx=1)
        ttk.Entry(row, textvariable=variable, width=9).pack(side="left", padx=5)
        for button_text, delta in (("+1", 1.0), ("+5", 5.0)):
            ttk.Button(
                row, text=button_text, width=4,
                command=lambda d=delta, v=variable: self.bump_container_trim(v, d),
            ).pack(side="left", padx=1)

    def _container_cal_toggle(self):
        self.container_cal_enabled = bool(self.container_cal_enabled_var.get())
        self._save_cube_calibration()
        self.update_container_calibration_status()

    def update_container_calibration_status(self):
        enabled = bool(self.container_cal_enabled_var.get())
        self.container_cal_var.set(
            f"Container XYZ calibration: {'ON' if enabled else 'OFF'} | "
            f"samples={len(self.container_cal_samples)} | "
            f"dX=[{self.container_coeff_x[0]:+.2f}, {self.container_coeff_x[1]:+.4f}X, {self.container_coeff_x[2]:+.4f}Y] | "
            f"dY=[{self.container_coeff_y[0]:+.2f}, {self.container_coeff_y[1]:+.4f}X, {self.container_coeff_y[2]:+.4f}Y] | "
            f"dZ=[{self.container_coeff_z[0]:+.2f}, {self.container_coeff_z[1]:+.4f}X, {self.container_coeff_z[2]:+.4f}Y]"
        )

    def _get_container_local_by_marker(self, marker_id, apply_calibration=True):
        containers = self.app_context.shared_data.get("containers_world", {}) or {}
        world = containers.get(marker_id)
        if world is None:
            world = containers.get(str(marker_id))
        if not isinstance(world, (list, tuple)) or len(world) < 2:
            return None

        scene_x = float(world[0])
        scene_y = -float(world[1])
        local = self.world_to_robot_local(scene_x, scene_y)
        if local is None:
            return None

        raw_x, raw_y = float(local[0]), float(local[1])
        x, y = raw_x, raw_y

        if apply_calibration and bool(self.container_cal_enabled_var.get()):
            basis = (1.0, raw_x, raw_y)
            dx = sum(c*b for c, b in zip(self.container_coeff_x, basis))
            dy = sum(c*b for c, b in zip(self.container_coeff_y, basis))
            x += dx
            y += dy

        return raw_x, raw_y, x, y

    def get_container_z_correction(self, raw_x, raw_y):
        if not bool(self.container_cal_enabled_var.get()):
            return 0.0
        basis = (1.0, float(raw_x), float(raw_y))
        return sum(c*b for c, b in zip(self.container_coeff_z, basis))

    def bump_container_trim(self, variable, delta):
        variable.set(float(variable.get()) + float(delta))
        # Live motion is deliberate for calibration.
        self.go_container_calibration()

    def go_container_calibration(self):
        marker_id = int(self.container_cal_marker_var.get())
        local = self._get_container_local_by_marker(marker_id, apply_calibration=False)
        if local is None:
            self.status_var.set(f"CONTAINER CAL: ID{marker_id} unavailable.")
            return

        raw_x, raw_y, _, _ = local
        learned_dx = learned_dy = learned_dz = 0.0
        if bool(self.container_cal_enabled_var.get()):
            basis = (1.0, raw_x, raw_y)
            learned_dx = sum(c*b for c, b in zip(self.container_coeff_x, basis))
            learned_dy = sum(c*b for c, b in zip(self.container_coeff_y, basis))
            learned_dz = sum(c*b for c, b in zip(self.container_coeff_z, basis))

        x = raw_x + learned_dx + float(self.container_trim_x_var.get())
        y = raw_y + learned_dy + float(self.container_trim_y_var.get())
        z = (
            float(self.drop_z_var.get())
            + learned_dz
            + float(self.container_trim_z_var.get())
        )

        result = self.solve_local_target(x, y, z, prefer_vertical=True)
        if not result.get("ok", False):
            self.status_var.set("CONTAINER CAL IK failed: " + str(result.get("message", "unknown")))
            return

        state = self.worker.get_state_snapshot()
        self._send_full_hold_frame(state.get("positions", {}), result["motors"])
        self.status_var.set(
            f"CONTAINER CAL ID{marker_id} | raw=({raw_x:.1f},{raw_y:.1f}) | "
            f"target=({x:.1f},{y:.1f},{z:.1f})"
        )

    def capture_container_calibration(self):
        marker_id = int(self.container_cal_marker_var.get())
        local = self._get_container_local_by_marker(marker_id, apply_calibration=False)
        if local is None:
            self.status_var.set(f"CAPTURE CONTAINER: ID{marker_id} unavailable.")
            return

        raw_x, raw_y, _, _ = local

        # Store the TOTAL correction that produced the manually verified target.
        # If an old learned model was enabled while refining, include it too.
        basis = (1.0, raw_x, raw_y)
        old_dx = old_dy = old_dz = 0.0
        if bool(self.container_cal_enabled_var.get()):
            old_dx = sum(c*b for c, b in zip(self.container_coeff_x, basis))
            old_dy = sum(c*b for c, b in zip(self.container_coeff_y, basis))
            old_dz = sum(c*b for c, b in zip(self.container_coeff_z, basis))

        sample = {
            "marker_id": marker_id,
            "raw_x": raw_x,
            "raw_y": raw_y,
            "dx": old_dx + float(self.container_trim_x_var.get()),
            "dy": old_dy + float(self.container_trim_y_var.get()),
            "dz": old_dz + float(self.container_trim_z_var.get()),
        }
        self.container_cal_samples.append(sample)
        self._save_cube_calibration()
        self.update_container_calibration_status()

        self.container_trim_x_var.set(0.0)
        self.container_trim_y_var.set(0.0)
        self.container_trim_z_var.set(0.0)
        self.status_var.set(
            f"Container sample captured at ({raw_x:.1f},{raw_y:.1f}) | "
            f"d=({sample['dx']:+.1f},{sample['dy']:+.1f},{sample['dz']:+.1f}) mm"
        )

    def fit_container_calibration(self):
        if len(self.container_cal_samples) < 3:
            self.status_var.set("FIT CONTAINER XYZ: need at least 3 samples in different positions.")
            return

        try:
            A = np.asarray(
                [[1.0, float(s["raw_x"]), float(s["raw_y"])]
                 for s in self.container_cal_samples],
                dtype=np.float64,
            )
            bx = np.asarray([float(s["dx"]) for s in self.container_cal_samples], dtype=np.float64)
            by = np.asarray([float(s["dy"]) for s in self.container_cal_samples], dtype=np.float64)
            bz = np.asarray([float(s["dz"]) for s in self.container_cal_samples], dtype=np.float64)

            cx, *_ = np.linalg.lstsq(A, bx, rcond=None)
            cy, *_ = np.linalg.lstsq(A, by, rcond=None)
            cz, *_ = np.linalg.lstsq(A, bz, rcond=None)

            self.container_coeff_x = [float(v) for v in cx]
            self.container_coeff_y = [float(v) for v in cy]
            self.container_coeff_z = [float(v) for v in cz]
            self.container_cal_enabled = True
            self.container_cal_enabled_var.set(True)
            self._save_cube_calibration()
            self.update_container_calibration_status()
            self.status_var.set(
                f"FIT CONTAINER XYZ complete from {len(self.container_cal_samples)} samples."
            )
        except Exception as exc:
            self.status_var.set(f"FIT CONTAINER XYZ failed: {exc}")

    def clear_container_calibration(self):
        self.container_cal_samples = []
        self.container_coeff_x = [0.0, 0.0, 0.0]
        self.container_coeff_y = [0.0, 0.0, 0.0]
        self.container_coeff_z = [0.0, 0.0, 0.0]
        self.container_cal_enabled = False
        self.container_cal_enabled_var.set(False)
        self.container_trim_x_var.set(0.0)
        self.container_trim_y_var.set(0.0)
        self.container_trim_z_var.set(0.0)
        self._save_cube_calibration()
        self.update_container_calibration_status()
        self.status_var.set("Container XYZ calibration cleared.")

    def get_container_local_for_locked_cube(self):
        if self.locked_cube is None:
            return None

        marker_id = self.container_map.get(
            str(self.locked_cube.get("class"))
        )
        if marker_id is None:
            return None

        local = self._get_container_local_by_marker(
            marker_id,
            apply_calibration=True,
        )
        if local is None:
            return None

        raw_x, raw_y, x, y = local
        return marker_id, float(x), float(y)

    def _solve_lift_from_grasp(self, target_z=None):
        """Simple Module-7-style lift: keep calibrated GRASP X/Y, change only Z."""
        grasp = self.get_grasp_target_local(include_manual_trim=False)
        if grasp is None:
            return {"ok": False, "message": "Lock a cube first / grasp target unavailable."}

        _, _, x, y, _ = grasp
        if target_z is None:
            target_z = float(self.lift_clear_z_var.get())

        return self.solve_local_target(
            float(x), float(y), float(target_z), prefer_vertical=True
        )

    def start_manual_grip_close(self):
        state = self.worker.get_state_snapshot()
        if not state.get("connected", False) or not state.get("torque_enabled", False):
            self.status_var.set("CLOSE GRIPPER: robot offline or torque OFF.")
            return
        self._reset_grip_close_state()
        self.manual_grip_active = True
        self.status_var.set("CLOSE GRIPPER: closing until configured resistance threshold.")

    def _update_manual_grip(self):
        if not self.manual_grip_active:
            return
        ok, gripped, info = self._close_gripper_until_strength()
        self.status_var.set("Manual grip | " + info)
        if not ok:
            self.manual_grip_active = False
            return
        if gripped:
            self.manual_grip_active = False
            self.status_var.set("Manual grip complete | " + info)

    def test_lift(self):
        """One direct Cartesian lift target; no waypoints, no re-detection, no XY change."""
        state = self.worker.get_state_snapshot()
        if not state.get("connected", False) or not state.get("torque_enabled", False):
            self.status_var.set("TEST LIFT: robot offline or torque OFF.")
            return

        result = self._solve_lift_from_grasp(float(self.lift_clear_z_var.get()))
        if not result.get("ok", False):
            self.status_var.set("TEST LIFT IK failed: " + str(result.get("message", "unknown")))
            return

        # Send the calibrated Module-7-style IK motor target. Preserve wrist-roll/gripper
        # by sending a full hold frame around the four IK joints.
        positions = state.get("positions", {})
        self._send_full_hold_frame(positions, result["motors"])
        self.status_var.set(
            f"TEST LIFT sent | same grasp X/Y=({result['local_x']:.1f},"
            f"{result['local_y']:.1f}) | Z={result['physical_z']:.1f} mm"
        )

    def go_container_manual(self):
        container = self.get_container_local_for_locked_cube()
        if container is None:
            self.status_var.set("GO CONTAINER: mapped container unavailable.")
            return
        marker_id, x, y = container
        raw_local = self._get_container_local_by_marker(marker_id, apply_calibration=False)
        dz = 0.0 if raw_local is None else self.get_container_z_correction(raw_local[0], raw_local[1])
        target_z = float(self.container_travel_z_var.get()) + dz
        result = self.solve_local_target(
            x, y, target_z, prefer_vertical=True
        )
        if not result.get("ok", False):
            self.status_var.set("GO CONTAINER IK failed: " + str(result.get("message", "unknown")))
            return
        state = self.worker.get_state_snapshot()
        self._send_full_hold_frame(state.get("positions", {}), result["motors"])
        self.status_var.set(
            f"GO CONTAINER sent | ID{marker_id} | X={x:.1f}, Y={y:.1f}, "
            f"Z={target_z:.1f} mm"
        )

    def release_manual(self):
        self.manual_grip_active = False
        self.command_gripper(float(self.gripper_open_var.get()))
        self.status_var.set("RELEASE sent: gripper opening.")

    def start_pick_place(self):
        state = self.worker.get_state_snapshot()

        if not state.get("connected", False):
            self.status_var.set("Pick & Place cannot start: robot offline.")
            return

        if not state.get("torque_enabled", False):
            self.status_var.set("Pick & Place cannot start: torque is OFF.")
            return

        if self.locked_cube is None:
            self.status_var.set("Pick & Place cannot start: LOCK CUBE first.")
            return

        if "gripper.pos" not in state.get("positions", {}):
            self.status_var.set(
                "Pick & Place cannot start: gripper position unavailable."
            )
            return

        container = self.get_container_local_for_locked_cube()
        if container is None:
            cls = str(self.locked_cube.get("class", "cube"))
            marker_id = self.container_map.get(cls)
            self.status_var.set(
                f"Pick & Place cannot start: container ID{marker_id} "
                f"for {cls} is unavailable from Module 3."
            )
            return

        # Verify the first approach IK before entering the state machine.
        test = self.solve_cube(
            physical_z=float(self.travel_z_var.get())
        )
        if not test.get("ok", False):
            self.status_var.set(
                "Pick & Place cannot start: approach IK failed: "
                + str(test.get("message", "unknown"))
            )
            return

        self.home_sequence_active = False
        self.manual_grip_active = False
        self.pick_place_active = True
        self.lift_progress_z = None
        self.lift_waypoints_z = []
        self.container_travel_waypoints = []
        self.container_waypoint_index = 0
        self.safe_lift_targets = []
        self.safe_lift_index = 0
        self.clear_wrist_orientation()
        self._reset_grip_close_state()
        self._set_pick_state("OPEN", "starting")

        self.status_var.set(
            "Pick & Place started. Watch the phase/status line."
        )

    def stop_pick_place(self):
        self.pick_place_active = False
        self.pick_place_state = "IDLE"
        self.pick_place_state_started_at = None
        self.pick_place_var.set(
            "Pick & Place: STOPPED"
        )

    def _set_pick_state(self, state, message):
        self.pick_place_state = str(state)
        self.pick_place_state_started_at = time.monotonic()

        # A new phase gets a fresh frozen target.
        self.pick_phase_result = None
        self.pick_phase_last_positions = None
        self.pick_phase_last_motion_at = time.monotonic()
        self.grasp_snap_started_at = None

        self.pick_place_var.set(
            f"Pick & Place: {state} | {message}"
        )

    def _pick_fail(self, message):
        self.pick_place_active = False
        self.pick_place_state = "ERROR"
        self.pick_phase_result = None
        self.pick_place_var.set(f"Pick & Place: ERROR | {message}")
        self.status_var.set(f"Pick & Place stopped: {message}")

    def _pick_state_age(self):
        if self.pick_place_state_started_at is None:
            return 0.0
        return time.monotonic() - self.pick_place_state_started_at

    def _update_pick_place(self):
        if not self.pick_place_active:
            return

        if self.locked_cube is None:
            self._pick_fail("cube lock lost")
            return

        robot = self.worker.get_state_snapshot()

        if not robot.get("connected", False):
            self._pick_fail("robot disconnected")
            return

        if not robot.get("torque_enabled", False):
            self._pick_fail("torque turned OFF")
            return

        state = self.pick_place_state
        age = self._pick_state_age()

        if state == "OPEN":
            arrived, info = self._step_gripper_toward(
                float(self.gripper_open_var.get())
            )

            self.pick_place_var.set(
                f"Pick & Place: OPEN | {info}"
            )

            if arrived or age > 4.0:
                self._set_pick_state(
                    "APPROACH",
                    "gripper open / solving frozen approach target"
                )
            return

        if state == "APPROACH":
            if self.pick_phase_result is None:
                self.pick_phase_result = self.solve_cube(
                    physical_z=float(self.travel_z_var.get())
                )

                if not self.pick_phase_result.get("ok", False):
                    self._pick_fail(
                        "approach IK: "
                        + str(
                            self.pick_phase_result.get(
                                "message",
                                "failed",
                            )
                        )
                    )
                    return

            arrived, info = self._move_arm_step_to_result(
                self.pick_phase_result,
                max_step_deg=5.0,
                arrival_deg=self.pick_arrival_deg,
                stall_accept_deg=max(1.0, float(self.approach_accept_var.get())),
            )

            self.pick_place_var.set(
                f"Pick & Place: APPROACH | {info} | {age:.1f}s"
            )

            if arrived:
                self.start_wrist_observation()
                self._set_pick_state(
                    "OBSERVE",
                    "waiting for stable wrist angle"
                )

            elif age > max(1.0, float(self.motion_timeout_var.get())):
                self._pick_fail(
                    "approach did not converge | " + info
                )

            return

        if state == "OBSERVE":
            tracking = self.app_context.shared_data.get(
                "wrist_tracking_target"
            )

            if self.wrist_locked_angle_deg is not None:
                self._set_pick_state(
                    "ALIGN",
                    "orientation frozen"
                )
                return

            if isinstance(tracking, dict):
                self.pick_place_var.set(
                    "Pick & Place: OBSERVE | "
                    f"{tracking.get('class', 'cube')} | "
                    f"angle≈{float(tracking.get('angle_deg', 0.0)):.1f}° | "
                    f"{age:.1f}s"
                )
            else:
                self.pick_place_var.set(
                    "Pick & Place: OBSERVE | "
                    f"waiting for wrist detection | {age:.1f}s"
                )

            if age > 8.0:
                self._pick_fail(
                    "wrist observation never became stable; check Module 9"
                )

            return

        if state == "ALIGN":
            done = self.align_wrist_once()

            self.pick_place_var.set(
                f"Pick & Place: ALIGN | "
                f"{self.wrist_var.get()} | {age:.1f}s"
            )

            if done:
                self._set_pick_state(
                    "DESCEND",
                    "wrist aligned / solving frozen descent target"
                )

            elif age > 8.0:
                self._pick_fail(
                    "wrist alignment did not converge"
                )

            return

        if state == "DESCEND":
            if self.pick_phase_result is None:
                if not bool(
                    self.grasp_layer_enabled_var.get()
                ):
                    self._pick_fail(
                        "GRASP XYZ calibration is not fitted/enabled. "
                        "Calibrate the grasp layer first."
                    )
                    return

                self.pick_phase_result = self.solve_grasp_target(
                    include_manual_trim=False
                )

                if not self.pick_phase_result.get("ok", False):
                    self._pick_fail(
                        "descent/grasp IK: "
                        + str(
                            self.pick_phase_result.get(
                                "message",
                                "failed",
                            )
                        )
                    )
                    return

            arrived, info = self._move_arm_step_to_result(
                self.pick_phase_result,
                max_step_deg=3.5,
                arrival_deg=3.0,
                stall_accept_deg=max(1.0, float(self.grasp_accept_var.get())),
                final_snap_deg=max(0.1, float(self.grasp_final_snap_var.get())),
            )

            # If the exact FINAL SNAP command has been sent but a loaded joint
            # still refuses to move the last few degrees, do not wait forever.
            # We keep the normal Grasp tolerance strict, but allow a separate,
            # small post-snap residual before starting the gripper close.
            now_snap = time.monotonic()
            if "FINAL SNAP" in info:
                if self.grasp_snap_started_at is None:
                    self.grasp_snap_started_at = now_snap

                try:
                    snap_max_text = info.split("max=", 1)[1].split("°", 1)[0]
                    snap_max_error = abs(float(snap_max_text))
                except Exception:
                    snap_max_error = 999.0

                snap_age = now_snap - self.grasp_snap_started_at
                snap_accept = max(
                    0.1,
                    float(self.grasp_snap_accept_var.get()),
                )

                if snap_age >= 1.5 and snap_max_error <= snap_accept:
                    self._set_pick_state(
                        "GRIP",
                        (
                            "final snap accepted | "
                            f"residual={snap_max_error:.1f}° "
                            f"after {snap_age:.1f}s"
                        ),
                    )
                    return
            else:
                self.grasp_snap_started_at = None

            grasp_target = self.get_grasp_target_local(
                include_manual_trim=False
            )

            if grasp_target is not None:
                _, _, gx, gy, gz = grasp_target
                target_text = (
                    f"target=({gx:.1f},{gy:.1f},{gz:.1f}) mm"
                )
            else:
                target_text = "target=unavailable"

            stalled_for = (
                time.monotonic()
                - (
                    self.pick_phase_last_motion_at
                    if self.pick_phase_last_motion_at is not None
                    else time.monotonic()
                )
            )

            self.pick_place_var.set(
                f"Pick & Place: DESCEND | "
                f"{target_text} | "
                f"snap≤{float(self.grasp_final_snap_var.get()):.1f}° | "
                f"snap accept≤{float(self.grasp_snap_accept_var.get()):.1f}° | "
                f"{info} | stall={stalled_for:.1f}s | {age:.1f}s"
            )

            if arrived:
                self._set_pick_state(
                    "GRIP",
                    "at grasp height / close enough accepted"
                )

            elif (
                stalled_for >= self.descend_stall_accept_after_s
                and "max=" in info
            ):
                # _move_arm_step_to_result already knows the current max error.
                # If it has physically stopped and the residual is within the
                # dedicated descent tolerance, force the transition instead of
                # waiting for the global 18 s timeout.
                try:
                    max_part = info.split("max=", 1)[1].split("°", 1)[0]
                    max_error = abs(float(max_part))
                except Exception:
                    max_error = 999.0

                if max_error <= max(1.0, float(self.grasp_accept_var.get())):
                    self._set_pick_state(
                        "GRIP",
                        (
                            "descent accepted after stall | "
                            f"max residual={max_error:.1f}°"
                        )
                    )
                    return

            if age > max(1.0, float(self.motion_timeout_var.get())):
                self._pick_fail(
                    "descent did not converge | " + info
                )

            return

        if state == "GRIP":
            ok, gripped, info = (
                self._close_gripper_until_strength()
            )

            self.pick_place_var.set(
                f"Pick & Place: GRIP | {info}"
            )

            if not ok:
                self._pick_fail(info)
                return

            if gripped:
                # Simple lift: the next state solves ONE Cartesian target using
                # the calibrated GRASP X/Y and only changes Z.
                self._set_pick_state(
                    "LIFT",
                    "grip detected -> direct Cartesian lift"
                )

            elif age > 5.0:
                self._pick_fail(
                    "gripper close timeout / no reliable contact"
                )

            return

        if state == "LIFT":
            clear_z = max(
                70.0,
                float(self.lift_clear_z_var.get()),
            )

            # Build exactly 3 vertical targets from the calibrated grasp pose.
            # Same X/Y, only Z changes. Each step uses the current stall-driven
            # acceptance, so no degree threshold can block the sequence.
            if not self.safe_lift_targets:
                grasp = self.get_grasp_target_local(
                    include_manual_trim=False
                )
                if grasp is None:
                    self._pick_fail(
                        "grasp target unavailable for safe lift"
                    )
                    return

                _, _, _, _, grasp_z = grasp
                start_z = float(grasp_z)

                # Three meaningful lift stages.
                z1 = max(start_z + 15.0, 30.0)
                z2 = max(start_z + 35.0, 50.0)
                z3 = clear_z

                # Keep them strictly increasing and never above final by accident.
                values = []
                for z in (z1, z2, z3):
                    z = min(float(z), float(clear_z))
                    if not values or z > values[-1] + 0.5:
                        values.append(z)

                if not values:
                    values = [clear_z]

                self.safe_lift_targets = values
                self.safe_lift_index = 0

            if self.safe_lift_index >= len(self.safe_lift_targets):
                container = self.get_container_local_for_locked_cube()
                if container is None:
                    self._pick_fail(
                        "mapped container disappeared after safe lift"
                    )
                    return

                marker_id, _, _ = container
                self.safe_lift_targets = []
                self.safe_lift_index = 0
                self._set_pick_state(
                    "CONTAINER",
                    f"safe lift complete -> ID{marker_id}"
                )
                return

            target_z = self.safe_lift_targets[self.safe_lift_index]

            if self.pick_phase_result is None:
                self.pick_phase_result = self._solve_lift_from_grasp(
                    target_z
                )

                if not self.pick_phase_result.get("ok", False):
                    self._pick_fail(
                        "safe lift IK: "
                        + str(
                            self.pick_phase_result.get(
                                "message",
                                "failed",
                            )
                        )
                    )
                    return

                self.pick_place_state_started_at = time.monotonic()
                age = 0.0

            arrived, info = self._move_arm_step_to_result(
                self.pick_phase_result,
                max_step_deg=5.0,
                arrival_deg=3.0,
                stall_accept_deg=max(
                    1.0,
                    float(self.lift_accept_var.get()),
                ),
            )

            step_no = self.safe_lift_index + 1
            step_count = len(self.safe_lift_targets)

            self.pick_place_var.set(
                f"Pick & Place: SAFE LIFT | "
                f"step {step_no}/{step_count} | "
                f"Z={target_z:.0f} mm | {info} | {age:.1f}s"
            )

            if arrived:
                self.safe_lift_index += 1
                self.pick_phase_result = None
                self.pick_phase_last_positions = None
                self.pick_phase_last_motion_at = time.monotonic()
                self.pick_place_state_started_at = time.monotonic()

            elif age > max(
                1.0,
                float(self.motion_timeout_var.get()),
            ):
                self._pick_fail(
                    "safe lift step did not converge | " + info
                )

            return

        if state == "CONTAINER":
            container = self.get_container_local_for_locked_cube()

            if container is None:
                self._pick_fail(
                    "mapped container disappeared"
                )
                return

            marker_id, container_x, container_y = container
            raw_container = self._get_container_local_by_marker(marker_id, apply_calibration=False)
            container_dz = (
                0.0 if raw_container is None
                else self.get_container_z_correction(raw_container[0], raw_container[1])
            )

            if not self.container_travel_waypoints:
                cube_local = self.get_locked_target_local()

                if cube_local is None:
                    self._pick_fail(
                        "cube target unavailable for container travel"
                    )
                    return

                _, _, cube_x, cube_y = cube_local

                self.container_travel_waypoints = [
                    (
                        cube_x + 0.5 * (container_x - cube_x),
                        cube_y + 0.5 * (container_y - cube_y),
                    ),
                    (
                        container_x,
                        container_y,
                    ),
                ]
                self.container_waypoint_index = 0

            if self.container_waypoint_index >= len(
                self.container_travel_waypoints
            ):
                self._set_pick_state(
                    "DROP",
                    f"above container ID{marker_id}"
                )
                return

            waypoint_x, waypoint_y = (
                self.container_travel_waypoints[
                    self.container_waypoint_index
                ]
            )

            if self.pick_phase_result is None:
                self.pick_phase_result = self.solve_local_target(
                    waypoint_x,
                    waypoint_y,
                    (float(self.container_travel_z_var.get()) + container_dz),
                    prefer_vertical=True,
                )

                self.pick_phase_result[
                    "_container_id"
                ] = marker_id
                self.pick_phase_result[
                    "_container_waypoint"
                ] = self.container_waypoint_index + 1

                self.pick_phase_last_positions = None
                self.pick_phase_last_motion_at = time.monotonic()
                self.pick_place_state_started_at = time.monotonic()
                age = 0.0

                if not self.pick_phase_result.get("ok", False):
                    self._pick_fail(
                        "container IK: "
                        + str(
                            self.pick_phase_result.get(
                                "message",
                                "failed",
                            )
                        )
                    )
                    return

            waypoint_number = int(
                self.pick_phase_result.get(
                    "_container_waypoint",
                    self.container_waypoint_index + 1,
                )
            )

            is_final_waypoint = (
                self.container_waypoint_index
                >= len(self.container_travel_waypoints) - 1
            )

            arrived, info = self._move_arm_step_to_result(
                self.pick_phase_result,
                max_step_deg=(7.0 if not is_final_waypoint else 6.0),
                arrival_deg=(4.0 if not is_final_waypoint else 3.0),
                stall_accept_deg=(
                    max(1.0, float(self.container_accept_var.get()) + 2.0)
                    if not is_final_waypoint
                    else max(1.0, float(self.container_accept_var.get()))
                ),
            )

            self.pick_place_var.set(
                f"Pick & Place: CONTAINER | ID{marker_id} | "
                f"waypoint {waypoint_number}/{len(self.container_travel_waypoints)} | "
                f"{info} | {age:.1f}s"
            )

            if arrived:
                self.container_waypoint_index += 1
                self.pick_phase_result = None
                self.pick_phase_last_positions = None
                self.pick_phase_last_motion_at = time.monotonic()
                self.pick_place_state_started_at = time.monotonic()

                if self.container_waypoint_index >= len(
                    self.container_travel_waypoints
                ):
                    self._set_pick_state(
                        "DROP",
                        f"above container ID{marker_id}"
                    )

            elif age > max(1.0, float(self.motion_timeout_var.get())):
                self._pick_fail(
                    "container did not converge | " + info
                )

            return

        if state == "DROP":
            container = self.get_container_local_for_locked_cube()

            if container is None:
                self._pick_fail(
                    "mapped container disappeared"
                )
                return

            marker_id, x, y = container
            raw_container = self._get_container_local_by_marker(marker_id, apply_calibration=False)
            container_dz = (
                0.0 if raw_container is None
                else self.get_container_z_correction(raw_container[0], raw_container[1])
            )

            if self.pick_phase_result is None:
                self.pick_phase_result = self.solve_local_target(
                    x,
                    y,
                    (float(self.drop_z_var.get()) + container_dz),
                    prefer_vertical=True,
                )
                self.pick_phase_result["_container_id"] = marker_id

                if not self.pick_phase_result.get("ok", False):
                    self._pick_fail(
                        "drop IK: "
                        + str(
                            self.pick_phase_result.get(
                                "message",
                                "failed",
                            )
                        )
                    )
                    return

            arrived, info = self._move_arm_step_to_result(
                self.pick_phase_result,
                max_step_deg=4.0,
                arrival_deg=4.0,
                stall_accept_deg=max(1.0, float(self.drop_accept_var.get())),
            )

            self.pick_place_var.set(
                f"Pick & Place: DROP | ID{marker_id} | "
                f"accept≤{float(self.drop_accept_var.get()):.1f}° | "
                f"{info} | {age:.1f}s"
            )

            if arrived:
                self._set_pick_state(
                    "RELEASE",
                    "at drop height"
                )

            elif age > max(1.0, float(self.motion_timeout_var.get())):
                # Dropping is intentionally tolerant. The cube only needs to be
                # safely over the container; unlike grasping, exact joint
                # convergence is not required before opening the gripper.
                state_snapshot = self.worker.get_state_snapshot()
                positions = state_snapshot.get("positions", {})
                errors = []

                for joint, target in self.pick_phase_result.get("motors", {}).items():
                    if joint in positions:
                        errors.append(
                            abs(float(target) - float(positions[joint]))
                        )

                max_error = max(errors) if errors else 999.0

                if max_error <= max(1.0, float(self.drop_accept_var.get())):
                    self.app_context.log(
                        f"DROP accepted near container after timeout: "
                        f"max residual {max_error:.1f} deg"
                    )
                    self._set_pick_state(
                        "RELEASE",
                        f"drop accepted near target ({max_error:.1f} deg)"
                    )
                else:
                    self._pick_fail(
                        "drop did not converge | " + info
                    )

            return

        if state == "LIFT_AFTER_DROP":
            container = self.get_container_local_for_locked_cube()

            if container is None:
                self._pick_fail(
                    "mapped container disappeared"
                )
                return

            marker_id, x, y = container
            raw_container = self._get_container_local_by_marker(marker_id, apply_calibration=False)
            container_dz = (
                0.0 if raw_container is None
                else self.get_container_z_correction(raw_container[0], raw_container[1])
            )

            if self.pick_phase_result is None:
                self.pick_phase_result = self.solve_local_target(
                    x,
                    y,
                    (float(self.container_travel_z_var.get()) + container_dz),
                    prefer_vertical=True,
                )

                if not self.pick_phase_result.get("ok", False):
                    self._pick_fail(
                        "lift-after-drop IK: "
                        + str(
                            self.pick_phase_result.get(
                                "message",
                                "failed",
                            )
                        )
                    )
                    return

            arrived, info = self._move_arm_step_to_result(
                self.pick_phase_result,
                max_step_deg=6.0,
                arrival_deg=3.0,
                stall_accept_deg=8.0,
            )

            self.pick_place_var.set(
                f"Pick & Place: LIFT AFTER DROP | "
                f"{info} | {age:.1f}s"
            )

            if arrived:
                self.pick_place_active = False
                self._set_pick_state(
                    "HOME",
                    "drop complete; safe HOME started"
                )
                self.go_home()

            elif age > max(1.0, float(self.motion_timeout_var.get())):
                self._pick_fail(
                    "lift-after-drop did not converge | " + info
                )

            return

        if state == "RELEASE":
            arrived, info = self._step_gripper_toward(
                float(self.gripper_open_var.get())
            )

            self.pick_place_var.set(
                f"Pick & Place: RELEASE | {info}"
            )

            if arrived or age > 4.0:
                self._reset_grip_close_state()
                self._set_pick_state(
                    "LIFT_AFTER_DROP",
                    "cube released / solving frozen lift target"
                )

            return

    # ------------------------------------------------------
    # Live status
    # ------------------------------------------------------

    def update_loop(self):
        try:
            state = self.worker.get_state_snapshot()

            if not state.get("connected", False):
                self.robot_var.set("Robot: offline")
            else:
                self.robot_var.set(
                    "Robot: connected | "
                    + (
                        "Torque ON"
                        if state.get("torque_enabled", False)
                        else "FREE MOVE"
                    )
                )

            if self.locked_cube is not None:
                self.update_corrected_info()

            self.update_cube_calibration_status()
            self.update_grasp_calibration_status()

            if self.home_sequence_active:
                self._update_home_sequence()

            if self.pick_place_active:
                self._update_pick_place()
            elif self.manual_grip_active:
                self._update_manual_grip()

            # Compact live wrist status from Module 9.
            if self.wrist_locked_angle_deg is None:
                stable = self.app_context.shared_data.get(
                    "wrist_stable_target"
                )

                tracking = self.app_context.shared_data.get(
                    "wrist_tracking_target"
                )

                if (
                    self.wrist_observation_active
                    and isinstance(stable, dict)
                ):
                    # Observation phase ends automatically as soon as Module 9
                    # has a stable angle. We freeze, but DO NOT rotate yet.
                    self.freeze_wrist_orientation()

                elif isinstance(stable, dict):
                    self.wrist_var.set(
                        "Wrist: READY | "
                        f"{stable.get('class', 'cube')} | "
                        f"angle={float(stable.get('angle_deg', 0.0)):.1f}° | "
                        f"conf={float(stable.get('confidence', 0.0)):.2f}"
                    )

                elif isinstance(tracking, dict):
                    self.wrist_var.set(
                        "Wrist: OBSERVING | "
                        f"{tracking.get('class', 'cube')} | "
                        f"angle≈{float(tracking.get('angle_deg', 0.0)):.1f}° | "
                        "waiting for stable"
                    )

        except Exception as exc:
            self.status_var.set(
                f"Cube test loop error: {exc}"
            )

        self.frame.after(
            self.UPDATE_MS,
            self.update_loop,
        )

    def get_frame(self):
        return self.frame

    def shutdown(self):
        pass

READY_FILE = PROJECT_DIR / "data" / "module8_ready_pose.json"

class ModulePanel:
    """
    Integrated cube calibration and pick-and-place module.

    Provides:
      - READY and HOME pose handling;
      - calibrated cube approach and grasp;
      - wrist-camera orientation alignment;
      - gripper resistance-based closing;
      - calibrated container placement;
      - direct transport between task poses.

    The calibrated low-level pick-and-place logic remains centralized
    in this module and is also used by the autonomous controller.
    """

    title = "Cube Pick + READY + Test Auto"

    UPDATE_MS = 100
    READY_STALL_S = 1.5
    READY_MOTION_EPS = 0.20
    READY_TIMEOUT_S = 18.0

    # READY from HOME:
    # 1) rotate base toward workspace FIRST,
    # 2) raise shoulder,
    # 3) shape wrist,
    # 4) then elbow,
    # 5) wrist roll,
    # 6) gripper LAST.
    #
    # No Cartesian lift is done before base rotation because HOME is already
    # a known safe pose and the camera support is close to that path.
    READY_PHASES = (
        "shoulder_pan.pos",
        "wrist_flex.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )

    READY_STEP = {
        "shoulder_lift.pos": 6.0,
        "elbow_flex.pos": 6.0,
        "wrist_flex.pos": 6.0,
        "wrist_roll.pos": 8.0,
        "shoulder_pan.pos": 6.0,
        "gripper.pos": 6.0,
    }

    def __init__(self, parent, app_context):
        self.app_context = app_context
        self.worker = app_context.robot_worker

        self.frame = ttk.Frame(parent, padding=6)

        # --------------------------------------------------
        # Integrated controls extend the core pick-and-place UI.
        # --------------------------------------------------
        self.ready_pose = self._load_ready_pose()

        # Direct transport poses are generated from the calibrated IK
        # targets at runtime, then commanded in one shot like READY.
        self._direct_pose_active = False
        self._direct_pose_target = None
        self._direct_pose_label = ""
        self._direct_pose_last_positions = None
        self._direct_pose_last_motion_at = None
        self._direct_pose_started_at = None

        # Optional external speed control for DIRECT transport phases.
        # 1.0 = nominal transport speed. Module 10 may set 0.20..1.00..
        self.transport_speed_scale = 1.0
        self._direct_pose_step_target = None
        self._direct_pose_final_target = None
        self._direct_pose_next_send_at = 0.0

        self.ready_active = False

        # Direct HOME uses the same successful principle as READY:
        # one saved motor pose -> one move_to_positions command.
        self.home_direct_active = False
        self.home_direct_target = None
        self.home_direct_last_positions = None
        self.home_direct_last_motion_at = None
        self.home_direct_started_at = None

        self.ready_stage = "IDLE"
        self.ready_phase_index = 0
        self.ready_last_positions = None
        self.ready_last_motion_at = None
        self.ready_started_at = None
        self.ready_vertical_waypoints = []
        self.ready_vertical_index = 0
        self.ready_vertical_result = None

        # Test-auto PRE-APPROACH deliberately uses the same direct command
        # as the manual GO ABOVE CUBE action. We wait until the
        # encoders physically settle, then start the normal pick-and-place sequence.
        self.pre_approach_active = False
        self.pre_approach_last_positions = None
        self.pre_approach_last_motion_at = None
        self.pre_approach_started_at = None

        self.direct_grasp_active = False
        self.direct_grasp_last_positions = None
        self.direct_grasp_last_motion_at = None
        self.direct_grasp_started_at = None

        self.test_auto_active = False
        self.test_auto_state = "IDLE"
        self.test_auto_started_at = None
        self.test_auto_processed = []
        self.test_auto_max_cubes_var = tk.IntVar(master=self.frame, value=20)

        self.transport_override_active = False
        self.transport_override_state = "IDLE"
        self.transport_container_result = None
        self.transport_lift_result = None
        self.transport_drop_result = None

        self.ready_status_var = tk.StringVar(
            master=self.frame,
            value="READY: saved" if self.ready_pose else "READY: not captured",
        )
        self.auto_status_var = tk.StringVar(
            master=self.frame,
            value="Test Auto: IDLE",
        )

        # --------------------------------------------------
        # Core calibrated pick-and-place engine.
        # --------------------------------------------------
        engine_host = ttk.Frame(self.frame)
        engine_host.pack(fill="both", expand=True)

        self.engine = BaseModulePanel(
            engine_host,
            app_context,
        )
        self.engine.get_frame().pack(fill="both", expand=True)

        # Keep original HOME behavior for all manual work.
        # During Test Auto only, the successful drop must go to READY,
        # not HOME. The core sequence calls self.go_home() after
        # LIFT_AFTER_DROP, so a small dispatcher changes only that
        # final automatic transition.
        self._engine_original_go_home = self.engine.go_home

        def _engine_home_dispatch():
            if self.test_auto_active:
                self.engine.home_sequence_active = False
                self.engine.home_sequence_phase = None
                self.engine.pick_place_active = False
                self.engine.pick_place_state = "COMPLETE"
                self.engine.pick_place_var.set(
                    "Pick & Place: COMPLETE | Test Auto -> READY"
                )
                return

            return self._engine_original_go_home()

        self.engine.go_home = _engine_home_dispatch


        # During integrated Test Auto, skip the incremental LIFT_AFTER_DROP
        # phase after RELEASE. The wrapper detects COMPLETE and sends the
        # direct READY pose.
        self._engine_original_set_pick_state = self.engine._set_pick_state

        def _engine_pick_state_dispatch(state, message):
            if (
                self.test_auto_active
                and str(state) == "LIFT_AFTER_DROP"
            ):
                self.engine.pick_place_active = False
                self.engine.pick_place_state = "COMPLETE"
                self.engine.pick_place_state_started_at = time.monotonic()
                self.engine.pick_place_var.set(
                    "Pick & Place: COMPLETE | RELEASE done -> direct READY"
                )
                return

            return self._engine_original_set_pick_state(
                state,
                message,
            )

        self.engine._set_pick_state = _engine_pick_state_dispatch

        # Integrate READY into section 1 (Robot / Home) and put the
        # test loop as a normal final section 9.
        self._build_integrated_ui()

        self.frame.after(
            self.UPDATE_MS,
            self._integrated_update_loop,
        )

        self.app_context.log(
            "Module 8 loaded | calibrated pick-and-place ready."
        )

    # ======================================================
    # Attribute bridge
    # ======================================================

    def __getattr__(self, name):
        engine = self.__dict__.get("engine")
        if engine is not None and hasattr(engine, name):
            return getattr(engine, name)
        raise AttributeError(name)

    # ======================================================
    # Integrated UI
    # ======================================================

    def _build_integrated_ui(self):
        engine_root = self.engine.get_frame()

        # --------------------------------------------------
        # 1) READY controls directly inside "1. Robot / Home"
        # --------------------------------------------------
        home_box = None

        def find_home_box(widget):
            nonlocal home_box
            for child in widget.winfo_children():
                try:
                    if isinstance(child, ttk.LabelFrame):
                        text = str(child.cget("text"))
                        if text.startswith("1. Robot / Home"):
                            home_box = child
                            return
                except Exception:
                    pass
                find_home_box(child)
                if home_box is not None:
                    return

        find_home_box(engine_root)

        if home_box is not None:
            ready_row = ttk.Frame(home_box)
            ready_row.pack(fill="x", pady=(5, 0))

            ttk.Separator(
                ready_row,
                orient="horizontal",
            ).pack(fill="x", pady=(0, 5))

            controls = ttk.Frame(ready_row)
            controls.pack(fill="x")

            ttk.Label(
                controls,
                text="Ready pose:",
                font=("Segoe UI", 9, "bold"),
            ).pack(side="left", padx=(0, 8))

            ttk.Button(
                controls,
                text="CAPTURE READY",
                command=self.capture_ready,
            ).pack(side="left", padx=(0, 5))

            ttk.Button(
                controls,
                text="GO READY",
                command=self.go_ready,
            ).pack(side="left")

            ttk.Label(
                ready_row,
                textvariable=self.ready_status_var,
                wraplength=1000,
            ).pack(anchor="w", pady=(4, 0))

        # --------------------------------------------------
        # 2) Section 9 goes into the SAME scrollable content container
        #    that already contains sections 1..8.
        #
        # Find section 8 recursively, then use its parent as the correct
        # scroll-content frame and pack section 9 after section 8.
        # --------------------------------------------------
        section8 = None

        def find_section8(widget):
            nonlocal section8
            for child in widget.winfo_children():
                try:
                    if isinstance(child, ttk.LabelFrame):
                        text = str(child.cget("text")).strip()
                        if text.startswith("8."):
                            section8 = child
                            return
                except Exception:
                    pass
                find_section8(child)
                if section8 is not None:
                    return

        find_section8(engine_root)

        if section8 is not None:
            content_parent = section8.master
        else:
            # Fallback: find the deepest parent containing the most numbered
            # LabelFrames; this keeps section 9 inside the original scroll area.
            best_parent = engine_root
            best_count = -1

            def inspect(widget):
                nonlocal best_parent, best_count
                count = 0
                for child in widget.winfo_children():
                    try:
                        if isinstance(child, ttk.LabelFrame):
                            text = str(child.cget("text")).strip()
                            if text and text[0].isdigit():
                                count += 1
                    except Exception:
                        pass
                if count > best_count:
                    best_count = count
                    best_parent = widget
                for child in widget.winfo_children():
                    inspect(child)

            inspect(engine_root)
            content_parent = best_parent

        auto_box = ttk.LabelFrame(
            content_parent,
            text="9. Test Auto Loop",
            padding=8,
        )
        auto_box.pack(fill="x", pady=(4, 8))

        ttk.Label(
            auto_box,
            text=(
                "Test loop: READY -> exact GO ABOVE -> exact START PICK + PLACE -> "
                "direct one-shot LIFT -> direct one-shot CONTAINER -> direct one-shot DROP -> RELEASE -> "
                "direct READY -> next cube. HOME is skipped between cubes."
            ),
            wraplength=1050,
        ).pack(anchor="w", pady=(0, 6))

        controls = ttk.Frame(auto_box)
        controls.pack(fill="x")

        ttk.Button(
            controls,
            text="START AUTO",
            command=self.start_test_auto,
        ).pack(side="left", padx=(0, 5))

        ttk.Button(
            controls,
            text="STOP AUTO",
            command=self.stop_test_auto,
        ).pack(side="left", padx=(0, 14))

        ttk.Label(
            controls,
            text="Max cubes:",
        ).pack(side="left")

        ttk.Entry(
            controls,
            textvariable=self.test_auto_max_cubes_var,
            width=5,
        ).pack(side="left", padx=(5, 0))

        ttk.Label(
            auto_box,
            textvariable=self.auto_status_var,
            wraplength=1050,
        ).pack(anchor="w", pady=(6, 0))

        # Keep the original Module-8 Status section as the final section,
        # after the new Test Auto Loop.
        status_box = None

        for child in content_parent.winfo_children():
            try:
                if isinstance(child, ttk.LabelFrame):
                    text = str(child.cget("text")).strip()
                    if text == "Status":
                        status_box = child
                        break
            except Exception:
                pass

        if status_box is not None:
            status_box.pack_forget()
            status_box.pack(fill="x", pady=(0, 8))

    # ======================================================
    # READY persistence
    # ======================================================

    @staticmethod
    def _load_ready_pose():
        try:
            if not READY_FILE.exists():
                return None

            data = json.loads(
                READY_FILE.read_text(encoding="utf-8")
            )
            pose = data.get("ready_pose")

            if not isinstance(pose, dict) or not pose:
                return None

            return {
                str(k): float(v)
                for k, v in pose.items()
            }

        except Exception:
            return None

    def _save_ready_pose(self):
        READY_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        READY_FILE.write_text(
            json.dumps(
                {"ready_pose": self.ready_pose},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def capture_ready(self):
        if self.test_auto_active:
            self.auto_status_var.set(
                "Test Auto: stop loop before CAPTURE READY"
            )
            return

        state = self.worker.get_state_snapshot()

        if not state.get("connected", False):
            self.ready_status_var.set(
                "READY: cannot capture | robot offline"
            )
            return

        positions = state.get("positions", {})

        required = (
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
        )

        missing = [
            joint
            for joint in required
            if joint not in positions
        ]

        if missing:
            self.ready_status_var.set(
                "READY: missing " + ", ".join(missing)
            )
            return

        pose = {
            joint: float(positions[joint])
            for joint in self.READY_PHASES
            if joint in positions
        }

        self.ready_pose = pose
        self._save_ready_pose()

        self.ready_status_var.set(
            "READY: captured | "
            + " | ".join(
                f"{joint.replace('.pos', '')}={value:.1f}"
                for joint, value in pose.items()
            )
        )

    # ======================================================
    # READY motion
    # ======================================================

    def go_ready(self):
        """
        READY is just a normal captured motor pose.
        No custom staged trajectory, no FK path, no joint choreography.
        Send the exact captured pose once and wait until the real encoders settle.
        """
        if not self.ready_pose:
            self.ready_status_var.set(
                "READY: not saved | use CAPTURE READY first"
            )
            return False

        state = self.worker.get_state_snapshot()

        if not state.get("connected", False):
            self.ready_status_var.set(
                "READY: robot offline"
            )
            return False

        self.ready_active = True
        self.ready_stage = "DIRECT"
        self.ready_last_positions = None
        self.ready_last_motion_at = time.monotonic()
        self.ready_started_at = time.monotonic()

        self.worker.send(
            commands.move_to_positions(
                dict(self.ready_pose)
            )
        )

        self.ready_status_var.set(
            "READY: direct captured pose command sent"
        )
        return True

    def _update_ready_motion(self):
        if not self.ready_active:
            return

        state = self.worker.get_state_snapshot()

        if not state.get("connected", False):
            self.ready_active = False
            self.ready_status_var.set(
                "READY: robot disconnected"
            )
            return

        positions = state.get("positions", {})
        now = time.monotonic()

        tracked = {
            joint: float(positions[joint])
            for joint in self.ready_pose
            if joint in positions
        }

        if not tracked:
            self.ready_active = False
            self.ready_status_var.set(
                "READY: encoder positions unavailable"
            )
            return

        if self.ready_last_positions is None:
            self.ready_last_positions = dict(tracked)
            self.ready_last_motion_at = now
        else:
            max_motion = max(
                abs(
                    value
                    - self.ready_last_positions.get(
                        joint,
                        value,
                    )
                )
                for joint, value in tracked.items()
            )

            if max_motion >= self.READY_MOTION_EPS:
                self.ready_last_positions = dict(tracked)
                self.ready_last_motion_at = now

        stalled = now - float(
            self.ready_last_motion_at or now
        )

        max_error = max(
            abs(
                float(self.ready_pose[joint])
                - float(positions[joint])
            )
            for joint in self.ready_pose
            if joint in positions
        )

        self.ready_status_var.set(
            f"READY: moving | residual={max_error:.1f}° | "
            f"stall={stalled:.1f}s"
        )

        if max_error <= 2.5 or stalled >= self.READY_STALL_S:
            self.ready_active = False
            self.ready_stage = "IDLE"
            self.ready_status_var.set(
                f"READY: reached | residual={max_error:.1f}°"
            )
            return

        age = now - float(
            self.ready_started_at or now
        )

        if age > self.READY_TIMEOUT_S:
            self.ready_active = False
            self.ready_stage = "IDLE"
            self.ready_status_var.set(
                "READY: timeout"
            )

            if self.test_auto_active:
                self._test_auto_fail(
                    "READY timeout"
                )

    # ======================================================
    # Direct pose motion helper
    # ======================================================

    def _start_direct_pose(self, pose, label):
        """
        Direct transport pose.

        READY remains exactly original/direct.
        For post-grasp transport only, Module 10 may reduce
        self.transport_speed_scale. We then interpolate ALL joints together
        along the same joint-space line; no one-joint choreography is added.
        """
        if not isinstance(pose, dict) or not pose:
            return False

        self._direct_pose_label = str(label)
        self._direct_pose_final_target = {
            str(k): float(v)
            for k, v in pose.items()
        }
        self._direct_pose_target = dict(self._direct_pose_final_target)
        self._direct_pose_last_positions = None
        self._direct_pose_last_motion_at = time.monotonic()
        self._direct_pose_started_at = time.monotonic()
        self._direct_pose_next_send_at = 0.0
        self._direct_pose_active = True

        # READY uses the captured pose directly and is never interpolated.
        is_ready = "READY" in self._direct_pose_label.upper()

        try:
            speed = float(self.transport_speed_scale)
        except Exception:
            speed = 1.0

        speed = max(0.20, min(1.0, speed))

        state = self.worker.get_state_snapshot()
        current = state.get("positions", {})

        if is_ready or speed >= 0.999:
            self._direct_pose_step_target = None
            self.worker.send(
                commands.move_to_positions(
                    dict(self._direct_pose_final_target)
                )
            )
            return True

        # At reduced speed send full-pose joint-space interpolation.
        self._direct_pose_step_target = {
            joint: float(current.get(joint, target))
            for joint, target in self._direct_pose_final_target.items()
        }

        self._send_next_direct_pose_step()
        return True

    def _send_next_direct_pose_step(self):
        if self._direct_pose_step_target is None:
            return

        try:
            speed = float(self.transport_speed_scale)
        except Exception:
            speed = 1.0

        speed = max(0.20, min(1.0, speed))

        # Approx. 2.5..12 deg per 100 ms, all joints simultaneously.
        max_step = 2.5 + 9.5 * speed

        finished = True
        next_pose = {}

        for joint, final in self._direct_pose_final_target.items():
            current_cmd = float(
                self._direct_pose_step_target.get(joint, final)
            )
            error = float(final) - current_cmd

            if abs(error) > max_step:
                finished = False
                current_cmd += math.copysign(max_step, error)
            else:
                current_cmd = float(final)

            next_pose[joint] = current_cmd

        self._direct_pose_step_target = dict(next_pose)

        self.worker.send(
            commands.move_to_positions(
                dict(next_pose)
            )
        )

        self._direct_pose_next_send_at = time.monotonic() + 0.10

        if finished:
            self._direct_pose_step_target = None

    def _update_direct_pose(self):
        if not getattr(self, "_direct_pose_active", False):
            return True

        now = time.monotonic()

        if (
            self._direct_pose_step_target is not None
            and now >= float(self._direct_pose_next_send_at or 0.0)
        ):
            self._send_next_direct_pose_step()

        state = self.worker.get_state_snapshot()

        if not state.get("connected", False):
            self._direct_pose_active = False
            return False

        positions = state.get("positions", {})

        tracked = {
            joint: float(positions[joint])
            for joint in self._direct_pose_final_target
            if joint in positions
        }

        if not tracked:
            self._direct_pose_active = False
            return False

        if self._direct_pose_last_positions is None:
            self._direct_pose_last_positions = dict(tracked)
            self._direct_pose_last_motion_at = now
        else:
            max_motion = max(
                abs(
                    value
                    - self._direct_pose_last_positions.get(joint, value)
                )
                for joint, value in tracked.items()
            )

            if max_motion >= 0.20:
                self._direct_pose_last_positions = dict(tracked)
                self._direct_pose_last_motion_at = now

        stalled = now - float(
            self._direct_pose_last_motion_at or now
        )

        max_error = max(
            abs(
                float(self._direct_pose_final_target[joint])
                - float(positions[joint])
            )
            for joint in self._direct_pose_final_target
            if joint in positions
        )

        # If interpolation is still feeding commands, do not finish early.
        if self._direct_pose_step_target is None:
            if max_error <= 2.5 or stalled >= 1.5:
                self._direct_pose_active = False
                return True

        age = now - float(
            self._direct_pose_started_at or now
        )
        if age > 30.0:
            self._direct_pose_active = False
            return False

        return None

    # ======================================================
    # Direct HOME + hand-safety pause API
    # ======================================================

    def go_home_direct(self):
        """
        HOME exactly like the successful READY motion:
        send the saved HOME motor pose once, then only observe encoders.
        """
        state = self.worker.get_state_snapshot()

        if not state.get("connected", False):
            self.ready_status_var.set(
                "HOME: robot offline"
            )
            return False

        try:
            home = self.engine.get_home_pose()
        except Exception:
            home = None

        if not isinstance(home, dict) or not home:
            self.ready_status_var.set(
                "HOME: no saved HOME pose"
            )
            return False

        # Cancel the old staged HOME if it was running.
        try:
            self.engine.home_sequence_active = False
        except Exception:
            pass

        self.home_direct_target = {
            str(k): float(v)
            for k, v in home.items()
        }
        self.home_direct_active = True
        self.home_direct_last_positions = None
        self.home_direct_last_motion_at = time.monotonic()
        self.home_direct_started_at = time.monotonic()

        self.worker.send(
            commands.move_to_positions(
                dict(self.home_direct_target)
            )
        )

        self.ready_status_var.set(
            "HOME: direct captured pose command sent"
        )
        return True

    def _update_home_direct(self):
        if not self.home_direct_active:
            return

        state = self.worker.get_state_snapshot()

        if not state.get("connected", False):
            self.home_direct_active = False
            self.ready_status_var.set(
                "HOME: robot disconnected"
            )
            return

        positions = state.get("positions", {})
        now = time.monotonic()

        tracked = {
            joint: float(positions[joint])
            for joint in self.home_direct_target
            if joint in positions
        }

        if not tracked:
            self.home_direct_active = False
            self.ready_status_var.set(
                "HOME: encoder positions unavailable"
            )
            return

        if self.home_direct_last_positions is None:
            self.home_direct_last_positions = dict(tracked)
            self.home_direct_last_motion_at = now
        else:
            max_motion = max(
                abs(
                    value
                    - self.home_direct_last_positions.get(
                        joint,
                        value,
                    )
                )
                for joint, value in tracked.items()
            )

            if max_motion >= self.READY_MOTION_EPS:
                self.home_direct_last_positions = dict(tracked)
                self.home_direct_last_motion_at = now

        stalled = now - float(
            self.home_direct_last_motion_at or now
        )

        max_error = max(
            abs(
                float(self.home_direct_target[joint])
                - float(positions[joint])
            )
            for joint in self.home_direct_target
            if joint in positions
        )

        self.ready_status_var.set(
            f"HOME: moving | residual={max_error:.1f}° | "
            f"stall={stalled:.1f}s"
        )

        if max_error <= 2.5 or stalled >= self.READY_STALL_S:
            self.home_direct_active = False
            self.ready_status_var.set(
                f"HOME: reached | residual={max_error:.1f}°"
            )
            return

        if (
            now - float(self.home_direct_started_at or now)
            > self.READY_TIMEOUT_S
        ):
            self.home_direct_active = False
            self.ready_status_var.set(
                "HOME: timeout"
            )

    def pause_motion_hold(self):
        """
        Immediate cooperative pause for Module 10 hand safety.
        Stops wrapper-owned motion and holds the current measured pose.
        A later go_home_direct() restarts HOME from wherever the arm stopped.
        """
        self.ready_active = False
        self.home_direct_active = False
        self.pre_approach_active = False
        self.transport_override_active = False
        self._direct_pose_active = False

        try:
            self.engine.pick_place_active = False
            self.engine.home_sequence_active = False
            self.engine.manual_grip_active = False
        except Exception:
            pass

        try:
            self.worker.send(
                commands.hold_current()
            )
        except Exception:
            # Fallback: hold exact current measured positions.
            try:
                state = self.worker.get_state_snapshot()
                positions = state.get("positions", {})
                targets = {
                    str(k): float(v)
                    for k, v in positions.items()
                    if str(k).endswith(".pos")
                }
                if targets:
                    self.worker.send(
                        commands.move_to_positions(targets)
                    )
            except Exception:
                pass

        self.auto_status_var.set(
            "Test Auto: PAUSED / HOLD"
        )
        return True

    # ======================================================
    # Test auto target helpers
    # ======================================================

    def _live_cube_candidates(self):
        detections = self.engine.get_cube_detections()
        candidates = []

        for item in detections:
            try:
                cls = str(
                    item.get(
                        "class",
                        item.get("class_name", ""),
                    )
                )

                world = item.get("world")
                if (
                    not cls.startswith("cube_")
                    or not isinstance(world, (list, tuple))
                    or len(world) < 2
                ):
                    continue

                x = float(world[0])
                y = float(world[1])
                conf = float(
                    item.get(
                        "confidence",
                        item.get("conf", 0.0),
                    )
                )

            except Exception:
                continue

            # Session memory: do not immediately re-pick the same original
            # cube coordinate if camera/container filtering is one frame late.
            already_done = False

            for done in self.test_auto_processed:
                if (
                    done["class"] == cls
                    and math.hypot(
                        x - done["x"],
                        y - done["y"],
                    ) <= 30.0
                ):
                    already_done = True
                    break

            if already_done:
                continue

            candidates.append(
                {
                    "class": cls,
                    "x": x,
                    "y": y,
                    "confidence": conf,
                }
            )

        return candidates

    def _lock_test_cube(self, cube):
        self.engine.locked_cube = {
            "class": str(cube["class"]),
            "confidence": float(cube["confidence"]),
            "camera_world_x": float(cube["x"]),
            "camera_world_y": float(cube["y"]),
        }

        self.engine.correction_x_var.set(0.0)
        self.engine.correction_y_var.set(0.0)
        self.engine.update_corrected_info()
        self.engine.clear_wrist_orientation()

        self.app_context.shared_data[
            "wrist_requested_class"
        ] = str(cube["class"])

    # ======================================================
    # Simple direct transport
    # ======================================================

    def _solve_direct_lift(self):
        """
        One-shot lift using the SAME calibrated grasp X/Y as Module 8.
        solve_grasp_target() returns local_x/local_y directly.
        """
        grasp = self.engine.solve_grasp_target(
            include_manual_trim=False
        )
        if not grasp.get("ok", False):
            return grasp

        if "local_x" not in grasp or "local_y" not in grasp:
            return {
                "ok": False,
                "message": "grasp solver did not return local_x/local_y",
            }

        x = float(grasp["local_x"])
        y = float(grasp["local_y"])
        z = max(
            70.0,
            float(self.engine.lift_clear_z_var.get()),
        )

        return self.engine.solve_local_target(
            x,
            y,
            z,
            prefer_vertical=True,
        )

    def _solve_direct_container(self):
        """
        One-shot move above the mapped container using the calibrated
        container coordinates from the core pick-and-place engine.
        """
        container = self.engine.get_container_local_for_locked_cube()

        if container is None:
            return {
                "ok": False,
                "message": "mapped container unavailable",
            }

        marker_id, container_x, container_y = container

        raw_container = self.engine._get_container_local_by_marker(
            marker_id,
            apply_calibration=False,
        )

        container_dz = (
            0.0
            if raw_container is None
            else self.engine.get_container_z_correction(
                raw_container[0],
                raw_container[1],
            )
        )

        result = self.engine.solve_local_target(
            float(container_x),
            float(container_y),
            float(self.engine.container_travel_z_var.get())
            + float(container_dz),
            prefer_vertical=True,
        )

        if result.get("ok", False):
            result["_container_id"] = marker_id

        return result

    def _solve_direct_drop(self):
        """
        Direct drop pose using the same calibrated container XY and the
        configured DROP Z. Also include the live manual container Z trim,
        matching GO CONTAINER CAL behavior.
        """
        container = self.engine.get_container_local_for_locked_cube()
        if container is None:
            return {
                "ok": False,
                "message": "mapped container unavailable",
            }

        marker_id, container_x, container_y = container

        raw_container = self.engine._get_container_local_by_marker(
            marker_id,
            apply_calibration=False,
        )

        container_dz = (
            0.0
            if raw_container is None
            else self.engine.get_container_z_correction(
                raw_container[0],
                raw_container[1],
            )
        )

        # Preserve any currently verified manual Z trim as well.
        trim_z = 0.0
        try:
            trim_z = float(self.engine.container_trim_z_var.get())
        except Exception:
            pass

        result = self.engine.solve_local_target(
            float(container_x),
            float(container_y),
            float(self.engine.drop_z_var.get())
            + float(container_dz)
            + float(trim_z),
            prefer_vertical=True,
        )

        if result.get("ok", False):
            result["_container_id"] = marker_id
            result["_drop_z"] = (
                float(self.engine.drop_z_var.get())
                + float(container_dz)
                + float(trim_z)
            )

        return result

    def _start_direct_drop(self):
        result = self._solve_direct_drop()
        if not result.get("ok", False):
            return False, str(
                result.get("message", "drop IK failed")
            )

        self.transport_drop_result = result
        ok = self._start_direct_pose(
            result["motors"],
            "DIRECT DROP",
        )
        return ok, "direct drop"

    def _start_direct_lift(self):
        result = self._solve_direct_lift()
        if not result.get("ok", False):
            return False, str(result.get("message", "lift IK failed"))

        self.transport_lift_result = result
        ok = self._start_direct_pose(
            result["motors"],
            "DIRECT LIFT",
        )
        return ok, "direct lift"

    def _start_direct_container(self):
        result = self._solve_direct_container()
        if not result.get("ok", False):
            return False, str(result.get("message", "container IK failed"))

        self.transport_container_result = result
        ok = self._start_direct_pose(
            result["motors"],
            "DIRECT CONTAINER",
        )
        return ok, "direct container"

    # ======================================================
    # Exact manual GO ABOVE CUBE for auto
    # ======================================================

    def _start_manual_go_above(self):
        """
        Call the exact same public method as the working GUI button:
        GO ABOVE CUBE -> engine.go_to_cube().
        """
        try:
            self.engine.go_to_cube()
        except Exception as exc:
            self._test_auto_fail(
                "GO ABOVE CUBE failed: " + str(exc)
            )
            return False

        self.pre_approach_active = True
        self.pre_approach_last_positions = None
        self.pre_approach_last_motion_at = time.monotonic()
        self.pre_approach_started_at = time.monotonic()

        self.auto_status_var.set(
            "Test Auto: exact manual GO ABOVE CUBE"
        )
        return True

    def _update_manual_go_above(self):
        if not self.pre_approach_active:
            return True

        state = self.worker.get_state_snapshot()

        if not state.get("connected", False):
            self._test_auto_fail(
                "robot disconnected during GO ABOVE"
            )
            return False

        positions = state.get("positions", {})
        now = time.monotonic()

        tracked = {
            joint: float(positions[joint])
            for joint in (
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow_flex.pos",
                "wrist_flex.pos",
            )
            if joint in positions
        }

        if not tracked:
            self._test_auto_fail(
                "no encoder positions during GO ABOVE"
            )
            return False

        if self.pre_approach_last_positions is None:
            self.pre_approach_last_positions = dict(tracked)
            self.pre_approach_last_motion_at = now
        else:
            max_motion = max(
                abs(
                    value
                    - self.pre_approach_last_positions.get(
                        joint,
                        value,
                    )
                )
                for joint, value in tracked.items()
            )

            if max_motion >= 0.20:
                self.pre_approach_last_positions = dict(tracked)
                self.pre_approach_last_motion_at = now

        stalled = now - float(
            self.pre_approach_last_motion_at or now
        )

        self.auto_status_var.set(
            f"Test Auto: GO ABOVE | stall={stalled:.1f}s"
        )

        if stalled >= 1.5:
            self.pre_approach_active = False
            return True

        age = now - float(
            self.pre_approach_started_at or now
        )

        if age > max(
            6.0,
            float(self.engine.motion_timeout_var.get()),
        ):
            self._test_auto_fail(
                "GO ABOVE CUBE did not settle"
            )
            return False

        return None

    # ======================================================
    # Test auto
    # ======================================================

    def start_test_auto(self):
        if self.test_auto_active:
            return

        if not self.ready_pose:
            self.auto_status_var.set(
                "Test Auto: CAPTURE READY first"
            )
            return

        state = self.worker.get_state_snapshot()

        if not state.get("connected", False):
            self.auto_status_var.set(
                "Test Auto: robot offline"
            )
            return

        if not state.get("torque_enabled", False):
            self.auto_status_var.set(
                "Test Auto: torque OFF"
            )
            return

        self.engine.stop_pick_place()
        self.test_auto_processed = []
        self.test_auto_active = True
        self.test_auto_state = "GO_READY"
        self.test_auto_started_at = time.monotonic()

        if not self.go_ready():
            self._test_auto_fail(
                "READY could not start"
            )
            return

        self.auto_status_var.set(
            "Test Auto: START -> READY"
        )

    def stop_test_auto(self):
        self.test_auto_active = False
        self.test_auto_state = "IDLE"

        try:
            self.engine.stop_pick_place()
        except Exception:
            pass

        self.ready_active = False
        self.pre_approach_active = False
        self.transport_override_active = False
        self.transport_override_state = "IDLE"
        self._direct_pose_active = False
        self.direct_grasp_active = False

        try:
            self.worker.send(
                commands.hold_current()
            )
        except Exception:
            pass

        self.auto_status_var.set(
            "Test Auto: STOPPED"
        )

    def _test_auto_fail(self, message):
        self.test_auto_active = False
        self.test_auto_state = "ERROR"
        self.ready_active = False
        self.pre_approach_active = False
        self.transport_override_active = False
        self.transport_override_state = "IDLE"
        self._direct_pose_active = False
        self.direct_grasp_active = False

        try:
            self.engine.stop_pick_place()
        except Exception:
            pass

        try:
            self.worker.send(
                commands.hold_current()
            )
        except Exception:
            pass

        self.auto_status_var.set(
            "Test Auto ERROR: " + str(message)
        )

    def _update_test_auto(self):
        if not self.test_auto_active:
            return

        # Hand safety is intentionally NOT handled here.
        # Module 10 is the single owner of:
        # HAND -> HOLD -> clear timer -> HOME -> RESCAN -> continue.
        # Keeping a second hand state machine in Module 8 caused false
        # Test Auto ERRORs during normal operator interaction.

        state = self.test_auto_state

        if state == "GO_READY":
            if self.ready_active:
                self.auto_status_var.set(
                    "Test Auto: waiting READY"
                )
                return

            self.test_auto_state = "SELECT"
            self.auto_status_var.set(
                "Test Auto: READY -> select cube"
            )
            return

        if state == "SELECT":
            limit = max(
                1,
                int(self.test_auto_max_cubes_var.get()),
            )

            if len(self.test_auto_processed) >= limit:
                self.test_auto_active = False
                self.test_auto_state = "DONE"
                self.auto_status_var.set(
                    f"Test Auto: DONE | limit {limit}"
                )
                return

            candidates = self._live_cube_candidates()

            if not candidates:
                self.test_auto_active = False
                self.test_auto_state = "DONE"
                self.auto_status_var.set(
                    f"Test Auto: DONE | no cubes | "
                    f"processed={len(self.test_auto_processed)}"
                )
                return

            cube = max(
                candidates,
                key=lambda item: item["confidence"],
            )

            self._lock_test_cube(cube)
            self.test_auto_current = dict(cube)

            # EXACT manual sequence step 1.
            if not self._start_manual_go_above():
                return

            self.test_auto_state = "GO_ABOVE"
            return

        if state == "GO_ABOVE":
            result = self._update_manual_go_above()

            if result is None:
                return
            if result is False:
                return

            # EXACT manual sequence step 2:
            # same START PICK + PLACE button/method.
            self.engine.start_pick_place()

            if not bool(self.engine.pick_place_active):
                self._test_auto_fail(
                    "START PICK + PLACE was rejected after GO ABOVE"
                )
                return

            self.test_auto_state = "PICK"
            self.auto_status_var.set(
                "Test Auto: GO ABOVE complete -> START PICK + PLACE"
            )
            return

        if state == "PICK":
            if self.transport_override_active:
                return

            if bool(self.engine.pick_place_active):
                self.auto_status_var.set(
                    "Test Auto: " + self.engine.pick_place_var.get()
                )
                return

            pick_state = str(
                self.engine.pick_place_state
            )

            if pick_state == "ERROR":
                self._test_auto_fail(
                    self.engine.pick_place_var.get()
                )
                return

            # Test-auto final HOME is intercepted by the wrapper and becomes
            # COMPLETE. Therefore the next motion is READY, never HOME.
            if pick_state == "COMPLETE":
                self.test_auto_processed.append(
                    dict(self.test_auto_current)
                )

                try:
                    self.engine.clear_lock()
                except Exception:
                    pass

                if not self.go_ready():
                    self._test_auto_fail(
                        "could not start READY after drop"
                    )
                    return

                self.test_auto_state = "GO_READY"
                self.auto_status_var.set(
                    f"Test Auto: cube done "
                    f"{len(self.test_auto_processed)} | "
                    "drop -> READY -> next cube"
                )
                return

            self._test_auto_fail(
                f"unexpected pick state: {pick_state}"
            )
            return

    # ======================================================
    # Transport override updater
    # ======================================================

    def _update_transport_override(self):
        if not self.test_auto_active:
            return

        # Catch the moment the normal engine has finished gripping and wants
        # to begin LIFT. Suspend its incremental transport before it can drag.
        if (
            not self.transport_override_active
            and bool(self.engine.pick_place_active)
            and str(self.engine.pick_place_state) == "LIFT"
        ):
            self.engine.pick_place_active = False

            ok, info = self._start_direct_lift()
            if not ok:
                self._test_auto_fail("direct lift failed: " + info)
                return

            self.transport_override_active = True
            self.transport_override_state = "LIFT"
            self.auto_status_var.set(
                "Test Auto: DIRECT LIFT"
            )
            return

        if not self.transport_override_active:
            return

        if self.transport_override_state == "LIFT":
            result = self._update_direct_pose()
            if result is None:
                self.auto_status_var.set(
                    "Test Auto: DIRECT LIFT"
                )
                return
            if result is False:
                self._test_auto_fail("direct lift timeout")
                return

            ok, info = self._start_direct_container()
            if not ok:
                self._test_auto_fail(
                    "direct container failed: " + info
                )
                return

            self.transport_override_state = "CONTAINER"
            self.auto_status_var.set(
                "Test Auto: DIRECT CONTAINER"
            )
            return

        if self.transport_override_state == "CONTAINER":
            result = self._update_direct_pose()
            if result is None:
                self.auto_status_var.set(
                    "Test Auto: DIRECT CONTAINER"
                )
                return
            if result is False:
                self._test_auto_fail("direct container timeout")
                return

            ok, info = self._start_direct_drop()
            if not ok:
                self._test_auto_fail(
                    "direct drop failed: " + info
                )
                return

            self.transport_override_state = "DROP"
            self.auto_status_var.set(
                "Test Auto: DIRECT DROP"
            )
            return

        if self.transport_override_state == "DROP":
            result = self._update_direct_pose()
            if result is None:
                drop_z = None
                if isinstance(self.transport_drop_result, dict):
                    drop_z = self.transport_drop_result.get("_drop_z")

                if drop_z is None:
                    self.auto_status_var.set(
                        "Test Auto: DIRECT DROP"
                    )
                else:
                    self.auto_status_var.set(
                        f"Test Auto: DIRECT DROP | Z={float(drop_z):.1f} mm"
                    )
                return

            if result is False:
                self._test_auto_fail("direct drop timeout")
                return

            # Only RELEASE remains in the original state machine.
            self.transport_override_active = False
            self.transport_override_state = "IDLE"
            self.engine.pick_place_active = True
            self.engine._set_pick_state(
                "RELEASE",
                "direct drop pose reached"
            )
            self.auto_status_var.set(
                "Test Auto: direct drop reached -> RELEASE"
            )
            return

    # ======================================================
    # Integrated update
    # ======================================================

    def _integrated_update_loop(self):
        try:
            if self.ready_active:
                self._update_ready_motion()

            if self.home_direct_active:
                self._update_home_direct()

            self._update_transport_override()
            self._update_test_auto()

        except Exception as exc:
            self._test_auto_fail(
                f"loop exception: {exc}"
            )

        self.frame.after(
            self.UPDATE_MS,
            self._integrated_update_loop,
        )

    # ======================================================
    # Module API / forwarding
    # ======================================================

    def get_frame(self):
        return self.frame

    # Explicit forwards useful for a future lightweight Module 10.
    def start_pick_place(self):
        return self.engine.start_pick_place()

    def stop_pick_place(self):
        return self.engine.stop_pick_place()

    def go_home(self):
        return self.engine.go_home()

    def get_home_pose(self):
        return self.engine.get_home_pose()

    def clear_lock(self):
        return self.engine.clear_lock()

    @property
    def locked_cube(self):
        return self.engine.locked_cube

    @locked_cube.setter
    def locked_cube(self, value):
        self.engine.locked_cube = value

    @property
    def pick_place_active(self):
        return self.engine.pick_place_active

    @property
    def pick_place_state(self):
        return self.engine.pick_place_state

    @property
    def home_sequence_active(self):
        return self.engine.home_sequence_active

    @property
    def pick_place_var(self):
        return self.engine.pick_place_var

    def shutdown(self):
        self.test_auto_active = False
        self.ready_active = False

        try:
            self.engine.go_home = self._engine_original_go_home
        except Exception:
            pass

        try:
            self.engine._set_pick_state = self._engine_original_set_pick_state
        except Exception:
            pass

        try:
            self.engine.shutdown()
        except Exception:
            pass
