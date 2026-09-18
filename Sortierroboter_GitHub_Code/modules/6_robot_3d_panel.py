import importlib.util
import json
import math
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from core import commands
from robotics.forward_kinematics import (
    JointAngles,
    RobotGeometry,
    forward_kinematics,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODULES_DIR = PROJECT_DIR / "modules"
IK_CALIBRATION_FILE = PROJECT_DIR / "data" / "ik_calibration.json"
IK_TARGET_CALIBRATION_FILE = PROJECT_DIR / "data" / "ik_target_calibration.json"
OBJECT_GEOMETRY_FILE = PROJECT_DIR / "data" / "object_geometry.json"

JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
]

DEFAULT_PORT = "COM6"
DEFAULT_ROBOT_ID = "main"


def normalize_angle_deg(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


def rounded_tuple(values, digits=2):
    return tuple(round(float(v), digits) for v in values)


class ModulePanel:
    title = "SO-101 Control Center"

    STATUS_POLL_MS = 100
    MIN_REDRAW_MS = 250

    def __init__(self, parent, app_context):
        self.app_context = app_context
        self.worker = app_context.robot_worker
        self.frame = ttk.Frame(parent, padding=8)

        self.old_calibration = self._load_json(
            IK_CALIBRATION_FILE,
            {"geometry": {}, "poses": {}},
        )
        self.target_calibration = self._load_json(
            IK_TARGET_CALIBRATION_FILE,
            {},
        )

        geom = self.target_calibration.get("geometry", {})
        old_geom = self.old_calibration.get("geometry", {})
        mapping = self.target_calibration.get("model_mapping", {})

        # --------------------------------------------------
        # Robot connection
        # --------------------------------------------------

        self.port_var = tk.StringVar(value=DEFAULT_PORT)
        self.robot_id_var = tk.StringVar(value=DEFAULT_ROBOT_ID)

        self.robot_status_var = tk.StringVar(value="Robot: offline")
        self.torque_status_var = tk.StringVar(value="Torque: —")
        self.status_var = tk.StringVar(value="Ready")

        # --------------------------------------------------
        # Geometry / mapping
        # --------------------------------------------------

        self.l1 = float(geom.get("L1", old_geom.get("L1", 116.0)))
        self.l2 = float(geom.get("L2", old_geom.get("L2", 135.0)))
        self.l3 = float(geom.get("L3", old_geom.get("L3", 165.0)))
        self.shoulder_height = float(geom.get("shoulder_height", 120.0))
        self.base_to_shoulder = float(
            geom.get("base_to_shoulder_offset", 30.0)
        )

        self.signs = {
            joint: float(mapping.get(joint, {}).get("sign", default))
            for joint, default in {
                "shoulder_pan.pos": 1.0,
                "shoulder_lift.pos": -1.0,
                "elbow_flex.pos": -1.0,
                "wrist_flex.pos": -1.0,
            }.items()
        }

        self.offsets = {
            joint: float(mapping.get(joint, {}).get("offset", default))
            for joint, default in {
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 150.0,
                "elbow_flex.pos": 210.0,
                "wrist_flex.pos": -60.0,
            }.items()
        }

        # --------------------------------------------------
        # Main virtual robot source
        # --------------------------------------------------

        self.mode_var = tk.StringVar(value="Robot / IK")

        self.manual_base_var = tk.DoubleVar(value=0.0)
        self.manual_shoulder_var = tk.DoubleVar(value=20.0)
        self.manual_elbow_var = tk.DoubleVar(value=-60.0)
        self.manual_wrist_var = tk.DoubleVar(value=-50.0)

        self.show_encoder_debug_var = tk.BooleanVar(value=False)

        self.target_source_var = tk.StringVar(
            value=str(
                self.app_context.shared_data.get("ik_target_source", "ID8")
            )
        )

        # Last valid ID8 position for 3D visualization.
        # If the robot arm occludes the marker, keep showing the last seen target.
        self.last_id8_target_world = None

        # --------------------------------------------------
        # Scene display / object geometry
        # --------------------------------------------------

        object_geometry = self._load_json(
            OBJECT_GEOMETRY_FILE,
            {
                "cube_size_mm": 19.0,
                "container_size_x_mm": 120.0,
                "container_size_y_mm": 120.0,
                "container_height_mm": 60.0,
            },
        )

        self.show_workspace_var = tk.BooleanVar(value=True)
        self.show_target_var = tk.BooleanVar(value=True)
        self.show_containers_var = tk.BooleanVar(value=True)
        self.show_cubes_var = tk.BooleanVar(value=True)
        self.show_base_heading_var = tk.BooleanVar(value=True)
        self.show_joint_labels_var = tk.BooleanVar(value=False)
        self.show_tcp_var = tk.BooleanVar(value=True)
        self.show_tcp_target_error_var = tk.BooleanVar(value=True)
        self.grasp_tcp_length_var = tk.DoubleVar(value=160.0)
        self.view_focus_var = tk.StringVar(value="Robot + Target")
        self.view_zoom_var = tk.DoubleVar(value=600.0)
        
        self.cube_size_var = tk.DoubleVar(
            value=float(object_geometry.get("cube_size_mm", 19.0))
        )
        self.container_size_x_var = tk.DoubleVar(
            value=float(object_geometry.get("container_size_x_mm", 120.0))
        )
        self.container_size_y_var = tk.DoubleVar(
            value=float(object_geometry.get("container_size_y_mm", 120.0))
        )
        self.container_height_var = tk.DoubleVar(
            value=float(object_geometry.get("container_height_mm", 60.0))
        )

        self.publish_object_geometry()
        self.app_context.shared_data["grasp_tcp_length_mm"] = float(
            self.grasp_tcp_length_var.get()
        )

        # --------------------------------------------------
        # Info
        # --------------------------------------------------

        self.ik_info_var = tk.StringVar(value="Robot / IK: —")
        self.encoder_info_var = tk.StringVar(value="Encoder debug: —")
        self.base_info_var = tk.StringVar(value="Base: —")
        self.target_info_var = tk.StringVar(value="ID8: —")
        self.gripper_info_var = tk.StringVar(value="Gripper: —")
        self.tcp_info_var = tk.StringVar(value="Grasp TCP: —")
        self.object_info_var = tk.StringVar(value="Objects: —")
        self.calibration_info_var = tk.StringVar(value="Calibration: —")

        # --------------------------------------------------
        # Modules
        # --------------------------------------------------

        self.available_modules = {}
        self.module_select_var = tk.StringVar()
        self.open_module_windows = {}

        # --------------------------------------------------
        # Rendering state
        # --------------------------------------------------

        self.last_model_revision = None
        self.last_scene_signature = None
        self.last_draw_monotonic = 0.0
        self.scene_dirty = True
        self.pending_draw_after_id = None

        self.user_view_locked = False
        self._mouse_interacting = False
        self._saved_view = None

        self.build_ui()
        self.refresh_module_list()

        # First scene draw should happen as soon as Tk is ready.
        self.frame.after(50, self.request_scene_update)
        self.frame.after(self.STATUS_POLL_MS, self.update_loop)

    # ======================================================
    # Loading / calibration sync
    # ======================================================

    @staticmethod
    def _load_json(path, default):
        path = Path(path)
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def sync_model_calibration(self):
        data = self.app_context.shared_data.get("robot_model_calibration")
        if not isinstance(data, dict):
            return False

        revision = data.get("revision")
        if revision is not None and revision == self.last_model_revision:
            return False

        changed = False

        geom = data.get("geometry", {})
        mapping = data.get("mapping", {})

        def update_attr(name, key):
            nonlocal changed
            if key not in geom:
                return
            value = float(geom[key])
            if abs(float(getattr(self, name)) - value) > 1e-9:
                setattr(self, name, value)
                changed = True

        update_attr("l1", "L1")
        update_attr("l2", "L2")
        update_attr("l3", "L3")
        update_attr("shoulder_height", "shoulder_height")
        update_attr("base_to_shoulder", "base_to_shoulder_offset")

        for joint in JOINTS:
            item = mapping.get(joint)
            if not isinstance(item, dict):
                continue

            if "sign" in item:
                value = float(item["sign"])
                if self.signs[joint] != value:
                    self.signs[joint] = value
                    changed = True

            if "offset" in item:
                value = float(item["offset"])
                if self.offsets[joint] != value:
                    self.offsets[joint] = value
                    changed = True

        self.last_model_revision = revision
        return changed

    # ======================================================
    # UI
    # ======================================================

    def build_ui(self):
        # Header
        header = ttk.Frame(self.frame)
        header.pack(fill="x", pady=(0, 7))

        ttk.Label(
            header,
            text="SO-101 CONTROL CENTER",
            font=("Segoe UI", 18, "bold"),
        ).pack(side="left")

        ttk.Label(
            header,
            textvariable=self.status_var,
        ).pack(side="right")

        # Robot toolbar
        robot_bar = ttk.LabelFrame(self.frame, text="Robot", padding=7)
        robot_bar.pack(fill="x", pady=(0, 7))

        ttk.Label(robot_bar, text="Port").pack(side="left")
        ttk.Entry(robot_bar, textvariable=self.port_var, width=8).pack(
            side="left", padx=(4, 10)
        )

        ttk.Label(robot_bar, text="ID").pack(side="left")
        ttk.Entry(robot_bar, textvariable=self.robot_id_var, width=8).pack(
            side="left", padx=(4, 10)
        )

        ttk.Button(
            robot_bar,
            text="Connect",
            command=self.connect_robot,
        ).pack(side="left", padx=2)

        ttk.Button(
            robot_bar,
            text="Disconnect",
            command=self.disconnect_robot,
        ).pack(side="left", padx=2)

        ttk.Button(
            robot_bar,
            text="Free Move",
            command=self.free_move,
        ).pack(side="left", padx=(12, 2))

        ttk.Button(
            robot_bar,
            text="Hold",
            command=self.hold_robot,
        ).pack(side="left", padx=2)

        ttk.Label(
            robot_bar,
            textvariable=self.robot_status_var,
        ).pack(side="right", padx=5)

        ttk.Label(
            robot_bar,
            textvariable=self.torque_status_var,
        ).pack(side="right", padx=8)

        # Module launcher
        launcher = ttk.LabelFrame(
            self.frame,
            text="Modules",
            padding=7,
        )
        launcher.pack(fill="x", pady=(0, 7))

        self.module_combo = ttk.Combobox(
            launcher,
            textvariable=self.module_select_var,
            state="readonly",
            width=34,
        )
        self.module_combo.pack(side="left", padx=(0, 5))

        ttk.Button(
            launcher,
            text="Open",
            command=self.open_selected_module,
        ).pack(side="left", padx=2)

        ttk.Button(
            launcher,
            text="Refresh",
            command=self.refresh_module_list,
        ).pack(side="left", padx=2)

        # Main body
        body = ttk.PanedWindow(self.frame, orient="horizontal")
        body.pack(fill="both", expand=True)

        plot_frame = ttk.Frame(body)
        side_outer = ttk.Frame(body)

        body.add(plot_frame, weight=5)
        body.add(side_outer, weight=2)

        self.figure = Figure(figsize=(11, 8.5), dpi=100)
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.canvas.mpl_connect("button_press_event", self._on_3d_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_3d_mouse_release)
        self.canvas.mpl_connect("scroll_event", self._on_3d_scroll)

        self._build_sidebar(side_outer)

    def _build_sidebar(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0, width=390)
        scrollbar = ttk.Scrollbar(
            parent,
            orient="vertical",
            command=canvas.yview,
        )
        controls = ttk.Frame(canvas, padding=8)

        window_id = canvas.create_window(
            (0, 0),
            window=controls,
            anchor="nw",
        )

        controls.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(window_id, width=e.width),
        )

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Source
        source = ttk.LabelFrame(controls, text="Main robot", padding=8)
        source.pack(fill="x", pady=(0, 8))

        for text in ("Robot / IK", "Manual"):
            ttk.Radiobutton(
                source,
                text=text,
                value=text,
                variable=self.mode_var,
                command=self.request_scene_update,
            ).pack(anchor="w")

        ttk.Checkbutton(
            source,
            text="Show encoder reconstruction",
            variable=self.show_encoder_debug_var,
            command=self.request_scene_update,
        ).pack(anchor="w", pady=(5, 0))

        target_source = ttk.LabelFrame(
            controls,
            text="IK Target Source",
            padding=8,
        )
        target_source.pack(fill="x", pady=(0, 8))

        ttk.Radiobutton(
            target_source,
            text="ID8 marker",
            value="ID8",
            variable=self.target_source_var,
            command=self.on_target_source_changed,
        ).pack(anchor="w")

        ttk.Radiobutton(
            target_source,
            text="Mouse target from Camera",
            value="Mouse",
            variable=self.target_source_var,
            command=self.on_target_source_changed,
        ).pack(anchor="w")

        # Manual
        manual = ttk.LabelFrame(
            controls,
            text="Manual virtual angles",
            padding=8,
        )
        manual.pack(fill="x", pady=(0, 8))

        self._add_slider(manual, "Base", self.manual_base_var)
        self._add_slider(manual, "Shoulder", self.manual_shoulder_var)
        self._add_slider(manual, "Elbow", self.manual_elbow_var)
        self._add_slider(manual, "Wrist", self.manual_wrist_var)


        # View / zoom
        view_box = ttk.LabelFrame(
            controls,
            text="3D View",
            padding=8,
        )
        view_box.pack(fill="x", pady=(0, 8))

        ttk.Label(view_box, text="Focus:").pack(anchor="w")

        focus_combo = ttk.Combobox(
            view_box,
            textvariable=self.view_focus_var,
            state="readonly",
            values=[
                "Robot + Target",
                "Robot",
                "Target",
                "Full Scene",
            ],
            width=22,
        )
        focus_combo.pack(fill="x", pady=(2, 5))
        focus_combo.bind(
            "<<ComboboxSelected>>",
            self._on_focus_changed,
        )

        zoom_row = ttk.Frame(view_box)
        zoom_row.pack(fill="x", pady=2)

        ttk.Button(
            zoom_row,
            text="-",
            width=4,
            command=self.zoom_out,
        ).pack(side="left")

        ttk.Scale(
            zoom_row,
            from_=350.0,
            to=1200.0,
            variable=self.view_zoom_var,
            orient="horizontal",
            command=self._on_zoom_scale_changed,
        ).pack(side="left", fill="x", expand=True, padx=5)

        ttk.Button(
            zoom_row,
            text="+",
            width=4,
            command=self.zoom_in,
        ).pack(side="left")

        preset_row = ttk.Frame(view_box)
        preset_row.pack(fill="x", pady=(5, 0))

        ttk.Button(preset_row, text="Top", command=lambda: self.set_view_preset("Top")).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(preset_row, text="Front", command=lambda: self.set_view_preset("Front")).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(preset_row, text="Side", command=lambda: self.set_view_preset("Side")).pack(side="left", fill="x", expand=True, padx=(2, 0))

        ttk.Button(
            view_box,
            text="Reset / Auto Frame",
            command=self.reset_3d_view,
        ).pack(fill="x", pady=(5, 0))

        ttk.Label(
            view_box,
            text=(
                "Drag to rotate. Manual pan/zoom is preserved across live redraws. "
                "Reset / Auto Frame restores Focus + Zoom framing."
            ),
            wraplength=340,
        ).pack(anchor="w", pady=(5, 0))

        # Display
        display = ttk.LabelFrame(controls, text="Scene", padding=8)
        display.pack(fill="x", pady=(0, 8))

        for text, var in [
            ("Workspace", self.show_workspace_var),
            ("ID8 target", self.show_target_var),
            ("Containers", self.show_containers_var),
            ("Cubes", self.show_cubes_var),
            ("Base heading", self.show_base_heading_var),
            ("Joint labels", self.show_joint_labels_var),
            ("Grasp TCP", self.show_tcp_var),
            ("TCP -> target error", self.show_tcp_target_error_var),
        ]:
            ttk.Checkbutton(
                display,
                text=text,
                variable=var,
                command=self.request_scene_update,
            ).pack(anchor="w")

        # Object dimensions
        objects = ttk.LabelFrame(
            controls,
            text="Object dimensions [mm]",
            padding=8,
        )
        objects.pack(fill="x", pady=(0, 8))


        self._add_entry(
            objects,
            "Grasp TCP from wrist",
            self.grasp_tcp_length_var,
            on_change=self.on_tcp_geometry_changed,
        )

        ttk.Label(
            objects,
            text=(
                "Measured tip: L3 = "
                f"{self.l3:.1f} mm. Preferred grasp point = 160 mm."
            ),
            wraplength=340,
        ).pack(anchor="w", pady=(0, 5))

        self._add_entry(
            objects,
            "Cube size",
            self.cube_size_var,
            on_change=self.on_object_geometry_changed,
        )
        self._add_entry(
            objects,
            "Container X size",
            self.container_size_x_var,
            on_change=self.on_object_geometry_changed,
        )
        self._add_entry(
            objects,
            "Container Y size",
            self.container_size_y_var,
            on_change=self.on_object_geometry_changed,
        )
        self._add_entry(
            objects,
            "Container height",
            self.container_height_var,
            on_change=self.on_object_geometry_changed,
        )

        ttk.Button(
            objects,
            text="Save Object Dimensions",
            command=self.save_object_geometry,
        ).pack(fill="x", pady=(6, 0))

        # Live scene data
        info = ttk.LabelFrame(controls, text="Live scene data", padding=8)
        info.pack(fill="x")

        for variable in [
            self.ik_info_var,
            self.encoder_info_var,
            self.base_info_var,
            self.target_info_var,
            self.gripper_info_var,
            self.tcp_info_var,
            self.object_info_var,
            self.calibration_info_var,
        ]:
            ttk.Label(
                info,
                textvariable=variable,
                wraplength=360,
            ).pack(anchor="w", pady=2)

    def _add_slider(self, parent, label, variable):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)

        ttk.Label(row, text=label, width=10).pack(side="left")

        ttk.Scale(
            row,
            from_=-180,
            to=180,
            variable=variable,
            orient="horizontal",
            command=lambda _: self.request_scene_update(),
        ).pack(side="left", fill="x", expand=True)

        ttk.Label(row, textvariable=variable, width=10).pack(side="right")

    def _add_entry(
        self,
        parent,
        label,
        variable,
        on_change=None,
    ):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)

        ttk.Label(
            row,
            text=label,
            width=20,
        ).pack(side="left")

        entry = ttk.Entry(
            row,
            textvariable=variable,
            width=10,
        )
        entry.pack(side="left")

        def changed(_event=None):
            if on_change is not None:
                on_change()
            else:
                self.request_scene_update()

        entry.bind("<Return>", changed)
        entry.bind("<FocusOut>", changed)


    # ======================================================
    # 3D view / TCP helpers
    # ======================================================

    def _capture_current_view(self):
        try:
            return {
                "elev": float(getattr(self.axis, "elev", 26.0)),
                "azim": float(getattr(self.axis, "azim", -58.0)),
                "xlim": tuple(float(v) for v in self.axis.get_xlim3d()),
                "ylim": tuple(float(v) for v in self.axis.get_ylim3d()),
                "zlim": tuple(float(v) for v in self.axis.get_zlim3d()),
            }
        except Exception:
            return None

    def _restore_saved_view(self):
        if not self.user_view_locked or not isinstance(self._saved_view, dict):
            return False
        try:
            self.axis.view_init(elev=self._saved_view["elev"], azim=self._saved_view["azim"])
            self.axis.set_xlim3d(*self._saved_view["xlim"])
            self.axis.set_ylim3d(*self._saved_view["ylim"])
            self.axis.set_zlim3d(*self._saved_view["zlim"])
            return True
        except Exception:
            return False

    def _on_3d_mouse_press(self, event):
        if event.inaxes is self.axis:
            self._mouse_interacting = True

    def _on_3d_mouse_release(self, event):
        if not self._mouse_interacting:
            return
        self._mouse_interacting = False
        self._saved_view = self._capture_current_view()
        if self._saved_view is not None:
            self.user_view_locked = True

        # Apply any scene changes that accumulated while dragging, while
        # restoring exactly the view the user just selected.
        if self.scene_dirty:
            self.frame.after(1, self._try_redraw_scene)

    def _on_3d_scroll(self, event):
        if event.inaxes is not self.axis:
            return
        def remember():
            self._saved_view = self._capture_current_view()
            if self._saved_view is not None:
                self.user_view_locked = True
        self.frame.after(15, remember)

    def _on_zoom_scale_changed(self, _value=None):
        self.user_view_locked = False
        self._saved_view = None
        self.last_scene_signature = None
        self.request_scene_update()

    def _on_focus_changed(self, _event=None):
        self.user_view_locked = False
        self._saved_view = None
        self.last_scene_signature = None
        self.request_scene_update()

    def set_view_preset(self, preset):
        views = {"Top": (90.0, -90.0), "Front": (8.0, -90.0), "Side": (8.0, 0.0)}
        if preset not in views:
            return
        elev, azim = views[preset]
        try:
            self.axis.view_init(elev=elev, azim=azim)
            self._saved_view = self._capture_current_view()
            self.user_view_locked = True
            self.canvas.draw_idle()
        except Exception:
            pass

    def zoom_in(self):
        # '+' = show MORE area horizontally.
        self.user_view_locked = False
        self._saved_view = None

        current = float(self.view_zoom_var.get())
        self.view_zoom_var.set(
            min(1200.0, current + 75.0)
        )

        self.last_scene_signature = None
        self.request_scene_update()

    def zoom_out(self):
        # '-' = show LESS area horizontally.
        self.user_view_locked = False
        self._saved_view = None

        current = float(self.view_zoom_var.get())
        self.view_zoom_var.set(
            max(350.0, current - 75.0)
        )

        self.last_scene_signature = None
        self.request_scene_update()

    def reset_3d_view(self):
        self.user_view_locked = False
        self._saved_view = None
        self.view_focus_var.set("Robot + Target")
        self.view_zoom_var.set(600.0)
        try:
            self.axis.view_init(elev=26, azim=-58)
        except Exception:
            pass
        self.last_scene_signature = None
        self.request_scene_update()

    def on_tcp_geometry_changed(self):
        try:
            value = float(self.grasp_tcp_length_var.get())
        except Exception:
            return

        value = max(0.0, min(float(self.l3), value))
        self.grasp_tcp_length_var.set(value)
        self.app_context.shared_data["grasp_tcp_length_mm"] = value
        self.request_scene_update()

    def get_grasp_tcp_local(self, fk):
        wrist = fk["wrist"]
        tip = fk["gripper"]

        l3 = max(1e-6, float(self.l3))
        tcp_length = max(
            0.0,
            min(l3, float(self.grasp_tcp_length_var.get())),
        )
        ratio = tcp_length / l3

        return (
            float(wrist[0]) + ratio * (float(tip[0]) - float(wrist[0])),
            float(wrist[1]) + ratio * (float(tip[1]) - float(wrist[1])),
            float(wrist[2]) + ratio * (float(tip[2]) - float(wrist[2])),
        )

    # ======================================================
    # Object geometry
    # ======================================================

    def get_object_geometry(self):
        return {
            "cube_size_mm": float(self.cube_size_var.get()),
            "container_size_x_mm": float(self.container_size_x_var.get()),
            "container_size_y_mm": float(self.container_size_y_var.get()),
            "container_height_mm": float(self.container_height_var.get()),
        }

    def publish_object_geometry(self):
        self.app_context.shared_data[
            "object_geometry"
        ] = self.get_object_geometry()

    def save_object_geometry(self):
        try:
            data = self.get_object_geometry()

            OBJECT_GEOMETRY_FILE.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            OBJECT_GEOMETRY_FILE.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8",
            )

            self.publish_object_geometry()
            self.status_var.set("Object dimensions saved.")
            self.request_scene_update()

        except Exception as error:
            self.status_var.set(
                f"Object geometry error: {error}"
            )

    def on_object_geometry_changed(self):
        self.publish_object_geometry()
        self.request_scene_update()

    # ======================================================
    # Robot controls
    # ======================================================

    def connect_robot(self):
        self.worker.send(
            commands.connect(
                port=self.port_var.get().strip(),
                robot_id=self.robot_id_var.get().strip(),
            )
        )

    def disconnect_robot(self):
        self.worker.send(commands.disconnect())

    def free_move(self):
        self.worker.send(commands.free_move())

    def hold_robot(self):
        self.worker.send(commands.hold_current())

    # ======================================================
    # Modules
    # ======================================================

    def refresh_module_list(self):
        self.available_modules = {}

        if not MODULES_DIR.exists():
            self.module_combo["values"] = []
            self.module_select_var.set("")
            return

        current = Path(__file__).resolve()

        for path in sorted(MODULES_DIR.glob("*.py")):
            if path.name.startswith("_"):
                continue
            if path.resolve() == current:
                continue

            display_name = (
                path.stem
                .replace("_panel", "")
                .replace("_", " ")
                .strip()
                .title()
            )

            self.available_modules[display_name] = path

        names = list(self.available_modules)
        self.module_combo["values"] = names

        preferred = next(
            (name for name in names if "Camera" in name),
            None,
        )
        if preferred is None:
            preferred = next(
                (name for name in names if "Ik Target" in name),
                None,
            )

        if preferred:
            self.module_select_var.set(preferred)
        elif names:
            self.module_select_var.set(names[0])
        else:
            self.module_select_var.set("")

    def open_selected_module(self):
        module_name = self.module_select_var.get().strip()
        path = self.available_modules.get(module_name)

        if path is None:
            self.status_var.set("Select a module.")
            return

        existing = self.open_module_windows.get(module_name)
        if existing:
            window = existing.get("window")
            if window is not None and window.winfo_exists():
                window.lift()
                window.focus_force()
                return

        try:
            spec = importlib.util.spec_from_file_location(
                "floating_" + path.stem,
                path,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Could not load module.")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            panel_class = getattr(module, "ModulePanel", None)
            if panel_class is None:
                raise RuntimeError("Module has no ModulePanel.")

            window = tk.Toplevel(self.frame)
            window.title(getattr(panel_class, "title", module_name))
            window.geometry("1050x850")

            panel = panel_class(window, self.app_context)
            panel_frame = (
                panel.get_frame()
                if hasattr(panel, "get_frame")
                else panel.frame
            )
            panel_frame.pack(fill="both", expand=True)

            self.open_module_windows[module_name] = {
                "window": window,
                "panel": panel,
            }

            def close_window():
                try:
                    if hasattr(panel, "shutdown"):
                        panel.shutdown()
                except Exception:
                    pass

                try:
                    window.destroy()
                except Exception:
                    pass

                self.open_module_windows.pop(module_name, None)

            window.protocol("WM_DELETE_WINDOW", close_window)

            self.status_var.set(f"Opened: {module_name}")

        except Exception as error:
            self.status_var.set(f"Module error: {error}")
            self.app_context.log(f"[Control Center launcher] {error}")

    def on_target_source_changed(self):
        source = self.target_source_var.get()
        if source not in ("ID8", "Mouse"):
            source = "ID8"
            self.target_source_var.set(source)

        self.app_context.shared_data["ik_target_source"] = source
        self.request_scene_update()

    # ======================================================
    # Kinematics / model data
    # ======================================================

    def get_geometry(self):
        return RobotGeometry(
            l1=float(self.l1),
            l2=float(self.l2),
            l3=float(self.l3),
            base_height=float(self.shoulder_height),
            base_to_shoulder_offset=float(self.base_to_shoulder),
        )

    def get_neutral_pose(self):
        return self.old_calibration.get("poses", {}).get("neutral", {})

    def motor_to_model(self, joint, motor_position):
        neutral = float(self.get_neutral_pose().get(joint, 0.0))
        delta = normalize_angle_deg(float(motor_position) - neutral)

        return normalize_angle_deg(
            self.signs[joint] * delta
            + self.offsets[joint]
        )

    def get_encoder_angles(self):
        state = self.worker.get_state_snapshot()
        positions = state.get("positions", {})

        if not all(joint in positions for joint in JOINTS):
            return None

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

    def get_live_angles(self):
        data = self.app_context.shared_data.get(
            "robot_live_angles"
        )

        if not isinstance(data, dict):
            return None

        try:
            return JointAngles(
                base=float(data["base"]),
                shoulder=float(data["shoulder"]),
                elbow=float(data["elbow"]),
                wrist=float(data["wrist"]),
            )
        except Exception:
            return None

    def get_ik_angles(self):
        data = self.app_context.shared_data.get("ik_solution_angles")

        if not isinstance(data, dict):
            return None

        try:
            return JointAngles(
                base=float(data["base"]),
                shoulder=float(data["shoulder"]),
                elbow=float(data["elbow"]),
                wrist=float(data["wrist"]),
            )
        except Exception:
            return None

    def get_manual_angles(self):
        return JointAngles(
            base=float(self.manual_base_var.get()),
            shoulder=float(self.manual_shoulder_var.get()),
            elbow=float(self.manual_elbow_var.get()),
            wrist=float(self.manual_wrist_var.get()),
        )

    def get_main_angles(self):
        if self.mode_var.get() == "Manual":
            return self.get_manual_angles()

        # Main 3D robot follows the REAL encoder reconstruction first.
        # IK remains a fallback only when no live encoder pose is published.
        live = self.get_live_angles()
        if live is not None:
            return live

        ik = self.get_ik_angles()
        if ik is not None:
            return ik

        return None

    # ======================================================
    # Coordinate transforms
    # ======================================================

    @staticmethod
    def camera_world_to_scene(x, y):
        # Established project convention:
        # camera/world x = x
        # camera/world y -> scene y = -y
        return float(x), -float(y)

    def get_workspace_polygon(self):
        data = self.app_context.shared_data.get("workspace_polygon_world")

        if not data or len(data) < 3:
            return None

        return [
            self.camera_world_to_scene(point[0], point[1])
            for point in data
        ]

    def get_marker_base_world(self):
        data = self.app_context.shared_data.get("robot_base_world")
        if data is None:
            return None

        return self.camera_world_to_scene(data[0], data[1])

    def get_frame_calibration(self):
        data = self.app_context.shared_data.get("robot_frame_calibration")

        if isinstance(data, dict):
            return {
                "offset_x": float(data.get("offset_x", 0.0)),
                "offset_y": float(data.get("offset_y", 0.0)),
                "heading_offset_deg": float(
                    data.get("heading_offset_deg", 0.0)
                ),
            }

        # Fallback to saved target calibration so the main 3D
        # still starts correctly before IK Target panel is opened.
        frame = self.target_calibration.get("robot_frame", {})

        return {
            "offset_x": float(frame.get("offset_x", 0.0)),
            "offset_y": float(frame.get("offset_y", 0.0)),
            "heading_offset_deg": float(
                frame.get("heading_offset_deg", 0.0)
            ),
        }

    def get_heading(self):
        marker_angle = self.app_context.shared_data.get(
            "robot_base_marker_angle_deg"
        )
        correction = self.get_frame_calibration()["heading_offset_deg"]

        if marker_angle is None:
            return correction

        return normalize_angle_deg(
            -float(marker_angle) + correction
        )

    @staticmethod
    def robot_local_to_world(local_x, local_y, base_world, heading_deg):
        bx, by = base_world
        h = math.radians(heading_deg)

        fx, fy = math.cos(h), math.sin(h)
        rx, ry = math.sin(h), -math.cos(h)

        return (
            bx + float(local_y) * fx + float(local_x) * rx,
            by + float(local_y) * fy + float(local_x) * ry,
        )

    def get_robot_origin_world(self):
        """
        Return corrected robot-frame origin in scene coordinates.

        In IK target code:
            local = raw_local - frame_offset

        Therefore world position of the corrected local origin is
        the marker origin shifted by +frame_offset in robot axes.
        """
        marker_base = self.get_marker_base_world()
        if marker_base is None:
            return None

        frame = self.get_frame_calibration()
        heading = self.get_heading()

        return self.robot_local_to_world(
            frame["offset_x"],
            frame["offset_y"],
            marker_base,
            heading,
        )

    def get_target_world(self):
        # SMART GO in Module 7 freezes the target before the arm can occlude ID8.
        # Prefer that frozen world target whenever it exists.
        locked = self.app_context.shared_data.get(
            "ik_locked_target_world"
        )

        if (
            isinstance(locked, (tuple, list))
            and len(locked) >= 3
        ):
            return (
                float(locked[0]),
                float(locked[1]),
                float(locked[2]),
            )

        source = self.target_source_var.get()

        if source == "Mouse":
            target = self.app_context.shared_data.get("target_world")

            if target is None:
                return None

            x, y = self.camera_world_to_scene(
                target[0],
                target[1],
            )

            z = float(
                self.app_context.shared_data.get(
                    "ik_target_z",
                    50.0,
                )
            )

            return x, y, z

        # ID8 source:
        # refresh cache when visible; otherwise keep last valid position.
        target = self.app_context.shared_data.get(
            "marker_target_world"
        )

        if target is not None:
            x, y = self.camera_world_to_scene(
                target[0],
                target[1],
            )

            self.last_id8_target_world = (
                float(x),
                float(y),
            )

        if self.last_id8_target_world is None:
            return None

        z = float(
            self.app_context.shared_data.get(
                "ik_target_z",
                50.0,
            )
        )

        return (
            float(self.last_id8_target_world[0]),
            float(self.last_id8_target_world[1]),
            z,
        )

    def get_target_label(self):
        locked = self.app_context.shared_data.get(
            "ik_locked_target_world"
        )

        if (
            isinstance(locked, (tuple, list))
            and len(locked) >= 3
        ):
            return "LOCKED TARGET"

        if self.target_source_var.get() == "Mouse":
            return "MOUSE TARGET"

        current = self.app_context.shared_data.get(
            "marker_target_world"
        )

        if current is None and self.last_id8_target_world is not None:
            return "ID8 TARGET (LAST SEEN)"

        return "ID8 TARGET"

    # ======================================================
    # Scene signature / redraw scheduling
    # ======================================================

    def _angles_signature(self, angles):
        if angles is None:
            return None

        return (
            round(float(angles.base), 2),
            round(float(angles.shoulder), 2),
            round(float(angles.elbow), 2),
            round(float(angles.wrist), 2),
        )

    def _workspace_signature(self):
        polygon = self.get_workspace_polygon()
        if not polygon:
            return None

        return tuple(
            rounded_tuple(point, 1)
            for point in polygon
        )

    def _containers_signature(self):
        data = self.app_context.shared_data.get("containers_world", {})
        if not isinstance(data, dict):
            return None

        items = []
        for marker_id, position in sorted(
            data.items(),
            key=lambda item: str(item[0]),
        ):
            if position is None:
                continue
            items.append(
                (
                    str(marker_id),
                    round(float(position[0]), 1),
                    round(float(position[1]), 1),
                )
            )

        return tuple(items)

    def get_scene_cubes(self):
        # Canonical live cube source from Module 3.
        cubes = self.app_context.shared_data.get(
            "cube_detections",
            [],
        )

        if isinstance(cubes, (list, tuple)):
            valid = []

            for cube in cubes:
                if not isinstance(cube, dict):
                    continue

                world = cube.get("world")

                if (
                    not isinstance(world, (list, tuple))
                    or len(world) < 2
                ):
                    continue

                cube_class = str(
                    cube.get("class", "")
                )

                if not cube_class.startswith("cube_"):
                    continue

                valid.append(
                    {
                        "class": cube_class,
                        "confidence": float(
                            cube.get("confidence", 0.0)
                        ),
                        "world": (
                            float(world[0]),
                            float(world[1]),
                        ),
                        "size_mm": float(
                            self.cube_size_var.get()
                        ),
                    }
                )

            if valid:
                return valid

        legacy = self.app_context.shared_data.get(
            "cubes_world",
            [],
        )

        if isinstance(legacy, dict):
            legacy = list(legacy.values())

        if not isinstance(legacy, (list, tuple)):
            return []

        return list(legacy)

    def _cubes_signature(self):
        cubes = self.get_scene_cubes()
        result = []

        for cube in cubes:
            if isinstance(cube, dict):
                x = cube.get("world_x")
                y = cube.get("world_y")

                if x is None or y is None:
                    world = cube.get("world")
                    if isinstance(world, (list, tuple)) and len(world) >= 2:
                        x, y = world[0], world[1]

                result.append(
                    (
                        str(
                            cube.get(
                                "class",
                                cube.get("label", "cube"),
                            )
                        ),
                        None if x is None else round(float(x) / 5.0) * 5.0,
                        None if y is None else round(float(y) / 5.0) * 5.0,
                        round(
                            float(
                                cube.get(
                                    "size_mm",
                                    self.cube_size_var.get(),
                                )
                            ),
                            1,
                        ),
                    )
                )

            elif isinstance(cube, (list, tuple)) and len(cube) >= 2:
                result.append(
                    (
                        "cube",
                        round(float(cube[0]) / 5.0) * 5.0,
                        round(float(cube[1]) / 5.0) * 5.0,
                        round(float(self.cube_size_var.get()), 1),
                        0.0,
                    )
                )

        return tuple(result)

    def build_scene_signature(self):
        marker_base = self.get_marker_base_world()
        robot_origin = self.get_robot_origin_world()
        target = self.get_target_world()

        main_angles = self.get_main_angles()

        encoder_angles = None
        if self.show_encoder_debug_var.get():
            encoder_angles = self.get_encoder_angles()

        frame = self.get_frame_calibration()

        return (
            self.mode_var.get(),
            self._angles_signature(main_angles),
            self._angles_signature(encoder_angles),

            None if marker_base is None else rounded_tuple(marker_base, 1),
            None if robot_origin is None else rounded_tuple(robot_origin, 1),
            round(float(self.get_heading()), 2),

            self._workspace_signature(),

            None if target is None else rounded_tuple(target, 1),

            self._containers_signature(),
            self._cubes_signature(),

            (
                bool(self.show_workspace_var.get()),
                bool(self.show_target_var.get()),
                bool(self.show_containers_var.get()),
                bool(self.show_cubes_var.get()),
                bool(self.show_base_heading_var.get()),
                bool(self.show_encoder_debug_var.get()),
                bool(self.show_joint_labels_var.get()),
                bool(self.show_tcp_var.get()),
                bool(self.show_tcp_target_error_var.get()),
            ),

            (
                self.view_focus_var.get(),
                round(float(self.view_zoom_var.get()), 3),
                round(float(self.grasp_tcp_length_var.get()), 2),
                bool(self.user_view_locked),
            ),

            (
                round(float(self.l1), 2),
                round(float(self.l2), 2),
                round(float(self.l3), 2),
                round(float(self.shoulder_height), 2),
                round(float(self.base_to_shoulder), 2),
            ),

            (
                round(float(frame["offset_x"]), 2),
                round(float(frame["offset_y"]), 2),
                round(float(frame["heading_offset_deg"]), 2),
            ),

            (
                round(float(self.cube_size_var.get()), 1),
                round(float(self.container_size_x_var.get()), 1),
                round(float(self.container_size_y_var.get()), 1),
                round(float(self.container_height_var.get()), 1),
            ),
        )

    def request_scene_update(self):
        self.scene_dirty = True
        self._try_redraw_scene()

    def _try_redraw_scene(self):
        if not self.scene_dirty:
            return

        # Do not clear/rebuild the Matplotlib 3D axes while the user is
        # actively dragging the view. Camera/AI updates can otherwise
        # fight with the mouse interaction and make the scene feel stuck.
        if self._mouse_interacting:
            return

        now = time.monotonic()
        elapsed_ms = (
            now - self.last_draw_monotonic
        ) * 1000.0

        if elapsed_ms < self.MIN_REDRAW_MS:
            remaining = max(
                1,
                int(self.MIN_REDRAW_MS - elapsed_ms),
            )

            if self.pending_draw_after_id is None:
                self.pending_draw_after_id = self.frame.after(
                    remaining,
                    self._scheduled_redraw,
                )
            return

        self._perform_redraw()

    def _scheduled_redraw(self):
        self.pending_draw_after_id = None
        self._try_redraw_scene()

    def _perform_redraw(self):
        signature = self.build_scene_signature()

        # Dirty can be set manually by controls. If the semantic
        # scene did not actually change, do not clear/redraw axes.
        if signature == self.last_scene_signature:
            self.scene_dirty = False
            return

        try:
            self.draw_scene()
            self.last_scene_signature = signature
            self.last_draw_monotonic = time.monotonic()
            self.scene_dirty = False
        except Exception as error:
            self.status_var.set(f"3D error: {error}")
            self.app_context.log(f"[Control Center draw] {error}")

    # ======================================================
    # Scene drawing
    # ======================================================

    def draw_scene(self):
        current_view = self._capture_current_view()
        self.axis.clear()
        if current_view is not None:
            self.axis.view_init(elev=current_view["elev"], azim=current_view["azim"])
        else:
            self.axis.view_init(elev=26, azim=-58)

        polygon = self.get_workspace_polygon()
        marker_base = self.get_marker_base_world()
        robot_origin = self.get_robot_origin_world()
        heading = self.get_heading()

        self._draw_table_grid(polygon)

        if self.show_workspace_var.get():
            self._draw_workspace(polygon)

        if robot_origin is not None:
            self._draw_robot_base(
                robot_origin,
                heading,
            )

        container_count = 0
        cube_count = 0

        if self.show_containers_var.get():
            container_count = self._draw_containers()

        if self.show_cubes_var.get():
            cube_count = self._draw_cubes()

        target = None
        if self.show_target_var.get():
            target = self._draw_target()

        main_angles = self.get_main_angles()
        main_gripper_world = None

        if robot_origin is not None and main_angles is not None:
            main_fk = forward_kinematics(
                main_angles,
                self.get_geometry(),
            )
            main_gripper_world = self._draw_arm(
                main_fk,
                robot_origin,
                heading,
                linewidth=4.2,
                alpha=1.0,
                label_prefix="",
            )

            tcp_local = self.get_grasp_tcp_local(main_fk)
            tcp_xy = self.robot_local_to_world(
                tcp_local[0],
                tcp_local[1],
                robot_origin,
                heading,
            )
            main_tcp_world = (
                tcp_xy[0],
                tcp_xy[1],
                float(tcp_local[2]),
            )

            if self.show_tcp_var.get():
                self.axis.scatter(
                    [main_tcp_world[0]],
                    [main_tcp_world[1]],
                    [main_tcp_world[2]],
                    marker="*",
                    s=280,
                    depthshade=False,
                )
                self.axis.text(
                    main_tcp_world[0],
                    main_tcp_world[1],
                    main_tcp_world[2] + 10,
                    "GRASP TCP",
                    fontweight="bold",
                )
                self.axis.plot(
                    [main_tcp_world[0], main_gripper_world[0]],
                    [main_tcp_world[1], main_gripper_world[1]],
                    [main_tcp_world[2], main_gripper_world[2]],
                    linewidth=5.0,
                    alpha=0.75,
                )

            if target is not None:
                dx = main_tcp_world[0] - target[0]
                dy = main_tcp_world[1] - target[1]
                dz = main_tcp_world[2] - target[2]
                error = math.sqrt(dx * dx + dy * dy + dz * dz)

                self.tcp_info_var.set(
                    "GRASP TCP | "
                    f"World=({main_tcp_world[0]:.1f}, "
                    f"{main_tcp_world[1]:.1f}, "
                    f"{main_tcp_world[2]:.1f}) mm | "
                    f"Target error={error:.1f} mm | "
                    f"dX={dx:+.1f}, dY={dy:+.1f}, dZ={dz:+.1f}"
                )

                if self.show_tcp_target_error_var.get():
                    self.axis.plot(
                        [main_tcp_world[0], target[0]],
                        [main_tcp_world[1], target[1]],
                        [main_tcp_world[2], target[2]],
                        linestyle="--",
                        linewidth=2.2,
                        alpha=0.9,
                    )
                    mid = (
                        (main_tcp_world[0] + target[0]) / 2.0,
                        (main_tcp_world[1] + target[1]) / 2.0,
                        (main_tcp_world[2] + target[2]) / 2.0,
                    )
                    self.axis.text(
                        mid[0],
                        mid[1],
                        mid[2] + 10,
                        f"TCP error {error:.1f} mm",
                    )
            else:
                self.tcp_info_var.set(
                    "GRASP TCP | "
                    f"World=({main_tcp_world[0]:.1f}, "
                    f"{main_tcp_world[1]:.1f}, "
                    f"{main_tcp_world[2]:.1f}) mm"
                )

        if (
            robot_origin is not None
            and self.show_encoder_debug_var.get()
        ):
            encoder_angles = self.get_encoder_angles()

            if encoder_angles is not None:
                encoder_fk = forward_kinematics(
                    encoder_angles,
                    self.get_geometry(),
                )
                self._draw_arm(
                    encoder_fk,
                    robot_origin,
                    heading,
                    linewidth=1.8,
                    alpha=0.55,
                    label_prefix="ENC ",
                    draw_labels=False,
                )

        if (
            target is not None
            and main_gripper_world is not None
            and not self.show_tcp_var.get()
        ):
            self.axis.plot(
                [main_gripper_world[0], target[0]],
                [main_gripper_world[1], target[1]],
                [main_gripper_world[2], target[2]],
                linestyle="--",
                linewidth=1.0,
                alpha=0.45,
            )

        self.object_info_var.set(
            f"Objects: {cube_count} cube(s), "
            f"{container_count} container(s)"
        )

        self._configure_axes(
            polygon,
            robot_origin or marker_base,
            target,
        )

        if self.user_view_locked:
            self._restore_saved_view()
            self._apply_equal_data_aspect()

        self.canvas.draw_idle()

    def _draw_table_grid(self, polygon):
        if not polygon:
            return

        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        step = 50.0

        x = math.floor(min_x / step) * step
        while x <= max_x:
            self.axis.plot(
                [x, x],
                [min_y, max_y],
                [0, 0],
                linewidth=0.35,
                alpha=0.35,
            )
            x += step

        y = math.floor(min_y / step) * step
        while y <= max_y:
            self.axis.plot(
                [min_x, max_x],
                [y, y],
                [0, 0],
                linewidth=0.35,
                alpha=0.35,
            )
            y += step

    def _draw_workspace(self, polygon):
        if not polygon:
            return

        closed = polygon + [polygon[0]]

        self.axis.plot(
            [p[0] for p in closed],
            [p[1] for p in closed],
            [0.0] * len(closed),
            linewidth=3.0,
        )

        for index, point in enumerate(polygon):
            self.axis.scatter(
                [point[0]],
                [point[1]],
                [0],
                s=45,
            )
            self.axis.text(
                point[0],
                point[1],
                5,
                f"ID{index}",
            )

    def _draw_robot_base(self, base, heading):
        bx, by = base

        # One clear visual origin only:
        # the corrected robot-frame origin used by FK/IK.
        self.axis.scatter(
            [bx],
            [by],
            [0],
            s=230,
        )

        self.axis.text(
            bx,
            by,
            8,
            "ROBOT BASE",
        )

        if self.show_base_heading_var.get():
            length = 85.0
            h = math.radians(heading)

            dx = length * math.cos(h)
            dy = length * math.sin(h)

            self.axis.quiver(
                bx,
                by,
                5,
                dx,
                dy,
                0,
                arrow_length_ratio=0.18,
            )

    def _draw_target(self):
        target = self.get_target_world()

        if target is None:
            return None

        x, y, z = target

        self.axis.scatter(
            [x],
            [y],
            [z],
            marker="x",
            s=170,
        )

        self.axis.plot(
            [x, x],
            [y, y],
            [0, z],
            linestyle="--",
            linewidth=1.0,
        )

        self.axis.text(
            x,
            y,
            z + 8,
            self.get_target_label(),
        )

        return target

    def _draw_arm(
        self,
        fk,
        robot_origin,
        heading,
        linewidth,
        alpha,
        label_prefix="",
        draw_labels=True,
    ):
        points = {}

        for name in [
            "base",
            "shoulder",
            "elbow",
            "wrist",
            "gripper",
        ]:
            lx, ly, lz = fk[name]
            wx, wy = self.robot_local_to_world(
                lx,
                ly,
                robot_origin,
                heading,
            )

            points[name] = (
                wx,
                wy,
                float(lz),
            )

        chain = [
            points[name]
            for name in [
                "base",
                "shoulder",
                "elbow",
                "wrist",
                "gripper",
            ]
        ]

        self.axis.plot(
            [p[0] for p in chain],
            [p[1] for p in chain],
            [p[2] for p in chain],
            marker="o",
            linewidth=linewidth,
            alpha=alpha,
        )

        # Vertical column from table to base rotation axis height.
        bx, by = robot_origin

        self.axis.plot(
            [bx, bx],
            [by, by],
            [0, self.shoulder_height],
            linestyle=":",
            linewidth=max(0.8, linewidth * 0.35),
            alpha=alpha,
        )

        if draw_labels:
            names = (
                ["shoulder", "elbow", "wrist", "gripper"]
                if self.show_joint_labels_var.get()
                else ["gripper"]
            )

            for name in names:
                if name == "gripper" and self.show_tcp_var.get() and not label_prefix:
                    continue
                p = points[name]
                label = "TIP" if name == "gripper" else name
                self.axis.text(
                    p[0],
                    p[1],
                    p[2] + 5,
                    label_prefix + label,
                )

        return points["gripper"]

    def _draw_containers(self):
        containers = self.app_context.shared_data.get(
            "containers_world",
            {},
        )

        if not isinstance(containers, dict):
            return 0

        size_x = float(self.container_size_x_var.get())
        size_y = float(self.container_size_y_var.get())
        height = float(self.container_height_var.get())
        count = 0

        for marker_id, position in containers.items():
            if position is None:
                continue

            x, y = self.camera_world_to_scene(
                position[0],
                position[1],
            )

            self._draw_solid_box(
                center_x=x,
                center_y=y,
                bottom_z=0.0,
                size_x=size_x,
                size_y=size_y,
                size_z=height,
                alpha=0.16,
            )

            self.axis.text(
                x,
                y,
                height + 5,
                f"Container ID{marker_id}",
            )

            count += 1

        return count

    def _draw_cubes(self):
        cubes = self.get_scene_cubes()

        default_size = float(
            self.cube_size_var.get()
        )

        class_colors = {
            "cube_red": "red",
            "cube_green": "green",
            "cube_blue": "blue",
            "cube_yellow": "gold",
        }

        count = 0

        for index, cube in enumerate(cubes):
            x = None
            y = None
            size = default_size
            label = f"Cube {index + 1}"
            confidence = None

            if isinstance(cube, dict):
                x = cube.get("world_x")
                y = cube.get("world_y")

                if x is None or y is None:
                    world = cube.get("world")
                    if isinstance(world, (list, tuple)) and len(world) >= 2:
                        x, y = world[0], world[1]

                size = float(
                    cube.get(
                        "size_mm",
                        default_size,
                    )
                )

                label = str(
                    cube.get(
                        "class",
                        cube.get("label", label),
                    )
                )

                confidence = cube.get("confidence")

            elif isinstance(cube, (list, tuple)) and len(cube) >= 2:
                x, y = cube[0], cube[1]

            if x is None or y is None:
                continue

            x, y = self.camera_world_to_scene(x, y)

            self._draw_solid_box(
                center_x=x,
                center_y=y,
                bottom_z=0.0,
                size_x=size,
                size_y=size,
                size_z=size,
                alpha=0.55,
                face_color=class_colors.get(
                    label,
                    "gray",
                ),
            )

            text = label
            if confidence is not None:
                text += f" {float(confidence):.2f}"

            self.axis.text(
                x,
                y,
                size + 4,
                text,
            )

            count += 1

        return count

    def _draw_solid_box(
        self,
        center_x,
        center_y,
        bottom_z,
        size_x,
        size_y,
        size_z,
        alpha,
        face_color=None,
    ):
        hx = float(size_x) / 2.0
        hy = float(size_y) / 2.0

        x0 = float(center_x) - hx
        x1 = float(center_x) + hx

        y0 = float(center_y) - hy
        y1 = float(center_y) + hy

        z0 = float(bottom_z)
        z1 = z0 + float(size_z)

        corners = [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),

            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ]

        faces = [
            [corners[i] for i in (0, 1, 2, 3)],
            [corners[i] for i in (4, 5, 6, 7)],
            [corners[i] for i in (0, 1, 5, 4)],
            [corners[i] for i in (1, 2, 6, 5)],
            [corners[i] for i in (2, 3, 7, 6)],
            [corners[i] for i in (3, 0, 4, 7)],
        ]

        collection = Poly3DCollection(
            faces,
            alpha=alpha,
            linewidths=0.8,
            edgecolors="black",
            facecolors=(
                face_color
                if face_color is not None
                else None
            ),
        )

        self.axis.add_collection3d(collection)

    def _configure_axes(self, polygon, base_world, target):
        focus = self.view_focus_var.get()

        # Horizontal field width in millimetres.
        # This is intentionally independent from Z framing:
        # + widens X/Y, - narrows X/Y, Z stays fixed.
        requested_span = max(
            350.0,
            min(
                1200.0,
                float(self.view_zoom_var.get()),
            ),
        )

        if focus == "Full Scene":
            xs = []
            ys = []

            if polygon:
                xs.extend(point[0] for point in polygon)
                ys.extend(point[1] for point in polygon)

            if base_world is not None:
                xs.append(base_world[0])
                ys.append(base_world[1])

            if target is not None:
                xs.append(target[0])
                ys.append(target[1])

            if not xs:
                xs = [-300.0, 300.0]
                ys = [-300.0, 300.0]

            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            required_span = max(
                max(xs) - min(xs),
                max(ys) - min(ys),
                400.0,
            )

        elif focus == "Target" and target is not None:
            cx = float(target[0])
            cy = float(target[1])
            required_span = 300.0

        elif focus == "Robot" and base_world is not None:
            cx = float(base_world[0])
            cy = float(base_world[1])
            required_span = 500.0

        else:
            if base_world is not None and target is not None:
                cx = (float(base_world[0]) + float(target[0])) / 2.0
                cy = (float(base_world[1]) + float(target[1])) / 2.0
                separation = math.hypot(
                    float(target[0]) - float(base_world[0]),
                    float(target[1]) - float(base_world[1]),
                )
                required_span = max(300.0, separation + 250.0)
            elif base_world is not None:
                cx = float(base_world[0])
                cy = float(base_world[1])
                required_span = 500.0
            elif target is not None:
                cx = float(target[0])
                cy = float(target[1])
                required_span = 350.0
            else:
                cx = 0.0
                cy = 0.0
                required_span = 500.0

        # Focus chooses the centre. The +/- control chooses the actual
        # horizontal width. For Full Scene, never crop known scene geometry.
        if focus == "Full Scene":
            span = max(
                requested_span,
                required_span + 60.0,
            )
        else:
            span = requested_span

        self.axis.set_xlim(
            cx - span / 2.0,
            cx + span / 2.0,
        )
        self.axis.set_ylim(
            cy - span / 2.0,
            cy + span / 2.0,
        )

        # Fixed tabletop-oriented vertical range.
        # +/- changes ONLY X/Y width; it never stretches the scene upward.
        scene_z_top = 320.0

        if target is not None:
            scene_z_top = max(
                scene_z_top,
                float(target[2]) + 80.0,
            )

        try:
            scene_z_top = max(
                scene_z_top,
                float(self.container_height_var.get()) + 60.0,
            )
        except Exception:
            pass

        self.axis.set_zlim(-10.0, scene_z_top)

        self.axis.set_xlabel("World X [mm]")
        self.axis.set_ylabel("World Y [mm]")
        self.axis.set_zlabel("Z [mm]")
        self.axis.set_title(
            f"SO-101 Digital Twin — {self.mode_var.get()} | "
            f"TCP={float(self.grasp_tcp_length_var.get()):.0f} mm"
        )
        self._apply_equal_data_aspect()

    def _apply_equal_data_aspect(self):
        """Keep equal visual scale per millimetre on X/Y/Z."""
        try:
            x0, x1 = self.axis.get_xlim3d()
            y0, y1 = self.axis.get_ylim3d()
            z0, z1 = self.axis.get_zlim3d()
            xspan = max(1.0, abs(float(x1) - float(x0)))
            yspan = max(1.0, abs(float(y1) - float(y0)))
            zspan = max(1.0, abs(float(z1) - float(z0)))
            self.axis.set_box_aspect((xspan, yspan, zspan))
        except Exception:
            pass

    # ======================================================
    # Polling / info
    # ======================================================

    def update_live_info(self):
        state = self.worker.get_state_snapshot()

        connected = bool(state.get("connected", False))
        torque = bool(state.get("torque_enabled", False))

        self.robot_status_var.set(
            "Robot: connected"
            if connected
            else "Robot: offline"
        )

        if not connected:
            self.torque_status_var.set("Torque: —")
        elif torque:
            self.torque_status_var.set("Torque: ON")
        else:
            self.torque_status_var.set("Torque: FREE")

        ik = self.get_ik_angles()

        if ik is None:
            self.ik_info_var.set("Robot / IK: —")
        else:
            self.ik_info_var.set(
                "Robot / IK | "
                f"B={ik.base:.1f}° "
                f"S={ik.shoulder:.1f}° "
                f"E={ik.elbow:.1f}° "
                f"W={ik.wrist:.1f}°"
            )

        encoder = self.get_encoder_angles()

        if encoder is None:
            self.encoder_info_var.set("Encoder debug: —")
        else:
            self.encoder_info_var.set(
                "Encoder | "
                f"B={encoder.base:.1f}° "
                f"S={encoder.shoulder:.1f}° "
                f"E={encoder.elbow:.1f}° "
                f"W={encoder.wrist:.1f}°"
            )

        marker_base = self.get_marker_base_world()
        robot_origin = self.get_robot_origin_world()

        if marker_base is None:
            self.base_info_var.set("Base: —")
        else:
            text = (
                f"ID4=({marker_base[0]:.1f}, "
                f"{marker_base[1]:.1f}) "
                f"Heading={self.get_heading():.1f}°"
            )

            if robot_origin is not None:
                text += (
                    f" | Robot origin=({robot_origin[0]:.1f}, "
                    f"{robot_origin[1]:.1f})"
                )

            self.base_info_var.set(text)

        target = self.get_target_world()

        source_name = (
            "Mouse"
            if self.target_source_var.get() == "Mouse"
            else "ID8"
        )

        if target is None:
            self.target_info_var.set(
                f"{source_name}: —"
            )
        else:
            self.target_info_var.set(
                f"{source_name}: X={target[0]:.1f}, "
                f"Y={target[1]:.1f}, "
                f"Z={target[2]:.1f}"
            )

        main_angles = self.get_main_angles()

        if main_angles is None:
            self.gripper_info_var.set("Gripper: —")
        else:
            fk = forward_kinematics(
                main_angles,
                self.get_geometry(),
            )
            g = fk["gripper"]

            self.gripper_info_var.set(
                "Gripper local: "
                f"X={g[0]:.1f}, "
                f"Y={g[1]:.1f}, "
                f"Z={g[2]:.1f}"
            )

        frame = self.get_frame_calibration()

        self.calibration_info_var.set(
            "Calibration | "
            f"L1={self.l1:.1f} "
            f"L2={self.l2:.1f} "
            f"L3={self.l3:.1f} | "
            f"Frame X={frame['offset_x']:.1f} "
            f"Y={frame['offset_y']:.1f} "
            f"H={frame['heading_offset_deg']:.1f}°"
        )

    def update_loop(self):
        try:
            calibration_changed = self.sync_model_calibration()
            if calibration_changed:
                self.scene_dirty = True

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
                self.scene_dirty = True

            self.publish_object_geometry()
            self.update_live_info()

            signature = self.build_scene_signature()

            if signature != self.last_scene_signature:
                self.scene_dirty = True

            self._try_redraw_scene()

        except Exception as error:
            self.app_context.log(f"[Control Center loop] {error}")

        self.frame.after(
            self.STATUS_POLL_MS,
            self.update_loop,
        )

    # ======================================================
    # Module API
    # ======================================================

    def get_frame(self):
        return self.frame

    def shutdown(self):
        if self.pending_draw_after_id is not None:
            try:
                self.frame.after_cancel(self.pending_draw_after_id)
            except Exception:
                pass
            self.pending_draw_after_id = None

        for data in list(self.open_module_windows.values()):
            panel = data.get("panel")
            window = data.get("window")

            try:
                if panel is not None and hasattr(panel, "shutdown"):
                    panel.shutdown()
            except Exception:
                pass

            try:
                if window is not None and window.winfo_exists():
                    window.destroy()
            except Exception:
                pass

        self.open_module_windows.clear()
