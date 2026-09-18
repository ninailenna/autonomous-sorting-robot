from __future__ import annotations

import math
import time
import tkinter as tk
from tkinter import ttk


class ModulePanel:
    """
    Module 10: lightweight autonomous sorter.

    IMPORTANT:
    - Module 10 does NOT command robot joints.
    - Module 10 does NOT create a hidden Module 8.
    - Module 10 uses the already-open live Module 8.
    - All physical motion remains owned by Module 8.

    Flow:
        freeze scene snapshot
        -> choose cube
        -> ask live Module 8 to run its Test Auto for exactly 1 cube
        -> wait until Module 8 finishes
        -> mark cube DONE
        -> repeat
        -> when no cubes remain: wait for hand/change, then rescan
    """

    title = "Autonomous Sorting"
    UPDATE_MS = 150

    # Session matching radius: prevents the same original detection from being
    # selected twice if it remains visible for a short time after sorting.
    DONE_RADIUS_MM = 30.0

    HAND_CONFIRM_FRAMES = 3
    HAND_CLEAR_FRAMES = 3

    def __init__(self, parent, app_context):
        self.app_context = app_context
        self.worker = app_context.robot_worker
        self.frame = ttk.Frame(parent, padding=8)

        self.running = False
        self.state = "IDLE"
        self.active_cube = None
        self.done_cubes = []
        self.scene_snapshot = []

        self.hand_seen_frames = 0
        self.hand_clear_frames = 0
        self.change_wait_started_at = None
        self.hand_clear_started_at = None
        self.module8_was_active = False
        self.pending_native_cube = None
        self.home_reason = None

        # Hand monitoring must remain active even after a sorting ERROR.
        # Example: failed grasp -> Test Auto ERROR. If the user then reaches
        # into the workspace, HOLD immediately; after the hand is stably gone,
        # send DIRECT HOME even though the autonomous run itself is stopped.
        self.error_latched = False
        self.safety_hand_active = False
        self.safety_hand_clear_started_at = None
        self.safety_home_after_error_active = False

        # If a hand interrupted a normal autonomous run, resume with
        # HOME -> rescan delay -> SCAN. If the task had already ERRORed,
        # do HOME only and keep sorting stopped.
        self.safety_resume_after_home = False

        self.mode_var = tk.StringVar(value="ALL")
        self.rescan_delay_var = tk.DoubleVar(value=10.0)

        # Hand must stay absent for this many seconds before HOME resumes.
        # This filters short YOLO dropouts/flicker.
        self.hand_clear_delay_var = tk.DoubleVar(value=1.5)

        self.transport_speed_var = tk.DoubleVar(value=70.0)
        self.ready_home_speed_var = tk.DoubleVar(value=85.0)

        self.status_var = tk.StringVar(value="Module 10: IDLE")
        self.hand_scene_var = tk.StringVar(value="HAND IN SCENE: NO")
        self.scene_var = tk.StringVar(value="Scene: not scanned")
        self.module8_var = tk.StringVar(value="Module 8: searching...")

        self._build_ui()
        self.frame.after(self.UPDATE_MS, self.update_loop)

    # --------------------------------------------------
    # UI
    # --------------------------------------------------

    def _build_ui(self):
        controls = ttk.LabelFrame(
            self.frame,
            text="Autonomous Sorting",
            padding=8,
        )
        controls.pack(fill="x")

        # --------------------------------------------------
        # Row 1: main actions + mode/timers
        # --------------------------------------------------
        row1 = ttk.Frame(controls)
        row1.pack(fill="x")

        ttk.Button(
            row1,
            text="START",
            command=self.start,
        ).pack(side="left")

        ttk.Button(
            row1,
            text="STOP",
            command=self.stop,
        ).pack(side="left", padx=(6, 0))

        ttk.Button(
            row1,
            text="RESCAN NOW",
            command=self.rescan_now,
        ).pack(side="left", padx=(16, 0))

        ttk.Button(
            row1,
            text="CLEAR DONE",
            command=self.clear_done,
        ).pack(side="left", padx=(6, 0))

        ttk.Label(
            row1,
            text="Sort:",
        ).pack(side="left", padx=(18, 4))

        ttk.Combobox(
            row1,
            textvariable=self.mode_var,
            state="readonly",
            width=12,
            values=("ALL", "RED", "YELLOW", "GREEN", "BLUE"),
        ).pack(side="left")

        ttk.Label(
            row1,
            text="Rescan after hand:",
        ).pack(side="left", padx=(18, 4))

        ttk.Spinbox(
            row1,
            from_=0,
            to=60,
            increment=1,
            width=6,
            textvariable=self.rescan_delay_var,
        ).pack(side="left")

        ttk.Label(
            row1,
            text="s",
        ).pack(side="left", padx=(2, 0))

        ttk.Label(
            row1,
            text="Hand clear:",
        ).pack(side="left", padx=(14, 4))

        ttk.Spinbox(
            row1,
            from_=0.2,
            to=10.0,
            increment=0.1,
            width=5,
            textvariable=self.hand_clear_delay_var,
        ).pack(side="left")

        ttk.Label(
            row1,
            text="s",
        ).pack(side="left", padx=(2, 0))

        # --------------------------------------------------
        # Row 2: speed controls
        # --------------------------------------------------
        row2 = ttk.Frame(controls)
        row2.pack(fill="x", pady=(7, 0))

        ttk.Label(
            row2,
            text="After-grasp speed:",
        ).pack(side="left")

        ttk.Scale(
            row2,
            from_=20,
            to=100,
            variable=self.transport_speed_var,
            orient="horizontal",
            length=180,
        ).pack(side="left", padx=(6, 0))

        self.speed_value_label = ttk.Label(
            row2,
            text="70%",
            width=5,
        )
        self.speed_value_label.pack(side="left", padx=(4, 16))

        ttk.Label(
            row2,
            text="READY/HOME speed:",
        ).pack(side="left")

        ttk.Scale(
            row2,
            from_=20,
            to=100,
            variable=self.ready_home_speed_var,
            orient="horizontal",
            length=180,
        ).pack(side="left", padx=(6, 0))

        self.ready_home_speed_label = ttk.Label(
            row2,
            text="85%",
            width=5,
        )
        self.ready_home_speed_label.pack(side="left", padx=(4, 0))

        ttk.Label(
            controls,
            textvariable=self.status_var,
        ).pack(anchor="w", pady=(8, 0))

        ttk.Label(
            controls,
            textvariable=self.module8_var,
        ).pack(anchor="w", pady=(2, 0))

        scene = ttk.LabelFrame(
            self.frame,
            text="Frozen Scene Memory",
            padding=8,
        )
        scene.pack(fill="both", expand=True, pady=(8, 0))

        ttk.Label(
            scene,
            textvariable=self.scene_var,
            justify="left",
        ).pack(anchor="w")

        ttk.Label(
            scene,
            textvariable=self.hand_scene_var,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))

        self.canvas = tk.Canvas(
            scene,
            width=620,
            height=420,
            background="black",
            highlightthickness=1,
        )
        self.canvas.pack(fill="both", expand=True, pady=(8, 0))
        self.canvas.bind(
            "<Configure>",
            lambda _e: self._draw_scene(),
        )

    # --------------------------------------------------
    # Live Module 8 lookup
    # --------------------------------------------------

    def _find_live_module8(self):
        """
        Find the already-open Module 8 panel from Module 6/main panel.
        Never instantiate a second engine.
        """
        app = getattr(self.app_context, "app", None)
        main_panel = getattr(app, "main_panel", None)
        windows = getattr(main_panel, "open_module_windows", None)

        if not isinstance(windows, dict):
            return None

        candidates = []

        for key, value in windows.items():
            objs = []

            if value is not None:
                objs.append(value)

            if isinstance(value, dict):
                objs.extend(value.values())

            for obj in objs:
                panel = getattr(obj, "panel", None)
                if panel is not None:
                    objs.append(panel)

                module_panel = getattr(obj, "module_panel", None)
                if module_panel is not None:
                    objs.append(module_panel)

            for obj in objs:
                if obj is None:
                    continue

                # Our integrated Module 8 wrapper has an embedded .engine and
                # the test-auto API. This avoids accidentally selecting Module 3/7/etc.
                if (
                    hasattr(obj, "engine")
                    and hasattr(obj, "start_test_auto")
                    and hasattr(obj, "test_auto_active")
                ):
                    candidates.append(obj)

        if not candidates:
            return None

        # Prefer the newest/open wrapper if duplicates somehow exist.
        return candidates[-1]

    # --------------------------------------------------
    # Scene helpers
    # --------------------------------------------------

    @staticmethod
    def _cube_xy(cube):
        for pair in (
            ("x", "y"),
            ("world_x", "world_y"),
            ("x_mm", "y_mm"),
            ("center_world_x", "center_world_y"),
            ("workspace_x", "workspace_y"),
        ):
            if pair[0] in cube and pair[1] in cube:
                try:
                    return float(cube[pair[0]]), float(cube[pair[1]])
                except Exception:
                    pass

        for key in ("world", "world_center"):
            center = cube.get(key)
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                try:
                    return float(center[0]), float(center[1])
                except Exception:
                    pass

        return None

    @staticmethod
    def _cube_class(cube):
        return str(
            cube.get(
                "class",
                cube.get("class_name", ""),
            )
        ).lower()

    def _mode_accepts(self, cube):
        mode = self.mode_var.get().strip().upper()

        if mode == "ALL":
            return True

        cls = self._cube_class(cube)
        return mode.lower() in cls

    def _is_done(self, cube):
        xy = self._cube_xy(cube)
        if xy is None:
            return False

        x, y = xy

        for done in self.done_cubes:
            dxy = self._cube_xy(done)
            if dxy is None:
                continue

            dx, dy = dxy

            if math.hypot(x - dx, y - dy) <= self.DONE_RADIUS_MM:
                return True

        return False

    def _live_cubes(self):
        cubes = self.app_context.shared_data.get(
            "cube_detections",
            [],
        )

        if not isinstance(cubes, list):
            return []

        return [
            dict(c)
            for c in cubes
            if isinstance(c, dict)
        ]

    def _freeze_scene(self):
        self.scene_snapshot = [
            cube
            for cube in self._live_cubes()
            if self._mode_accepts(cube)
            and not self._is_done(cube)
        ]

        self.scene_snapshot.sort(
            key=lambda c: float(c.get("confidence", 0.0)),
            reverse=True,
        )

        self.scene_var.set(
            f"Scene: {len(self.scene_snapshot)} candidate(s) | "
            f"DONE={len(self.done_cubes)} | mode={self.mode_var.get()}"
        )
        self._draw_scene()

    def _next_cube(self):
        for cube in self.scene_snapshot:
            if not self._is_done(cube):
                return cube
        return None

    # --------------------------------------------------
    # Drawing
    # --------------------------------------------------

    def _draw_scene(self):
        if not hasattr(self, "canvas"):
            return

        self.canvas.delete("all")
        w = max(240, int(self.canvas.winfo_width()))
        h = max(180, int(self.canvas.winfo_height()))
        shared = self.app_context.shared_data

        polygon = shared.get("workspace_polygon_world")
        base = shared.get("robot_base_world")
        base_angle = shared.get("robot_base_marker_angle_deg")
        containers = shared.get("containers_world", {}) or {}
        hands = [
            d for d in shared.get("hand_detections", [])
            if isinstance(d, dict)
        ]
        hand_inside = bool(
            shared.get("hand_inside_workspace", False)
        )

        self.hand_scene_var.set(
            "HAND IN SCENE: "
            + ("YES | ROBOT HOLD" if hand_inside
               else "YES | outside workspace" if hands
               else "NO")
        )

        # --------------------------------------------------
        # FIXED transform: workspace polygon ONLY.
        # Hands/cubes never change map scale, so the map cannot jump.
        # --------------------------------------------------
        workspace_points = []

        if isinstance(polygon, (list, tuple)):
            for p in polygon:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    workspace_points.append(
                        (float(p[0]), float(p[1]))
                    )

        if len(workspace_points) < 3:
            self.canvas.create_text(
                w/2,
                h/2,
                text="Workspace not available from Module 3",
                fill="#dddddd",
                font=("TkDefaultFont", 11),
            )
            return

        xs = [p[0] for p in workspace_points]
        ys = [p[1] for p in workspace_points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        span_x = max(100.0, max_x-min_x)
        span_y = max(100.0, max_y-min_y)

        # Fixed 8% workspace padding.
        min_x -= span_x * 0.08
        max_x += span_x * 0.08
        min_y -= span_y * 0.08
        max_y += span_y * 0.08

        span_x = max_x-min_x
        span_y = max_y-min_y

        margin = 34.0
        scale = min(
            max(1.0, w-2*margin) / span_x,
            max(1.0, h-2*margin) / span_y,
        )

        draw_w = span_x * scale
        draw_h = span_y * scale
        ox = (w-draw_w)/2.0
        oy = (h-draw_h)/2.0

        def ws(x, y):
            # Fixed operator view with a 180° workspace orientation.
            return (
                ox + (max_x-float(x))*scale,
                oy + (max_y-float(y))*scale,
            )

        # --------------------------------------------------
        # Background + subtle world grid
        # --------------------------------------------------
        self.canvas.create_rectangle(
            ox,
            oy,
            ox+draw_w,
            oy+draw_h,
            fill="#101318",
            outline="#2f3947",
            width=2,
        )

        grid_step = 50.0
        gx = math.ceil(min_x/grid_step)*grid_step
        while gx <= max_x:
            x1, y1 = ws(gx, min_y)
            x2, y2 = ws(gx, max_y)
            self.canvas.create_line(
                x1, y1, x2, y2,
                fill="#1d2632",
                width=1,
            )
            gx += grid_step

        gy = math.ceil(min_y/grid_step)*grid_step
        while gy <= max_y:
            x1, y1 = ws(min_x, gy)
            x2, y2 = ws(max_x, gy)
            self.canvas.create_line(
                x1, y1, x2, y2,
                fill="#1d2632",
                width=1,
            )
            gy += grid_step

        # Workspace polygon
        coords = []
        for p in workspace_points:
            sx, sy = ws(*p)
            coords.extend([sx, sy])

        self.canvas.create_polygon(
            coords,
            fill="#171d25",
            outline="#9aa7b7",
            width=3,
        )

        self.canvas.create_text(
            ox+12,
            oy+10,
            anchor="nw",
            text=self.hand_scene_var.get(),
            fill="#ff67d6" if hand_inside else "#c7d0dc",
            font=("TkDefaultFont", 11, "bold"),
        )

        # --------------------------------------------------
        # Containers
        # --------------------------------------------------
        id_colors = {
            5: "#ff5353",
            6: "#ffd84e",
            7: "#5ee06e",
            19: "#58a8ff",
        }
        id_names = {
            5: "RED",
            6: "YELLOW",
            7: "GREEN",
            19: "BLUE",
        }

        for marker_id, center in containers.items():
            try:
                marker_id = int(marker_id)
                cx, cy = float(center[0]), float(center[1])
            except Exception:
                continue

            sx, sy = ws(cx, cy)
            half = 25.0*scale
            color = id_colors.get(marker_id, "#dddddd")

            self.canvas.create_rectangle(
                sx-half, sy-half,
                sx+half, sy+half,
                fill="#11151b",
                outline=color,
                width=4,
            )
            self.canvas.create_text(
                sx,
                sy,
                text=id_names.get(marker_id, f"ID{marker_id}"),
                fill=color,
                font=("TkDefaultFont", 9, "bold"),
            )

        # --------------------------------------------------
        # Robot base
        # --------------------------------------------------
        if isinstance(base, (list, tuple)) and len(base) >= 2:
            bx, by = float(base[0]), float(base[1])
            sx, sy = ws(bx, by)
            r = 20

            self.canvas.create_oval(
                sx-r, sy-r, sx+r, sy+r,
                fill="#ff9d42",
                outline="#fff3df",
                width=3,
            )
            self.canvas.create_text(
                sx,
                sy,
                text="BASE",
                fill="#111111",
                font=("TkDefaultFont", 8, "bold"),
            )

            if base_angle is not None:
                try:
                    a = math.radians(float(base_angle))
                    ex, ey = ws(
                        bx + 60.0*math.cos(a),
                        by + 60.0*math.sin(a),
                    )
                    self.canvas.create_line(
                        sx, sy, ex, ey,
                        fill="#ffb96f",
                        width=5,
                        arrow="last",
                    )
                except Exception:
                    pass

        # --------------------------------------------------
        # Cubes
        # --------------------------------------------------
        cube_colors = {
            "cube_red": "#ff5353",
            "cube_yellow": "#ffd84e",
            "cube_green": "#5ee06e",
            "cube_blue": "#58a8ff",
        }

        active_xy = (
            self._cube_xy(self.active_cube)
            if self.active_cube is not None
            else None
        )

        for cube in self.scene_snapshot:
            xy = self._cube_xy(cube)
            if xy is None:
                continue

            sx, sy = ws(*xy)
            cls = self._cube_class(cube)
            color = cube_colors.get(cls, "#ffffff")

            half = max(7.0, 9.5*scale)
            self.canvas.create_rectangle(
                sx-half, sy-half,
                sx+half, sy+half,
                fill=color,
                outline="#ffffff",
                width=2,
            )

            if active_xy is not None and math.hypot(
                xy[0]-active_xy[0],
                xy[1]-active_xy[1],
            ) <= 12.0:
                self.canvas.create_rectangle(
                    sx-half-7, sy-half-7,
                    sx+half+7, sy+half+7,
                    outline="#ffffff",
                    width=3,
                )
                self.canvas.create_text(
                    sx,
                    sy-half-17,
                    text="TARGET",
                    fill="#ffffff",
                    font=("TkDefaultFont", 9, "bold"),
                )

        # DONE
        for cube in self.done_cubes:
            xy = self._cube_xy(cube)
            if xy is None:
                continue

            sx, sy = ws(*xy)
            r = 12
            self.canvas.create_line(
                sx-r, sy-r, sx+r, sy+r,
                fill="#ffffff",
                width=3,
            )
            self.canvas.create_line(
                sx-r, sy+r, sx+r, sy-r,
                fill="#ffffff",
                width=3,
            )

        # --------------------------------------------------
        # LIVE hand bbox
        # --------------------------------------------------
        # World center is exact from Module 3.
        # Camera bbox orientation is ~90° relative to this world-map view,
        # therefore bbox width/height are intentionally SWAPPED here.
        camera_poly = shared.get("workspace_camera_shape")
        px_per_mm_x = px_per_mm_y = None

        try:
            if (
                isinstance(camera_poly, (list, tuple))
                and len(camera_poly) >= 4
            ):
                px_x = [float(p[0]) for p in camera_poly]
                px_y = [float(p[1]) for p in camera_poly]

                pspan_x = max(px_x)-min(px_x)
                pspan_y = max(px_y)-min(px_y)

                world_span_x = max(
                    1.0,
                    max(p[0] for p in workspace_points)
                    - min(p[0] for p in workspace_points),
                )
                world_span_y = max(
                    1.0,
                    max(p[1] for p in workspace_points)
                    - min(p[1] for p in workspace_points),
                )

                px_per_mm_x = pspan_x/world_span_x
                px_per_mm_y = pspan_y/world_span_y
        except Exception:
            pass

        for hand in hands:
            hxy = self._cube_xy(hand)
            if hxy is None:
                continue

            sx, sy = ws(*hxy)
            half_w_mm = 35.0
            half_h_mm = 35.0

            bbox = hand.get("bbox_px")

            if (
                isinstance(bbox, (list, tuple))
                and len(bbox) >= 4
                and px_per_mm_x
                and px_per_mm_y
            ):
                pixel_w = abs(float(bbox[2])-float(bbox[0]))
                pixel_h = abs(float(bbox[3])-float(bbox[1]))

                # 90° rotation relative to camera view:
                half_w_mm = max(
                    15.0,
                    pixel_h/px_per_mm_y/2.0,
                )
                half_h_mm = max(
                    15.0,
                    pixel_w/px_per_mm_x/2.0,
                )

            hw = half_w_mm*scale
            hh = half_h_mm*scale

            self.canvas.create_rectangle(
                sx-hw, sy-hh,
                sx+hw, sy+hh,
                outline="#ff58d3",
                width=4,
                dash=(8, 4),
            )
            self.canvas.create_text(
                sx-hw+5,
                sy-hh+5,
                anchor="nw",
                text="HAND",
                fill="#ff7cdd",
                font=("TkDefaultFont", 10, "bold"),
            )

    # --------------------------------------------------
    # Controls
    # --------------------------------------------------

    def start(self):
        module8 = self._find_live_module8()

        if module8 is None:
            self.status_var.set(
                "Module 10 ERROR: open Module 8 first."
            )
            return

        self.running = True
        self.error_latched = False
        self.safety_hand_active = False
        self.safety_hand_clear_started_at = None
        self.safety_home_after_error_active = False
        self.safety_resume_after_home = False
        self.state = "START_HOME"
        self.active_cube = None
        self.pending_native_cube = None
        self.module8_was_active = False
        self.hand_seen_frames = 0
        self.hand_clear_frames = 0
        self.change_wait_started_at = None
        self.hand_clear_started_at = None
        self.home_reason = "START"

        try:
            module8.stop_test_auto()
        except Exception:
            pass

        try:
            if not module8.go_home_direct():
                raise RuntimeError("direct HOME command rejected")
        except Exception as exc:
            self.running = False
            self.state = "ERROR"
            self.status_var.set(
                f"Module 10 ERROR: HOME start failed: {exc}"
            )
            return

        self.status_var.set(
            "Module 10: START -> HOME -> SCAN"
        )

    def stop(self):
        self.running = False
        self.error_latched = False
        self.safety_resume_after_home = False
        self.state = "IDLE"
        self.active_cube = None
        self.module8_was_active = False

        module8 = self._find_live_module8()
        if module8 is not None:
            try:
                module8.stop_test_auto()
            except Exception:
                pass

        self.status_var.set(
            "Module 10: STOPPED"
        )
        self._draw_scene()

    def rescan_now(self):
        self._freeze_scene()

        if self.running:
            self.state = "SELECT"

        self.status_var.set(
            "Module 10: scene rescanned"
        )

    def clear_done(self):
        self.done_cubes = []
        self.active_cube = None
        self._freeze_scene()
        self.status_var.set(
            "Module 10: DONE memory cleared"
        )

    # --------------------------------------------------
    # Module 8 handoff
    # --------------------------------------------------

    def _select_native_target(self, module8):
        """
        Native Module-8 candidates are the single source of truth for the
        robot target. This prevents the UI saying YELLOW while Module 8 later
        re-selects BLUE.
        """
        try:
            candidates = module8._live_cube_candidates()
        except Exception:
            return None

        if not candidates:
            return None

        mode = self.mode_var.get().strip().upper()

        if mode != "ALL":
            wanted = "cube_" + mode.lower()
            candidates = [
                c for c in candidates
                if self._cube_class(c) == wanted
            ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda c: float(c.get("confidence", 0.0)),
        )

    def _scene_cube_for_native(self, native_cube):
        if native_cube is None:
            return None

        nxy = self._cube_xy(native_cube)
        ncls = self._cube_class(native_cube)

        pool = [
            c for c in self.scene_snapshot
            if self._cube_class(c) == ncls
        ]

        if not pool:
            return dict(native_cube)

        if nxy is None:
            return pool[0]

        candidates = [
            (self._cube_xy(c), c)
            for c in pool
        ]
        candidates = [
            (xy, c)
            for xy, c in candidates
            if xy is not None
        ]

        if not candidates:
            return pool[0]

        return min(
            candidates,
            key=lambda item: math.hypot(
                item[0][0]-nxy[0],
                item[0][1]-nxy[1],
            ),
        )[1]

    def _prepare_one_cube_in_module8(self, module8):
        native = self._select_native_target(module8)

        if native is None:
            return False

        self.pending_native_cube = dict(native)
        self.active_cube = self._scene_cube_for_native(native)
        self._draw_scene()

        try:
            module8.stop_test_auto()
        except Exception:
            pass

        # Move to READY first. The exact target is locked only AFTER READY,
        # so Module 8 cannot enter its own SELECT state and change colors.
        try:
            if not module8.go_ready():
                self.status_var.set(
                    "Module 10 ERROR: Module 8 READY failed."
                )
                return False
        except Exception as exc:
            self.status_var.set(
                f"Module 10 ERROR: READY failed: {exc}"
            )
            return False

        self.state = "WAIT_READY_TARGET"
        self.status_var.set(
            "Module 10: READY for "
            + self._cube_class(native).replace("cube_", "").upper()
        )
        return True

    def _launch_forced_target(self, module8):
        native = self.pending_native_cube
        if native is None:
            return False

        try:
            module8._lock_test_cube(native)
            module8.test_auto_max_cubes_var.set(1)
            module8.test_auto_processed = []
            module8.test_auto_current = dict(native)

            # Start the exact same GO ABOVE path, but skip Module 8's SELECT.
            if not module8._start_manual_go_above():
                return False

            module8.test_auto_active = True
            module8.test_auto_state = "GO_ABOVE"
            module8.test_auto_started_at = time.monotonic()

            self.module8_was_active = True
            self.status_var.set(
                "Module 10: TARGET "
                + self._cube_class(native).replace("cube_", "").upper()
                + " -> Module 8"
            )
            return True

        except Exception as exc:
            self.status_var.set(
                f"Module 10 ERROR: forced target start failed: {exc}"
            )
            return False

    # --------------------------------------------------
    # Always-on hand monitoring / error recovery
    # --------------------------------------------------

    def _update_always_on_hand_safety(self, module8, hand):
        """
        Runs even when autonomous sorting has stopped with ERROR.

        Hand-interaction behavior:
            hand appears -> HOLD immediately
            hand remains -> keep waiting
            hand disappears -> debounce using Hand clear timer
            clear long enough -> DIRECT HOME
            hand reappears during HOME -> HOLD again
            clear again -> restart DIRECT HOME

        Returns True when this helper currently owns the robot flow and the
        normal autonomous state machine should not run this tick.
        """
        if module8 is None:
            return False

        # Hand entered at any time, including after a latched Module-8 error.
        if hand:
            # Remember whether this hand event interrupted a normal run.
            # Do NOT request automatic resume after a latched task error.
            # Any real hand interaction is treated as an operator reset:
            # after the hand clears we always go HOME, rescan, and continue.
            self.safety_resume_after_home = True

            if (
                not self.safety_hand_active
                or self.safety_home_after_error_active
            ):
                try:
                    module8.stop_test_auto()
                except Exception:
                    pass

                try:
                    module8.pause_motion_hold()
                except Exception:
                    pass

            self.safety_hand_active = True
            self.safety_hand_clear_started_at = None
            self.safety_home_after_error_active = False

            # Abort the current autonomous task, but do not lose safety logic.
            if self.running:
                self.running = False

            if self.error_latched:
                self.status_var.set(
                    "Module 10: ERROR latched | HAND -> HOLD | "
                    "waiting for hand to leave"
                )
            else:
                self.status_var.set(
                    "Module 10: HAND -> HOLD | waiting for hand to leave"
                )

            return True

        # Hand has disappeared after previously being present.
        if self.safety_hand_active:
            if self.safety_hand_clear_started_at is None:
                self.safety_hand_clear_started_at = time.monotonic()

            clear_delay = max(
                0.2,
                float(self.hand_clear_delay_var.get()),
            )
            clear_age = (
                time.monotonic()
                - self.safety_hand_clear_started_at
            )

            if clear_age < clear_delay:
                self.status_var.set(
                    f"Module 10: hand missing | confirm clear "
                    f"{clear_age:.1f}/{clear_delay:.1f}s"
                )
                return True

            # Stable clear -> direct HOME, even if sorting previously ERRORed.
            try:
                if not module8.go_home_direct():
                    raise RuntimeError(
                        "direct HOME command rejected"
                    )
            except Exception as exc:
                self.status_var.set(
                    f"Module 10 RECOVERY ERROR: HOME failed: {exc}"
                )
                return True

            self.safety_hand_active = False
            self.safety_hand_clear_started_at = None
            self.safety_home_after_error_active = True

            # Hand intervention is an explicit scene reset.
            self.error_latched = False

            self.status_var.set(
                "Module 10: hand clear -> DIRECT HOME -> RESCAN"
            )
            return True

        # HOME started by the safety recovery. Keep ownership until it finishes.
        if self.safety_home_after_error_active:
            if hand:
                # Normally caught above, but keep this explicit.
                try:
                    module8.pause_motion_hold()
                except Exception:
                    pass

                self.safety_home_after_error_active = False
                self.safety_hand_active = True
                self.safety_hand_clear_started_at = None
                self.status_var.set(
                    "Module 10: HAND during safety HOME -> HOLD"
                )
                return True

            if bool(
                getattr(
                    module8,
                    "home_direct_active",
                    False,
                )
            ):
                self.status_var.set(
                    "Module 10: safety DIRECT HOME moving"
                    + (" | previous task ERROR" if self.error_latched else "")
                )
                return True

            self.safety_home_after_error_active = False

            if self.safety_resume_after_home:
                self.safety_resume_after_home = False
                self.error_latched = False
                self.running = True
                self.done_cubes = []
                self.active_cube = None
                self.pending_native_cube = None
                self.change_wait_started_at = time.monotonic()
                self.state = "CHANGE_DELAY"

                self.status_var.set(
                    f"Module 10: HOME reached -> RESCAN in "
                    f"{float(self.rescan_delay_var.get()):.0f}s"
                )
                return True

            self.status_var.set(
                "Module 10: robot HOME | safety recovery complete"
            )
            return True

        return False

    # --------------------------------------------------
    # State machine
    # --------------------------------------------------

    def update_loop(self):
        try:
            module8 = self._find_live_module8()

            try:
                speed_pct = max(
                    20.0,
                    min(100.0, float(self.transport_speed_var.get())),
                )
                self.speed_value_label.configure(
                    text=f"{speed_pct:.0f}%"
                )
            except Exception:
                speed_pct = 70.0

            try:
                ready_home_pct = max(
                    20.0,
                    min(100.0, float(self.ready_home_speed_var.get())),
                )
                self.ready_home_speed_label.configure(
                    text=f"{ready_home_pct:.0f}%"
                )
            except Exception:
                ready_home_pct = 85.0

            if module8 is None:
                self.module8_var.set("Module 8: not open")
            else:
                if hasattr(module8, "transport_speed_scale"):
                    module8.transport_speed_scale = speed_pct / 100.0

                if hasattr(module8, "ready_home_speed_scale"):
                    module8.ready_home_speed_scale = (
                        ready_home_pct / 100.0
                    )

                self.module8_var.set(
                    "Module 8: "
                    + str(
                        getattr(
                            module8,
                            "auto_status_var",
                            tk.StringVar(value="ready"),
                        ).get()
                    )
                )

            hand = bool(
                self.app_context.shared_data.get(
                    "hand_inside_workspace",
                    False,
                )
            )

            # This runs even after a task ERROR or explicit autonomous stop.
            if self._update_always_on_hand_safety(
                module8,
                hand,
            ):
                return

            if not self.running:
                return

            if module8 is None:
                self.running = False
                self.state = "ERROR"
                self.status_var.set(
                    "Module 10 ERROR: Module 8 is not open."
                )
                return

            # --------------------------------------------------
            # START / HOME / SCAN
            # --------------------------------------------------
            if self.state == "START_HOME":
                if bool(
                    getattr(
                        module8,
                        "home_direct_active",
                        False,
                    )
                ):
                    self.status_var.set(
                        "Module 10: START -> moving HOME"
                    )
                    return

                self.state = "SCAN"
                self.status_var.set(
                    "Module 10: HOME reached -> SCAN"
                )
                return

            if self.state == "SCAN":
                self._freeze_scene()
                self.state = "SELECT"
                return

            # --------------------------------------------------
            # Select exact Module-8 target
            # --------------------------------------------------
            if self.state == "SELECT":
                if not self._prepare_one_cube_in_module8(module8):
                    self.state = "WAIT_HAND"
                    self.hand_seen_frames = 0
                    self.hand_clear_frames = 0
                    self.status_var.set(
                        "Module 10: no cubes -> waiting for hand/change"
                    )
                    return
                return

            if self.state == "WAIT_READY_TARGET":
                if bool(module8.ready_active):
                    return

                if not self._launch_forced_target(module8):
                    self.running = False
                    self.state = "ERROR"
                    return

                self.state = "WAIT_MODULE8"
                return

            # --------------------------------------------------
            # Wait for one exact cube task
            # --------------------------------------------------
            if self.state == "WAIT_MODULE8":
                active = bool(
                    getattr(module8, "test_auto_active", False)
                )

                if active:
                    self.module8_was_active = True
                    return

                if not self.module8_was_active:
                    return

                m8_state = str(
                    getattr(module8, "test_auto_state", "")
                )
                m8_status = str(
                    getattr(
                        module8,
                        "auto_status_var",
                        tk.StringVar(value=""),
                    ).get()
                )

                if "ERROR" in m8_status.upper() or m8_state == "ERROR":
                    self.running = False
                    self.error_latched = True
                    self.state = "ERROR"
                    self.status_var.set(
                        "Module 10: Module 8 failed | "
                        + m8_status
                        + " | hand safety still ACTIVE"
                    )
                    return

                if self.active_cube is not None:
                    self.done_cubes.append(
                        dict(self.active_cube)
                    )

                self.active_cube = None
                self.pending_native_cube = None
                self.module8_was_active = False

                # Module 8 ends in READY. Refresh detections and continue.
                self._freeze_scene()
                self.state = "SELECT"
                self.status_var.set(
                    f"Module 10: cube complete | DONE={len(self.done_cubes)}"
                )
                return

            # --------------------------------------------------
            # No cubes: wait for user hand/change.
            # --------------------------------------------------
            if self.state == "WAIT_HAND":
                if hand:
                    self.hand_seen_frames += 1
                    if self.hand_seen_frames >= self.HAND_CONFIRM_FRAMES:
                        self.state = "HAND_WAIT"
                        self.status_var.set(
                            "Module 10: hand confirmed -> robot stopped"
                        )
                    return

                self.hand_seen_frames = 0
                return

            # --------------------------------------------------
            # Hand safety/resume
            # --------------------------------------------------
            if self.state == "HAND_WAIT":
                if hand:
                    self.hand_clear_frames = 0
                    self.hand_clear_started_at = None
                    self.status_var.set(
                        "Module 10: HAND IN SCENE | robot HOLD"
                    )
                    return

                # YOLO can miss the hand for a frame or two.
                # Start/restart a real-time debounce period.
                if self.hand_clear_started_at is None:
                    self.hand_clear_started_at = time.monotonic()

                clear_delay = max(
                    0.2,
                    float(self.hand_clear_delay_var.get()),
                )
                clear_age = time.monotonic() - self.hand_clear_started_at

                self.status_var.set(
                    f"Module 10: hand missing | confirm clear "
                    f"{clear_age:.1f}/{clear_delay:.1f}s"
                )

                if clear_age < clear_delay:
                    return

                # Hand has been continuously absent long enough -> HOME.
                try:
                    if not module8.go_home_direct():
                        raise RuntimeError(
                            "direct HOME command rejected"
                        )
                except Exception as exc:
                    self.running = False
                    self.state = "ERROR"
                    self.status_var.set(
                        f"Module 10 ERROR: HOME after hand failed: {exc}"
                    )
                    return

                self.hand_clear_started_at = None
                self.home_reason = "HAND"
                self.state = "HAND_HOME"
                self.status_var.set(
                    "Module 10: hand confirmed clear -> HOME for rescan"
                )
                return

            if self.state == "HAND_HOME":
                if hand:
                    # Hand returned while HOME was moving:
                    # HOLD immediately, wait, then restart direct HOME after clear.
                    try:
                        module8.pause_motion_hold()
                    except Exception:
                        pass

                    self.state = "HAND_WAIT"
                    self.hand_clear_frames = 0
                    self.hand_clear_started_at = None
                    self.status_var.set(
                        "Module 10: HAND during HOME -> HOLD"
                    )
                    return

                if bool(
                    getattr(
                        module8,
                        "home_direct_active",
                        False,
                    )
                ):
                    self.status_var.set(
                        "Module 10: moving HOME for rescan"
                    )
                    return

                self.change_wait_started_at = time.monotonic()
                self.state = "CHANGE_DELAY"
                self.status_var.set(
                    f"Module 10: HOME reached -> rescan in "
                    f"{float(self.rescan_delay_var.get()):.0f}s"
                )
                return

            if self.state == "CHANGE_DELAY":
                if hand:
                    self.state = "HAND_WAIT"
                    self.hand_clear_frames = 0
                    self.hand_clear_started_at = None
                    self.change_wait_started_at = None
                    return

                delay = max(
                    0.0,
                    float(self.rescan_delay_var.get()),
                )
                age = time.monotonic() - float(
                    self.change_wait_started_at
                    or time.monotonic()
                )

                self.status_var.set(
                    f"Module 10: HOME | RESCAN in "
                    f"{max(0.0, delay-age):.1f}s"
                )

                if age >= delay:
                    # New physical scene after hand interaction = new session.
                    self.done_cubes = []
                    self.active_cube = None
                    self.pending_native_cube = None
                    self.state = "SCAN"
                    self.status_var.set(
                        "Module 10: RESCAN NOW"
                    )
                return

        except Exception as exc:
            self.running = False
            self.error_latched = True
            self.state = "ERROR"
            self.status_var.set(
                f"Module 10 ERROR: {exc} | hand safety still ACTIVE"
            )

        finally:
            try:
                self._draw_scene()
            except Exception:
                pass

            self.frame.after(
                self.UPDATE_MS,
                self.update_loop,
            )

    def get_frame(self):
        return self.frame

    def shutdown(self):
        self.running = False
