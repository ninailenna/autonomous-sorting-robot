import tkinter as tk
from pathlib import Path

from core.robot_worker import RobotWorker


PROJECT_DIR = Path(__file__).resolve().parent
MAIN_PANEL_FILE = PROJECT_DIR / "modules" / "6_robot_3d_panel.py"


class AppContext:
    """
    Shared application services/state.

    Main GUI intentionally stays small:
    - one RobotWorker
    - one shared_data dictionary
    - logging callback
    - the main 3D Control Center owns module launching
    """

    def __init__(self, app):
        self.app = app

        self.robot_worker = RobotWorker()
        self.robot_worker.start()

        self.shared_data = {
            # Camera / scene
            "workspace_polygon_world": None,
            "workspace_bounds": None,
            "workspace_camera_shape": None,

            "robot_base_world": None,
            "robot_base_marker_angle_deg": None,

            "containers_world": {},
            "cubes_world": [],

            "marker_target_world": None,

            # Optional mouse-selected target from camera panel
            "target_world": None,
            "target_inside_workspace": False,

            # Active IK target source: "ID8" or "Mouse"
            "ik_target_source": "ID8",

            # IK / model
            "ik_target_z": 50.0,
            "ik_solution_angles": None,

            # Published calibration
            "robot_model_calibration": None,
            "robot_frame_calibration": None,
        }

    def log(self, message):
        self.app.log(message)


class MainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SO-101 Control Center")
        self.root.geometry("1450x900")
        self.root.minsize(760, 480)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.context = AppContext(self)
        self.main_panel = None

        self._load_main_panel()

    def _load_main_panel(self):
        import importlib.util

        if not MAIN_PANEL_FILE.exists():
            raise FileNotFoundError(f"Main panel not found: {MAIN_PANEL_FILE}")

        spec = importlib.util.spec_from_file_location(
            "so101_main_control_center",
            MAIN_PANEL_FILE,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not create module spec for main panel.")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        panel_class = getattr(module, "ModulePanel", None)
        if panel_class is None:
            raise RuntimeError("Main panel file has no ModulePanel class.")

        self.main_panel = panel_class(self.root, self.context)
        frame = (
            self.main_panel.get_frame()
            if hasattr(self.main_panel, "get_frame")
            else self.main_panel.frame
        )
        frame.pack(fill="both", expand=True)

    def log(self, message):
        # Main window has no separate log widget anymore.
        # Keep a single logging API for all modules.
        print(str(message))

    def on_close(self):
        try:
            if self.main_panel is not None and hasattr(self.main_panel, "shutdown"):
                self.main_panel.shutdown()
        except Exception as error:
            self.log(f"[shutdown] Main panel: {error}")

        try:
            self.context.robot_worker.shutdown()
        except Exception as error:
            self.log(f"[shutdown] Robot worker: {error}")

        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MainGUI(root)
    root.mainloop()
