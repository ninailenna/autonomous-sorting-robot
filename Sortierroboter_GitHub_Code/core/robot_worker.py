import queue
import threading
import time
from typing import Any

from robotics.calibration import (
    load_joint_limits,
    clamp_joint,
)

from core.commands import RobotCommand


# ============================================================
# ROBOT JOINTS
# ============================================================

JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


# ============================================================
# ROBOT WORKER
# ============================================================

class RobotWorker:

    def __init__(self):

        # --------------------------------------------------------
        # Communication
        # --------------------------------------------------------

        self.command_queue = queue.Queue()
        self.state_queue = queue.Queue()

        # --------------------------------------------------------
        # Thread
        # --------------------------------------------------------

        self.running = False
        self.thread = None

        # --------------------------------------------------------
        # Robot
        # --------------------------------------------------------

        self.robot = None
        self.connected = False

        # True = motors powered / holding
        # False = back-drivable
        self.torque_enabled = False

        # --------------------------------------------------------
        # Positions
        # --------------------------------------------------------

        self.current_positions = {}

        self.commanded_positions = {}

        # --------------------------------------------------------
        # Limits
        # --------------------------------------------------------

        self.joint_limits = (
            load_joint_limits()
        )

        # --------------------------------------------------------
        # Latest public state
        # --------------------------------------------------------

        self.last_state = {
            "connected": False,
            "torque_enabled": False,
            "positions": {},
            "commanded_positions": {},
        }

    # ============================================================
    # THREAD
    # ============================================================

    def start(self):

        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )

        self.thread.start()

    def shutdown(self):

        self.running = False

        self._safe_disconnect()

    # ============================================================
    # PUBLIC COMMAND API
    # ============================================================

    def send(
        self,
        command: RobotCommand,
    ):

        self.command_queue.put(
            command
        )

    # ============================================================
    # PUBLIC STATE API
    # ============================================================

    def get_latest_state(self):

        latest = None

        while not self.state_queue.empty():

            latest = (
                self.state_queue.get()
            )

        return latest

    def get_state_snapshot(self):

        return (
            self.last_state.copy()
        )

    # ============================================================
    # MAIN LOOP
    # ============================================================

    def _loop(self):

        while self.running:

            try:

                # ----------------------------------------------
                # Handle commands
                # ----------------------------------------------

                self._process_commands()

                # ----------------------------------------------
                # Update robot observation
                #
                # IMPORTANT:
                # Reading still works while torque is disabled.
                # This is needed for manual calibration.
                # ----------------------------------------------

                if self.connected:

                    self._read_state()

                time.sleep(0.08)

            except Exception as error:

                self._publish_state(
                    error=str(error)
                )

                time.sleep(0.3)

    # ============================================================
    # COMMAND PROCESSING
    # ============================================================

    def _process_commands(self):

        while not self.command_queue.empty():

            command = (
                self.command_queue.get()
            )

            self._handle_command(
                command
            )

    def _handle_command(
        self,
        command: RobotCommand,
    ):

        # --------------------------------------------------------
        # CONNECT
        # --------------------------------------------------------

        if command.type == "connect":

            self._connect(
                port=command.data.get(
                    "port",
                    "COM3",
                ),
                robot_id=command.data.get(
                    "robot_id",
                    "main",
                ),
            )

        # --------------------------------------------------------
        # DISCONNECT
        # --------------------------------------------------------

        elif command.type == "disconnect":

            self._safe_disconnect()

        # --------------------------------------------------------
        # STOP / HOLD
        # --------------------------------------------------------

        elif command.type in (
            "stop",
            "hold_current",
        ):

            self._hold_current()

        # --------------------------------------------------------
        # FREE MOVE
        # --------------------------------------------------------

        elif command.type == "free_move":

            self._free_move()

        # --------------------------------------------------------
        # ENABLE TORQUE
        # --------------------------------------------------------

        elif command.type == "enable_torque":

            self._enable_torque()

        # --------------------------------------------------------
        # MOVE ONE JOINT
        # --------------------------------------------------------

        elif command.type == "move_joint_delta":

            self._move_joint_delta(
                joint=
                    command.data["joint"],

                delta=float(
                    command.data["delta"]
                ),
            )

        # --------------------------------------------------------
        # MOVE MULTIPLE JOINTS
        # --------------------------------------------------------

        elif command.type == "move_to_positions":

            self._move_to_positions(
                command.data[
                    "positions"
                ]
            )

        # --------------------------------------------------------
        # REQUEST STATE
        # --------------------------------------------------------

        elif command.type == "request_state":

            self._read_state()

        # --------------------------------------------------------
        # UNKNOWN COMMAND
        # --------------------------------------------------------

        else:

            self._publish_state(
                error=(
                    "Unknown command: "
                    f"{command.type}"
                )
            )

    # ============================================================
    # CONNECT
    # ============================================================

    def _connect(
        self,
        port: str,
        robot_id: str,
    ):

        if self.connected:

            self._publish_state(
                info="Already connected"
            )

            return

        try:

            # Import only when actually needed.
            from lerobot.robots.so_follower.config_so_follower import (
                SO101FollowerConfig,
            )

            from lerobot.robots.utils import (
                make_robot_from_config,
            )

            cfg = SO101FollowerConfig(
                port=port,
                id=robot_id,
            )

            self.robot = (
                make_robot_from_config(
                    cfg
                )
            )

            # May ask for ENTER in terminal.
            self.robot.connect()

            self.connected = True

            # Robot normally connects with torque active.
            self.torque_enabled = True

            # Reload calibration every connection.
            self.joint_limits = (
                load_joint_limits()
            )

            self._publish_state(
                info=(
                    "Loaded calibration limits: "
                    f"{len(self.joint_limits)} joints"
                )
            )

            # Read actual encoder state.
            self._read_state()

            # Start command target exactly at
            # the real current robot pose.
            self.commanded_positions = (
                self.current_positions.copy()
            )

            self._publish_state(
                info=(
                    f"Connected on {port}"
                )
            )

        except Exception as error:

            self.robot = None
            self.connected = False
            self.torque_enabled = False

            self.current_positions = {}
            self.commanded_positions = {}

            self._publish_state(
                error=(
                    "Connect failed: "
                    f"{error}"
                )
            )

    # ============================================================
    # DISCONNECT
    # ============================================================

    def _safe_disconnect(self):

        try:

            if self.robot is not None:

                self.robot.disconnect()

        except Exception:

            pass

        self.robot = None

        self.connected = False
        self.torque_enabled = False

        self.current_positions = {}
        self.commanded_positions = {}

        self._publish_state(
            info="Disconnected"
        )

    # ============================================================
    # READ ROBOT
    # ============================================================

    def _read_state(self):

        if (
            not self.connected
            or self.robot is None
        ):

            self._publish_state()

            return

        try:

            observation = (
                self.robot
                .get_observation()
            )

            self.current_positions = {
                joint: float(
                    observation[joint]
                )
                for joint in JOINTS
                if joint in observation
            }

            self._publish_state()

        except Exception as error:

            self._publish_state(
                error=(
                    "Read state failed: "
                    f"{error}"
                )
            )

    # ============================================================
    # MOTOR BUS
    # ============================================================

    def _get_motor_bus(self):
        """
        SO101Follower currently exposes:

            self.robot.bus

        Keep a small fallback list so this worker
        remains usable if the wrapper changes.
        """

        if self.robot is None:

            return None

        for attribute_name in (
            "bus",
            "motors_bus",
            "motor_bus",
        ):

            if hasattr(
                self.robot,
                attribute_name,
            ):

                return getattr(
                    self.robot,
                    attribute_name,
                )

        return None

    # ============================================================
    # FREE MOVE
    # ============================================================

    def _free_move(self):
        if not self.connected or self.robot is None:
            self._publish_state(error="Robot not connected")
            return

        try:
            bus = self.robot.bus

            bus.disable_torque()

            self.torque_enabled = False

            self._publish_state(
                info="Free Move enabled - torque disabled"
            )

        except Exception as e:
            self._publish_state(
                error=f"Free Move failed: {e}"
            )

    # ============================================================
    # ENABLE TORQUE
    # ============================================================

    def _enable_torque(self):
        """
        Re-enable torque without jumping back
        to an old commanded position.

        Procedure:

        1. Read current manually-positioned pose.
        2. Enable torque.
        3. Command exactly that current pose.
        """

        if (
            not self.connected
            or self.robot is None
        ):

            self._publish_state(
                error="Robot not connected"
            )

            return

        try:

            bus = (
                self._get_motor_bus()
            )

            if bus is None:

                raise RuntimeError(
                    "Motor bus not found"
                )

            # Read pose BEFORE torque is enabled.
            self._read_state()

            hold_target = (
                self.current_positions.copy()
            )

            # Turn motors back on.
            bus.enable_torque()

            self.torque_enabled = True

            # Hold exactly where user left the arm.
            if hold_target:

                hold_target = (
                    self._apply_limits(
                        hold_target
                    )
                )

                self.robot.send_action(
                    hold_target.copy()
                )

                self.commanded_positions = (
                    hold_target.copy()
                )

            self._publish_state(
                info=(
                    "Torque enabled — "
                    "holding current position"
                )
            )

        except Exception as error:

            self._publish_state(
                error=(
                    "Enable torque failed: "
                    f"{error}"
                )
            )

    # ============================================================
    # HOLD
    # ============================================================

    def _hold_current(self):
        if not self.connected or self.robot is None:
            self._publish_state(error="Robot not connected")
            return

        try:
            bus = self.robot.bus

            # IMPORTANT:
            # Do NOT send_action() here.
            # Just enable torque at the current physical motor positions.
            bus.enable_torque()

            self.torque_enabled = True

            # Read the actual positions AFTER torque is enabled.
            self._read_state()

            # Synchronize our software target with the real arm.
            # But DO NOT send it to the motors.
            self.commanded_positions = self.current_positions.copy()

            self._publish_state(
                info="Torque enabled at current physical position"
            )

        except Exception as e:
            self._publish_state(
                error=f"Hold failed: {e}"
            )

    # ============================================================
    # LIMITS
    # ============================================================

    def _apply_limits(
        self,
        target,
    ):

        safe_target = (
            target.copy()
        )

        for joint, value in (
            safe_target.items()
        ):

            safe_target[joint] = (
                clamp_joint(
                    joint,
                    value,
                    self.joint_limits,
                )
            )

        return safe_target

    # ============================================================
    # MOVE JOINT DELTA
    # ============================================================

    def _move_joint_delta(
        self,
        joint: str,
        delta: float,
    ):

        if (
            not self.connected
            or self.robot is None
        ):

            self._publish_state(
                error="Robot not connected"
            )

            return

        if not self.torque_enabled:

            self._publish_state(
                error=(
                    "Cannot move robot while "
                    "Free Move is active"
                )
            )

            return

        if joint not in JOINTS:

            self._publish_state(
                error=(
                    f"Unknown joint: {joint}"
                )
            )

            return

        try:

            # ----------------------------------------------------
            # Initialize target
            # ----------------------------------------------------

            if not self.commanded_positions:

                self._read_state()

                self.commanded_positions = (
                    self.current_positions.copy()
                )

            target = (
                self.commanded_positions.copy()
            )

            if joint not in target:

                self._publish_state(
                    error=(
                        "Joint not available: "
                        f"{joint}"
                    )
                )

                return

            # ----------------------------------------------------
            # Apply delta
            # ----------------------------------------------------

            target[joint] += float(
                delta
            )

            # ----------------------------------------------------
            # SAFETY LIMITS
            # ----------------------------------------------------

            target = (
                self._apply_limits(
                    target
                )
            )

            # ----------------------------------------------------
            # Send
            # ----------------------------------------------------

            self.robot.send_action(
                target.copy()
            )

            # Target remains independent
            # from encoder observation.
            self.commanded_positions = (
                target.copy()
            )

            time.sleep(0.03)

            self._read_state()

            self._publish_state(
                info=(
                    f"Moved {joint} "
                    f"by {delta}"
                )
            )

        except Exception as error:

            self._publish_state(
                error=(
                    "Move joint failed: "
                    f"{error}"
                )
            )

    # ============================================================
    # MOVE TO POSITIONS
    # ============================================================

    def _move_to_positions(
        self,
        positions,
    ):

        if (
            not self.connected
            or self.robot is None
        ):
            self._publish_state(
                error="Robot not connected"
            )
            return

        if not self.commanded_positions:

            self._read_state()

            self.commanded_positions = (
                self.current_positions.copy()
            )

        target = (
            self.commanded_positions.copy()
        )

        for joint, value in positions.items():

            if joint in target:

                target[joint] = float(
                    value
                )

        self.robot.send_action(
            target.copy()
        )

        self.commanded_positions = (
            target.copy()
        )

        time.sleep(0.03)

        self._read_state()

        self._publish_state(
            info="Moved to positions"
        )

    # ============================================================
    # PUBLISH STATE
    # ============================================================

    def _publish_state(
        self,
        info: str | None = None,
        error: str | None = None,
    ):

        state: dict[str, Any] = {

            "connected":
                self.connected,

            "torque_enabled":
                self.torque_enabled,

            "positions":
                self.current_positions.copy(),

            "commanded_positions":
                self.commanded_positions.copy(),
        }

        if info:

            state["info"] = info

        if error:

            state["error"] = error

        self.last_state = (
            state.copy()
        )

        self.state_queue.put(
            state
        )
