from dataclasses import dataclass, field
from typing import Any


@dataclass
class RobotCommand:
    type: str
    data: dict[str, Any] = field(default_factory=dict)


def connect(
    port: str = "COM3",
    robot_id: str = "main",
) -> RobotCommand:
    return RobotCommand(
        type="connect",
        data={
            "port": port,
            "robot_id": robot_id,
        },
    )


def disconnect() -> RobotCommand:
    return RobotCommand(
        type="disconnect"
    )


def stop() -> RobotCommand:
    return RobotCommand(
        type="stop"
    )


def hold_current() -> RobotCommand:
    return RobotCommand(
        type="hold_current"
    )


def free_move() -> RobotCommand:
    return RobotCommand(
        type="free_move"
    )


def enable_torque() -> RobotCommand:
    return RobotCommand(
        type="enable_torque"
    )


def move_joint_delta(
    joint: str,
    delta: float,
) -> RobotCommand:
    return RobotCommand(
        type="move_joint_delta",
        data={
            "joint": joint,
            "delta": delta,
        },
    )


def move_to_positions(
    positions: dict[str, float],
) -> RobotCommand:
    return RobotCommand(
        type="move_to_positions",
        data={
            "positions": positions,
        },
    )


def request_state() -> RobotCommand:
    return RobotCommand(
        type="request_state"
    )
