import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = PROJECT_DIR / "data" / "main.json"


JOINT_TO_CALIB_NAME = {
    "shoulder_pan.pos": "shoulder_pan",
    "shoulder_lift.pos": "shoulder_lift",
    "elbow_flex.pos": "elbow_flex",
    "wrist_flex.pos": "wrist_flex",
    "wrist_roll.pos": "wrist_roll",
    "gripper.pos": "gripper",
}


def load_calibration(path=CALIBRATION_FILE):
    path = Path(path)

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_joint_limits(path=CALIBRATION_FILE):
    calibration = load_calibration(path)

    limits = {}

    for joint, calib_name in JOINT_TO_CALIB_NAME.items():
        if calib_name not in calibration:
            continue

        item = calibration[calib_name]

        limits[joint] = {
            "min": item.get("range_min"),
            "max": item.get("range_max"),
            "id": item.get("id"),
            "homing_offset": item.get("homing_offset"),
        }

    return limits


def clamp_joint(joint, value, limits):
    if joint not in limits:
        return value

    lo = limits[joint].get("min")
    hi = limits[joint].get("max")

    if lo is None or hi is None:
        return value

    lo = float(lo)
    hi = float(hi)

    if lo > hi:
        lo, hi = hi, lo

    return max(lo, min(hi, value))
