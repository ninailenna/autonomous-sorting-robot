import math
from dataclasses import dataclass


# ============================================================
# ROBOT GEOMETRY
# ============================================================

@dataclass
class RobotGeometry:
    # Shoulder -> Elbow
    l1: float = 116.0

    # Elbow -> Wrist
    l2: float = 135.0

    # Wrist -> Gripper center
    l3: float = 165.0

    # Table / base rotation axis -> shoulder axis height
    base_height: float = 120.0

    # Horizontal distance between:
    #
    # shoulder_pan rotation axis
    # and
    # shoulder_lift rotation axis
    #
    # At base=0 this offset points along +Y.
    base_to_shoulder_offset: float = 30.0


# ============================================================
# JOINT ANGLES
# ============================================================

@dataclass
class JointAngles:
    base: float = 0.0
    shoulder: float = 0.0
    elbow: float = 0.0
    wrist: float = 0.0


# ============================================================
# HELPERS
# ============================================================

def deg_to_rad(value: float) -> float:
    return math.radians(value)


def radial_to_xyz(
    radial: float,
    z: float,
    base_angle: float,
) -> tuple[float, float, float]:
    """
    Convert radial distance in the arm's vertical plane
    into XYZ coordinates.

    Coordinate system:

        X = right
        Y = forward
        Z = up

    With base=0°, the arm points along +Y.
    """

    x = radial * math.sin(base_angle)
    y = radial * math.cos(base_angle)

    return x, y, z


# ============================================================
# FORWARD KINEMATICS
# ============================================================

def forward_kinematics(
    angles: JointAngles,
    geometry: RobotGeometry,
) -> dict:
    """
    Forward kinematics for the SO-101 style arm.

    Coordinate system:

        X = right
        Y = forward
        Z = up

    Joint interpretation:

        base
            Rotation around the global Z axis.

        shoulder
            Absolute angle of L1 relative to horizontal.

        elbow
            Relative rotation of L2 relative to L1.

        wrist
            Relative rotation of L3 relative to L2.

    Important geometry:

        The shoulder_lift axis is NOT directly above
        the shoulder_pan axis.

        There is a horizontal radial offset:

            base_to_shoulder_offset

        Therefore the shoulder axis moves on a circle
        around the base axis when shoulder_pan rotates.
    """

    # --------------------------------------------------------
    # Angles
    # --------------------------------------------------------

    base = deg_to_rad(
        angles.base
    )

    shoulder = deg_to_rad(
        angles.shoulder
    )

    elbow = deg_to_rad(
        angles.elbow
    )

    wrist = deg_to_rad(
        angles.wrist
    )

    # ========================================================
    # BASE
    # ========================================================

    # Physical rotation axis of shoulder_pan.
    base_point = (
        0.0,
        0.0,
        0.0,
    )

    # ========================================================
    # SHOULDER OFFSET
    # ========================================================

    # The shoulder_lift axis sits at a horizontal radial
    # distance from the shoulder_pan axis.
    #
    # At base=0:
    #
    #     shoulder X = 0
    #     shoulder Y = +offset
    #
    # When base rotates, this point rotates around Z.

    shoulder_point = radial_to_xyz(
        radial=geometry.base_to_shoulder_offset,
        z=geometry.base_height,
        base_angle=base,
    )

    # ========================================================
    # ELBOW
    # ========================================================

    # Radial distance from SHOULDER axis produced by L1.
    l1_radial = (
        geometry.l1
        * math.cos(shoulder)
    )

    # Total radial distance measured from BASE rotation axis.
    radial_elbow = (
        geometry.base_to_shoulder_offset
        + l1_radial
    )

    z_elbow = (
        geometry.base_height
        + geometry.l1
        * math.sin(shoulder)
    )

    elbow_point = radial_to_xyz(
        radial=radial_elbow,
        z=z_elbow,
        base_angle=base,
    )

    # ========================================================
    # WRIST
    # ========================================================

    forearm_angle = (
        shoulder
        + elbow
    )

    l2_radial = (
        geometry.l2
        * math.cos(forearm_angle)
    )

    radial_wrist = (
        radial_elbow
        + l2_radial
    )

    z_wrist = (
        z_elbow
        + geometry.l2
        * math.sin(forearm_angle)
    )

    wrist_point = radial_to_xyz(
        radial=radial_wrist,
        z=z_wrist,
        base_angle=base,
    )

    # ========================================================
    # GRIPPER
    # ========================================================

    gripper_angle = (
        forearm_angle
        + wrist
    )

    l3_radial = (
        geometry.l3
        * math.cos(gripper_angle)
    )

    radial_gripper = (
        radial_wrist
        + l3_radial
    )

    z_gripper = (
        z_wrist
        + geometry.l3
        * math.sin(gripper_angle)
    )

    gripper_point = radial_to_xyz(
        radial=radial_gripper,
        z=z_gripper,
        base_angle=base,
    )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "base": base_point,

        "shoulder": shoulder_point,

        "elbow": elbow_point,

        "wrist": wrist_point,

        "gripper": gripper_point,

        "angles": {
            "base":
                angles.base,

            "shoulder":
                angles.shoulder,

            "elbow":
                angles.elbow,

            "wrist":
                angles.wrist,
        },

        "absolute_angles": {
            "shoulder":
                angles.shoulder,

            "forearm":
                angles.shoulder
                + angles.elbow,

            "gripper":
                angles.shoulder
                + angles.elbow
                + angles.wrist,
        },

        # Useful for debugging / visualization.
        "geometry_debug": {
            "base_to_shoulder_offset":
                geometry.base_to_shoulder_offset,

            "radial_elbow":
                radial_elbow,

            "radial_wrist":
                radial_wrist,

            "radial_gripper":
                radial_gripper,
        },
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    geometry = RobotGeometry(
        l1=116.0,
        l2=135.0,
        l3=165.0,
        base_height=120.0,
        base_to_shoulder_offset=30.0,
    )

    angles = JointAngles(
        base=0.0,
        shoulder=30.0,
        elbow=-60.0,
        wrist=-60.0,
    )

    result = forward_kinematics(
        angles,
        geometry,
    )

    print(
        "\nSO-101 Forward Kinematics\n"
    )

    for name in [
        "base",
        "shoulder",
        "elbow",
        "wrist",
        "gripper",
    ]:

        print(
            f"{name:10s}",
            result[name],
        )

    print(
        "\nGeometry debug:"
    )

    for key, value in (
        result[
            "geometry_debug"
        ].items()
    ):

        print(
            f"{key:28s}",
            value,
        )
