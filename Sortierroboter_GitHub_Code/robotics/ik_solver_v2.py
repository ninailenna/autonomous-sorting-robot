import math
from dataclasses import dataclass, field


# ================================================================
# HELPERS
# ================================================================

def clamp(
    value,
    minimum,
    maximum,
):
    return max(
        minimum,
        min(maximum, value),
    )


def normalize_angle_deg(
    angle,
):
    return (
        float(angle) + 180.0
    ) % 360.0 - 180.0


def angular_distance_deg(
    a,
    b,
):
    return abs(
        normalize_angle_deg(
            float(a) - float(b)
        )
    )


# ================================================================
# GEOMETRY
# ================================================================

@dataclass
class IKGeometry:

    # Shoulder -> Elbow
    l1: float = 116.0

    # Elbow -> Wrist
    l2: float = 135.0

    # Wrist motor axis -> tips of gripper jaws
    #
    # IMPORTANT:
    # L3 already ends at the physical gripper tip.
    l3: float = 165.0

    # Table -> shoulder_lift axis
    shoulder_height: float = 120.0

    # Horizontal offset:
    # shoulder_pan axis -> shoulder_lift axis
    base_to_shoulder_offset: float = 30.0


# ================================================================
# ANGLES
# ================================================================

@dataclass
class IKAngles:
    base: float
    shoulder: float
    elbow: float
    wrist: float


# ================================================================
# CANDIDATE
# ================================================================

@dataclass
class IKCandidate:

    angles: IKAngles

    # Absolute orientation of L3
    tool_angle_deg: float

    # Lower = better
    score: float

    # Debug geometry
    wrist_radial: float
    wrist_z_world: float

    elbow_z_world: float
    shoulder_z_world: float
    gripper_z_world: float

    elbow_configuration: str = ""


# ================================================================
# RESULT
# ================================================================

@dataclass
class IKResult:

    reachable: bool

    angles: IKAngles | None

    target_x: float
    target_y: float
    target_z: float

    radial_target: float

    wrist_radial: float
    wrist_z: float

    error_message: str = ""

    # All geometrically valid candidates
    candidates: list[IKCandidate] = field(
        default_factory=list
    )

    selected_tool_angle_deg: float | None = None

    tested_candidates: int = 0

    rejected_floor: int = 0

    rejected_reach: int = 0


# ================================================================
# SOLVER
# ================================================================

class IKSolverV2:
    """
    Multi-solution IK for SO-101.

    Coordinate system:

        X = right
        Y = forward
        Z = up

    FK convention:

        x = radial * sin(base)
        y = radial * cos(base)

    Therefore:

        base = atan2(x, y)

    Geometry:

        shoulder_pan axis
              |
              | base_to_shoulder_offset
              v
        shoulder_lift
              |
              L1
              |
            elbow
              |
              L2
              |
            wrist
              |
              L3
              |
        physical gripper tips

    L3 therefore ends at the actual gripper tips.

    Unlike the old solver, L3 is NOT forced to -90 degrees.

    The solver scans multiple absolute L3 orientations
    and both elbow configurations.
    """

    def __init__(
        self,
        geometry=None,
    ):

        self.geometry = (
            geometry
            or IKGeometry()
        )

        # ========================================================
        # TOOL ORIENTATION SEARCH
        # ========================================================

        # Absolute L3 angle relative to horizontal.
        #
        # 0°   = horizontal forward
        # -90° = vertically down
        # +90° = vertically up

        self.tool_angle_min_deg = -110.0
        self.tool_angle_max_deg = 30.0

        # 5° gives a good compromise for now.
        self.tool_angle_step_deg = 5.0

        # ========================================================
        # FLOOR
        # ========================================================

        self.table_z = 0.0

        # Since L3 ends at the real jaw tips,
        # this can be 0 mm.
        #
        # Use e.g. 2-5 mm later if we want safety margin.
        self.minimum_gripper_z = 0.0

        self.minimum_link_z = 0.0

        # ========================================================
        # SCORING
        # ========================================================

        # If current robot pose is unavailable,
        # prefer a somewhat downward tool orientation,
        # but DON'T force it.
        self.default_tool_angle_deg = -60.0

        # Small preference for negative elbow,
        # matching the useful physical configuration
        # we have seen so far.
        self.negative_elbow_bonus = 15.0

    # ============================================================
    # PUBLIC SOLVE
    # ============================================================

    def solve(
        self,
        x,
        y,
        z,
        current_angles=None,
    ):

        x = float(x)
        y = float(y)
        z = float(z)

        g = self.geometry

        # ========================================================
        # BASE
        # ========================================================

        base_angle = normalize_angle_deg(
            math.degrees(
                math.atan2(
                    x,
                    y,
                )
            )
        )

        radial_target = math.sqrt(
            x * x
            + y * y
        )

        # ========================================================
        # SEARCH ALL TOOL ANGLES
        # ========================================================

        candidates = []

        tested_candidates = 0

        rejected_floor = 0
        rejected_reach = 0

        tool_angle = (
            self.tool_angle_min_deg
        )

        while (
            tool_angle
            <= self.tool_angle_max_deg
            + 1e-9
        ):

            tool_rad = math.radians(
                tool_angle
            )

            # ----------------------------------------------------
            # L3 vector
            #
            # Wrist -> physical jaw tips
            # ----------------------------------------------------

            l3_radial = (
                g.l3
                * math.cos(
                    tool_rad
                )
            )

            l3_z = (
                g.l3
                * math.sin(
                    tool_rad
                )
            )

            # ----------------------------------------------------
            # Wrist point from base axis
            # ----------------------------------------------------

            wrist_radial_from_base = (
                radial_target
                - l3_radial
            )

            wrist_z_world = (
                z
                - l3_z
            )

            # ----------------------------------------------------
            # Convert to shoulder_lift coordinate system
            # ----------------------------------------------------

            wrist_radial = (
                wrist_radial_from_base
                - g.base_to_shoulder_offset
            )

            wrist_z_local = (
                wrist_z_world
                - g.shoulder_height
            )

            # ====================================================
            # L1 + L2 REACHABILITY
            # ====================================================

            distance_sq = (
                wrist_radial ** 2
                + wrist_z_local ** 2
            )

            distance = math.sqrt(
                distance_sq
            )

            max_reach = (
                g.l1 + g.l2
            )

            min_reach = abs(
                g.l1 - g.l2
            )

            if (
                distance
                > max_reach + 1e-6
                or
                distance
                < min_reach - 1e-6
            ):

                rejected_reach += 2

                tool_angle += (
                    self.tool_angle_step_deg
                )

                continue

            # ====================================================
            # ELBOW
            # ====================================================

            cos_elbow = (
                distance_sq
                - g.l1 ** 2
                - g.l2 ** 2
            ) / (
                2.0
                * g.l1
                * g.l2
            )

            cos_elbow = clamp(
                cos_elbow,
                -1.0,
                1.0,
            )

            elbow_abs = math.acos(
                cos_elbow
            )

            elbow_options = [
                (
                    elbow_abs,
                    "positive",
                ),
                (
                    -elbow_abs,
                    "negative",
                ),
            ]

            # ====================================================
            # BOTH ELBOW CONFIGURATIONS
            # ====================================================

            for (
                elbow_rad,
                elbow_configuration,
            ) in elbow_options:

                tested_candidates += 1

                # ------------------------------------------------
                # Shoulder
                # ------------------------------------------------

                shoulder_rad = (
                    math.atan2(
                        wrist_z_local,
                        wrist_radial,
                    )
                    - math.atan2(
                        g.l2
                        * math.sin(
                            elbow_rad
                        ),
                        g.l1
                        + g.l2
                        * math.cos(
                            elbow_rad
                        ),
                    )
                )

                shoulder_deg = (
                    math.degrees(
                        shoulder_rad
                    )
                )

                elbow_deg = (
                    math.degrees(
                        elbow_rad
                    )
                )

                # ------------------------------------------------
                # Wrist relative angle
                #
                # absolute L3 =
                # shoulder + elbow + wrist
                # ------------------------------------------------

                wrist_deg = (
                    tool_angle
                    - shoulder_deg
                    - elbow_deg
                )

                angles = IKAngles(

                    base=normalize_angle_deg(
                        base_angle
                    ),

                    shoulder=normalize_angle_deg(
                        shoulder_deg
                    ),

                    elbow=normalize_angle_deg(
                        elbow_deg
                    ),

                    wrist=normalize_angle_deg(
                        wrist_deg
                    ),
                )

                # =================================================
                # FORWARD GEOMETRY FOR FLOOR CHECK
                # =================================================

                shoulder_z_world = (
                    g.shoulder_height
                )

                elbow_z_world = (
                    g.shoulder_height
                    + g.l1
                    * math.sin(
                        shoulder_rad
                    )
                )

                forearm_angle_rad = (
                    shoulder_rad
                    + elbow_rad
                )

                calculated_wrist_z = (
                    elbow_z_world
                    + g.l2
                    * math.sin(
                        forearm_angle_rad
                    )
                )

                # Should be effectively the requested z.
                gripper_z_world = (
                    calculated_wrist_z
                    + g.l3
                    * math.sin(
                        tool_rad
                    )
                )

                # =================================================
                # FLOOR COLLISION
                # =================================================
                #
                # Links are straight segments.
                #
                # Z varies linearly along a straight link.
                # Therefore if both endpoints are >= floor,
                # the entire segment is >= floor.
                # =================================================

                floor_ok = self._floor_check(
                    shoulder_z_world=
                        shoulder_z_world,

                    elbow_z_world=
                        elbow_z_world,

                    wrist_z_world=
                        calculated_wrist_z,

                    gripper_z_world=
                        gripper_z_world,
                )

                if not floor_ok:

                    rejected_floor += 1

                    continue

                # =================================================
                # SCORE
                # =================================================

                score = self._score_candidate(
                    angles=angles,
                    tool_angle_deg=
                        tool_angle,
                    current_angles=
                        current_angles,
                    elbow_configuration=
                        elbow_configuration,
                )

                candidate = IKCandidate(

                    angles=angles,

                    tool_angle_deg=
                        float(tool_angle),

                    score=float(
                        score
                    ),

                    wrist_radial=float(
                        wrist_radial
                    ),

                    wrist_z_world=float(
                        calculated_wrist_z
                    ),

                    elbow_z_world=float(
                        elbow_z_world
                    ),

                    shoulder_z_world=float(
                        shoulder_z_world
                    ),

                    gripper_z_world=float(
                        gripper_z_world
                    ),

                    elbow_configuration=
                        elbow_configuration,
                )

                candidates.append(
                    candidate
                )

            tool_angle += (
                self.tool_angle_step_deg
            )

        # ========================================================
        # NO VALID SOLUTION
        # ========================================================

        if not candidates:

            return IKResult(

                reachable=False,

                angles=None,

                target_x=x,
                target_y=y,
                target_z=z,

                radial_target=
                    radial_target,

                wrist_radial=0.0,
                wrist_z=0.0,

                error_message=(
                    "No valid IK solution. "
                    f"Tested={tested_candidates}, "
                    f"floor rejected={rejected_floor}, "
                    f"reach rejected={rejected_reach}."
                ),

                candidates=[],

                selected_tool_angle_deg=None,

                tested_candidates=
                    tested_candidates,

                rejected_floor=
                    rejected_floor,

                rejected_reach=
                    rejected_reach,
            )

        # ========================================================
        # SORT BEST -> WORST
        # ========================================================

        candidates.sort(
            key=lambda candidate:
                candidate.score
        )

        selected = (
            candidates[0]
        )

        return IKResult(

            reachable=True,

            angles=
                selected.angles,

            target_x=x,
            target_y=y,
            target_z=z,

            radial_target=
                radial_target,

            wrist_radial=
                selected.wrist_radial,

            wrist_z=
                selected.wrist_z_world,

            error_message="",

            candidates=
                candidates,

            selected_tool_angle_deg=
                selected.tool_angle_deg,

            tested_candidates=
                tested_candidates,

            rejected_floor=
                rejected_floor,

            rejected_reach=
                rejected_reach,
        )

    # ============================================================
    # FLOOR CHECK
    # ============================================================

    def _floor_check(
        self,
        shoulder_z_world,
        elbow_z_world,
        wrist_z_world,
        gripper_z_world,
    ):

        if (
            shoulder_z_world
            < self.minimum_link_z
        ):
            return False

        if (
            elbow_z_world
            < self.minimum_link_z
        ):
            return False

        if (
            wrist_z_world
            < self.minimum_link_z
        ):
            return False

        # IMPORTANT:
        #
        # gripper point = actual jaw tips
        #
        # because L3 was measured:
        #
        # wrist axis -> tips of claws
        #
        if (
            gripper_z_world
            < self.minimum_gripper_z
        ):
            return False

        return True

    # ============================================================
    # SCORING
    # ============================================================

    def _score_candidate(
        self,
        angles,
        tool_angle_deg,
        current_angles,
        elbow_configuration,
    ):

        # --------------------------------------------------------
        # If robot state exists:
        # choose pose requiring least movement.
        # --------------------------------------------------------

        if current_angles is not None:

            score = (

                angular_distance_deg(
                    angles.base,
                    current_angles.base,
                )

                + angular_distance_deg(
                    angles.shoulder,
                    current_angles.shoulder,
                )

                + angular_distance_deg(
                    angles.elbow,
                    current_angles.elbow,
                )

                + angular_distance_deg(
                    angles.wrist,
                    current_angles.wrist,
                )
            )

            # Very small penalty for extreme tool orientation.
            score += (
                0.05
                * abs(
                    tool_angle_deg
                    - self.default_tool_angle_deg
                )
            )

            return score

        # --------------------------------------------------------
        # No current robot pose:
        #
        # prefer reasonable downward tool orientation,
        # but still allow the full search range.
        # --------------------------------------------------------

        score = abs(
            tool_angle_deg
            - self.default_tool_angle_deg
        )

        # Prefer negative elbow slightly.
        if (
            elbow_configuration
            == "negative"
        ):

            score -= (
                self.negative_elbow_bonus
            )

        # Avoid unnecessarily extreme individual joints.
        score += (
            0.10
            * abs(
                angles.shoulder
            )
        )

        score += (
            0.05
            * abs(
                angles.wrist
            )
        )

        return score


# ================================================================
# MOTOR MAPPING
# ================================================================

class MotorMapper:
    """
    Current motor -> model mapping.

    model_angle =
        sign * (motor_pos - neutral_pos)
        + offset

    inverse:

    motor_pos =
        neutral_pos
        + sign * (model_angle - offset)
    """

    def __init__(self):

        self.mapping = {

            "shoulder_pan.pos": {
                "sign": 1.0,
                "offset": 0.0,
            },

            "shoulder_lift.pos": {
                "sign": -1.0,
                "offset": 150.0,
            },

            "elbow_flex.pos": {
                "sign": -1.0,
                "offset": 210.0,
            },

            "wrist_flex.pos": {
                "sign": -1.0,
                "offset": -60.0,
            },
        }

    def angle_to_motor(
        self,
        joint,
        model_angle,
        neutral_pos,
    ):

        data = (
            self.mapping[
                joint
            ]
        )

        sign = float(
            data["sign"]
        )

        offset = float(
            data["offset"]
        )

        relative_motor = (
            sign
            * (
                float(model_angle)
                - offset
            )
        )

        target = (
            float(neutral_pos)
            + relative_motor
        )

        return normalize_angle_deg(
            target
        )

    def angles_to_motor_positions(
        self,
        angles,
        neutral_pose,
    ):

        result = {}

        angle_map = {

            "shoulder_pan.pos":
                angles.base,

            "shoulder_lift.pos":
                angles.shoulder,

            "elbow_flex.pos":
                angles.elbow,

            "wrist_flex.pos":
                angles.wrist,
        }

        for joint, angle in (
            angle_map.items()
        ):

            neutral = (
                neutral_pose.get(
                    joint
                )
            )

            if neutral is None:

                continue

            result[joint] = (
                self.angle_to_motor(
                    joint=joint,
                    model_angle=angle,
                    neutral_pos=neutral,
                )
            )

        return result


# ================================================================
# STANDALONE TEST
# ================================================================

if __name__ == "__main__":

    geometry = IKGeometry(

        l1=116.0,

        l2=135.0,

        l3=165.0,

        shoulder_height=120.0,

        base_to_shoulder_offset=30.0,
    )

    solver = IKSolverV2(
        geometry
    )

    # For table targets:
    solver.minimum_gripper_z = 0.0
    solver.minimum_link_z = 0.0

    targets = [
        (0, 200, 0),
        (50, 200, 0),
        (-50, 200, 0),
        (0, 250, 0),
        (0, 300, 10),
    ]

    print()
    print(
        "SO-101 Multi-Solution IK"
    )

    print(
        "L3 = wrist axis -> physical gripper tips"
    )

    for target in targets:

        result = solver.solve(
            x=target[0],
            y=target[1],
            z=target[2],
        )

        print()
        print(
            "=" * 70
        )

        print(
            "Target:",
            target,
        )

        print(
            "Reachable:",
            result.reachable,
        )

        print(
            "Candidates:",
            len(
                result.candidates
            ),
        )

        print(
            "Tested:",
            result.tested_candidates,
        )

        print(
            "Floor rejected:",
            result.rejected_floor,
        )

        print(
            "Reach rejected:",
            result.rejected_reach,
        )

        if result.reachable:

            print(
                "Selected tool angle:",
                result.selected_tool_angle_deg,
            )

            print(
                "Base:",
                result.angles.base,
            )

            print(
                "Shoulder:",
                result.angles.shoulder,
            )

            print(
                "Elbow:",
                result.angles.elbow,
            )

            print(
                "Wrist:",
                result.angles.wrist,
            )

            print()
            print(
                "Top 5 candidates:"
            )

            for candidate in (
                result.candidates[:5]
            ):

                print(
                    f"tool={candidate.tool_angle_deg:6.1f}° | "
                    f"B={candidate.angles.base:7.1f} "
                    f"S={candidate.angles.shoulder:7.1f} "
                    f"E={candidate.angles.elbow:7.1f} "
                    f"W={candidate.angles.wrist:7.1f} | "
                    f"score={candidate.score:.1f}"
                )

        else:

            print(
                result.error_message
            )
