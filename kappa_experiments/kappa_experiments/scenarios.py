"""Predefined corridor scenarios for the lab floor.

Ported from kappa-motion-planner/experiments/bicycle_journal_paper/
maps_for_robot_experiments.py (develop branch). Geometry is unchanged; only the
plotting was removed and the four experiments are exposed through a registry so
the ROS node can select one by number.

All coordinates are in the projector / Vive frame ("map").
"""
from dataclasses import dataclass, field
from math import cos, pi, sin
from typing import Callable, Dict, List

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.corridor_geometry import get_corridor_from_vector
from kappa_planner.helpers.poses import compute_end_pose, compute_start_pose
from kappa_planner.vehicle import Unicycle

# ---------------------------------------------------------------------------
# Floor projection bounds and robot parameters
# ---------------------------------------------------------------------------
X_MIN = -0.9
X_MAX = 2.65
Y_MIN = -1.25
Y_MAX = 4.9

FLOOR_CLEARANCE = 0.20

ROBOT_V_MAX = 0.5
ROBOT_V_MIN = 0.0
ROBOT_OMEGA_MAX = 2.0
ROBOT_OMEGA_MIN = -2.0
ROBOT_WIDTH = 0.34
ROBOT_LENGTH = 0.34


def build_unicycle(v_max=ROBOT_V_MAX, v_min=ROBOT_V_MIN,
                   omega_max=ROBOT_OMEGA_MAX, omega_min=ROBOT_OMEGA_MIN,
                   width=ROBOT_WIDTH, length=ROBOT_LENGTH) -> Unicycle:
    unicycle = Unicycle(model='Rosbot circular')
    unicycle.update(
        width=width,
        length=length,
        v_max=v_max,
        v_min=v_min,
        omega_max=omega_max,
        omega_min=omega_min,
    )
    return unicycle


def robot_radius(unicycle: Unicycle) -> float:
    return 0.5 * max(unicycle.width, unicycle.length)


def turning_radius(unicycle: Unicycle) -> float:
    return unicycle.v_max / unicycle.omega_max


@dataclass
class Scenario:
    number: int
    title: str
    corridor_list: List[CorridorWorld]
    start_pose: List[float]
    goal_pose: List[float]
    notes: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bounds check (unchanged)
# ---------------------------------------------------------------------------
def assert_experiment_fits_floor(corridor_list, initial_pose, final_pose, radius):
    for idx, corridor in enumerate(corridor_list, start=1):
        x_values = corridor.corners[:, 0]
        y_values = corridor.corners[:, 1]
        if (x_values.min() < X_MIN or x_values.max() > X_MAX or
                y_values.min() < Y_MIN or y_values.max() > Y_MAX):
            raise ValueError(f'Corridor {idx} exceeds the projected floor bounds.')

    for pose_name, pose in (('start', initial_pose), ('goal', final_pose)):
        x, y, _ = pose
        if (x - radius < X_MIN or x + radius > X_MAX or
                y - radius < Y_MIN or y + radius > Y_MAX):
            raise ValueError(f'The {pose_name} pose footprint exceeds the projected floor bounds.')


# ---------------------------------------------------------------------------
# Shared geometry for experiments 1 and 2
# ---------------------------------------------------------------------------
def _right_angle_corridors(width=0.90):
    right_clearance = FLOOR_CLEARANCE

    horizontal_y = Y_MIN + FLOOR_CLEARANCE + 0.5 * width
    horizontal_start_x = X_MIN + FLOOR_CLEARANCE
    vertical_center_x = X_MAX - right_clearance - 0.5 * width
    horizontal_end_x = vertical_center_x + 0.5 * width

    horizontal_length = horizontal_end_x - horizontal_start_x
    corridor1 = CorridorWorld(
        width=width,
        height=horizontal_length,
        center=[horizontal_start_x + 0.5 * horizontal_length, horizontal_y],
        tilt=0.0,
    )

    vertical_tail_y = horizontal_y - 0.5 * width
    vertical_top_y = Y_MAX - FLOOR_CLEARANCE - 0.5 * width - FLOOR_CLEARANCE
    vertical_length = vertical_top_y - vertical_tail_y
    corridor2 = CorridorWorld(
        width=width,
        height=vertical_length,
        center=[vertical_center_x, vertical_tail_y + 0.5 * vertical_length],
        tilt=pi / 2,
    )
    return [corridor1, corridor2]


# ---------------------------------------------------------------------------
# Experiment 1: right-angle turn, start/goal far from the junction
# ---------------------------------------------------------------------------
def experiment_1(unicycle: Unicycle) -> Scenario:
    corridor_list = _right_angle_corridors()
    initial_pose = [-0.15, -0.35, -3 * pi / 4]
    final_pose = [2.1, 3.1, -pi / 6]
    assert_experiment_fits_floor(corridor_list, initial_pose, final_pose, robot_radius(unicycle))
    return Scenario(1, 'Experiment 1 - Right-Angle Turn', corridor_list, initial_pose, final_pose)


# ---------------------------------------------------------------------------
# Experiment 2: same corridors, start/goal close to the junction circle
# ---------------------------------------------------------------------------
def experiment_2(unicycle: Unicycle) -> Scenario:
    corridor_list = _right_angle_corridors()
    initial_pose = [1.33, -0.54, pi /2 ] #3 * pi / 4]
    final_pose = [1.92, 0.62, pi]
    assert_experiment_fits_floor(corridor_list, initial_pose, final_pose, robot_radius(unicycle))
    return Scenario(2, 'Experiment 2 - Right-Angle Turn (overlapping circles)',
                    corridor_list, initial_pose, final_pose)


# ---------------------------------------------------------------------------
# Experiment 3: multiple corridors with different orientations
# ---------------------------------------------------------------------------
# def experiment_3(unicycle: Unicycle) -> Scenario:
#     width = 0.80
#     add_height = 0.45

#     phi1 = 0.0
#     phi2 = pi / 3
#     phi3 = pi
#     phi4 = pi / 2
#     phi5 = phi4 - pi / 2

#     length1 = 1.50
#     length2 = 2.00
#     length3 = 1.40
#     length4 = 2.80
#     length5 = 1.20

#     corridor1 = CorridorWorld(width=width, height=length1, center=[-0.05, -0.70], tilt=phi1)

#     corridors = [corridor1]
#     for phi, length in ((phi2, length2), (phi3, length3), (phi4, length4), (phi5, length5)):
#         tail = corridors[-1].head
#         head = [tail[0] + length * cos(phi), tail[1] + length * sin(phi)]
#         corridors.append(get_corridor_from_vector(tail, head, width, add_height=add_height))

#     initial_pose = compute_start_pose(corridors[0], unicycle, 0.35)
#     initial_pose[2] = -pi / 2
#     final_pose = compute_end_pose(corridors[-1], unicycle, 0.35)
#     final_pose[2] = -pi / 2

#     assert_experiment_fits_floor(corridors, initial_pose, final_pose, robot_radius(unicycle))
#     return Scenario(3, 'Experiment 3 - Multiple Corridors', corridors,
#                     list(initial_pose), list(final_pose))

def experiment_3(unicycle: Unicycle) -> Scenario:
    width = 0.80
    add_height = 0.45          # only used for the pose-reference corridors

    phi1 = 0.0
    phi2 = pi / 3
    phi3 = pi
    phi4 = pi / 2
    phi5 = phi4 - pi / 2

    length1 = 1.50
    length2 = 2.00
    length3 = 1.40
    length4 = 2.80
    length5 = 1.20

    # Original path (UNCHANGED): junctions from the segment chain
    tail1 = [-0.05 - 0.5 * length1, -0.70]          # (-0.80, -0.70)
    segs = [(phi1, length1), (phi2, length2), (phi3, length3),
            (phi4, length4), (phi5, length5)]
    joints = [tail1]
    for phi, length in segs:
        p = joints[-1]
        joints.append([p[0] + length * cos(phi), p[1] + length * sin(phi)])
    # joints = tail1, j1, j2, j3, j4, head5

    # Visual extensions per corridor end [m], in each corridor's LOCAL frame
    # (+x = tail->head). Junctions, widths, start and goal are untouched.
    EXTS = [
        (0.00, 0.40),   # c1: +x ~0.40 (grows through j1)
        (0.05, 0.30),   # c2: -x few cm, +x 0.30
        (0.00, 0.50),   # c3: +x 0.50 (global -x, since phi3 = pi)
        (0.40, 0.40),   # c4: both ends 0.40
        (0.30, 0.00),   # c5: -x 0.30 (global -x, phi5 = 0)
    ]

    def seg_corridor(tail, phi, seg_len, ext_tail, ext_head, w):
        total = seg_len + ext_tail + ext_head
        t0 = [tail[0] - ext_tail * cos(phi), tail[1] - ext_tail * sin(phi)]
        center = [t0[0] + 0.5 * total * cos(phi), t0[1] + 0.5 * total * sin(phi)]
        return CorridorWorld(width=w, height=total, center=center, tilt=phi)

    corridors = [
        seg_corridor(joints[i], segs[i][0], segs[i][1], EXTS[i][0], EXTS[i][1], width)
        for i in range(5)
    ]

    # Start/goal from the ORIGINAL construction -> poses bit-identical to before
    corridor1_ref = CorridorWorld(width=width, height=length1,
                                  center=[-0.05, -0.70], tilt=phi1)
    corridor5_ref = get_corridor_from_vector(joints[4], joints[5], width,
                                             add_height=add_height)
    initial_pose = compute_start_pose(corridor1_ref, unicycle, 0.35)
    initial_pose[2] = -pi / 2
    final_pose = compute_end_pose(corridor5_ref, unicycle, 0.35)
    final_pose[2] = -pi / 2

    assert_experiment_fits_floor(corridors, initial_pose, final_pose, robot_radius(unicycle))
    return Scenario(3, 'Experiment 3 - Multiple Corridors', corridors,
                    list(initial_pose), list(final_pose),
                    notes={'joints': joints, 'extensions': EXTS})

# ---------------------------------------------------------------------------
# Experiment 4: narrow corridors
# ---------------------------------------------------------------------------
# def experiment_4(unicycle: Unicycle) -> Scenario:
#     phi1 = 0.0
#     phi2 = pi / 2
#     phi3 = pi / 6

#     length1 = 1.40
#     length2 = 2.30
#     length3 = 1.70

#     add_height = 0.40

#     R = turning_radius(unicycle)
#     r = robot_radius(unicycle)
#     rho = R + r

#     beta1 = abs(phi2 - phi1) / 2
#     beta2 = abs(phi3 - phi2) / 2
#     q1 = (R - r) * cos(beta1)
#     q2 = (R - r) * cos(beta2)
#     width_min_90 = rho - q1
#     width_min_60 = rho - q2

#     width2 = max(width_min_90, width_min_60)
#     width1 = 0.50
#     width3 = 0.50

#     corridor1 = CorridorWorld(width=width1, height=length1, center=[-0.05, -0.70], tilt=phi1)

#     tail = corridor1.head
#     head = [tail[0] + length2 * cos(phi2), tail[1] + length2 * sin(phi2)]
#     corridor2 = get_corridor_from_vector(tail, head, width2, add_height=add_height)

#     tail = corridor2.head
#     head = [tail[0] + length3 * cos(phi3), tail[1] + length3 * sin(phi3)]
#     corridor3 = get_corridor_from_vector(tail, head, width3, add_height=add_height)

#     corridors = [corridor1, corridor2, corridor3]
#     initial_pose = compute_start_pose(corridor1, unicycle, 0.35)
#     final_pose = compute_end_pose(corridor3, unicycle, 0.35)

#     assert_experiment_fits_floor(corridors, initial_pose, final_pose, robot_radius(unicycle))
#     return Scenario(4, 'Experiment 4 - Narrow Corridors', corridors,
#                     list(initial_pose), list(final_pose),
#                     notes={'width_min_90': width_min_90, 'width_min_60': width_min_60,
#                            'width2': width2})

def experiment_4(unicycle: Unicycle) -> Scenario:
    phi1 = 0.0
    phi2 = pi / 2
    phi3 = pi / 6

    # Original segment lengths and junctions (UNCHANGED):
    # c1: (-0.75, -0.70) --phi1--> j1 = (0.65, -0.70)
    # c2:  j1            --phi2--> j2 = (0.65,  1.60)
    # c3:  j2            --phi3--> 1.70 m at 30 deg
    length1 = 1.40
    length2 = 2.30
    length3 = 1.70
    tail1 = [-0.75, -0.70]
    j1 = [tail1[0] + length1 * cos(phi1), tail1[1] + length1 * sin(phi1)]
    j2 = [j1[0] + length2 * cos(phi2), j1[1] + length2 * sin(phi2)]
    head3 = [j2[0] + length3 * cos(phi3), j2[1] + length3 * sin(phi3)]

    # Visual extensions per corridor end [m]. These ONLY lengthen the drawn
    # rectangles along their own axes; junctions, tilts, widths, start and
    # goal are untouched.
    EXT1_HEAD = 0.60     # corridor 1 grows in +x, through j1
    EXT2_TAIL = 0.45     # corridor 2 grows in -y, below j1 (floor bound: y >= -1.25)
    EXT2_HEAD = 0.60     # corridor 2 grows in +y, above j2
    EXT3_TAIL = 0.50     # corridor 3 grows backward through j2 (mostly -x)
    EXT3_HEAD = 0.10     # corridor 3 grows a bit forward (x bound: 2.65)

    def seg_corridor(tail, phi, seg_len, ext_tail, ext_head, width):
        total = seg_len + ext_tail + ext_head
        t0 = [tail[0] - ext_tail * cos(phi), tail[1] - ext_tail * sin(phi)]
        center = [t0[0] + 0.5 * total * cos(phi), t0[1] + 0.5 * total * sin(phi)]
        return CorridorWorld(width=width, height=total, center=center, tilt=phi)

    # ----------------------------------------------------------------
    # Minimum admissible widths (unchanged)
    # ----------------------------------------------------------------
    R = turning_radius(unicycle)
    r = robot_radius(unicycle)
    rho = R + r

    beta1 = abs(phi2 - phi1) / 2
    beta2 = abs(phi3 - phi2) / 2
    q1 = (R - r) * cos(beta1)
    q2 = (R - r) * cos(beta2)
    width_min_90 = rho - q1
    width_min_60 = rho - q2

    width2 = max(width_min_90, width_min_60)
    width1 = 0.50
    width3 = 0.50

    corridor1 = seg_corridor(tail1, phi1, length1, 0.0, EXT1_HEAD, width1)
    corridor2 = seg_corridor(j1, phi2, length2, EXT2_TAIL, EXT2_HEAD, width2)
    corridor3 = seg_corridor(j2, phi3, length3, EXT3_TAIL, EXT3_HEAD, width3)
    corridors = [corridor1, corridor2, corridor3]

    # Start/goal from the ORIGINAL corridor construction, so the poses are
    # bit-identical to the previous scenario 4 regardless of the extensions.
    corridor1_ref = CorridorWorld(width=width1, height=length1,
                                  center=[-0.05, -0.70], tilt=phi1)
    corridor3_ref = get_corridor_from_vector(j2, head3, width3, add_height=0.40)
    initial_pose = compute_start_pose(corridor1_ref, unicycle, 0.35)
    final_pose = compute_end_pose(corridor3_ref, unicycle, 0.35)

    assert_experiment_fits_floor(corridors, initial_pose, final_pose, robot_radius(unicycle))
    return Scenario(4, 'Experiment 4 - Narrow Corridors', corridors,
                    list(initial_pose), list(final_pose),
                    notes={'width_min_90': width_min_90, 'width_min_60': width_min_60,
                           'width2': width2, 'j1': j1, 'j2': j2,
                           'extensions': [EXT1_HEAD, EXT2_TAIL, EXT2_HEAD,
                                          EXT3_TAIL, EXT3_HEAD]})

SCENARIOS: Dict[int, Callable[[Unicycle], Scenario]] = {
    1: experiment_1,
    2: experiment_2,
    3: experiment_3,
    4: experiment_4,
}


def build_scenario(number: int, unicycle: Unicycle) -> Scenario:
    if number not in SCENARIOS:
        raise ValueError(f'Unknown scenario {number}; available: {sorted(SCENARIOS)}')
    return SCENARIOS[number](unicycle)
