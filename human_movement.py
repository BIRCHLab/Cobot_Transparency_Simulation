import numpy as np
import os 
import csv
import sympy as sp
from kinematics import damped_pinv, wrap_to_pi
import sympy as sp
from kinematics import Rx,Ry,Rz, Txyz

h_start = np.array([
    0, 0, 0, 0.0708, 0, 1.5, 0                   # human
])

fe_traj_params = h_start.copy()
fe_traj_params[5] =  2.0

multijoint_traj_params = np.array([0, 0, -.3, .9, 1.2, 0.5, 0.3])

h2m_traj_params = np.array([0, 0, 0, 0.7, 0.73, 2.4, .5])

fr_distance_WF = np.array([0.010, 0.0, 0.0])

LSC = 0.17
LUA = 0.34
LFA = 0.305
LFAA = 0.20
FA_offset = 0.036


q1, q2, q3, q4, q5, q6, q7 = sp.symbols('q1 q2 q3 q4 q5 q6 q7')
q = sp.Matrix([q1, q2, q3, q4, q5, q6, q7])
# LSC, LUA = sp.symbols('LSC LUA', real=True)


def HT1HB(t):
    T = Txyz(0, 0.015, 0)
    return T * Rx(t)

def HT21(t):
    return Rz(t)

def HT32(t):
    T = Txyz(0, LSC, 0)
    return T * Rx(t)

def HT43(t):
    return Ry(t)

def HT54(t):
    return Rz(t)

def HT65(t):
    T = Txyz(0, 0, -LUA)
    return T * Ry(t)

def HT76(t):
    return Rz(t)

def HTFAA7():
    return Txyz(FA_offset, 0, -LFAA) * sp.Matrix([
                                            [0, 0, -1, 0],
                                            [1, 0, 0, 0],
                                            [0, -1, 0, 0],
                                            [0, 0, 0, 1]
                                            ]) # add one more that transforms to attachment

def HTFAAHB(ts):
    return (
        HT1HB(ts[0]) *
        HT21(ts[1]) *
        HT32(ts[2]) *
        HT43(ts[3]) *
        HT54(ts[4]) *
        HT65(ts[5]) *
        HT76(ts[6]) *
        HTFAA7()
    )

TFAAHB = sp.simplify(HTFAAHB(q))

# Create a lambdified transform function once
HTFAAHB_num = sp.lambdify(q, TFAAHB, "numpy")

# Optional convenience wrapper to preserve the old API
def HTFAAHB_eval(ts):
    return np.array(
        HTFAAHB_num(*ts),
        dtype=np.float64
    )

def get_H_jacobian_func():
    from ur5e_analytic import jacobian_from_transform
    J_sym = jacobian_from_transform(TFAAHB,q)
    return sp.lambdify(q,J_sym,"numpy")

# Build once
H_JACOBIAN_FUNC = get_H_jacobian_func()



def H_INV_KIN(
    hand_position_trajectory,
    q0=None,
    max_iters=200,
    tol=1e-5,
    step_size=0.5,
    damping=1e-4,
    nullspace_gain=0.05,
    q_nominal=None,
    joint_limits=None,
):
    """
    Numerical inverse kinematics using Jacobian pseudoinverse.

    Parameters
    ----------
    hand_position_trajectory : np.ndarray
        Either:
            (4,4) homogeneous transform
        or
            (T,4,4) trajectory

    q0 : (7,) initial guess
    max_iters : int
    tol : float
    step_size : float
    damping : float
    nullspace_gain : float
        Gain for secondary nullspace objective.

    q_nominal : (7,)
        Preferred posture for nullspace optimization.

    joint_limits : tuple(lower, upper)
        Optional joint limits.

    Returns
    -------
    qs : (7,) or (T,7)
    """

    # -------------------------------------------------------------
    # Normalize trajectory shape
    # -------------------------------------------------------------

    Hs = np.asarray(hand_position_trajectory)

    single_input = False

    if Hs.shape[-2] == Hs.shape[-1] and Hs.ndim == 2:
        Hs = Hs[None, ...]
        single_input = True

    T = Hs.shape[0]

    if Hs.shape[1] == 4:
        Hs = Hs[:,0:3,-1] # select just the position vec

    # -------------------------------------------------------------
    # Defaults
    # -------------------------------------------------------------

    if q0 is None:
        q = np.zeros(7)
    else:
        q = np.array(q0, dtype=float).copy()

    if q_nominal is None:
        q_nominal = np.zeros(7)

    qs = np.zeros((T, 7))

    # -------------------------------------------------------------
    # Solve each timestep sequentially
    # -------------------------------------------------------------
    for t in range(T):
        xd = Hs[t, :]
        print(t)
        for _ in range(max_iters):

            # -------------------------------------------------
            # Forward kinematics
            # -------------------------------------------------

            H = HTFAAHB_eval(q)
            x = H[0:3, 3]

            # Position error
            e = xd - x
            # Converged
            if np.linalg.norm(e) < tol:
                break

            # -------------------------------------------------
            # Jacobian
            # -------------------------------------------------

            J_full = np.array(
                H_JACOBIAN_FUNC(*q),
                dtype=np.float64
            )

            # Position Jacobian only
            J = J_full[0:3, :]

            # -------------------------------------------------
            # Primary task
            # -------------------------------------------------

            J_pinv = damped_pinv(J, damping)

            dq_task = J_pinv @ e

            # -------------------------------------------------
            # Nullspace optimization
            # -------------------------------------------------

            #
            # Secondary objective:
            # keep joints near q_nominal
            #
            # gradient of:
            #
            #   0.5 * ||q - q_nominal||²
            #
            grad_secondary = -(q - q_nominal)

            #
            # Nullspace projector
            #
            N = np.eye(7) - J_pinv @ J

            dq_null = nullspace_gain * (N @ grad_secondary)

            # -------------------------------------------------
            # Combined update
            # -------------------------------------------------

            dq = step_size * dq_task + dq_null

            q = q + dq 

            q = wrap_to_pi(q)

            # -------------------------------------------------
            # Joint limits
            # -------------------------------------------------

            if joint_limits is not None:
                q_lower, q_upper = joint_limits
                q = np.clip(q, q_lower, q_upper)

        qs[t] = q

    if single_input:
        return qs[0]

    return qs


def minjerk_cartesian_trajectory(
    waypoints,               # (N,4,4) or (N,3)
    time_vector,             # (T,)
    waypoint_timestamps,     # (N,)
):
    """
    Generate a Cartesian minimum-jerk trajectory.

    Position only is interpolated.
    Orientation is held piecewise constant.

    Parameters
    ----------
    waypoints : ndarray
        Either:
            (N,3)      Cartesian positions
        or
            (N,4,4)    Homogeneous transforms

    time_vector : ndarray, shape (T,)
        Full trajectory time vector.

    waypoint_timestamps : ndarray, shape (N,)
        Timestamp associated with each waypoint.

    Returns
    -------
    traj : ndarray
        If input was:
            (N,3)   -> returns (T,3)
            (N,4,4) -> returns (T,4,4)
    """

    waypoints = np.asarray(waypoints, dtype=float)
    time_vector = np.asarray(time_vector, dtype=float)
    waypoint_timestamps = np.asarray(
        waypoint_timestamps,
        dtype=float
    )

    # ---------------------------------------------------------
    # Parse waypoint format
    # ---------------------------------------------------------

    return_transforms = False

    if waypoints.ndim == 2 and waypoints.shape[1] == 3:

        positions = waypoints

    elif waypoints.ndim == 3 and waypoints.shape[1:] == (4, 4):

        return_transforms = True

        positions = waypoints[:, 0:3, 3]

    else:
        raise ValueError(
            "waypoints must have shape (N,3) or (N,4,4)"
        )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if time_vector.ndim != 1:
        raise ValueError("time_vector must be 1D")

    if waypoint_timestamps.ndim != 1:
        raise ValueError("waypoint_timestamps must be 1D")

    if len(positions) != len(waypoint_timestamps):
        raise ValueError(
            "waypoints and waypoint_timestamps "
            "must have same length"
        )

    if np.any(np.diff(time_vector) <= 0):
        raise ValueError(
            "time_vector must be strictly increasing"
        )

    if np.any(np.diff(waypoint_timestamps) <= 0):
        raise ValueError(
            "waypoint_timestamps must be strictly increasing"
        )

    if waypoint_timestamps[0] < time_vector[0]:
        raise ValueError(
            "waypoint timestamps must lie within time_vector"
        )

    if waypoint_timestamps[-1] > time_vector[-1]:
        raise ValueError(
            "waypoint timestamps must lie within time_vector"
        )

    n_waypoints = len(positions)

    # ---------------------------------------------------------
    # Allocate outputs
    # ---------------------------------------------------------

    pos_traj = np.zeros((len(time_vector), 3))

    if return_transforms:

        H_traj = np.repeat(
            np.eye(4)[None, :, :],
            len(time_vector),
            axis=0
        )

    # ---------------------------------------------------------
    # Build trajectory segment-by-segment
    # ---------------------------------------------------------

    for seg in range(n_waypoints - 1):

        p0 = positions[seg]
        pf = positions[seg + 1]

        t0 = waypoint_timestamps[seg]
        tf = waypoint_timestamps[seg + 1]

        T = tf - t0

        mask = (
            (time_vector >= t0) &
            (time_vector < tf)
        )

        t = time_vector[mask] - t0

        # -----------------------------------------------------
        # Hold segment
        # -----------------------------------------------------

        if np.allclose(p0, pf):

            pos_traj[mask] = p0

        else:

            tau = t / T

            s = (
                10 * tau**3
                - 15 * tau**4
                + 6 * tau**5
            )

            pos_traj[mask] = (
                p0 + np.outer(s, (pf - p0))
            )

        # -----------------------------------------------------
        # Orientation handling
        # -----------------------------------------------------

        if return_transforms:

            #
            # Piecewise constant orientation
            #
            # Could later replace with quaternion SLERP
            #
            H_traj[mask, 0:3, 0:3] = (
                waypoints[seg, 0:3, 0:3]
            )

    # ---------------------------------------------------------
    # Final sample
    # ---------------------------------------------------------

    pos_traj[
        time_vector >= waypoint_timestamps[-1]
    ] = positions[-1]

    if return_transforms:

        H_traj[
            time_vector >= waypoint_timestamps[-1],
            0:3,
            0:3
        ] = waypoints[-1, 0:3, 0:3]

        H_traj[:, 0:3, 3] = pos_traj

        return H_traj

    return pos_traj


def minjerk_joint_trajectory(
    waypoints,
    time_vector,
    waypoint_timestamps,
):
    """
    Generate a multi-segment minimum-jerk trajectory in joint space.

    Parameters
    ----------
    waypoints : array-like, shape (N, ndim)
        Joint-space waypoints.

        Repeated waypoints automatically create holds.

    time_vector : array-like, shape (M,)
        Full trajectory time vector.

        Example:
            np.linspace(0, 10, 1000)

    waypoint_timestamps : array-like, shape (N,)
        Timestamp for each waypoint.

        Example:
            [0.0, 2.0, 5.0, 10.0]

    Returns
    -------
    q_des : ndarray, shape (M, ndim)
        Minimum-jerk trajectory sampled at time_vector.
    """

    waypoints = np.asarray(waypoints, dtype=float)
    time_vector = np.asarray(time_vector, dtype=float)
    waypoint_timestamps = np.asarray(
        waypoint_timestamps,
        dtype=float
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    if waypoints.ndim != 2:
        raise ValueError(
            "waypoints must have shape (N, ndim)"
        )

    if time_vector.ndim != 1:
        raise ValueError(
            "time_vector must be 1D"
        )

    if waypoint_timestamps.ndim != 1:
        raise ValueError(
            "waypoint_timestamps must be 1D"
        )

    if len(waypoints) != len(waypoint_timestamps):
        raise ValueError(
            "waypoints and waypoint_timestamps "
            "must have same length"
        )

    if np.any(np.diff(time_vector) <= 0):
        raise ValueError(
            "time_vector must be strictly increasing"
        )

    if np.any(np.diff(waypoint_timestamps) <= 0):
        raise ValueError(
            "waypoint_timestamps must be strictly increasing"
        )

    if waypoint_timestamps[0] < time_vector[0]:
        raise ValueError(
            "waypoint timestamps must lie within time_vector"
        )

    if waypoint_timestamps[-1] > time_vector[-1]:
        raise ValueError(
            "waypoint timestamps must lie within time_vector"
        )

    n_waypoints, ndim = waypoints.shape

    q_des = np.zeros((len(time_vector), ndim))

    # ---------------------------------------------------------
    # Build trajectory segment-by-segment
    # ---------------------------------------------------------
    for seg in range(n_waypoints - 1):

        q0 = waypoints[seg]
        qf = waypoints[seg + 1]

        t0 = waypoint_timestamps[seg]
        tf = waypoint_timestamps[seg + 1]

        T = tf - t0

        # Samples belonging to this segment
        mask = (
            (time_vector >= t0) &
            (time_vector < tf)
        )

        t = time_vector[mask] - t0

        # -----------------------------------------------------
        # Hold segment
        # -----------------------------------------------------
        if np.allclose(q0, qf):

            q_des[mask] = q0
            continue

        # -----------------------------------------------------
        # Minimum jerk interpolation
        # -----------------------------------------------------
        tau = t / T

        s = (
            10 * tau**3
            - 15 * tau**4
            + 6 * tau**5
        )

        q_des[mask] = (
            q0 + np.outer(s, (qf - q0))
        )

    # Final sample
    q_des[time_vector >= waypoint_timestamps[-1]] = waypoints[-1]

    return q_des

def sinusoidal_trajectory(n_steps,
                              n_cycles,
                              hold_steps,
                              ndim=4,
                              end=[0, 0, 0, 0.2, 0.7, 0.9, 0 ],
                              start=[0, 0, 0, 0, 0, 0, 0 ],
                              ):
    """
    Generate a trajectory producing a circular hand motion with
    stationary holds at the start and end.
    """

    if start is None:
        start = np.zeros(ndim)
    else:
        start = np.asarray(start)
        
    amplitude = end.copy()-start.copy()

    q_des = np.tile(start, (n_steps, 1))

    motion_steps = n_steps - 2 * hold_steps
    if motion_steps <= 0:
        return q_des

    t = np.arange(motion_steps)
    phase = 2 * np.pi * n_cycles * t / motion_steps

    for i in range(motion_steps):
        k = i + hold_steps

        # shoulder abduction
        q_des[k, :] = start[:] + amplitude[:] * (-1/2)*(np.cos(phase[i]) - 1)

    return q_des


# DEFINE physiologically realistic motor signs
# positive signs

def weakness(torque_traj,
             joint_multipliers=np.array([0, 0, 0, 0, 0, .5, 0]), # 
             ):
    
    weakness_traj = torque_traj * joint_multipliers[np.newaxis,:]

    return - (torque_traj - weakness_traj) # returns torques you can superimpose over the actual input

def synergy(torque_traj,
                    coupling_vec= np.array([0.1, 0, 0, 0, .1, .1, 0])
                    ):
    """This function takes a portion of activation for each DOF and redistributes that weighting even across all n coupled DOF
    For example:

    0.5 of shoulder abduction (F1), and 0.25 of elbow flexion (F2)
    here, n =2

    so half of F1 would be removed from the original F1 (F1_0).
    Then 0.5*F1/n would be added to each DOF.

    Similarly for F2, except 0.25*F2/n would be added to each DOF
    

    Args:
        torque_traj (_type_): _description_
        coupling_vec (_type_, optional): _description_. Defaults to np.array([.1, 0, 0, 0, .1, .1, 0]).
    """
    I = np.eye(C.shape[0])
    C = coupling_vec[:,np.newaxis] @ coupling_vec[np.newaxis,:]
    C[I==1] = 1
    
    n_coupled = np.sum(coupling_vec>0)
    print(f"N coupled: {n_coupled}")

    if len(torque_traj.shape) < 2:
        torque_traj = torque_traj[np.newaxis,:] # add time dimension if not full traj

    torque_traj_0 = torque_traj.copy()

    return np.einsum('ij,tj->ti', C, torque_traj)


def flexor_synergy():
    pass

def tremor(torque_traj,
           weighting_vec= np.array([.1, 0, 0, 0, .1, .1, 0])
           ):
    pass

def get_human_traj(folder):

    filepath = os.path.join(folder,'human_arm_attachment_pose_WF.csv')
    vectors = []
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # discard header

        for row in reader:
            # Remove empty padding cells
            vals = [float(x) for x in row if x != ""]
            vectors.append(np.array(vals))

    length = len(vectors)

    return np.array(vectors).reshape((length,4,4))