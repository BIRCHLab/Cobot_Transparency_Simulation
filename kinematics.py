import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy as np
import sympy as sp


def Rx(t):
    return sp.Matrix([
        [1,          0,           0, 0],
        [0,  sp.cos(t), -sp.sin(t), 0],
        [0,  sp.sin(t),  sp.cos(t), 0],
        [0,          0,           0, 1]
    ])

def Ry(t):
    return sp.Matrix([
        [ sp.cos(t), 0, sp.sin(t), 0],
        [          0, 1,         0, 0],
        [-sp.sin(t), 0, sp.cos(t), 0],
        [          0, 0,         0, 1]
    ])

def Rz(t):
    return sp.Matrix([
        [sp.cos(t), -sp.sin(t), 0, 0],
        [sp.sin(t),  sp.cos(t), 0, 0],
        [0,                    0, 1, 0],
        [0,                    0, 0, 1]
    ])

def Txyz(x=0, y=0, z=0):
    return sp.Matrix([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])


# transforming between different pose representations

def transform_to_pos_quat(T):
    """
    Convert 4x4 homogeneous transform to position + quaternion.

    Returns
    -------
    pos : (3,)
    quat : (4,)  # [w, x, y, z]
    """
    pos = T[:3, 3]
    R_mat = T[:3, :3]

    rot = R.from_matrix(R_mat)

    # SciPy returns quaternion as [x, y, z, w]
    quat_xyzw = rot.as_quat()

    # Reorder to [w, x, y, z] to match your original convention
    quat_wxyz = np.array([
        quat_xyzw[3],
        quat_xyzw[0],
        quat_xyzw[1],
        quat_xyzw[2]
    ])

    return np.concatenate([pos, quat_wxyz])

def transform_to_pos_rpy(T, degrees=False, order='xyz'):
    """
    Convert 4x4 homogeneous transform to position + roll-pitch-yaw.

    Parameters
    ----------
    T : (4,4) array
    degrees : bool
        If True, return angles in degrees (default: radians)
    order : str
        Euler angle convention (default: 'xyz' → roll, pitch, yaw)

    Returns
    -------
    out : (6,)
        [x, y, z, roll, pitch, yaw]
    """
    pos = T[:3, 3]
    R_mat = T[:3, :3]

    rot = R.from_matrix(R_mat)
    rpy = rot.as_euler(order, degrees=degrees)

    return np.concatenate([pos, rpy])

def pos_quat_to_transform(pos_quat):
    """
    Convert position + quaternion to 4x4 homogeneous transform.

    Parameters
    ----------
    pos : (3,)
    quat : (4,)  # [w, x, y, z]
    """
    pos = pos_quat[0:3]
    quat = pos_quat[3:]

    w, x, y, z = quat
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos

    return T

def damped_pinv(J, damping=1e-4):
    """
    Damped least-squares pseudoinverse.
    """
    m, n = J.shape

    if m >= n:
        return np.linalg.inv(J.T @ J + damping * np.eye(n)) @ J.T
    else:
        return J.T @ np.linalg.inv(J @ J.T + damping * np.eye(m))
    
def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def so3_log(R): # Rodrigues formula
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    
    if np.isclose(theta, 0):
        return np.zeros((3, 3))
    
    return (theta / (2 * np.sin(theta))) * (R - R.T)

def get_w_from_T(T, dt):
    """
    T: array of shape (N, 4, 4)
    dt: timestep

    Returns:
        w: array of shape (N, 3) (spatial angular velocity)
    """
    N = T.shape[0]
    w = np.zeros((N, 3))

    for k in range(N - 1):
        RA = T[k, :3, :3]
        RB = T[k + 1, :3, :3]

        # spatial relative rotation
        R_rel = RB @ RA.T

        # log map
        w_hat = so3_log(R_rel) / dt

        # vee operator
        w[k] = np.array([
            w_hat[2, 1],
            w_hat[0, 2],
            w_hat[1, 0]
        ])

    # pad last value to keep same length
    w[-1] = w[-2]

    return w