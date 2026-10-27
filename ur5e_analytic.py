#%%
import numpy as np
import sympy as sp
import mujoco as mj
# from numba import njit
import pickle
import copy
from scipy.spatial.transform import Rotation as R
import sys

import time

np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)


def sym2num(x): return np.array(x).astype(np.float64)

# parameters from https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
theta = sp.symbols('t1 t2 t3 t4 t5 t6')
ee_state_var = sp.Matrix(sp.symbols('x y z roll pitch yaw'))
theta_sym = sp.Matrix(theta)


def joint_vel_lim():
    return np.ones(6)*np.pi # rad/s

# from UR website
# CoM = np.array([[0.0,       -0.02561,   0.00193],
#                 [0.2125,    0.0,        0.11336],
#                 [0.15,      0.0,        0.0265],
#                 [0.0,       -0.0018,    0.01634],
#                 [0.0,       0.0018,     0.01634],
#                 [0.0,       0.0,        -0.001159],
#                 ])


COM_sim = np.array([[0,         0,      0], # shoulder link
                    [0.2125,    0,      0.138], #upper_arm_link
                    [0.196,     0,      0], # forearm_link
                    [0,    0,     0], # wrist_1_joint
                    [0,         0,      0], # wrist_2_link
                    [0,         0,      0.0771683], # wrist_3_link
                    ])

inertia = np.array([np.array([[0, 0, 0,], 
                            [0, 0, 0],
                            [0, 0, 0],]) for _ in range(6)])

inertia[-1] = np.array([[0, 0, 0,], 
                        [0, 0, 0],
                        [0, 0, 0.0002],])



                   
np.diag(v=np.array([]))

LB = 0.1625
a2 = 0.4250
a3 = 0.3922
d4 = 0.1333 # number that works with menajerie cad....
d5 = 0.0997


def get_dh_params():
    return  np.array([
                # a i-1,        alpha i-1,      di,                 thetai (home offset)])
                [0.0,           0.0,            0.0         ,       0.0,    ],
                [0.0,           np.pi/2,        0.0         ,       np.pi/2,],
                [a2,            0.0,            0.0         ,       0.0,    ],
                [a3,            0.0,            d4          ,       -np.pi/2,],
                [0.0,           -np.pi/2,       d5          ,       0.0,    ],
                [0.0,           np.pi/2,        0           ,       0.0,    ],])
# LTP = 0.0996 + 0.0254
LTP = 0.1250

def invert_transformation(t):
    t_new = t.copy()

    t_new[0:3,0:3] = t[0:3,0:3].T
    t_new[0:3,-1] = -t_new[0:3,0:3] @ t[0:3,-1]

    return t_new

# 3 singularity conditions
# sin(t3) = 0
# sin(t5) = 0
# a2*sin(t2) + a3*sin(t2+t3) + d5*sin(t2+t3+t4)

# the final decision: T_staticframe_body_frame!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# or TAB reads as Frame B expressed in A

TWF0 = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, LB],
                [0, 0, 0, 1]])

T6TP = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, LTP],
                [0, 0, 0, 1]])



T0WF = invert_transformation(TWF0)
TTP6 = invert_transformation(T6TP)

# T_fix = np.array([[-1, 0,  0,  0],
                #   [0, -1,  0,  0],
                #   [0,  0,  1,  0],
                #   [0,  0,  0,  1]])

def get_robot_home_qpos():
    return np.array([0.0,
                    -1.5708,
                    0.0,
                    -1.5708,
                    0.0,
                    0.0,])


def dh(a,alpha,d,theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    T = np.array([[ ct, -st,  0,  a],
                  [ st*ca, ct*ca, -sa, -sa*d],
                  [ st*sa, ct*sa,  ca,  ca*d],
                  [ 0,     0,      0,   1]])
    return T

def dh_sym(a,alpha,d,theta):
    ca, sa = sp.cos(alpha), sp.sin(alpha)
    ct, st = sp.cos(theta), sp.sin(theta)

    T = sp.Matrix([[   ct,   -st,   0,     a],
                  [ st*ca, ct*ca, -sa, -sa*d],
                  [ st*sa, ct*sa,  ca,  ca*d],
                  [     0,     0,   0,     1]])
    
    return T

def get_transformation(joint_angles,N=7):

    T = np.eye(4)
    dh_params = get_dh_params()


    # TB0 = np.array([[1, 0, 0,  0],
    #                 [0, 1, 0,  0],
    #                 [0, 0, 1, LB],
    #                 [0, 0, 0,  1]])

    # TTP6 = np.array([[1, 0, 0,   0],
    #                  [0, 1, 0,   0],
    #                  [0, 0, 1, LTP],
    #                  [0, 0, 0,   1]])
    

    if N > 6:
        T_final = T6TP
        N = 6
    else:
        T_final = np.identity(4)

    for i in range(N):
        a, alpha, d, theta = dh_params[i]
        theta += joint_angles[i]

        T_i = dh(a,alpha,d,theta)

        T = T @ T_i
    
    return TWF0 @ T @ T_final

    # return TB0 @ T @ T6TP # this makes sure to go from world frame to TP


def ik(r,turn_off_warnings=False):
    # taken directly from: https://www.ohio.edu/mechanical-faculty/williams/html/pdf/UniversalKinematics.pdf

    # just to make the argument make sense but keep equations shorter

    E1 = r[1,-1]
    F1 = -r[0,-1]
    G1 = d4


    D1 = E1**2 + F1**2 - G1**2
    if D1 < -1e-9: D1 = np.nan
    else: D1 = max(D1, 0.0)

    theta1 = np.array([2*np.arctan((-F1+np.sqrt(D1))/(G1-E1)), 
                        2*np.arctan((-F1-np.sqrt(D1))/(G1-E1))]) # 2 solns in +- np.sqrt
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)

    theta6 = np.arctan2((r[0,1]*s1 - r[1,1]*c1), (r[1,0]*c1 - r[0,0]*s1))
    c6 = np.cos(theta6)
    s6 = np.sin(theta6)

    theta5 = np.arctan2((r[1,0]*c1-r[0,0]*s1)*c6 + (r[0,1]*s1-r[1,1]*c1)*s6, r[0,2]*s1-r[1,2]*c1)
    c5 = np.cos(theta5)
    s5 = np.sin(theta5)

    A = (r[2,0]*c6 - r[2,1]*s6)/c5
    B = (r[2,1]*c6 + r[2,0]*s6)

    a = -r[0,-1]*c1 - r[1,-1]*s1 - d5*A
    b = r[2,-1] -d5*B

    E2 = -2*a2*b
    F2 = -2*a2*a
    G2 = a2**2 + a**2 + b**2 - a3**2

    D2 = E2**2 + F2**2 - G2**2

    x = []
    for p in D2:
        if p < -1e-9: x.append(np.nan)
        else: x.append(max(p, 0.0))
    D2_clean = np.array(x)


    solutions = []
    for i,D in enumerate(D2_clean):

        sqrtD = np.sqrt(D)

        for sign in (+1, -1):
            theta2 = 2*np.arctan((-F2[i] + sign*sqrtD) / (G2[i] - E2[i]))

            s2 = np.sin(theta2)
            c2 = np.cos(theta2)

            theta3 = np.arctan2(a[i] - a2*s2,b[i] - a2*c2) - theta2

            theta4 = np.arctan2(A[i], B[i]) - theta2 - theta3
            # if a2*np.sin(theta2) + a3*np.sin(theta2+theta3) + d5*np.sin(theta2+theta3+theta4) == 0:
            #     print('here')
            solutions.append(np.array([theta1[i], theta2, theta3, theta4, theta5[i], theta6[i]]))
    res = np.stack(solutions,axis=0)
    if np.sum(np.isnan(res)) > 0 and not turn_off_warnings:
        print("IK DIDN'T LIKE THIS ONE")
    return res

def inverse_kinematics(tWF_TP,turn_off_warnings = False):
    # taken directly from: https://www.ohio.edu/mechanical-faculty/williams/html/pdf/UniversalKinematics.pdf
    T06 = T0WF @ tWF_TP @ TTP6 # just to make the argument make sense but keep equations shorter
    return ik(T06,turn_off_warnings = turn_off_warnings)

def get_site_frames(model,data):

    human_hand_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "human_hahuman_arm_attachment_sitend_site")
    proximal_interface_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "ee_attachment_site")
    distal_interface_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "distal_interface_site")

    proximal_interface_site_pos = data.site_xpos[proximal_interface_site_id]
    distal_interface_site_pos = data.site_xpos[distal_interface_site_id]
    human_attachment_site_pos = data.site_xpos[human_hand_site_id]

    proximal_interface_site_orientation = data.site_xmat[proximal_interface_site_id].reshape((3,3))
    distal_interface_site_orientation = data.site_xmat[distal_interface_site_id].reshape((3,3))
    human_attachment_site_orientation = data.site_xmat[human_hand_site_id].reshape((3,3))

    PI_T = transformation_mat(proximal_interface_site_pos, proximal_interface_site_orientation)
    DI_T = transformation_mat(distal_interface_site_pos, distal_interface_site_orientation)
    HA_T = transformation_mat(human_attachment_site_pos, human_attachment_site_orientation)
    
    return PI_T, DI_T, HA_T

def transformation_mat(x,R):
    t = np.zeros((4,4))
    t[0:3,0:3] = R
    t[0:3,-1] = x
    t[-1,-1] = 1
    return t

def site_transformation_mat(model,data,site_name):
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    site_pos = data.site_xpos[site_id].copy()
    site_rotation = data.site_xmat[site_id].copy().reshape((3,3))

    return transformation_mat(site_pos,site_rotation)

def transform_twist(T, V):
    R_mat = T[:3, :3]
    p = T[:3, 3]

    rot = R.from_matrix(R_mat)

    v = V[:3]
    w = V[3:]

    w_new = rot.apply(w)
    v_new = rot.apply(v) + np.cross(p, w_new)

    return np.concatenate([v_new, w_new])

def transform_wrench(T, F):
    R_mat = T[:3, :3]
    p = T[:3, 3]
    f = F[:3]
    tau = F[3:]

    f_new = R_mat @ f
    tau_new = R_mat @ tau + np.cross(p, f_new)

    return np.concatenate([f_new, tau_new])


def get_dh_params():
    return  np.array([
                # a i-1,        alpha i-1,      di,                 thetai (home offset)])
                [0.0,           0.0,            0.0         ,       0.0,    ],
                [0.0,           np.pi/2,        0.0         ,       np.pi/2,],
                [a2,            0.0,            0.0         ,       0.0,    ],
                [a3,            0.0,            d4          ,       -np.pi/2,],
                [0.0,           -np.pi/2,       d5          ,       0.0,    ],
                [0.0,           np.pi/2,        0           ,       0.0,    ],])

# FORWARD DYNAMICS

def get_transformation_symbolic(N=7):

    T06 = sp.eye(4)
    dh_params = get_dh_params()

    if N > 6:
        T_final = T6TP
        N = 6
    else:
        T_final = sp.Identity(4)


    for i in range(N):
        a, alpha, d, theta = dh_params[i]
        theta += theta_sym[i]
        T_i = dh_sym(a,alpha,d,theta)

        T06 = T06 @ T_i

    return TWF0 @ T06 @ T_final

def jacobian_from_transform(T, q):
    """
    Compute the geometric Jacobian from a homogeneous transform.

    Parameters
    ----------
    T : sympy.Matrix (4x4)
        Homogeneous transformation matrix T(q)
    q : sympy.Matrix or list
        Joint variables [q1, q2, ..., qn]

    Returns
    -------
    J : sympy.Matrix (6 x n)
        Geometric Jacobian [Jv; Jw]
    """
    q = sp.Matrix(q)

    # position of end-effector
    p = T[:3, 3]

    # rotation matrix
    R = T[:3, :3]

    # Linear velocity Jacobian
    Jv = p.jacobian(q)

    # Angular velocity Jacobian
    Jw = sp.zeros(3, len(q))
    for i in range(len(q)):
        dR = R.diff(q[i])
        omega_hat = dR * R.T
        Jw[:, i] = sp.Matrix([
            omega_hat[2, 1],
            omega_hat[0, 2],
            omega_hat[1, 0]
        ])

    return Jv.col_join(Jw)
    
def get_jacobian_func():
    T_WF_TP = get_transformation_symbolic()
    J_sym = jacobian_from_transform(T_WF_TP,theta_sym)
    return sp.lambdify(theta_sym,J_sym,"numpy")


# %% Dynamics Matrices

def Bijk_sym(M, i, j, k): 
    return 0.5 * (sp.diff(M[i,j], theta[k]) + sp.diff(M[i,k], theta[j]) - sp.diff(M[j,k], theta[i]))

def qq_sym(qdot_in):
    n = len(qdot_in)
    all_dem = []
    for i in range(n-1):
        for j in range(i+1,n):
            all_dem.append( qdot_in[i]*qdot_in[j])

    return sp.Matrix(all_dem)

def get_link_masses():
    return np.array([3.761, 8.058, 2.846, 1.37, 1.300, 0.365])  # kg

def transformation_mat_sym(x,R):
    t = sp.zeros(4,4)
    t[0:3,0:3] = sp.Matrix(R)
    t[0:3,3] = sp.Matrix(x)
    t[3,3] = 1
    return t

def get_COM_tranformation(link_index):
    T_WF_link = get_transformation_symbolic(link_index)
    T_link_COM = transformation_mat_sym(COM_sim[link_index],np.identity(3))
    return T_WF_link @ T_link_COM

I1 = np.diag(np.array([0.0102675, 0.0102675, 0.00666]))
I2 = np.diag(np.array([0.0151074, 0.133886, 0.133886]))
I3 = np.diag(np.array([0.004095, 0.0311796, 0.0311796]))
I4 = np.diag(np.array([0.0025599, 0.0021942, 0.0025599]))
I5 = np.diag(np.array([0.0025599, 0.0021942, 0.0025599]))
I6 = np.diag(np.array([9.90863e-05, 9.90863e-05, 0.000132134]))
link_inertia_matrices_body_frame = [I1,I2,I3,I4,I5,I6]

def M_sym(thetas):

    M = sp.ZeroMatrix(6,6) # todo make based on size of Jacobian
    link_mass = get_link_masses()
    for i,q in enumerate(thetas):
        T = get_COM_tranformation(i)
        print(f"\n{i}")
        Rwi = T[0:3,0:3]
        J_w = jacobian_from_transform(T,thetas) # world frame
        print(J_w)
        Mv = link_mass[i] * (J_w[0:3,:].T @ J_w[0:3,:])
        Mw = J_w[3:,:].T @ Rwi @ link_inertia_matrices_body_frame[i] @ Rwi.T @ J_w[3:,:]
        print(Mv)
        print(Mw)
        M +=  Mv + Mw
    return M

def G_sym(thetas):
    g_vec = sp.Matrix([0, 0, -9.81])

    big_J = []
    long_G = []

    G = sp.zeros(len(thetas),1)

    link_mass = get_link_masses()
    for i,q in enumerate(thetas):
        T = get_COM_tranformation(i)
        J = jacobian_from_transform(T,thetas)[0:3,:]
        G += J.T @ (link_mass[i]*g_vec.copy())
        big_J.append(J)
        long_G.append(link_mass[i]*g_vec.copy())
    
    return G

# Centrifugal Matrix, C
def C_sym(M,theta):

    C = sp.Matrix([])

    for n1 in range(len(theta)):
        row = []
        for n2 in range(len(theta)):
            row.append(Bijk_sym(M, n1, n2, n2))
        C = C.row_insert(n1, sp.Matrix([row])) 

    return C

# Coriolis Matrix, B
def B_sym(M,thetas):

    B = sp.Matrix([])

    for row in range(len(thetas)):
        row_list = []
        for col1 in range(len(thetas)-1):
            for col2 in range(col1+1,len(thetas)):
                row_list.append(Bijk_sym(M, row, col1, col2))
        
        B = B.row_insert(row, sp.Matrix([row_list]))

    return B

def gravity_comp_only(model, data):
    qvel_backup = data.qvel.copy()

    data.qvel[:] = 0.0
    data.qacc[:] = 0.0

    result = np.zeros(model.nv)  # <-- required output buffer
    mj.mj_rne(model, data, 0, result)

    data.qvel[:] = qvel_backup

    return result

def get_robot_dynamics_func(save_path = None, load_path = None, gravity_vec = False):
    """returns a numerical function that evaluates the robot dynamics equation:

    returns Tau = M(q)*qddot + C(q)*qdot*qdot + B(q)*qq + G(q)

    3/9/2026
    test_vec = [0 for i in range(18)]
    t1 = t.time()
    w(*test_vec)
    t2 = t.time()
    print(t2-t1)
    -> 0.2563340663909912 s

    Args:
        q (np.array): _description_
        qdot (np.array): _description_
        qddot (np.array): _description_

    Returns:
        scalar: Force due to dynamics
    """
    q = sp.symbols('t1 t2 t3 t4 t5 t6')
    qdot = sp.symbols('td1 td2 td3 td4 td5 td6')
    qddot = sp.symbols('tdd1 tdd2 tdd3 tdd4 tdd5 tdd6')
    
    state = list(q)
    state.extend(list(qdot))
    state.extend(list(qddot))

    if load_path is not None:
        with open(load_path,'rb') as f: 
            subs, reduced = pickle.load(f)
    else:
        M_sym_ = M_sym(q)
        C_sym_ = C_sym(M_sym_,q)
        B_sym_ = B_sym(M_sym_,q)
        G_sym_ = G_sym(q)
        qq_sym_ = qq_sym(qdot)
        qdot2 = sp.matrix_multiply_elementwise(sp.Matrix(qdot),sp.Matrix(qdot))
        dynamics_symbolic = (
                            (M_sym_ * sp.Matrix(qddot)) +
                            (C_sym_ * qdot2) +
                            (B_sym_ * qq_sym_))
        if gravity_vec: dynamics_symbolic = dynamics_symbolic + G_sym_
    
        # dynamics_simplified = sp.simplify(dynamics_symbolic) MAKES SUPER SLOW...
        subs, reduced = sp.cse(dynamics_symbolic)

        if save_path is not None:
            with open(save_path,'wb') as f:pickle.dump((subs, reduced),f)


    sub_syms = [s[0] for s in subs]

    # lambdify each temporary using state + previous temporaries
    sub_funcs = []
    for i, (sym, expr) in enumerate(subs):
        args = state + sub_syms[:i]
        sub_funcs.append(sp.lambdify(args, expr, "numpy"))

    # final reduced expression
    dynamics_core = sp.lambdify(state + sub_syms, reduced, "numpy")

    def dynamics_func(*args):
        temps = []
        for f in sub_funcs:
            val = f(*args, *temps)
            temps.append(val)

        return np.array(dynamics_core(*args, *temps)).flatten()
    return dynamics_func


def get_robot_dynamics_components_func(save_path=None, load_path=None, gravity_vec=False):
    """
    Returns a function:

        M, C_term, B_term, G, tau

    where:
        M        : (6x6)
        C_term   : (6,)   -> C(q) * (qdot ∘ qdot)
        B_term   : (6,)
        G        : (6,)
        tau      : (6,)

    """
    t_start = time.time()

    # =========================
    # SYMBOLS
    # =========================
    q = sp.symbols('t1 t2 t3 t4 t5 t6')
    qdot = sp.symbols('td1 td2 td3 td4 td5 td6')
    qddot = sp.symbols('tdd1 tdd2 tdd3 tdd4 tdd5 tdd6')

    state = list(q) + list(qdot) + list(qddot)

    # =========================
    # LOAD OR BUILD SYMBOLICS
    # =========================
    if load_path is not None:
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
    else:
        M_sym_ = M_sym(q)
        C_sym_ = C_sym(M_sym_, q)
        B_sym_ = B_sym(M_sym_, q)
        G_sym_ = G_sym(q)

        qq_sym_ = qq_sym(qdot)
        qdot2 = sp.matrix_multiply_elementwise(sp.Matrix(qdot), sp.Matrix(qdot))

        # individual components
        M_term = M_sym_ * sp.Matrix(qddot)
        C_term = C_sym_ * qdot2
        B_term = B_sym_ * qq_sym_
        G_term = G_sym_ if gravity_vec else sp.zeros(6, 1)

        tau = M_term + C_term + B_term + G_term

        # CSE each separately
        data = {
            "M": sp.cse(M_sym_),
            "M_term": sp.cse(M_term),
            "C_term": sp.cse(C_term),
            "B_term": sp.cse(B_term),
            "G_term": sp.cse(G_term),
            "tau": sp.cse(tau),
        }

        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)

    # =========================
    # BUILD NUMERICAL FUNCTIONS
    # =========================

    def build_func(cse_pair):
        subs, reduced = cse_pair
        sub_syms = [s[0] for s in subs]

        sub_funcs = []
        for i, (sym, expr) in enumerate(subs):
            args = state + sub_syms[:i]
            sub_funcs.append(sp.lambdify(args, expr, "numpy"))

        core = sp.lambdify(state + sub_syms, reduced, "numpy")

        def func(*args):
            temps = []
            for f in sub_funcs:
                temps.append(f(*args, *temps))
            return np.array(core(*args, *temps))

        return func

    M_func      = build_func(data["M"])
    Mterm_func  = build_func(data["M_term"])
    C_func      = build_func(data["C_term"])
    B_func      = build_func(data["B_term"])
    G_func      = build_func(data["G_term"])
    tau_func    = build_func(data["tau"])

    # =========================
    # FINAL WRAPPER
    # =========================

    def dynamics_components_func(*args):
        """
        args = [q(6), qdot(6), qddot(6)]
        """

        M      = M_func(*args)
        Mterm  = Mterm_func(*args).flatten()
        Cterm  = C_func(*args).flatten()
        Bterm  = B_func(*args).flatten()
        Gterm  = G_func(*args).flatten()
        tau    = tau_func(*args).flatten()

        return {
            "M": M,
            "M_term": Mterm,
            "C_term": Cterm,
            "B_term": Bterm,
            "G": Gterm,
            "tau": tau
        }
    print(f"Finished in {time.time()-t_start:.1f} seconds")

    return dynamics_components_func



# f1 = get_robot_dynamics_func(save_path=r'C:\Users\lahmanjh\Documents\Github\simbox\universal_robots_ur5e\forward_dynamics.sympy')
# f2 = get_robot_dynamics_func(load_path=r'C:\Users\lahmanjh\Documents\Github\simbox\universal_robots_ur5e\forward_dynamics.sympy')


#%%

# # TODO
# def get_inverse_robot_dynamics_func(save_path = None, load_path = None):
#     """returns a numerical function that evaluates the robot dynamics equation:

#     returns qddot = M(q)^-1(Tau - C(q)*qdot*qdot - B(q)*qq - G(q))

#     3/9/2026
#     test_vec = [0 for i in range(18)]
#     t1 = t.time()
#     w(*test_vec)
#     t2 = t.time()
#     print(t2-t1)
#     -> 0.2563340663909912 s

#     Args:
#         q (np.array): _description_
#         qdot (np.array): _description_
#         qddot (np.array): _description_

#     Returns:
#         scalar: Force due to dynamics
#     """
#     q = sp.symbols('t1 t2 t3 t4 t5 t6')
#     qdot = sp.symbols('td1 td2 td3 td4 td5 td6')
#     qddot = sp.symbols('tdd1 tdd2 tdd3 tdd4 tdd5 tdd6')
    
#     state = list(q)
#     state.extend(list(qdot))
#     state.extend(list(qddot))

#     if load_path is not None:
#         subs, reduced = sp.load(load_path)
#     else:
#         M_sym_ = M_sym(q)
#         C_sym_ = C_sym(M_sym_,q)
#         B_sym_ = B_sym(M_sym_,q)
#         G_sym_ = G_sym(q)
#         qq_sym_ = qq_sym(qdot)
#         qdot2 = sp.matrix_multiply_elementwise(sp.Matrix(qdot),sp.Matrix(qdot))
#         dynamics_symbolic = (
#                             (M_sym_ * sp.Matrix(qddot)) +
#                             (C_sym_ * qdot2) +
#                             (B_sym_ * qq_sym_) +
#                             G_sym_)
    
#         # dynamics_simplified = sp.simplify(dynamics_symbolic) MAKES SUPER SLOW...
#         subs, reduced = sp.cse(dynamics_symbolic)

#         if save_path is not None:
#             sp.save((subs, reduced),save_path)


#     subs, reduced = sp.cse(dynamics_symbolic)


#     sub_syms = [s[0] for s in subs]

#     # lambdify each temporary using state + previous temporaries
#     sub_funcs = []
#     for i, (sym, expr) in enumerate(subs):
#         args = state + sub_syms[:i]
#         sub_funcs.append(sp.lambdify(args, expr, "numpy"))

#     # final reduced expression
#     dynamics_core = sp.lambdify(state + sub_syms, reduced, "numpy")

#     def dynamics_func(*args):
#         temps = []
#         for f in sub_funcs:
#             val = f(*args, *temps)
#             temps.append(val)

#         return dynamics_core(*args, *temps)
#     return dynamics_func


#%%