import numpy as np
import matplotlib.pyplot as plt
from ur5e_analytic import inverse_kinematics


def rmse_pos(t_1, t_2):
    p1 = t_1[:, :3, 3]
    p2 = t_2[:, :3, 3]
    return np.sqrt(np.mean(np.sum((p1 - p2)**2, axis=1)))

def check_manipulability(human_pose_trajectory,time,save_path,ylim = None):
    J_traj = [[],[],[],[]]
    for pose in human_pose_trajectory:
        q = inverse_kinematics(pose)
        # for i,soln in enumerate(q):
            # J_traj[i].append(get_06_jacobian(soln)) # TODO FIX ME
    
    fig, axes = plt.subplots(len(J_traj), 1, sharex=True)

    for i, ax in enumerate(axes):
        try: ax.plot(time, np.linalg.det(np.array(J_traj[i])))
        except: ax.plot(time,np.zeros_like(time),label="FAILED")
        # ax.set_ylabel(f"{joint_names[i]}\n{ylabel}")
        ax.grid(True)
        if ylim is not None: ax.set_ylim(ylim)

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle("Determinant(J)")
    plt.tight_layout()
    if save_path is not None: plt.savefig(save_path+"_det_J_")
    else: plt.show(block=False)


    fig, axes = plt.subplots(len(J_traj), 1, sharex=True)

    for i, ax in enumerate(axes):
        try: ax.plot(time, np.linalg.cond(np.transpose(np.array(J_traj[i]),(0,2,1))))
        except: ax.plot(time,np.zeros_like(time),label="FAILED")
        # ax.set_ylabel(f"{joint_names[i]}\n{ylabel}")
        ax.grid(True)
        ax.legend()
        if ylim is not None: ax.set_ylim(ylim)

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('Condition Number J.T')
    plt.tight_layout()
    if save_path is not None: plt.savefig(save_path+"_cond_JT_")
    else: plt.show(block=False)

