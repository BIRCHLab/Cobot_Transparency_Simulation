import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import xml.etree.ElementTree as ET
from kinematics import get_w_from_T,so3_log

def plot_timeseries(time, 
                          data, 
                          data_des=None,
                          data_label="Actual",
                          data_des_label="Desired",
                          joint_names=None, 
                          ylabel="Value", 
                          xlabel="Time",
                          title="",
                          ylim = None,
                          ylims = None,
                          save_path=None,
                          yy=False,
                          xyz_colors = False,
                          show=False):

    """
    Plot a time x n matrix with one vertical subplot per joint.

    Parameters
    ----------
    time : (T,) array-like
        Time vector.
    data : (T, N) array-like
        Time x joint matrix, where N is the number of joints.
    joint_names : list of str, optional
        Names for each joint (length N). Defaults to Joint 1, Joint 2, ...
    ylabel : str, optional
        Y-axis label for each subplot.
    xlabel : str, optional
        X-axis label (shown only on the bottom subplot).
    """
    data = np.asarray(data)
    time = np.asarray(time)

    if data.ndim != 2:
        raise ValueError("data must be a 2D array of shape (T, N)")

    T, N = data.shape

    if joint_names is None:
        joint_names = [f"Joint {i+1}" for i in range(N)]
    if len(joint_names) != N:
        raise ValueError("joint_names must have length N")

    fig, axes = plt.subplots(N, 1, sharex=True, figsize=(8, (1.25 * N)+1))

    # Handle the case N == 1
    if N == 1:
        axes = [axes]

    colors = ['red','green','blue','red','green','blue']


    for i, ax in enumerate(axes):
        if data_des is not None:
            ax.plot(time, data[:, i],
                    label=data_label,
                    color=colors[i] if xyz_colors else None)

            if yy:
                ax2 = ax.twinx()
                ax2.plot(time, data_des[:, i],
                        label=data_des_label,
                        linestyle='--',
                        color='k')

                # optional: separate limits for desired
                if ylims is not None:
                    ax.set_ylim(ylims[i][0])
                    ax2.set_ylim(ylims[i][1])

                # Align zero lines between ax and ax2
                y1_min, y1_max = ax.get_ylim()

                # position of zero in primary axis (0–1 in axis coords)
                zero_pos = (0 - y1_min) / (y1_max - y1_min)

                y2_min, y2_max = ax2.get_ylim()

                # compute new limits for ax2 so zero aligns
                new_y2_min = -zero_pos * (y2_max - y2_min)
                new_y2_max = (1 - zero_pos) * (y2_max - y2_min)

                ax2.set_ylim(new_y2_min, new_y2_max)

                # combined legend
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2)
            else:
                ax.plot(time, data_des[:, i],
                        label=data_des_label,
                        linestyle='--',
                        color='k')
                ax.legend()
        else: ax.plot(time, data[:, i])
        ax.set_ylabel(f"{joint_names[i]}\n{ylabel}")
        ax.grid(True)
        if ylim is not None: ax.set_ylim(ylim)
        if ylims is not None: ax.set_ylim(ylims[i])

    axes[-1].set_xlabel(xlabel)
    plt.suptitle(title)
    plt.tight_layout()
    if save_path is not None: plt.savefig(save_path)
    if show: plt.show(block=False)


def plot_timeseries_segments(
    time,
    data_list,
    segments,
    dt,
    segment_names=None,
    labels=None,
    line_styles=None,
    line_colors=None,
    ylabel="Value",
    xlabel="Time (s)",
    title="",
    ylim=None,
    save_path=None,
    show=False,
):
    """
    Plot multiple datasets over specified time segments.

    Parameters
    ----------
    time : (T,)
        Time vector.

    data_list : list of (T,1) or (T,)
        Data series to plot.

    segments : list of tuples
        [(t_start, t_end), ...]

    segment_names : list of str
        Names for each segment subplot.

    labels, line_styles, line_colors
        Same meaning as plot_timeseries().

    ylim : tuple, optional
        Common y-limits.

    Creates a 1 x n_segments layout.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    time = np.asarray(time)

    if labels is None:
        labels = [f"Data {i+1}" for i in range(len(data_list))]

    if line_styles is None:
        line_styles = ['-'] * len(data_list)

    if line_colors is None:
        line_colors = [None] * len(data_list)

    if segment_names is None:
        segment_names = [f"Segment {i+1}" for i in range(len(segments))]

    n_segments = len(segments)

    fig, axes = plt.subplots(
        1,
        n_segments,
        figsize=(4*n_segments, 4),
        sharey=True
    )

    if n_segments == 1:
        axes = [axes]

    for seg_idx, ((t_start, t_end), ax) in enumerate(zip(segments, axes)):

        mask = (time >= t_start) & (time <= t_end)

        for k, data in enumerate(data_list):

            data = np.asarray(data)

            if data.ndim == 1:
                y = data[:, np.newaxis]
            else:
                y = data

            ax.plot(
                time[mask],
                y[mask],
                label=labels[k],
                linestyle=line_styles[k],
                color=line_colors[k]
            )

            print(f'SPARC: {title}')  
            from smoothness import sparc
            print(f"{labels[k]} : {sparc(movement=y[mask].flatten(),fs=1/dt)[0]:.3f}")

        ax.set_title(segment_names[seg_idx])
        ax.grid(True)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlabel(xlabel)

    axes[0].set_ylabel(ylabel)

    if len(data_list) > 1:
        axes[0].legend()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show(block=False)

    return fig, axes

def plot_force_torque(
    vec,
    time=None,
    quantity_name="Force",
    component_labels=("Fx", "Fy", "Fz"),
    units="N",
    fig1_title=None,
    fig2_title=None,
    interval_ms=50,
    save_path=None,
    show = False,
):
    """
    Plot force/torque data as time traces and a 3D animated vector.

    Parameters
    ----------
    vec : (T, 3) array_like
        Force or torque vector over time.
    time : (T,) array_like or None
        Time vector. If None, sample indices are used.
    quantity_name : str
        Name of the quantity (e.g., "Force", "Torque").
    component_labels : tuple of str
        Labels for the x, y, z components.
    units : str
        Units for labeling axes.
    fig1_title : str or None
        Title for the time-series figure.
    fig2_title : str or None
        Title for the 3D animation figure.
    interval_ms : int
        Animation frame interval in milliseconds.
    """

    vec = np.asarray(vec)
    assert vec.ndim == 2 and vec.shape[1] == 3, "vec must be (T, 3)"

    T = vec.shape[0]

    if time is None:
        time = np.arange(T)
    else:
        time = np.asarray(time)
        assert time.shape[0] == T, "time must match vec length"

    # =========================
    # Figure 1: Time traces
    # =========================
    magnitude = np.linalg.norm(vec, axis=1)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    colors = ['red','green','blue']

    for i in range(3):
        ax1.plot(time, vec[:, i], label=component_labels[i],color=colors[i])

    ax1.plot(time, magnitude, "k--", linewidth=2, label=f"|{quantity_name}|")

    ax1.set_xlabel("Time")
    ax1.set_ylabel(f"{quantity_name} [{units}]")
    ax1.legend()
    ax1.grid(True)

    if fig1_title is None:
        fig1_title = f"{quantity_name} Components and Magnitude vs Time"
    ax1.set_title(fig1_title)
    if save_path is not None: plt.savefig(save_path)
    if show: plt.show(block=False)



    # =========================
    # Figure 2: 3D animation
    # =========================
    if fig2_title is not None:
        fig2 = plt.figure(figsize=(6, 6))
        ax2 = fig2.add_subplot(projection="3d")

        max_range = np.max(np.linalg.norm(vec, axis=1))
        lim = 1.1 * max_range

        ax2.set_xlim([-lim, lim])
        ax2.set_ylim([-lim, lim])
        ax2.set_zlim([-lim, lim])

        ax2.set_xlabel(component_labels[0])
        ax2.set_ylabel(component_labels[1])
        ax2.set_zlabel(component_labels[2])

        ax2.set_title(fig2_title)

        # Initialize vector (line from origin)
        line, = ax2.plot([0, 0], [0, 0], [0, 0], linewidth=3)

        def update(frame):
            x, y, z = vec[frame]
            line.set_data([0, x], [0, y])
            line.set_3d_properties([0, z])
            return line,

        anim = FuncAnimation(
            fig2,
            update,
            frames=T,
            interval=interval_ms,
            blit=True,
        )

        fps = int(1000/interval_ms)

        anim.save(save_path+'_animation',
                    writer="ffmpeg",
                    fps=fps
                    )


        return fig1, fig2, anim

def base_plots(d,
               save_folder,
               time_vec,
               CONTROL_HUMAN,
               CONTROL_ROBOT,
               save_name,
               human_joint_limits,
               human_u_min,
               human_u_max,
               baseline_traj,
               ):
    

        # Verification Plots

    det_jacobian = np.linalg.det(d['Jacobian'])
    cond_jacobianT = np.linalg.cond(np.transpose(np.array(d['Jacobian']),(0,2,1)))

    dt = np.mean(np.diff(time_vec))

    cartesian_pose_error = d['robot_ee_pose_model_trajectory_WF'][:,0:3,-1] - d['robot_ee_pose_trajectory_WF'][:,0:3,-1]

    from eval import check_manipulability
    plot_timeseries(
        data_list=[
            d['robot_jacobian_TS_velocity_trajectory_WF'],
            d['robot_TS_velocity_trajectory_WF_orientation']
        ],
        labels=['EE Twist from Jacobian', 'Mujoco measured'],
        line_styles=['-', '--'],
        line_colors=[None, 'k'],
        time=time_vec,
        joint_names=['X','Y', 'Z', 'Roll','Pitch','Yaw'],
        ylabel='[m/s,rad/s]',
        title="EE Velocities in world frame",
        save_path=os.path.join(save_folder,"plt_robot_ee_twist_comparison")
    )

    # plot_timeseries(
    #     data_list=[cartesian_pose_error],
    #     time=time_vec,
    #     xlabel='Time (s)',
    #     joint_names=["X","Y","Z"],
    #     ylabel='Distance [m]',
    #     title="Forward Kinematics Error (Cartesian)",
    #     save_path=os.path.join(save_folder,"plt_dynamic_model_comp")
    # )

    check_manipulability(d['human_arm_attachment_pose_WF'],
                        time_vec,
                        save_path=os.path.join(save_folder,"manip_"))

    plot_timeseries(
        data_list=[det_jacobian[:,np.newaxis]],
        time=time_vec,
        ylabel='-',
        title="Det(Jacobian)",
        save_path=os.path.join(save_folder,"plt_jacobian_det")
    )


    plot_timeseries(
        data_list=[cond_jacobianT[:,np.newaxis]],
        time=time_vec,
        ylabel='-',
        title="Cond(Jacobian.T)",
        save_path=os.path.join(save_folder,"plt_jacobianT_cond")
    )

    plot_timeseries(
        data_list=[
            d['robot_TS_velocity_trajectory_WF'] - d['robot_jacobian_TS_velocity_trajectory_WF']
        ],
        labels=['EE Vel Error due to Jacobian'],
        line_styles=['-'],
        time=time_vec,
        joint_names=['X','Y', 'Z', 'Roll','Pitch','Yaw'],
        ylabel='[m/s,rad/s]',
        title="EE Velocities in WF Error due to Jacobian",
        save_path=os.path.join(save_folder,"plt_robot_jacobian_check")
    )

    ################## HUMAN PLOTS ##################

    plot_timeseries(
        data_list=(
            [d['human_joint_angles'], d['human_pos_desired']]
            if CONTROL_HUMAN else
            [d['human_joint_angles']]
        ),
        labels=["Actual", "Desired"] if CONTROL_HUMAN else ["Actual"],
        line_styles=['-', '--'] if CONTROL_HUMAN else ['-'],
        line_colors=[None, 'k'] if CONTROL_HUMAN else [None],
        time=time_vec,
        joint_names=[f'J{i+1}' for i in range(7)],
        ylabel='Angle [rad]',
        title="Human Arm Trajectory",
        ylims=human_joint_limits,
        show=True,
        save_path=os.path.join(save_folder,"plt_human_joint_traj")
    )

    plot_timeseries(
        data_list=(
            [d['human_joint_angles'], d['human_pos_desired']]
            if CONTROL_HUMAN else
            [d['human_joint_angles']]
        ),
        labels=["Actual", "Desired"] if CONTROL_HUMAN else ["Actual"],
        line_styles=['-', '--'] if CONTROL_HUMAN else ['-'],
        line_colors=[None, 'k'] if CONTROL_HUMAN else [None],
        time=time_vec,
        joint_names=[f'J{i+1}' for i in range(7)],
        ylabel='Angle [rad]',
        title="Human Arm Trajectory",
        show=True,
        save_path=os.path.join(save_folder,"plt_human_joint_traj_zoomed")
    )


    if CONTROL_HUMAN:

        plot_timeseries(
            data_list=[d['human_control_torques']],
            time=time_vec,
            joint_names=[f'J{i+1}' for i in range(7)],
            ylabel='Torque [Nm]',
            title="Human Control Torques",
            ylims=[(umin-1,umax+1) for umin,umax in zip(human_u_min,human_u_max)],
            save_path=os.path.join(save_folder,"plt_human_control_torques")
        )

        plot_timeseries(
            data_list=[d['human_control_torques']],
            time=time_vec,
            joint_names=[f'J{i+1}' for i in range(7)],
            ylabel='Torque [Nm]',
            title="Human Control Torques",
            save_path=os.path.join(save_folder,"plt_human_control_torques_zoomed")
        )
    ################## Robot PLOTS ##################

    robot_pose = d['robot_ee_pose_model_trajectory_WF'][:,0:3,-1]
    robot_r = d['robot_ee_pose_model_trajectory_WF'][:,0:3,0:3]
    human_pose = d['human_arm_attachment_pose_WF'][:,0:3,-1]
    human_r = d['human_arm_attachment_pose_WF'][:,0:3,0:3]
    deflection = robot_pose-human_pose



    plot_timeseries(
        data_list=[deflection],
        time=time_vec,
        joint_names=['X','Y','Z'],
        ylabel='Angle [rad]',
        title="Maximum Linear Interface Deflection",
        save_path=os.path.join(save_folder,"plt_attachment_interface_deflection")
    )


    plot_timeseries(
        data_list=[d['robot_joint_angle_trajectory']],
        time=time_vec,
        joint_names=[f'J{i+1}' for i in range(6)],
        ylabel='Angle [rad]',
        title="Robot Joint Angles",
        save_path=os.path.join(save_folder,"plt_robot_joint_pos_traj")
    )

    if CONTROL_ROBOT: des = d['robot_desired_JS_velocity_trajectory']
    else: des = None


    plot_timeseries(
        data_list=(
            [d['robot_joint_velocity_trajectory'], des]
            if des is not None else
            [d['robot_joint_velocity_trajectory']]
        ),
        labels=["Actual", "Desired"] if des is not None else ["Actual"],
        line_styles=['-', '--'] if des is not None else ['-'],
        line_colors=[None, 'k'] if des is not None else [None],
        time=time_vec,
        joint_names=[f'J{i+1}' for i in range(6)],
        ylabel='Angle [rad/s]',
        show=True,
        save_path=os.path.join(save_folder,"plt_robot_joint_velocities"),
        title="Robot Joint Velocity",
    )

    if CONTROL_ROBOT and des is not None:

        plot_timeseries(
            data_list=[des - d['robot_joint_velocity_trajectory']],
            time=time_vec,
            joint_names=[f'J{i+1}' for i in range(6)],
            ylabel='Angle [rad/s]',
            show=True,
            save_path=os.path.join(save_folder,"plt_robot_joint_velocities_ERROR"),
            title="Robot Joint Velocity Error (Des-Measured)",
        )


    plot_timeseries(
        data_list=[d['robot_control_torques']],
        time=time_vec,
        joint_names=[f'J{i+1}' for i in range(6)],
        ylabel='Torque [Nm]',
        title="Robot Control Torques",
        save_path=os.path.join(save_folder,"plt_robot_control_torques")
    )

    plot_force_torque(d['robot_force_torque_sensor_EE'][:,3:],
                    time=time_vec,
                    quantity_name='Torque',
                    fig1_title="Interaction Torque On Robot (Sensor Frame)",
                    interval_ms=(dt*1000),
                    units="N*m",
                    save_path=os.path.join(save_folder,"torque_vec")
                    )

    plot_force_torque(d['robot_force_torque_sensor_EE'][:,0:3],
                    time=time_vec,
                    quantity_name='Force',
                    fig1_title="Interaction Force On Robot (Sensor Frame)",
                    interval_ms=(dt*1000),
                    show=True,
                    save_path=os.path.join(save_folder,"force_vec")
                    )

    if CONTROL_ROBOT:
        des = d['robot_desired_TS_velocity_trajectory_WF']
        des2 = d['robot_desired_TS_velocity_trajectory_EE']
    else:
        des = None
        des2 = None 

    plot_timeseries(
        data_list=(
            [d['robot_TS_velocity_trajectory_EE'], des2]
            if des2 is not None else
            [d['robot_TS_velocity_trajectory_EE']]
        ),
        labels=["Actual", "Desired"] if des2 is not None else ["Actual"],
        line_styles=['-', '--'] if des2 is not None else ['-'],
        line_colors=[None, 'k'] if des2 is not None else [None],
        time=time_vec,
        joint_names=['X','Y', 'Z', 'Roll','Pitch','Yaw'],
        ylabel='[m/s,rad/s]',
        title="EE Velocities EE Frame",
        xyz_colors=True,
        save_path=os.path.join(save_folder,"_plt_robot_TS_velocities_EE")
    )


    plot_timeseries(
        data_list=[
            d['robot_force_torque_sensor_EE'],
            d['robot_TS_velocity_trajectory_EE']
        ],
        labels=["Torque", "Velocity"],
        line_styles=['-', '--'],
        line_colors=[None, 'k'],
        time=time_vec,
        joint_names=['X','Y', 'Z', 'Mx','My','Mz'],
        ylabel='[Force Units]',
        title=f"{save_name} Force -> Velocity (EE Frame)",
        xyz_colors=True,
        yy=True,
        save_path=os.path.join(save_folder, "_plt_robot_force_velocity")
    )

    plot_timeseries(
        data_list=[
            d['robot_force_torque_sensor_WF_orientation'],
            d['robot_TS_velocity_trajectory_WF_orientation']
        ],
        labels=["F/T", "Velocity"],
        line_styles=['-', '--'],
        line_colors=[None, 'k'],
        time=time_vec,
        joint_names=['X','Y', 'Z', 'Mx','My','Mz'],
        ylabel='[Force Units]',
        title=f"{save_name} Force -> Velocity (WF Orientation)",
        xyz_colors=True,
        yy=True,
        save_path=os.path.join(save_folder, "plt_robot_force_velocity_WF")
    )
    
    ################ Compare Traj ################

    from toolbox import get_value_from_dict
    from eval import rmse_pos
    from human_movement import get_human_traj

    plot_timeseries(
        data_list=[d['human_arm_velocity_WF_orientation']],
        time=time_vec,
        joint_names=['X','Y', 'Z', 'Roll','Pitch','Yaw'],
        ylabel='[m/s,rad/s]',
        title="Human Attachment Velocities WF",
        xyz_colors=True,
        show=True,
        save_path=os.path.join(save_folder,"plt_human_TS_velocities_EE")
    )

    xyz_colors = ['red', 'blue', 'green']


    plot_timeseries(
        data_list=[ d['robot_force_torque_sensor_EE'], 
                    d['robot_desired_wrench_EE'] ],
        labels=["Actual", "Desired"],
        time=time_vec,
        joint_names=['X [N]','Y [N]','Z [N]', 'Mx [Nm]','My [Nm]','Mz [Nm]'],
        # ylabel='[N,Nm]',
        xyz_colors=True,
        line_styles=['-', '--'],
        line_colors=['k','k'],
        title=f"Desired Wrench Tracking",
        show=True,
        save_path=os.path.join(save_folder,"_plt_wrench")
    )




#%%







def body_transform_from_xml_file(xml_path, body_name="HUMAN"):
    # Parse XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find the body
    body = root.find(f".//body[@name='{body_name}']")
    if body is None:
        raise ValueError(f"Body '{body_name}' not found")
    
    # Extract position
    pos = np.fromstring(body.attrib["pos"], sep=" ")
    
    # Extract quaternion (w, x, y, z)
    quat = np.fromstring(body.attrib["quat"], sep=" ")
    w, x, y, z = quat

    # Quaternion → rotation matrix
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])

    # Homogeneous transform
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos

    return T

def load_mujoco_camera_from_xml(xml_path, camera_name):
    """
    Reads a MuJoCo camera definition from an XML file.

    Parameters
    ----------
    xml_path : str
        Path to MuJoCo XML file
    camera_name : str
        Name of the camera to extract

    Returns
    -------
    dict with:
        "pos": np.array shape (3,)
        "xyaxes": np.array shape (6,)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find camera by name (anywhere in XML)
    cam_elem = None
    for cam in root.iter("camera"):
        if cam.get("name") == camera_name:
            cam_elem = cam
            break

    if cam_elem is None:
        raise ValueError(f"Camera '{camera_name}' not found in XML")

    # Parse position
    pos_str = cam_elem.get("pos")
    if pos_str is None:
        raise ValueError(f"Camera '{camera_name}' missing 'pos'")
    pos = np.fromstring(pos_str, sep=' ')

    # Parse xyaxes
    xyaxes_str = cam_elem.get("xyaxes")
    if xyaxes_str is None:
        raise ValueError(f"Camera '{camera_name}' missing 'xyaxes'")
    xyaxes = np.fromstring(xyaxes_str, sep=' ')

    if pos.shape != (3,):
        raise ValueError(f"'pos' must have 3 values, got {pos.shape}")
    if xyaxes.shape != (6,):
        raise ValueError(f"'xyaxes' must have 6 values, got {xyaxes.shape}")

    return {
        "pos": pos,
        "xyaxes": xyaxes,
    }

def plot_trajectories_3d_multi_view(
    trajectories,
    labels,
    filename_base,
    extract_position_fn=None,
    dpi=300,
    mujoco_camera=None,
    views = {
        "iso": (30, -60),
        "top_xy": (90, -90),
        "front_xz": (0, 90),
    },

    # ---- font / style controls ----
    label_fontsize=12,
    tick_fontsize=10,
    legend_fontsize=10,
    title_fontsize=12,
    font_family=None,
    colors='k',
    linestyles='-',


    # ---- axis label defaults ----
    xlabel="X (cm)",
    ylabel="Y (cm)",
    zlabel="Z (cm)",

    # ---- tick formatting ----
    tick_precision=3,
    max_ticks=5,

    # ---- spacing / overlap controls ----
    labelpad=8,
    tick_pad=2,
    figsize=(6, 6),
    tight_layout_pad=0.5,

    # ---- legend ----
    show_legend=True,
    show=True,
    legend_loc="best",
):

    if font_family is not None:
        plt.rcParams["font.family"] = font_family

    def default_extract(traj):
        if traj.ndim == 3 and traj.shape[1:] == (4, 4):
            return traj[:, 0:3, 3]
        elif traj.ndim == 2 and traj.shape[1] >= 3:
            return traj[:, 0:3]
        else:
            raise ValueError("Unsupported trajectory format")

    extract_fn = extract_position_fn or default_extract

    # ---- compute global bounds ----
    all_points = np.vstack([extract_fn(traj) for traj in trajectories])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    span = (maxs - mins).max() / 2.0

    if mujoco_camera is not None:
        elev, azim, roll = mujoco_camera_to_mpl_view(
            mujoco_camera["pos"],
            mujoco_camera["xyaxes"]
        )
        views["mujoco_view"] = (elev, azim, roll)

            # ---- Normalize style inputs to lists ----

    def to_list(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x] * len(trajectories)

    colors = to_list(colors)
    linestyles = to_list(linestyles)

    for view_name, angles in views.items():
        elev, azim = angles[0], angles[1]
        roll = angles[2] if len(angles) > 2 else None

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        fig.patch.set_alpha(0.0)
        ax.set_facecolor((1, 1, 1, 0))

        start_labeled = False
        end_labeled = False

        for i, traj in enumerate(trajectories):
            pos = extract_fn(traj)


            # start point
            ax.scatter(
                pos[0, 0], pos[0, 1], pos[0, 2],
                c='k', marker='o', s=60,
                label="Start/End" if not start_labeled else None
            )
            start_labeled = True

            # end point
            ax.scatter(
                pos[-1, 0], pos[-1, 1], pos[-1, 2],
                c='k', marker='o', s=60,
                # label="Start/End" if not end_labeled else None
            )
            end_labeled = True

            # trajectory line
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=labels[i], linestyle=linestyles[i], color=colors[i])


        # ---- equal scaling ----
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.set_box_aspect([1, 1, 1])

        # ---- labels ----
        ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=labelpad)
        ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=labelpad)
        ax.set_zlabel(zlabel, fontsize=label_fontsize, labelpad=labelpad)

        # ---- tick formatting ----
        formatter = FormatStrFormatter(f"%.{tick_precision}f")
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_major_locator(MaxNLocator(nbins=max_ticks))
            axis.set_major_formatter(formatter)

        ax.tick_params(axis='both', which='major',
                       labelsize=tick_fontsize, pad=tick_pad)

        # ---- clean panes ----
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.grid(True)

        if show_legend:
            ax.legend(fontsize=legend_fontsize, loc=legend_loc)

        # ---- view ----
        if roll is not None:
            ax.view_init(elev=elev, azim=azim, roll=roll)
            plt.tight_layout(pad=tight_layout_pad)

        else:
            ax.view_init(elev=elev, azim=azim)
            plt.tight_layout(pad=tight_layout_pad)

        # ---- save ----
        plt.savefig(f"{filename_base}_{view_name}.png",
                    dpi=dpi, transparent=True)
        plt.savefig(f"{filename_base}_{view_name}.pdf",
                    dpi=dpi, transparent=True)

        if show:
            plt.show()

def preview_trajectory_view(
    trajectories,
    labels,
    extract_position_fn=None,
):
    """
    Opens an interactive window so you can rotate the plot.
    Prints the final camera angles after you close it.
    """

    def default_extract(traj):
        if traj.ndim == 3 and traj.shape[1:] == (4, 4):
            return traj[:, 0:3, 3]
        elif traj.ndim == 2 and traj.shape[1] >= 3:
            return traj[:, 0:3]
        else:
            raise ValueError("Unsupported trajectory format")

    extract_fn = extract_position_fn or default_extract

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for traj, label in zip(trajectories, labels):
        pos = extract_fn(traj)
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=label)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    ax.grid(True)

    plt.show()

    print(f"elev={ax.elev}, azim={ax.azim}")

def mujoco_camera_to_mpl_view(pos, xyaxes):
    """
    Convert MuJoCo camera (pos, xyaxes) to Matplotlib (elev, azim, roll).
    """
    x_axis = np.array(xyaxes[:3])
    y_axis = np.array(xyaxes[3:])

    # Camera forward (z axis)
    z_axis = np.cross(x_axis, y_axis)

    # Normalize
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Matplotlib looks along -z_axis
    view_dir = -z_axis

    # Compute azimuth and elevation
    azim = np.degrees(np.arctan2(view_dir[1], view_dir[0]))
    elev = np.degrees(np.arcsin(view_dir[2]))

    # --- Roll computation ---
    # Project camera x-axis into image plane to estimate roll
    up_world = np.array([0, 0, 1])
    right_proj = x_axis - np.dot(x_axis, view_dir) * view_dir
    right_proj /= np.linalg.norm(right_proj)

    up_proj = np.cross(view_dir, right_proj)

    roll = np.degrees(np.arctan2(
        np.dot(up_proj, up_world),
        np.dot(right_proj, up_world)
    ))

    return elev, azim, roll

def plot_trajectory_differences(traj1, traj2, t,save_path = None):
    """
    Plot absolute differences between two trajectories.

    Parameters
    ----------
    traj1 : (N,3) np.ndarray
        First trajectory (x,y,z over time)
    traj2 : (N,3) np.ndarray
        Second trajectory (x,y,z over time)
    t : (N,) np.ndarray
        Time vector

    Behavior
    --------
    - Plots |dx|, |dy|, |dz| vs time
    - Plots ||d|| (Euclidean norm) vs time
    """
    traj1 = np.asarray(traj1)
    traj2 = np.asarray(traj2)
    t = np.asarray(t)

    if traj1.shape != traj2.shape:
        raise ValueError("Trajectories must have the same shape")
    if traj1.shape[1] != 3:
        raise ValueError("Trajectories must be of shape (N,3)")
    if t.shape[0] != traj1.shape[0]:
        raise ValueError("Time vector length must match trajectory length")

    diff = traj1 - traj2
    norm_diff = np.linalg.norm(diff, axis=1)

    # Plot components
    plt.figure()
    plt.plot(t, diff[:, 0], label='dx',color='red')
    plt.plot(t, diff[:, 1], label='dy',color='green')
    plt.plot(t, diff[:, 2], label='dz',color='blue')
    plt.plot(t, norm_diff, label='||d||',linestyle='--',color='k')
    plt.xlabel('Time')
    plt.ylabel('Error [m]')
    plt.title('Trajectory Differences')
    plt.legend()
    plt.grid()
    if save_path is not None: plt.savefig()
    else: plt.show()


def plot_trajectory_magnitude_differences(
    baseline_traj,
    comparison_trajs,
    t,
    dt,
    labels=None,

    figsize=(2, 2),
    dpi=300,
    tight_layout=True,
    margins=None,

    scale=1.0,
    offset=0.0,

    colors='k',
    linestyles='-',
    linewidths=1.0,
    alphas=1.0,

    xlabel='Time [s]',
    ylabel='Position Error Norm',
    ylabel2='Angular Velocity Error Norm',
    title=None,
    xlim=None,
    ylim=None,
    xticks=None,
    yticks=None,
    pad_legend = 0.1,
    tick_fontsize=6,
    label_fontsize=7,
    title_fontsize=8,

    grid=True,
    grid_style=':',
    grid_alpha=0.5,

    legend=True,
    legend_loc='best',
    legend_fontsize=6,
    legend_frame=False,

    spine_linewidth=0.5,
    hide_top_right_spines=True,

    save_path=None,
    bbox_inches='tight',
    transparent=True,

    show=True
):
    baseline_T = np.asarray(baseline_traj)
    comparisons_T = [np.asarray(traj) for traj in comparison_trajs]
    t = np.asarray(t)

    # position
    baseline_pos = baseline_T[:, 0:3, -1]
    comparisons_pos = [traj[:, 0:3, -1] for traj in comparisons_T]


    n = len(comparisons_T)

    def to_list(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x] * n

    colors = to_list(colors)
    linestyles = to_list(linestyles)
    linewidths = to_list(linewidths)
    alphas = to_list(alphas)

    if labels is None:
        labels = [f"traj_{i}" for i in range(n)]

    # position norms
    pos_norms = []
    for traj in comparisons_pos:
        diff = baseline_pos - traj
        norm = np.linalg.norm(diff, axis=1) * scale + offset
        pos_norms.append(norm)

    print(f"{save_path} maxmimum linear error = {np.max(np.array(pos_norms))}")
    print(f"{save_path} final linear error = {pos_norms[-1][-1]}")

    # angular norms
    ang_norms = []

    for traj in comparisons_T:
        norms = []

        for k in range(len(baseline_T)):
            R_base = baseline_T[k, :3, :3]
            R_comp = traj[k, :3, :3]

            R_err = R_comp @ R_base.T

            # enforce SO(3) numerically
            U, _, Vt = np.linalg.svd(R_err)
            R_err = U @ Vt

            # rotation vector via log map approximation
            cos_theta = (np.trace(R_err) - 1) / 2
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            theta = np.arccos(cos_theta)

            norms.append(theta)

        ang_norms.append(np.array(norms))

    print(f"{save_path} maxmimum angular error = {np.max(np.array(ang_norms))}")
    print(f"{save_path} final angular error = {ang_norms[-1][-1]}")
    
    

    # figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax2 = ax.twinx()

    # plot position (left y-axis)
    i = 0
    ax.plot(
        t,
        pos_norms[0],
        label=labels[i] ,
        color=colors[i],
        linestyle=linestyles[i],
        linewidth=linewidths[i],
        alpha=alphas[i],
    )

    # plot angular (right y-axis)
    i = 1
    ax2.plot(
        t,
        ang_norms[0],
        label=labels[i],
        color=colors[i],
        linestyle=linestyles[i],
        linewidth=linewidths[i],
        alpha=alphas[i],
    )

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax2.set_ylabel(ylabel2, fontsize=label_fontsize)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if xticks is not None: ax.set_xticks(xticks)
    if yticks is not None: ax.set_yticks(yticks)

    # ---- Add 10% headroom for legend ----
    if ylim is None:
        ylim = ax.get_ylim()

    ymin, ymax = ax.get_ylim()
    pad = pad_legend * (ymax - ymin)
    ax.set_ylim(ymin, ymax + pad)

    ymin2, ymax2 = ax2.get_ylim()
    pad2 = pad_legend * (ymax2 - ymin2)
    ax2.set_ylim(ymin2, ymax2 + pad2)

    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax2.tick_params(axis='y', labelsize=tick_fontsize)

    if grid:
        ax.grid(True, linestyle=grid_style, alpha=grid_alpha)

    if legend:
        lines, labels_ = ax.get_legend_handles_labels()
        lines2, labels2_ = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2,
                  labels_ + labels2_,
                  loc=legend_loc,
                  fontsize=legend_fontsize,
                  frameon=legend_frame)

    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)

    if hide_top_right_spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if margins is not None:
        plt.subplots_adjust(**margins)

    if tight_layout:
        plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path,
                    dpi=dpi,
                    bbox_inches=bbox_inches,
                    transparent=transparent)

    if show:
        plt.show()

    return fig, (ax, ax2)

def plot_twist_magnitude_differences(
    baseline_traj,
    comparison_trajs,
    t,
    dt,
    labels=None,

    figsize=(2, 2),
    dpi=300,
    tight_layout=True,
    margins=None,

    scale=1.0,
    offset=0.0,

    colors='k',
    linestyles='-',
    linewidths=1.0,
    alphas=1.0,

    xlabel='Time [s]',
    ylabel='Linear Velocity Error Norm',
    ylabel2='Angular Velocity Error Norm',
    title=None,
    xlim=None,
    ylim=None,
    xticks=None,
    yticks=None,
    pad_legend=0.1,
    tick_fontsize=6,
    label_fontsize=7,
    title_fontsize=8,

    grid=True,
    grid_style=':',
    grid_alpha=0.5,

    legend=True,
    legend_loc='best',
    legend_fontsize=6,
    legend_frame=False,

    spine_linewidth=0.5,
    hide_top_right_spines=True,

    save_path=None,
    bbox_inches='tight',
    transparent=True,

    show=True
):
    baseline = np.asarray(baseline_traj)
    comparisons = [np.asarray(traj) for traj in comparison_trajs]
    t = np.asarray(t)

    # split twist
    baseline_lin = baseline[:, :3]
    baseline_ang = baseline[:, 3:]

    comparisons_lin = [traj[:, :3] for traj in comparisons]
    comparisons_ang = [traj[:, 3:] for traj in comparisons]

    n = len(comparisons)

    def to_list(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x] * n

    colors = to_list(colors)
    linestyles = to_list(linestyles)
    linewidths = to_list(linewidths)
    alphas = to_list(alphas)

    if labels is None:
        labels = [f"traj_{i}" for i in range(n)]

    # ---- linear norms ----
    lin_norms = []
    for traj in comparisons_lin:
        diff = baseline_lin - traj
        norm = np.linalg.norm(diff, axis=1) * scale + offset
        lin_norms.append(norm)

    print(f"{save_path} max linear vel error = {np.max(np.array(lin_norms))}")
    print(f"{save_path} final linear vel error = {lin_norms[-1][-1]}")

    # ---- angular norms ----
    ang_norms = []
    for traj in comparisons_ang:
        diff = baseline_ang - traj
        norm = np.linalg.norm(diff, axis=1)
        ang_norms.append(norm)

    print(f"{save_path} max angular vel error = {np.max(np.array(ang_norms))}")
    print(f"{save_path} final angular vel error = {ang_norms[-1][-1]}")

    # ---- plotting ----
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax2 = ax.twinx()

    # linear
    ax.plot(
        t,
        lin_norms[0],
        label=labels[0],
        color=colors[0],
        linestyle=linestyles[0],
        linewidth=linewidths[0],
        alpha=alphas[0],
    )

    # angular
    ax2.plot(
        t,
        ang_norms[0],
        label=labels[1] if len(labels) > 1 else "angular",
        color=colors[1] if len(colors) > 1 else colors[0],
        linestyle=linestyles[1] if len(linestyles) > 1 else linestyles[0],
        linewidth=linewidths[1] if len(linewidths) > 1 else linewidths[0],
        alpha=alphas[1] if len(alphas) > 1 else alphas[0],
    )

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax2.set_ylabel(ylabel2, fontsize=label_fontsize)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if xticks is not None: ax.set_xticks(xticks)
    if yticks is not None: ax.set_yticks(yticks)

    # padding
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + pad_legend * (ymax - ymin))

    ymin2, ymax2 = ax2.get_ylim()
    ax2.set_ylim(ymin2, ymax2 + pad_legend * (ymax2 - ymin2))

    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax2.tick_params(axis='y', labelsize=tick_fontsize)

    if grid:
        ax.grid(True, linestyle=grid_style, alpha=grid_alpha)

    if legend:
        lines, labels_ = ax.get_legend_handles_labels()
        lines2, labels2_ = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2,
                  labels_ + labels2_,
                  loc=legend_loc,
                  fontsize=legend_fontsize,
                  frameon=legend_frame)

    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)

    if hide_top_right_spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if margins is not None:
        plt.subplots_adjust(**margins)

    if tight_layout:
        plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path,
                    dpi=dpi,
                    bbox_inches=bbox_inches,
                    transparent=transparent)

    if show:
        plt.show()

    return fig, (ax, ax2)

#%%
if __name__=="__main__":
    from toolbox import load_dict_of_vectors,get_value_from_dict
    import os

    runs = [r"C:\Users\lahmanjh\Documents\Github\simbox\universal_robots_ur5e\data\Flexion_extension\movement_only",
            r"C:\Users\lahmanjh\Documents\Github\simbox\universal_robots_ur5e\data\Flexion_extension_tuning\4_human_feed_baseline"
            ]
    run_names = ['Free Movement',"Controlled"]

    trajectories = [get_value_from_dict(run,'human_arm_attachment_pose_WF') for run in runs]

    preview_trajectory_view(trajectories=trajectories,labels=run_names)

    plot_trajectories_3d_multi_view(trajectories=trajectories,
                        labels=run_names,
                        dpi=600,
                        filename_base=os.path.join(r"C:\Users\lahmanjh\Documents\Github\simbox\paper_figures",'test'))


#%%



