#%%
import numpy as np
import mujoco as mj
import mujoco.viewer
import time
from ur5e_analytic import get_site_frames, site_transformation_mat
from ur5e_analytic import get_jacobian_func, get_transformation
from ur5e_analytic import inverse_kinematics, transform_twist, transform_wrench
import imageio
from tqdm import tqdm
import os
from controllers import PID, Admittance
from toolbox import dump_globals_to_txt
import matplotlib.pyplot as plt
import copy
from toolbox import plot_control_trajectories
from ploty_the_plotypus import plot_force_torque, plot_timeseries
from human_movement import sinusoidal_trajectory
from human_movement import fe_traj_params, h2m_traj_params, multijoint_traj_params


# ============================
# Configuration
# ============================

FE_movement_only = {"soln_config":0,
                    "data_folder": r".\data\Flexion_Extension",
                    "ATTACHED": False,
                    "CONTROL_ROBOT": True,
                    "CONTROL_HUMAN": True,
                    "HUMAN_TRAJECTORY": "Flexion-Extension",
                    "baseline_traj": None,
                    "optimized_placement": False,
                    "camera":'coverpage',
                    "save_name":"movement_only",
                    "plt_tag":"FE_movement_only",
                    "XML_PATH": r".\xml\ur5e_7dof_baseline.xml",
                    }

FE_initial_pose = {"soln_config":3,
                    "data_folder": r".\data\Flexion_Extension",
                    "ATTACHED": True,
                    "CONTROL_ROBOT": True,
                    "CONTROL_HUMAN": True,
                    "HUMAN_TRAJECTORY": r".\data\Flexion_Extension\movement_only",
                    "baseline_traj": r".\data\Flexion_Extension\movement_only",
                    "optimized_placement": False,
                    "camera":'coverpage',
                    "save_name":"initial_pose",
                    "plt_tag":"FE_intial_pose",
                    "XML_PATH": r".\xml\ur5e_7dof_baseline.xml",
                    }

FE_general_optimized = {"soln_config":0,
                    "data_folder": r".\data\Flexion_Extension",
                    "ATTACHED": True,
                    "CONTROL_ROBOT": True,
                    "CONTROL_HUMAN": True,
                    "HUMAN_TRAJECTORY": r".\data\Flexion_Extension\movement_only",
                    "baseline_traj": r".\data\Flexion_Extension\movement_only",
                    "optimized_placement": True,
                    "opt_tag":'FE',
                    "camera":'coverpage',
                    "save_name":"general_optimization",
                    "plt_tag":"FE_optimized",
                    "XML_PATH": r".\xml\optimized_general_ur5e_7dof_baseline.xml",
                    }

multijoint_movement_only = {"soln_config":0,
                    "data_folder": r".\data\Multijoint",
                    "ATTACHED": False,
                    "CONTROL_ROBOT": True,
                    "CONTROL_HUMAN": True,
                    "HUMAN_TRAJECTORY": "Multijoint",
                    "baseline_traj": None,
                    "optimized_placement": False,
                    "camera":'coverpage',
                    "save_name":"movement_only",
                    "plt_tag":"multijoint_movement_only",
                    "XML_PATH": r".\xml\ur5e_7dof_baseline.xml",
                    }

multijoint_intial_pose = {"soln_config":3,
                    "data_folder": r".\data\Multijoint",
                    "ATTACHED": True,
                    "CONTROL_ROBOT": True,
                    "CONTROL_HUMAN": True,
                    "HUMAN_TRAJECTORY": r".\data\Multijoint\movement_only",
                    "baseline_traj": r".\data\Multijoint\movement_only",
                    "optimized_placement": False,
                    "camera":'coverpage',
                    "save_name":"initial_pose",
                    "plt_tag":"multijoint_intial_pose",
                    "XML_PATH": r".\xml\ur5e_7dof_baseline.xml",
                    }

multijoint_general_optimized = {"soln_config":0,
                    "data_folder": r".\data\Multijoint",
                    "ATTACHED": True,
                    "CONTROL_ROBOT": True,
                    "CONTROL_HUMAN": True,
                    "HUMAN_TRAJECTORY": r".\data\Multijoint\movement_only",
                    "baseline_traj": r".\data\Multijoint\movement_only",
                    "optimized_placement": True,
                    "opt_tag":'MJ',
                    "camera":'coverpage',
                    "save_name":"general_optimization",
                    "plt_tag":"multijoint_optimized",
                    "XML_PATH": r".\xml\optimized_general_ur5e_7dof_baseline.xml",
                    }

h2m_movement_only = {"soln_config":0,
                    "data_folder": r".\data\Hand to Mouth",
                    "ATTACHED": False,
                    "CONTROL_ROBOT": True,
                    "CONTROL_HUMAN": True,
                    "HUMAN_TRAJECTORY": "Hand to Mouth",
                    "baseline_traj": None,
                    "optimized_placement": False,
                    "camera":'coverpage',
                    "save_name":"movement_only",
                    "plt_tag":"h2m_movement_only",
                    "XML_PATH": r".\xml\ur5e_7dof_baseline.xml",

                    }

h2m_intial_pose = {"soln_config":3,
                    "data_folder": r".\data\Hand to Mouth",
                    "ATTACHED": True,
                    "CONTROL_ROBOT": True,
                    "CONTROL_HUMAN": True,
                    "HUMAN_TRAJECTORY": r".\data\Hand to Mouth\movement_only",
                    "baseline_traj": r".\data\Hand to Mouth\movement_only",
                    "optimized_placement": False,
                    "camera":'coverpage',
                    "save_name":"initial_pose",
                    "plt_tag":"h2m_intial_pose",
                    "XML_PATH": r".\xml\ur5e_7dof_baseline.xml",
                    }

h2m_general_optimized = {"soln_config":0,
                    "data_folder": r".\data\Hand to Mouth",
                    "ATTACHED": True,
                    "CONTROL_ROBOT": True,
                    "CONTROL_HUMAN": True,
                    "HUMAN_TRAJECTORY": r".\data\Hand to Mouth\movement_only",
                    "baseline_traj": r".\data\Hand to Mouth\movement_only",
                    "optimized_placement": True,
                    "opt_tag":'H2M',
                    "camera":'coverpage',
                    "save_name":"general_optimization",
                    "plt_tag":"h2m_optimized",     
                    "XML_PATH": r".\xml\optimized_general_ur5e_7dof_baseline.xml",
                    }

sims = [
        FE_movement_only,
        FE_initial_pose,

        multijoint_movement_only,
        multijoint_intial_pose,

        h2m_movement_only,
        h2m_intial_pose,
        ]

# %%UNIVERSAL SETTINGS
CONDITION = "COUPLED"
backdrivable = True
backdrivable_scalar = 1


dt = 0.001                # simulation timestep
total_time = 5           # seconds
buffer_time = 1
n_steps = int(total_time / dt)
time_vec = np.linspace(0,total_time,n_steps)

kp = 1000
ki = 10

robot_velocity_control_gains = {
    "ki":np.array([1, 1, 1, 1, 1, 1])*ki,
    "kp":np.array([1, 1, 1, 1, 1, 1])*kp,
    "kd":np.array([1, 1, 1, 1, 1, 1])*0,
}

human_control_gains = {
                        "kp":2000*np.ones(7),
                        "kd":10*np.ones(7),
                        "ki":0*np.ones(7),
                        }


linear_d = 1
ratio = 1/(4*np.pi*.2)
virtual_admittance_damping = np.identity(6)
virtual_admittance_damping[0:3,0:3] = virtual_admittance_damping[0:3,0:3] * linear_d
virtual_admittance_damping[3:,3:] = virtual_admittance_damping[3:,3:] * linear_d * ratio
 
filter_vref = True
filter_cutoff = 10
filter_FT = True
ft_cutoff = 1
solref=np.array([0.05, 1.5]) 
solimp=np.array([0.0001, 0.95, 0.001, 0.5, 1])
wrench_applied = np.array([0, 0, 0, 0, 0, 0]) 
cancel_damping = False
 

gravity = True
controller_partition = True

RENDER = False
SAVE_VIDEO = True
seed = 42
n_controllable_joints = 6
render_real_time = False
render_manipulability = False
angular_scalar = .8 # red
linear_scalar = 0 # blue

deadband = None
filter_torques = False

from toolbox import LowPassFilter
if filter_torques: 
    lp_torque = LowPassFilter(cutoff_freq=20,dt=dt,dim=n_controllable_joints)
if filter_vref: 
    lp_vref = LowPassFilter(cutoff_freq=filter_cutoff,dt=dt,dim=n_controllable_joints)
if filter_FT: 
    lp_FT = LowPassFilter(cutoff_freq=ft_cutoff,dt=dt,dim=n_controllable_joints)


#%% ################# RUN SIMS #######################
for set_dict in sims:
    
    save_name = set_dict['save_name']
    soln_config = set_dict['soln_config']
    data_folder = set_dict['data_folder']
    camera = set_dict['camera'] 
    baseline_traj = set_dict['baseline_traj']
    optimized_placement = set_dict['optimized_placement']
    HUMAN_TRAJECTORY = set_dict['HUMAN_TRAJECTORY']
    ATTACHED =set_dict['ATTACHED']
    CONTROL_ROBOT = set_dict['CONTROL_ROBOT']
    CONTROL_HUMAN = set_dict['CONTROL_HUMAN']
    XML_PATH = set_dict['XML_PATH'] 


    # ============================
    # Load model
    # ============================

    if optimized_placement:
        folder, new_file = os.path.split(XML_PATH)
        folder = os.path.dirname(XML_PATH)
        print(f"Using optimized XML: {XML_PATH}")
        with open(os.path.join(folder,new_file.split('.')[0]+".txt")) as f:
            line = f.readline()
            soln_config = int(line[-2])
            print(f"Solution_Config: {soln_config}")

    model = mj.MjModel.from_xml_path(XML_PATH)
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    model.opt.timestep = dt
    if gravity: model.opt.gravity[:] = np.array([0, 0, -9.81])
    else: model.opt.gravity[:] = np.array([0, 0, 0])

    arm_weld_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_EQUALITY,"EE_arm_weld" )
    ee_body_id = mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY, 'interface_attachment_body')

    n_human_dof = data.ctrl.shape[0] - n_controllable_joints 
    human_u_min = model.actuator_ctrlrange[-n_human_dof:,0]
    human_u_max = model.actuator_ctrlrange[-n_human_dof:,1]
    robot_u_min = model.actuator_ctrlrange[0:n_controllable_joints,0]
    robot_u_max = model.actuator_ctrlrange[0:n_controllable_joints,1]
    robot_control_lims = model.actuator_ctrlrange[0:n_controllable_joints,:]

    human_joint_limits = [(model.jnt_range[i,0],model.jnt_range[i,1]) for i in range(-n_human_dof,0,1)]
    robot_joint_limits = [(model.jnt_range[i,0],model.jnt_range[i,1]) for i in range(0,n_controllable_joints,1)]
    robot_joint_limits_array = model.jnt_range[0:n_controllable_joints,:]
    home_qpos = model.keyframe('home').qpos


    save_folder = os.path.join(data_folder,save_name)
    os.makedirs(save_folder, exist_ok=True)
    dump_globals_to_txt(os.path.join(save_folder,'settings.txt'))


    ###### Reset FUNCTION #######

    def reset(model, 
            data,
            condition = 'COUPLED',
            q_init = None,
            ):
        data.eq_active[arm_weld_id] = 0

        print(f"Default solref: {model.eq_solref[arm_weld_id]}")
        print(f"Default solimp: {model.eq_solimp[arm_weld_id]}")

        model.eq_solref[arm_weld_id] = solref
        model.eq_solimp[arm_weld_id] = solimp

        if q_init is None: 
            q_init = home_qpos

        data.qpos[:] = q_init
        data.qvel[:] = 0.0

        mj.mj_forward(model, data) # make the model so that arm is in initial config
        if ATTACHED:
            HA_T = site_transformation_mat(model,data,'human_arm_attachment_site')

            q_init[0:6] = inverse_kinematics(HA_T)[soln_config,:]
                
            # test forward kinematics (bonus)
            test_angles = np.ones(6)
            T = get_transformation(test_angles)
            soln = inverse_kinematics(T)
            passed = False
            min_error = 1
            for s in soln:
                if np.sum(np.abs(test_angles-s)) < 0.0001: 
                    passed = True
                    min_error = np.sum(np.abs(test_angles-s))

            if passed: print(f'Forward and inverse kinematics seem to work | Error: {min_error}')
            else: print('DOUBLE CHECK FORWARD/INVERSE KINEMATICS')

            data.qpos[:] = q_init
            data.qvel[:] = 0.0
            data.eq_active[arm_weld_id] = 1
            mj.mj_forward(model, data) # make the model so that arm is in inital config AND robot is now attached


    ###### Controllers #######

    admittance_controller = Admittance(n_outputs=n_controllable_joints,
                                    virtual_inertia=np.zeros_like(virtual_admittance_damping),
                                    virtual_damping=virtual_admittance_damping,
                                    deadband=deadband,
                                    dt = dt)

    velocity_PID_controller = PID(n_outputs=6,
                                kp=robot_velocity_control_gains["kp"],
                                kd=robot_velocity_control_gains["kd"],
                                ki=robot_velocity_control_gains["ki"],
                                dt = dt)

    human_controller = PID(n_outputs=n_human_dof,
                        kp=human_control_gains['kp'],
                        kd=human_control_gains['kd'],
                        ki=human_control_gains['ki'],
                        dt = dt)
    

    ###### TRAJECTORY PLANNING ######

    wrench_des_TS_ee = np.array([0, 0, 0, 0, 0, 0])

    if CONTROL_HUMAN:

        

        end_param = np.zeros(n_human_dof)
        if HUMAN_TRAJECTORY == "Flexion-Extension": end_param = fe_traj_params
        elif HUMAN_TRAJECTORY == "Multijoint": end_param = multijoint_traj_params
        elif HUMAN_TRAJECTORY == "Hand to Mouth": end_param = h2m_traj_params


        human_pos_desired = sinusoidal_trajectory(n_steps,
                                                    n_cycles=1,
                                                    hold_steps=int(buffer_time/dt),
                                                    ndim=n_human_dof,
                                                    start=home_qpos[-n_human_dof:], 
                                                    end=end_param)
        
        if r'data' in HUMAN_TRAJECTORY:
            print('LOADED HUMAN TORQUE TRAJECTORY')
            from toolbox import save_dict_of_vectors,load_dict_of_vectors
            old_d = load_dict_of_vectors(HUMAN_TRAJECTORY)
            human_pos_desired = np.array(old_d['human_joint_angles'])
            loaded_human_u = np.array(old_d['human_control_torques'])
            time_vec = old_d['time']
            plot_control_trajectories(loaded_human_u,
                            t=time_vec,
                            save_path=os.path.join(save_folder,"plt_human_loaded_torques_traj"))
        
            
        plot_control_trajectories(human_pos_desired,
                                t=time_vec,
                                save_path=os.path.join(save_folder,"plt_human_desired_traj"))
        


    ###### SIMULATION SETTINGS ######

    reset(model,data,CONDITION)

    if SAVE_VIDEO:
        width, height = 1280, 720
        renderer = mj.Renderer(model, width=width, height=height)
        video = imageio.get_writer(
            os.path.join(save_folder,f"1_render_{save_name}.mp4"),
            format="ffmpeg",
            fps=int(1/dt),
            codec="libx264",
            quality=8
        )
        renderer_ngeom = renderer.scene.ngeom

    if RENDER:
        viewer = mujoco.viewer.launch_passive(model, data)

    d = {           
                # Kinetics
                "robot_force_torque_sensor_EE":[],
                "robot_force_torque_sensor_WF":[],
                "robot_force_torque_sensor_WF_orientation":[],
                "robot_joint_torques":[],

                # Joint Kinematics
                "robot_joint_angle_trajectory":[],
                "robot_joint_velocity_trajectory":[],
                "robot_joint_acceleration_trajectory":[],

                # Cartesian Kinematics
                "robot_ee_pose_trajectory_WF":[],
                "robot_ee_pose_model_trajectory_WF":[],

                # Robot Model
                "robot_model_predicted_acceleration":[],
                "robot_model_predicted_joint_torque":[],

                # Control Variables
                "robot_control_torques":[],
                "external_wrench_error":[],
                # "robot_desired_joint_angles":[],
                "robot_desired_JS_velocity_trajectory":[],
                "robot_desired_TS_velocity_trajectory_WF":[],
                "robot_jacobian_TS_velocity_trajectory_WF":[],
                "robot_TS_velocity_trajectory_WF":[],

                "robot_desired_TS_velocity_trajectory_EE":[],
                "robot_TS_velocity_trajectory_EE":[],
                "robot_TS_velocity_trajectory_WF":[],
                "robot_TS_velocity_trajectory_WF_orientation":[],

                # Human Vars
                "human_arm_attachment_pose_WF":[],
                "human_joint_angles":[],
                "human_control_torques":[],
                "human_control_torques_WF":[],
                "human_arm_velocity_WF_orientation":[],
                
                "human_joint_torques":[],

                # Metrics
                "Jacobian":[],
                "admittance_u":[],
                }

    ###### SIMULATE ######
    J_func = get_jacobian_func()

    if controller_partition:
        robot_model = mj.MjModel.from_xml_path(r".\xml\ur5e_alone.xml")
        robot_data = mj.MjData(robot_model)
        if gravity: robot_model.opt.gravity[:] = np.array([0, 0, -9.81])
        else: robot_model.opt.gravity[:] = np.array([0, 0, 0])
        mj.mj_forward(robot_model, robot_data)


    for step in tqdm(range(n_steps)):
        
        # get human/interface state
        T_EE_in_WF = site_transformation_mat(model,data,'distal_interface_site')
        T_EE_in_WF_orientation = T_EE_in_WF.copy()
        T_EE_in_WF_orientation[0:3,-1] = np.array([0, 0, 0]) # remove position component


        # MEASURE
        robot_force_torque_sensor_EE = -1*data.sensordata.copy()[0:6]
        if filter_FT: robot_force_torque_sensor_EE = lp_FT.update(robot_force_torque_sensor_EE)

        robot_force_torque_sensor_WF = transform_wrench(T_EE_in_WF,robot_force_torque_sensor_EE) 
        robot_force_torque_sensor_WF_orientation = transform_wrench(T_EE_in_WF_orientation,robot_force_torque_sensor_EE)
        unfiltered_FT_WF = transform_wrench(T_EE_in_WF_orientation,data.sensordata.copy()[0:6])

        robot_velocity_TS_EE = data.sensordata.copy()[6:] # this is actually in world coords for rotational quants???
        robot_velocity_TS_WF = transform_twist(T_EE_in_WF, robot_velocity_TS_EE)
        robot_velocity_TS_WF_orientation = transform_twist(T_EE_in_WF_orientation, robot_velocity_TS_EE)

        # robot state
        robot_joint_angles = data.qpos[0:n_controllable_joints].copy()
        robot_joint_velocities = data.qvel[0:n_controllable_joints].copy()
        robot_joint_accelerations = data.qacc[0:n_controllable_joints].copy()
        robot_joint_torques = data.qfrc_actuator + data.qfrc_bias + data.qfrc_constraint + data.qfrc_passive

        J_WF = J_func(*robot_joint_angles)
        robot_velocity_TS_WF_jacobian = J_WF @ robot_joint_velocities

        human_joint_angles = data.qpos[n_controllable_joints:].copy()
        human_arm_site_in_WF = site_transformation_mat(model,data,'human_arm_attachment_site')
        T_EE_in_WF_model = get_transformation(robot_joint_angles)
        
        site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "human_arm_attachment_site")

        # Get body id that the site belongs to
        human_attachment_body_id = model.site_bodyid[site_id]

        # Body spatial velocity (twist) in world frame
        human_arm_velocity_WF_orientation = data.cvel[human_attachment_body_id]  # (6,) -> [vx, vy, vz, wx, wy, wz]


        # CALCULATE CONTROL
        u = np.zeros(n_controllable_joints)
        human_u = np.zeros(n_human_dof)
        velocity_ref_JS = np.zeros(n_controllable_joints)
        external_wrench_error = np.zeros(6)
        velocity_ref_TS_WF_orientation = np.zeros(6)
        velocity_ref_TS_EE = np.zeros(6)
        qDes = np.zeros((soln_config+1,n_controllable_joints))
        u = np.zeros(n_controllable_joints)

        if CONTROL_ROBOT and (step+1)*dt > buffer_time:

            # measure the interaction force in the end effector frame
            external_wrench_TS_ee = robot_force_torque_sensor_EE # defining this immediately negative so that a pull on the EE measures as one.

            #calculate the velocity reference in task space
            external_wrench_error = external_wrench_TS_ee - wrench_des_TS_ee # see note in admittance controller for sign
            velocity_ref_TS_EE = admittance_controller.calculate(Fmeas=external_wrench_TS_ee, 
                                                                Fdes=wrench_des_TS_ee)
            
            if filter_vref:
                velocity_ref_TS_EE = lp_vref.update(velocity_ref_TS_EE)
            
            # transform velocity reference to the ground frame
            velocity_ref_TS_WF_orientation = transform_twist(T_EE_in_WF_orientation,velocity_ref_TS_EE)

            # SCALE TCP speed
            if np.linalg.norm(velocity_ref_TS_WF_orientation[0:3]) > 4:
                velocity_ref_TS_WF_orientation[0:3] = velocity_ref_TS_WF_orientation[0:3]*4/np.linalg.norm(velocity_ref_TS_WF_orientation[0:3])
            
            # invert jacobian
            velocity_ref_JS = np.linalg.pinv(J_WF) @ velocity_ref_TS_WF_orientation

            # SCALE TO joint rate limits
            if np.sum(np.abs(velocity_ref_JS) > np.pi):
                velocity_ref_JS = velocity_ref_JS * np.pi/np.max(velocity_ref_JS)

            u = velocity_PID_controller.calculate(robot_joint_velocities, velocity_ref_JS)

        if not backdrivable:
            joint_torque = J_WF.T @ unfiltered_FT_WF
            data.qfrc_applied[0:n_controllable_joints] = +backdrivable_scalar*joint_torque

        if controller_partition:
            data.xfrc_applied[human_attachment_body_id, :6] = wrench_applied

            # u = u + velocity_ref_JS/dt # for some reason it doesn't like this...
            robot_data.qpos[:] = data.qpos[:6].copy()
            robot_data.qvel[:] = data.qvel[:6].copy()
            mj.mj_forward(robot_model,robot_data)

            M_robot = np.zeros((robot_model.nv, robot_model.nv))
            mj.mj_fullM(robot_model, M_robot, robot_data.qM)
            bias_robot = robot_data.qfrc_bias.copy()
            damping_robot = robot_data.qfrc_damper.copy()

            if cancel_damping: u = ((M_robot @ u) + bias_robot + damping_robot)
            else:u = ((M_robot @ u) + bias_robot)
            # data.ctrl[:n_controllable_joints] = u

        if CONTROL_HUMAN: 
            human_u = human_controller.calculate(human_joint_angles,human_pos_desired[step,:])
            human_u = np.clip(human_u,human_u_min,human_u_max)

        
        if r'data' in HUMAN_TRAJECTORY:
            human_u = loaded_human_u[step,:]

        # APPLY CONTROL
        if filter_torques: u = lp_torque.update(u)
        u = u/101
        u = np.clip(u,robot_control_lims[0:n_controllable_joints,0],robot_control_lims[0:n_controllable_joints,1])

        if CONTROL_HUMAN: data.ctrl[-n_human_dof:] = human_u
        if CONTROL_ROBOT: data.ctrl[:n_controllable_joints] = u


        # LOGGING
        if True:
            # record relevant data
            d['robot_force_torque_sensor_EE'].append(robot_force_torque_sensor_EE.copy())
            d['robot_force_torque_sensor_WF'].append(robot_force_torque_sensor_WF.copy())
            d['robot_force_torque_sensor_WF_orientation'].append(robot_force_torque_sensor_WF_orientation.copy())
            
            d['robot_TS_velocity_trajectory_EE'].append(robot_velocity_TS_EE.copy())
            d['robot_TS_velocity_trajectory_WF'].append(robot_velocity_TS_WF.copy())
            d['robot_TS_velocity_trajectory_WF_orientation'].append(robot_velocity_TS_WF_orientation.copy())
            
            d['robot_joint_torques'].append(robot_joint_torques[0:n_controllable_joints].copy())

            # Kinematics
            d['robot_ee_pose_trajectory_WF'].append(T_EE_in_WF.copy())
            d['robot_ee_pose_model_trajectory_WF'].append(T_EE_in_WF_model.copy())
            
            d['robot_jacobian_TS_velocity_trajectory_WF'].append(robot_velocity_TS_WF_jacobian.copy())
            d['robot_joint_angle_trajectory'].append(robot_joint_angles.copy())
            d['robot_joint_velocity_trajectory'].append(robot_joint_velocities.copy())
            d['robot_joint_acceleration_trajectory'].append(robot_joint_accelerations.copy())

            # Control Variables
            d['robot_control_torques'].append(u.copy()*101) # multiply by gear ratio for effective torque
            d['Jacobian'].append(J_WF)

            # Human Vars
            d['human_joint_torques'].append(robot_joint_torques[-4:].copy())
            d['human_arm_attachment_pose_WF'].append(human_arm_site_in_WF.copy())
            d['human_joint_angles'].append(human_joint_angles.copy())
            d['human_control_torques'].append(human_u.copy())
            d['human_arm_velocity_WF_orientation'].append(human_arm_velocity_WF_orientation.copy())

            if CONTROL_ROBOT:
                d['external_wrench_error'].append(external_wrench_error.copy())
                d["robot_desired_JS_velocity_trajectory"].append(velocity_ref_JS.copy())
                d["robot_desired_TS_velocity_trajectory_WF"].append(velocity_ref_TS_WF_orientation.copy())
                d["robot_desired_TS_velocity_trajectory_EE"].append(velocity_ref_TS_EE.copy())


        if render_manipulability:
            from toolbox import render_manipulability_geometry
            render_manipulability_geometry(
                viewer,
                T_EE_in_WF,
                J_WF,
                linear_scalar=linear_scalar,
                angular_scalar=angular_scalar,
                renderer=renderer if SAVE_VIDEO else None,
                renderer_ngeom=renderer_ngeom if SAVE_VIDEO else None,
            )

        # STEP SIMULATION
        if True:

            mj.mj_step(model, data)

            if SAVE_VIDEO:
                renderer.update_scene(data, camera=camera)
                frame = renderer.render()          # (H, W, 3), uint8
                video.append_data(frame)

            # Render
            if RENDER:
                viewer.sync()

            # Optional real-time pacing
            if render_real_time: time.sleep(dt)


    if RENDER:
        viewer.close() 

    if SAVE_VIDEO:
        video.close()
        renderer.close()

    run_to_save = copy.deepcopy(d)
    for key, value in d.items():
        d[key] = np.array(d[key])

    # Verification Plots

    det_jacobian = np.linalg.det(d['Jacobian'])
    cond_jacobianT = np.linalg.cond(np.transpose(np.array(d['Jacobian']),(0,2,1)))

    cartesian_pose_error = d['robot_ee_pose_model_trajectory_WF'][:,0:3,-1] - d['robot_ee_pose_trajectory_WF'][:,0:3,-1]

    from eval import check_manipulability
    plot_timeseries(data=d['robot_jacobian_TS_velocity_trajectory_WF'], 
                    data_label='EE Twist from Jacobian',
                    time=time_vec,
                    data_des = d['robot_TS_velocity_trajectory_WF_orientation'],
                    data_des_label="Mujoco measured",
                    joint_names=['X','Y', 'Z', 'Roll','Pitch','Yaw'],
                    ylabel='[m/s,rad/s]',
                    title="EE Velocities in world frame",
                    save_path=os.path.join(save_folder,"plt_robot_ee_twist_comparison"))

    plot_timeseries(data=cartesian_pose_error,
                    time=time_vec,
                    xlabel='Time (s)',
                    joint_names=["X","Y","Z"],
                    ylabel='Distance [m]',
                    title="Forward Kinematics Error (Cartesian)",
                    save_path=os.path.join(save_folder,"plt_dynamic_model_comp"))

    check_manipulability(d['human_arm_attachment_pose_WF'],
                        time_vec,
                        save_path=os.path.join(save_folder,"manip_"))

    plot_timeseries(data=det_jacobian[:,np.newaxis], 
                        time=time_vec,
                        ylabel='-',
                        title="Det(Jacobian)",
                        save_path=os.path.join(save_folder,"plt_jacobian_det"))

    plot_timeseries(data=cond_jacobianT[:,np.newaxis], 
                        time=time_vec,
                        ylabel='-',
                        title="Cond(Jacobian.T)",
                        save_path=os.path.join(save_folder,"plt_jacobianT_cond"))

    plot_timeseries(data=d['robot_TS_velocity_trajectory_WF'] - d['robot_jacobian_TS_velocity_trajectory_WF'], 
                    data_label='EE Vel Error due to Jacobian',
                    time=time_vec,
                    joint_names=['X','Y', 'Z', 'Roll','Pitch','Yaw'],
                    ylabel='[m/s,rad/s]',
                    title="EE Velocities in WF Error due to Jacobian",
                    save_path=os.path.join(save_folder,"plt_robot_jacobian_check"))

    ################## HUMAN PLOTS ##################

    plot_timeseries(data=d['human_joint_angles'], 
                        time=time_vec,
                        data_des=human_pos_desired if CONTROL_HUMAN else None,
                        joint_names=[f'J{i+1}' for i in range(human_joint_angles.shape[-1])],
                        ylabel='Angle [rad]',
                        title="Human Arm Trajectory",
                        ylims=human_joint_limits,
                        show=True,
                        save_path=os.path.join(save_folder,"plt_human_joint_traj"))

    plot_timeseries(data=d['human_joint_angles'], 
                        time=time_vec,
                        data_des=human_pos_desired if CONTROL_HUMAN else None,
                        joint_names=[f'J{i+1}' for i in range(human_joint_angles.shape[-1])],
                        ylabel='Angle [rad]',
                        title="Human Arm Trajectory",
                        show=True,
                        save_path=os.path.join(save_folder,"plt_human_joint_traj_zoomed"))

    if CONTROL_HUMAN:
        plot_timeseries(data=d['human_control_torques'], 
                        time=time_vec,
                        joint_names=[f'J{i+1}' for i in range(n_human_dof)],
                        ylabel='Torque [Nm]',
                        title="Human Control Torques",
                        ylims=[(umin-1,umax+1) for umin,umax in zip(human_u_min,human_u_max)],
                        save_path=os.path.join(save_folder,"plt_human_control_torques"))

        plot_timeseries(data=d['human_control_torques'], 
                        time=time_vec,
                        joint_names=[f'J{i+1}' for i in range(n_human_dof)],
                        ylabel='Torque [Nm]',
                        title="Human Control Torques",
                        # ylims=[(umin-1,umax+1) for umin,umax in zip(human_u_min,human_u_max)],
                        save_path=os.path.join(save_folder,"plt_human_control_torques_zoomed"))

    ################## Robot PLOTS ##################

    robot_pose = d['robot_ee_pose_model_trajectory_WF'][:,0:3,-1]
    robot_r = d['robot_ee_pose_model_trajectory_WF'][:,0:3,0:3]
    human_pose = d['human_arm_attachment_pose_WF'][:,0:3,-1]
    human_r = d['human_arm_attachment_pose_WF'][:,0:3,0:3]
    deflection = robot_pose-human_pose

    plot_timeseries(data=deflection, 
                        time=time_vec,
                        joint_names=['X','Y','Z'],
                        ylabel='Angle [rad]',
                        title="Maximum Linear Interface Deflection",
                        save_path=os.path.join(save_folder,"plt_attachment_interface_deflection"))

    plot_timeseries(data=d['robot_joint_angle_trajectory'], 
                        time=time_vec,
                        joint_names=[f'J{i+1}' for i in range(6)],
                        ylabel='Angle [rad]',
                        title="Robot Joint Angles",
                        save_path=os.path.join(save_folder,"plt_robot_joint_pos_traj"))

    if CONTROL_ROBOT: des = d['robot_desired_JS_velocity_trajectory']
    else: des = None

    plot_timeseries(data=d['robot_joint_velocity_trajectory'], 
                        time = time_vec,
                        data_des=des,
                        joint_names=[f'J{i+1}' for i in range(n_controllable_joints)],
                        ylabel='Angle [rad/s]',
                        #   ylim=(-(np.pi+1),np.pi+1),
                        show=True,
                        save_path=os.path.join(save_folder,"plt_robot_joint_velocities"),
                        title="Robot Joint Velocity",)

    if CONTROL_ROBOT and des is not None:
        plot_timeseries(data=des-d['robot_joint_velocity_trajectory'], 
                            time = time_vec,
                            joint_names=[f'J{i+1}' for i in range(n_controllable_joints)],
                            ylabel='Angle [rad/s]',
                            show=True,
                            save_path=os.path.join(save_folder,"plt_robot_joint_velocities_ERROR"),
                            title="Robot Joint Velocity Error (Des-Measured)",)

    plot_timeseries(data=d['robot_control_torques'], 
                        time=time_vec,
                        joint_names=[f'J{i+1}' for i in range(n_controllable_joints)],
                        ylabel='Torque [Nm]',
                        title="Robot Control Torques",
                        #   ylims=[(umin*101,umax*101) for umin,umax in zip(robot_u_min,robot_u_max)],
                        save_path=os.path.join(save_folder,"plt_robot_control_torques"))

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

    plot_timeseries(data=d['robot_TS_velocity_trajectory_EE'], 
                        time=time_vec,
                        data_des= des2,
                        joint_names=['X','Y', 'Z', 'Roll','Pitch','Yaw'],
                        ylabel='[m/s,rad/s]',
                        title="EE Velocities EE Frame",
                        xyz_colors=True,
                        save_path=os.path.join(save_folder,"_plt_robot_TS_velocities_EE"))

    plot_timeseries(data=d['robot_force_torque_sensor_EE'], 
                    data_label="Torque",
                    time=time_vec,
                    data_des = d['robot_TS_velocity_trajectory_EE'],
                    data_des_label="Velocity",
                    joint_names=['X','Y', 'Z', 'Mx','My','Mz'],
                    ylabel='[Force Units]',
                    title=f"{save_name} Force -> Velocity (EE Frame)",
                    xyz_colors=True,
                    # ylim=((-5,5)),
                    yy="True",
                    save_path=os.path.join(save_folder, "_plt_robot_force_velocity"))

    plot_timeseries(data=d['robot_force_torque_sensor_WF_orientation'], 
                    data_label="F/T",
                    time=time_vec,
                    data_des = d['robot_TS_velocity_trajectory_WF_orientation'],
                    data_des_label="Velocity",
                    joint_names=['X','Y', 'Z', 'Mx','My','Mz'],
                    ylabel='[Force Units]',
                    title=f"{save_name} Force -> Velocity (WF Orientation)",
                    xyz_colors=True,
                    yy="True",
                    save_path=os.path.join(save_folder, "_plt_robot_force_velocity_WF"))



    # import pickle
    from toolbox import save_dict_of_vectors, load_dict_of_vectors
    import os
    import shutil

    shutil.copy(XML_PATH,os.path.join(save_folder,"model.xml"))

    run_to_save["time"] = time_vec
    save_dict_of_vectors(run_to_save,out_dir=save_folder)


    ################ Compare Traj ################

    from toolbox import get_value_from_dict
    from ploty_the_plotypus import plot_trajectories_3d_multi_view

    if baseline_traj is not None and optimized_placement:
        runs = [baseline_traj,
                save_folder
                ]
        run_names = ['Free',
                    "Controlled",
                    "Optimized Pose"
                    ]

    if baseline_traj is not None and not optimized_placement:
        runs = [baseline_traj,
                save_folder
                ]
        
        run_names = ['Free',
                    "Controlled"]

    else:
        runs = [save_folder
                ]
        run_names = ['Free']

    trajectories = [get_value_from_dict(run,'human_arm_attachment_pose_WF').reshape(-1, 4, 4) for run in runs]


    plot_trajectories_3d_multi_view(trajectories=trajectories,
                                    labels=run_names,
                                    dpi=600,
                                    filename_base=os.path.join(save_folder,'test'))

    plot_timeseries(data=d['human_arm_velocity_WF_orientation'], 
                        time=time_vec,
                        joint_names=['X','Y', 'Z', 'Roll','Pitch','Yaw'],
                        ylabel='[m/s,rad/s]',
                        title="Human Attachment Velocities WF",
                        xyz_colors=True,
                        show=True,
                        save_path=os.path.join(save_folder,"plt_human_TS_velocities_EE"))



    plt.close('all')


#%% ########## plot 3D trajectories ###########

from ur5e_analytic import invert_transformation
from toolbox import get_value_from_dict
from eval import rmse_pos
from ploty_the_plotypus import plot_trajectories_3d_multi_view, load_mujoco_camera_from_xml, plot_trajectory_magnitude_differences, body_transform_from_xml_file,plot_twist_magnitude_differences


def slice_time_window(time_vector, dt, t_start=0.5, t_end=2.5):
    """
    Returns indices and slice corresponding to [t_start, t_end].

    Parameters
    ----------
    time_vector : (N,) array
    dt : float
    t_start : float
    t_end : float

    Returns
    -------
    idx_start : int
    idx_end : int
    sl : slice
    """
    idx_start = int(np.floor(t_start / dt))
    idx_end   = int(np.ceil(t_end / dt))

    # clamp to valid bounds
    idx_start = max(idx_start, 0)
    idx_end   = min(idx_end, len(time_vector))

    return idx_start, idx_end, slice(idx_start, idx_end)

# list of sets of dictionary settings
sims = [
        [
        FE_movement_only,
        FE_initial_pose,
        ],
        [
        multijoint_movement_only,
        multijoint_intial_pose,
        ],
        [
        h2m_movement_only,
        h2m_intial_pose,
        ],
        ]

camera_names= ['left_iso','front','left_iso']
camera_views = [load_mujoco_camera_from_xml(r".\xml\ur5e_7dof_baseline.xml",name) for name in camera_names]
views = [
        {
        "v": (15, 64),
        },

        {
        "v": (55, 60),

        },

        {
        "v": (20, 70),
        }
        ]

condition_i = 0

separate_traj = [np.array([-2, 0, 0]), np.array([-8, 0, 0]), np.array([-6, 0, 0])]
# separate_traj = [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])]

all_twists = []
all_trajectories = []
all_pos_linear = []
all_pos_angular = []
all_twist_linear = []
all_twist_angular = []
all_time = []
all_titles = []
all_names = []


for traj_condition,mj_cam in zip(sims,camera_views):
    if traj_condition[0] is None: 
        condition_i +=1
        continue

    run_paths = [sd['data_folder'] for sd in traj_condition]
    run_names = ['Free', 'Coupled']

    trajectories = []
    twist_trajectories = []
    FT_trajectories = []

    T_adjust = np.identity(4)
    time_vec = get_value_from_dict(os.path.join(traj_condition[0]['data_folder'], traj_condition[0]['save_name']),'time')

    i0, i1, sl = slice_time_window(time_vec, dt,t_start=buffer_time,t_end=total_time-buffer_time)

    for i,run in enumerate(traj_condition):
        run_dict_fp = os.path.join(run['data_folder'], run['save_name'])
        trajectories.append(get_value_from_dict(run_dict_fp,'human_arm_attachment_pose_WF').reshape(-1, 4, 4)[sl]) # in centimeters!
        time_vec = time_vec[0:trajectories[0].shape[0]]

        cvel = get_value_from_dict(run_dict_fp,'human_arm_velocity_WF_orientation')[sl]
        twist = np.hstack([cvel[:,3:],cvel[:,0:3]])

        twist_trajectories.append(twist) # in centimeters!
        FT_trajectories.append(get_value_from_dict(run_dict_fp,'robot_force_torque_sensor_WF_orientation')[sl]) # in centimeters!


        run_xml = run['XML_PATH']
        print(run_xml)
        print(body_transform_from_xml_file(run_xml,body_name='HUMAN'))
        transform_to_zero_pose = invert_transformation(body_transform_from_xml_file(run_xml,body_name='HUMAN'))
        trajectories[i] = transform_to_zero_pose[np.newaxis,:,:] @ trajectories[i]

        #normalize to start at zero zero zero...
        attachment_zero = invert_transformation(trajectories[i][0,:,:])
        trajectories[i] = attachment_zero[np.newaxis,:,:] @ trajectories[i]

        trajectories[-1][:,0:3,-1] *= 100 # in CM! 

        if i > 0:
            val = rmse_pos(trajectories[0], trajectories[i])
            filename = f"RMSE.txt"
            filepath = os.path.join(run_dict_fp, filename)

            # create empty file (or write the value inside if you want)
            with open(filepath, "w") as f:
                f.write(f"{val:.6f}\n")

    
    save_name = os.path.split(traj_condition[0]['data_folder'])[-1]
    linestyles=['-','--']
    colors=['blue','orange']

    plot_trajectory_magnitude_differences(trajectories[0],
                                          trajectories[1:],
                                          t = time_vec,
                                          dt=dt,
                                          labels=['Linear', 'Angular'],
                                          figsize=(4.5,3.1),
                                          legend_loc='upper left',
                                          ylabel='Linear Error Norm [cm]',
                                          ylabel2='Angular Error Norm [rad]',
                                          linestyles=linestyles,
                                          linewidths=[1, 1],
                                          alphas=[1, 1],
                                          pad_legend = .4,
                                          colors = colors,
                                          legend_fontsize=13.5,
                                          transparent=True,
                                          label_fontsize=16,
                                          tick_fontsize=14,
                                          save_path= os.path.join(r".\paper_figures",save_name+'_error')
                                          )
    
    plot_twist_magnitude_differences(twist_trajectories[0],
                                          twist_trajectories[1:],
                                          t = time_vec,
                                          dt=dt,
                                          labels=['Linear', 'Angular'],
                                          figsize=(4.5,3.1),
                                          legend_loc='upper left',
                                        #   ylabel='Linear Error Norm [cm]',
                                        #   ylabel2='Angular Error Norm [rad]',
                                          linestyles=linestyles,
                                          linewidths=[1, 1],
                                          alphas=[1, 1],
                                          pad_legend = .4,
                                          colors = colors,
                                          legend_fontsize=13.5,
                                          transparent=True,
                                          label_fontsize=16,
                                          tick_fontsize=14,
                                          save_path= os.path.join(r".\paper_figures",save_name+'_error')
                                          )

    # MOVE ALL TRAJECTORIES TO NOT OVERLAP

    shifted_trajectories = copy.deepcopy(trajectories)
    for i,traj in enumerate(shifted_trajectories):
        shifted_trajectories[i] = T_adjust[np.newaxis,:,:] @ shifted_trajectories[i]
        T_adjust[0:3,-1] = T_adjust[0:3,-1] + separate_traj[condition_i] # for next cycle
        
    line_styles = [':', '--']
    line_colors = ['k', 'blue']
    
    plot_trajectories_3d_multi_view(trajectories=shifted_trajectories,
                                    labels=run_names,
                                    dpi=1200,
                                    tight_layout_pad=1,
                                    tick_precision=0,
                                    show=True,
                                    linestyles=line_styles,
                                    legend_loc='upper right',
                                    colors=line_colors,
                                    label_fontsize=20,
                                    tick_fontsize=18,
                                    legend_fontsize=16,
                                    font_family='Arial',
                                    views = views[condition_i],
                                    filename_base=os.path.join(r".\paper_figures",save_name))
    
    

    names = run_names.copy()
    names.pop(0) # shoud be two names now

    i = 0
    trajectory_names = ['Single-Joint Movement','Multi-Joint Reach', 'Hand to Mouth']

    pos_lin_list = []
    pos_ang_list = []
    twist_lin_list = []
    twist_ang_list = []

    for i in range(1, len(trajectories)):
        # --- position error ---
        T_ref = trajectories[0]
        T_cmp = trajectories[i]

        # position difference
        dp = T_ref[:, 0:3, -1] - T_cmp[:, 0:3, -1]
        pos_lin_list.append(np.linalg.norm(dp, axis=1))


        norms = []

        for k in range(len(T_ref)):
            R_base = T_ref[k, :3, :3]
            R_comp = T_cmp[k, :3, :3]

            R_err = R_comp @ R_base.T

            # enforce SO(3) numerically
            U, _, Vt = np.linalg.svd(R_err)
            R_err = U @ Vt

            # rotation vector via log map approximation
            cos_theta = (np.trace(R_err) - 1) / 2
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            theta = np.arccos(cos_theta)

            norms.append(theta)

        pos_ang_list.append(np.array(norms))

        # --- twist error ---
        V_ref = twist_trajectories[0]
        V_cmp = twist_trajectories[i]
        # V_cmp = np.zeros_like(twist_trajectories[0])

        dv = V_ref[:, 0:3] - V_cmp[:, 0:3]
        dw = V_ref[:, 3:6] - V_cmp[:, 3:6]

        twist_lin_list.append(np.linalg.norm(dv, axis=1)*100)
        twist_ang_list.append(np.linalg.norm(dw, axis=1))

    all_pos_linear.append(pos_lin_list)
    all_pos_angular.append(pos_ang_list)
    all_twist_linear.append(twist_lin_list)
    all_twist_angular.append(twist_ang_list)

    all_twists.append(twist_trajectories)
    all_trajectories.append(trajectories)
    all_time.append(time_vec.copy())
    all_titles.append(trajectory_names[condition_i])
    all_names = names.copy()

    condition_i += 1


# %%

fig, axes = plt.subplots(2, 3, figsize=(10, 4.5), dpi=900, sharex=True)

line_width = 1.0
line_styles = ['-', '--', '-.']
colors = ['tab:blue', 'tab:orange']
alpha = 0.9

ylabel_pos_left = 'Position \n Error Norm [cm]'
ylabel_pos_right = 'Orientation \n Error Norm [rad]'
ylabel_twist_left = 'Linear Velocity \n Error Norm [cm/s]'
ylabel_twist_right = 'Angular Velocity \n Error Norm [rad/s]'
xlabel = 'Time [s]'

legend_handles = []
legend_labels = []

for cond in range(3):

    # --- TOP ROW: POSITION ERRORS ---
    ax1 = axes[0, cond]
    ax2 = ax1.twinx()

    time_vec = all_time[cond]
    lin_list = all_pos_linear[cond]
    ang_list = all_pos_angular[cond]

    title = all_titles[cond]

    for i, (lin, ang) in enumerate(zip(lin_list, ang_list)):

        linestyle = line_styles[i % len(line_styles)]

        # ---- PRINT STATS (POSITION) ----
        print(f"\n[COND {cond} | POSITION | TRACE {i}]")

        print(f"Linear error:")
        print(f"  min   : {np.min(lin):.6f}")
        print(f"  max   : {np.max(lin):.6f}")
        print(f"  mean  : {np.mean(lin):.6f}")
        print(f"  final : {lin[-1]:.6f}")

        print(f"Angular error:")
        print(f"  min   : {np.min(ang):.6f}")
        print(f"  max   : {np.max(ang):.6f}")
        print(f"  mean  : {np.mean(ang):.6f}")
        print(f"  final : {ang[-1]:.6f}")

        h1 = ax1.plot(
            time_vec, lin,
            linestyle=linestyle,
            color='blue',
            linewidth=line_width,
            alpha=alpha,
            label="Linear"
        )[0]

        h2 = ax2.plot(
            time_vec, ang,
            linestyle='--',
            color='orange',
            linewidth=line_width,
            alpha=alpha,
            label="Angular"
        )[0]

    if cond == 0:
        legend_handles += [h1, h2]
        legend_labels += ['Linear', 'Angular']

    ax1.set_title(title)

    if cond == 0:
        ax1.set_ylabel(ylabel_pos_left)
    else:
        ax1.set_ylabel("")

    if cond == 2:
        ax2.set_ylabel(ylabel_pos_right)
    else:
        ax2.set_ylabel("")

    ax1.grid(True, alpha=0.3)

    # --- BOTTOM ROW: TWIST ERRORS ---
    ax3 = axes[1, cond]
    ax4 = ax3.twinx()

    lin_list = all_twist_linear[cond]
    ang_list = all_twist_angular[cond]

    for i, (lin, ang) in enumerate(zip(lin_list, ang_list)):

        linestyle = line_styles[i % len(line_styles)]

        # ---- PRINT STATS (TWIST) ----
        print(f"\n[COND {cond} | TWIST | TRACE {i}]")

        print(f"Linear twist:")
        print(f"  min   : {np.min(lin):.6f}")
        print(f"  max   : {np.max(lin):.6f}")
        print(f"  mean  : {np.mean(lin):.6f}")
        print(f"  final : {lin[-1]:.6f}")

        print(f"Angular twist:")
        print(f"  min   : {np.min(ang):.6f}")
        print(f"  max   : {np.max(ang):.6f}")
        print(f"  mean  : {np.mean(ang):.6f}")
        print(f"  final : {ang[-1]:.6f}")

        ax3.plot(
            time_vec, lin,
            linestyle=linestyle,
            color='blue',
            linewidth=line_width,
            alpha=alpha
        )

        ax4.plot(
            time_vec, ang,
            linestyle='--',
            color='orange',
            linewidth=line_width,
            alpha=alpha
        )

    ax3.set_xlabel(xlabel)

    if cond == 0:
        ax3.set_ylabel(ylabel_twist_left)
    else:
        ax3.set_ylabel("")

    if cond == 2:
        ax4.set_ylabel(ylabel_twist_right)
    else:
        ax4.set_ylabel("")

    ax3.grid(True, alpha=0.3)

# shared legend
fig.legend(
    legend_handles,
    legend_labels,
    loc='upper center',
    ncol=2,
    frameon=False,
    fontsize=12
)


save_path = r".\paper_figures\final_plot"


# plt.tight_layout(rect=[0, 0, 0.9, 0.92],h_pad=6)
plt.tight_layout(rect=[0, 0, 1, 0.92],h_pad=3)
plt.subplots_adjust(wspace=0.55)  # increase this value for more space

plt.savefig(save_path + '.png', dpi=900, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.savefig(save_path + '.svg', bbox_inches='tight')

plt.show()

#%%