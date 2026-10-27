import mujoco
import mujoco as mj
import numpy as np
from PIL import Image

# =========================
# USER SETTINGS
# =========================
XML_PATH = r"C:\Users\lahmanjh\Documents\Github\simbox\universal_robots_ur5e\xml\ur5e_7dof_pictures.xml"

# Joint angles (in qpos order)

from human_movement import fe_traj_params,multijoint_traj_params,h2m_traj_params

start = np.array([
    0, 0, 0, 0, 0, 0,  # UR5e
    0, 0, 0, 0.2708, 0, 1.3, 0                   # human
])

fe_end = start.copy()
fe_end[6:] = fe_traj_params

multijoint_end = start.copy()
multijoint_end[6:] = multijoint_traj_params

h2m_end = start.copy()
h2m_end[6:] = h2m_traj_params

robot_home = np.array([
    0, 0, 0, 0, 0, 0,  # UR5e
    0, 0, 0, 0.2708, 0, 1.3, 0                   # human
])


poses =     [start,         fe_end,         multijoint_end,     start,         start,        h2m_end,      robot_home,     start,       start,   ]
cameras =   ['right_iso',    'right_iso',   'front',        'front',   'right_iso',  'right_iso',  "urdh_home",    'arm_zoom',  'coverpage',   ]
# cameras =   ['right_iso',    'right_iso',   'right_iso',        'right_iso',   'right_iso',  'right_iso',  "urdh_home",    'arm_zoom',  'coverpage',   ]
names =     ['fe_start',    'fe_end',       'mj_end',            'mj_start',   'h2m_start',  'h2m_end',    "dh_pic",      'arm_zoom',    'cover',     ]
alphas =    [.4,            1,              1,                   .4,           .4,           1,            1,              1,           '1',        ]
keeps =     [(1,1),         (1,1),          (1,1),               (1,1),        (1,1),        (1,1),        (1,0),          (0,1),       (1,1),    ] # first index flag to keep robot, second human

for QPOS,cam_name,name,alpha,keep in zip(poses, cameras, names, alphas, keeps): 

    OUTPUT_PATH = f"C:\\Users\\lahmanjh\\Documents\\Github\\simbox\\paper_figures\\render_{name}.png"
    IMAGE_WIDTH = 3840
    IMAGE_HEIGHT = 3840

    ROBOT_ALPHA = alpha   # transparency for robot + human (0=transparent, 1=opaque)

    WHITE_THRESHOLD = 200  # threshold for background removal


    # =========================
    # LOAD MODEL
    # =========================
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # set pose
    # data.qpos[:] = QPOS
    # mujoco.mj_forward(model, data)

    def reset(model, 
            data,
            condition = 'COUPLED',
            q_init = None,
            attached =True,
            soln_config = 3,
            ):
        from ur5e_analytic import site_transformation_mat, inverse_kinematics
        data.qpos[:] = q_init
        data.qvel[:] = 0.0

        mj.mj_forward(model, data) # make the model so that arm is in initial config
        if attached:
            HA_T = site_transformation_mat(model,data,'human_arm_attachment_site')

            q_init[0:6] = inverse_kinematics(HA_T)[soln_config,:]

            data.qpos[:] = q_init
            data.qvel[:] = 0.0

            mj.mj_forward(model, data) # make the model so that arm is in inital config AND robot is now attached

    reset(model,data,q_init=QPOS,attached=bool(keep[1]))

    robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    human_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "HUMAN")

    robot_ids = [gid for gid in range(model.ngeom) if model.geom_group[gid] == 1]
    human_ids = [gid for gid in range(model.ngeom) if model.geom_group[gid] == 2]

    if keep[0] and keep[1]: keep_ids = set(robot_ids+human_ids)
    elif keep[0]: keep_ids = set(robot_ids)
    elif keep[1]: keep_ids = set(human_ids)
    # =========================
    # SET COLORS
    # =========================
    # Robot + human: keep color but adjust alpha later
    # Background: force pure white

    # for gid in background_geom_ids:
    #     model.geom_rgba[gid] = np.array([1, 1, 1, 1])


    # KEY_COLOR = np.array([1.0, 0.0, 1.0, 1.0])  # magenta

    # for gid in background_geom_ids:
    #     model.geom_rgba[gid] = KEY_COLOR


    # # =========================
    # # RENDER (USE XML CAMERA)
    # # =========================
    # renderer = mujoco.Renderer(model, IMAGE_WIDTH, IMAGE_HEIGHT)
    # renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

    # # get camera id from XML
    # cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "isometric")

    # renderer.update_scene(data, camera=cam_id)

    # rgb = renderer.render()
    # rgb = renderer.render()


    # =========================
    # RENDER SEGMENTATION
    # =========================
    renderer = mujoco.Renderer(model, IMAGE_WIDTH, IMAGE_HEIGHT)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

    # --- segmentation pass ---
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

    renderer.update_scene(data, camera=cam_id)
    seg = renderer.render()

    # --- normal render ---
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0

    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render()

    # =========================
    # POST-PROCESS (REMOVE WHITE BACKGROUND)
    # =========================
    img = rgb.copy()

    # create alpha channel
    # alpha = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)

    # key_rgb = np.array([255, 0, 255])

    # # distance-based mask (more robust than threshold per channel)
    # dist = np.linalg.norm(img.astype(np.int16) - key_rgb, axis=-1)

    # mask = dist < 50  # tolerance

    # alpha[mask] = 0.0
    # alpha[~mask] = ROBOT_ALPHA

    # # combine RGBA
    # rgba = np.dstack((img, (alpha * 255).astype(np.uint8)))

    # initialize alpha
    alpha = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)

    seg = seg.astype(np.uint32)

    geom_ids = (
        seg[..., 0]
        + (seg[..., 1] << 8)
        + (seg[..., 2] << 16)
    ) - 1  # background becomes -1


    # define which geoms to KEEP (robot + human)
    keep_ids = set(keep_ids)

    mask = np.isin(geom_ids, list(keep_ids))
    alpha[mask] = ROBOT_ALPHA
    alpha[~mask] = 0.0

    rgba = np.dstack((rgb, (alpha * 255).astype(np.uint8)))
    Image.fromarray(rgba).save(OUTPUT_PATH)

    # save
    Image.fromarray(rgba).save(OUTPUT_PATH)

    print(f"Saved image to {OUTPUT_PATH}")
    renderer.close()