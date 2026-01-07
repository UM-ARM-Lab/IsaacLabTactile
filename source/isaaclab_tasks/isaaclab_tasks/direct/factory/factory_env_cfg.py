# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Example command to run with tactile sensor enabled:
python scripts/reinforcement_learning/rl_games/train.py --task Isaac-Factory-NutThread-Direct-v0 --enable_cameras env.enable_tactile_sensor=true env.read_tactile_sensor=true

python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rl_games/train.py --task Isaac-Factory-NutThread-Direct-v0 --enable_cameras env.enable_tactile_sensor=true env.read_tactile_sensor=true --num_envs 256 --distributed --headless

Note: Hydra requires lowercase boolean values (true/false), not Python's True/False.

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sensors import TiledCameraCfg, VisuoTactileSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from .factory_tasks_cfg import ASSET_DIR, FactoryTask, GearMesh, NutThread, PegInsert, TestTask
from omegaconf import OmegaConf

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "held_pos": 3,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "gripper_pos": 2,
    "gripper_vel": 2,
    "scales": 1,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
    "scales": 1,
}


@configclass
class ObsRandCfg:
    fixed_asset_pos = [0.001, 0.001, 0.001]


@configclass
class ObsHistoryCfg:
    """Configuration for observation history."""
    history_length: int = 0
    flatten_history_dim: bool = True


@configclass
class CtrlCfg:
    ema_factor = 0.2

    pos_action_bounds = [0.05, 0.05, 0.05]
    rot_action_bounds = [1.0, 1.0, 1.0]

    pos_action_threshold = [0.02, 0.02, 0.02]
    rot_action_threshold = [0.097, 0.097, 0.097]

    reset_joints = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
    reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale = 10.0
    default_task_prop_gains = [100, 100, 100, 30, 30, 30]

    # Null space parameters.
    default_dof_pos_tensor = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    kp_null = 10.0
    kd_null = 6.3246
    use_full_rotation: bool = False


@configclass
class FactoryEnvCfg(DirectRLEnvCfg):
    decimation = 8
    action_space = 6
    # num_*: will be overwritten to correspond to obs_order, state_order.
    observation_space = 21
    state_space = 72
    obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
    ]

    task_name: str = "peg_insert"  # peg_insert, gear_mesh, nut_thread
    task: FactoryTask = FactoryTask()
    obs_rand: ObsRandCfg = ObsRandCfg()
    obs_history: ObsHistoryCfg = ObsHistoryCfg()
    ctrl: CtrlCfg = CtrlCfg()
    
    episode_length_s = 10.0  # Probably need to override.
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**29,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # Viewer settings
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.25, 0.1, 0.2), lookat=(0.0, 0.0, 0.04),
        origin_type="asset_root", asset_name="fixed_asset"
        )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, clone_in_fabric=False)

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/franka_mimic.usd",  # Will be overridden in __post_init__
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=-0.001),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=0.04,
                # stiffness=7500.0, # og param
                # damping=173.0,
                # stiffness=100.0, # params worked for my initial visual transfer exp
                # damping=10.0,
                stiffness = 500.0,
                damping = 40.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )

    obs_cam = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.8, 0.2, 0.15),
            rot=[0.18913, -0.25231, -0.70188, 0.6387],
            convention="ros",
        ),
        data_types=["distance_to_image_plane", "rgb"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.0001, 0.5)),
        width=224,
        height=224,
        # width=1200,
        # height=1200,
    )

    # TacSL Tactile Sensor
    tactile_cam = VisuoTactileSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_leftfinger/tactile_sensor",
        history_length=0,
        debug_vis=False,
        # Sensor configuration
        sensor_type="gelsight_r15",
        enable_camera_tactile=True,
        enable_force_field=True,
        # Elastomer configuration
        elastomer_rigid_body="elastomer",
        elastomer_tactile_mesh="elastomer/visuals",
        elastomer_tip_link_name="elastomer_tip",
        # Force field configuration
        num_tactile_rows=20,
        num_tactile_cols=25,
        tactile_margin=0.003,
        # Indenter configuration (will be set based on indenter type)
        indenter_rigid_body=None,  # Will be updated based on indenter type
        indenter_sdf_mesh=None,  # Will be updated based on indenter type
        # Force field physics parameters
        tactile_kn=1.0,
        tactile_mu=2.0,
        tactile_kt=0.1,
        # Compliant dynamics
        compliance_stiffness=350.0,
        compliant_damping=1.0,
        # Camera configuration
        camera_cfg=TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/panda_leftfinger/elastomer_tip/cam",
            update_period=1 / 60,  # 60 Hz
            # height=320,
            # width=240,
            height=80,
            width=60,
            data_types=["distance_to_image_plane"],
            spawn=None,  # the camera is already spawned in the scene, properties are set in the gelsight_r15_finger.usd file
        ),
        # Debug Visualization
        trimesh_vis_tactile_points=False,
        visualize_sdf_closest_pts=False,
        visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/TactileSensorDebugPts",
            markers={
                "debug_pts": sim_utils.SphereCfg(
                    radius=0.0002,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
            },
        ),
    )
    # To enable experiments with cfg dicts
    params = None

    # My parameters
    enable_tactile_sensor: bool = False
    read_tactile_sensor: bool = False # this is actually not taking effect, unless we change obs_order to include tactile_taxim
    enable_obs_camera: bool = False
    use_compliant_gripper: bool = True
    use_gelsight_finger: bool = True

    def update_env_params(self):
        # return
        """Set default environment parameters."""
        # Initialize params structure
        if self.params is None:
            self.params = OmegaConf.create({})
        params = self.params
        
        params.env = params.get("env", OmegaConf.create({}))
        env = params.env
        if env.get("enable_tactile_sensor", None) is not None:
            self.enable_tactile_sensor = env.enable_tactile_sensor
        if env.get("read_tactile_sensor", None) is not None:
            self.read_tactile_sensor = env.read_tactile_sensor
        if env.get("enable_obs_camera", None) is not None:
            self.enable_obs_camera = env.enable_obs_camera
        if env.get("use_compliant_gripper", None) is not None:
            self.use_compliant_gripper = env.use_compliant_gripper
        if env.get("use_gelsight_finger", None) is not None:
            self.use_gelsight_finger = env.use_gelsight_finger
        if env.get("obs_history", None) is not None and env["obs_history"].get("history_length", None) is not None:
            self.obs_history.history_length = env["obs_history"]["history_length"]

        task = env.get("task", OmegaConf.create({}))
        if task.get("held_asset_rot_noise", None) is not None:
            self.task.held_asset_rot_noise = OmegaConf.to_container(task.held_asset_rot_noise, resolve=True)
        if task.get("hand_init_pos", None) is not None:
            self.task.hand_init_pos = OmegaConf.to_container(task.hand_init_pos, resolve=True)
        if task.get("hand_init_pos_noise", None) is not None:
            self.task.hand_init_pos_noise = OmegaConf.to_container(task.hand_init_pos_noise, resolve=True)
        if task.get("hand_init_orn", None) is not None:
            self.task.hand_init_orn = OmegaConf.to_container(task.hand_init_orn, resolve=True)
        if task.get("hand_init_orn_noise", None) is not None:
            self.task.hand_init_orn_noise = OmegaConf.to_container(task.hand_init_orn_noise, resolve=True)
        ctrl = env.get("ctrl", OmegaConf.create({}))
        if ctrl.get("ema_factor", None) is not None:
            self.ctrl.ema_factor = ctrl.ema_factor
        pos_action_bounds = ctrl.get("pos_action_bounds", None)
        if pos_action_bounds is not None:
            self.ctrl.pos_action_bounds = OmegaConf.to_container(pos_action_bounds, resolve=True)
        rot_action_bounds = ctrl.get("rot_action_bounds", None)
        if rot_action_bounds is not None:
            self.ctrl.rot_action_bounds = OmegaConf.to_container(rot_action_bounds, resolve=True)
        use_full_rotation = ctrl.get("use_full_rotation", None)
        if use_full_rotation is not None:
            self.ctrl.use_full_rotation = use_full_rotation

    def __post_init__(self):
        """Post initialization."""
        self.update_env_params()
        self.sim.render_interval = self.decimation

        # self.episode_length_s = 24   # 24, 10 for sim quality test
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "fixed_asset"
        # self.viewer.eye = (0.1, 0.1, 0.06)
        # self.viewer.lookat = (0, 0.0, 0.04)
        self.viewer.eye = (0.37, 0.1, 0.12)
        self.viewer.lookat = (0.0, 0.0, 0.03)
        self.viewer.resolution = (720, 720)


@configclass
class FactoryTaskPegInsertCfg(FactoryEnvCfg):
    task_name = "peg_insert"
    task = PegInsert()
    episode_length_s = 10.0


@configclass
class FactoryTaskGearMeshCfg(FactoryEnvCfg):
    task_name = "gear_mesh"
    task = GearMesh()
    episode_length_s = 20.0


@configclass
class FactoryTaskNutThreadCfg(FactoryEnvCfg):
    task_name = "nut_thread"
    task = NutThread()
    episode_length_s = 30.0


@configclass
class FactoryTaskTestCfg(FactoryEnvCfg):
    """Configuration for test environment with operational space control (end-effector control)."""
    task_name = "test"
    task = TestTask()
    episode_length_s = 10.0
    
    # Operational space control: 6 DOF end-effector control (3 pos + 3 rot) + 1 DOF gripper
    action_space = 7
    
    # Observations for test environment (end-effector pose and velocities)
    # Note: These values will be computed automatically from obs_order and state_order in __init__
    observation_space = 20  # Placeholder; auto-computed in TestEnv.__init__
    state_space = 27  # Placeholder: 3 pos + 4 quat + 3 linvel + 3 angvel + 7 joint_pos + 7 prev_actions = 27
    
    obs_order: list = ["fingertip_pos", "fingertip_quat", "ee_linvel", "ee_angvel", "gripper_pos"]
    state_order: list = ["fingertip_pos", "fingertip_quat", "ee_linvel", "ee_angvel", "joint_pos"]
    
    def __post_init__(self):
        """Post initialization - override viewer to track robot instead of fixed_asset."""
        super().__post_init__()
        
        # Keep torque control (stiffness=0) since we use operational space control
        # No need to override actuators - use default torque control from FactoryEnvCfg
        
        # Lower gripper stiffness for more compliant gripper control
        self.robot.actuators["panda_hand"].stiffness = 50.0  # Reduced from 7500.0
        self.robot.actuators["panda_hand"].damping = 5.0  # Reduced from 173.0 (proportional to stiffness reduction)

        # Double the position action threshold only for the test environment
        self.ctrl.pos_action_threshold = [v * 2 for v in self.ctrl.pos_action_threshold]
        
        # Override viewer to track robot since test environment has no fixed_asset
        self.viewer.asset_name = "robot"
        self.viewer.eye = (1.0, 1.0, 1.3)
        self.viewer.lookat = (0.55, 0.0, 0.3)
