# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example script demonstrating the TacSL tactile sensor implementation in IsaacLab.

This script shows how to use the TactileSensor for both camera-based and force field
tactile sensing with the gelsight finger setup.

.. code-block:: bash

    # Usage
    python tacsl_example.py --enable_cameras --num_envs 16 --indenter_type nut --save_viz --use_tactile_taxim --use_tactile_ff

"""

import argparse
import math
import numpy as np
import os
import torch

import cv2

from isaaclab.app import AppLauncher
from isaaclab.utils.timer import Timer
# from force_tool.visualization.plot_utils import save_numpy_video, get_img_from_fig

# Add argparse arguments
parser = argparse.ArgumentParser(description="TacSL tactile sensor example.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument("--tactile_kn", type=float, default=1.0, help="Tactile normal stiffness.")
parser.add_argument("--tactile_kt", type=float, default=0.1, help="Tactile tangential stiffness.")
parser.add_argument("--tactile_mu", type=float, default=2.0, help="Tactile friction coefficient.")
parser.add_argument("--tactile_compliance_stiffness", type=float, default=150.0, help="Tactile compliance stiffness.")
parser.add_argument("--tactile_compliant_damping", type=float, default=1.0, help="Tactile compliant damping.")
parser.add_argument("--save_viz", action="store_true", help="Visualize tactile data.")
parser.add_argument("--save_viz_dir", type=str, default="tactile_record", help="Directory to save tactile data.")
parser.add_argument("--use_tactile_taxim", action="store_true", help="Use tactile taxim sensor data collection.")
parser.add_argument("--use_tactile_ff", action="store_true", help="Use tactile force field sensor data collection.")
parser.add_argument("--debug_sdf_closest_pts", action="store_true", help="Visualize closest SDF points.")
parser.add_argument("--debug_tactile_sensor_pts", action="store_true", help="Visualize tactile sensor points.")
parser.add_argument("--trimesh_vis_tactile_points", action="store_true", help="Visualize tactile points using trimesh.")
parser.add_argument(
    "--indenter_type", type=str, default="nut", choices=["none", "cube", "nut"], help="Type of indenter to use."
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# Import our TactileSensor
from isaaclab.sensors import TiledCameraCfg, VisuoTactileSensorCfg
from isaaclab.sensors.tacsl_sensor.visuotactile_viz_utils import visualize_tactile_shear_image
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@configclass
class TactileSensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with tactile sensors on the robot."""

    # Ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Robot with tactile sensor
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/TacSL/gelsight_r15_finger/gelsight_r15_finger.usd",
            # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/TacSL/gelsight_mini_finger/gelsight_mini_finger.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=-0.0005),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(math.sqrt(2) / 2, -math.sqrt(2) / 2, 0.0, 0.0),  # 90° rotation
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )

    # Camera configuration for tactile sensing

    # TacSL Tactile Sensor
    tactile_sensor = VisuoTactileSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/tactile_sensor",
        history_length=0,
        debug_vis=args_cli.debug_tactile_sensor_pts or args_cli.debug_sdf_closest_pts,
        # Sensor configuration
        sensor_type="gelsight_r15",
        enable_camera_tactile=args_cli.use_tactile_taxim,
        enable_force_field=args_cli.use_tactile_ff,
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
        tactile_kn=args_cli.tactile_kn,
        tactile_mu=args_cli.tactile_mu,
        tactile_kt=args_cli.tactile_kt,
        # Compliant dynamics
        compliance_stiffness=args_cli.tactile_compliance_stiffness,
        compliant_damping=args_cli.tactile_compliant_damping,
        # Camera configuration
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/elastomer_tip/cam",
            update_period=1 / 200,  # 60 Hz
            height=320,
            width=240,
            data_types=["distance_to_image_plane"],
            spawn=None,  # the camera is already spawned in the scene, properties are set in the gelsight_r15_finger.usd file
        ),
        # Debug Visualization
        trimesh_vis_tactile_points=args_cli.trimesh_vis_tactile_points,
        visualize_sdf_closest_pts=args_cli.debug_sdf_closest_pts,
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

    # Tiled camera to look at the nut
    nut_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/nut_camera",
        update_period=1 / 200,  # 60 Hz
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.05, 0.65),  # Position above the nut
            # rot=(0.065, 0.141, 0.987, 0.0355),  # Looking down
            # rot=(-0.0015, -0.0086, 0.1736, 0.9848),  # Looking down
            rot=(0.706, -0.037, 0.0246, 0.7067),
            convention="opengl"
        ),
    )


@configclass
class CubeTactileSceneCfg(TactileSensorsSceneCfg):
    """Scene with cube indenter."""

    # Cube indenter
    indenter = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/indenter",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.00327211),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0 + 0.06776, 0.51), rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class NutTactileSceneCfg(TactileSensorsSceneCfg):
    """Scene with nut indenter."""

    # Nut indenter
    indenter = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/indenter",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_nut_m16.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                max_angular_velocity=180.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0 + 0.06776, 0.498),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )



def mkdir_helper(dir_path):
    tactile_img_folder = dir_path
    os.makedirs(tactile_img_folder, exist_ok=True)
    return tactile_img_folder


def collect_frame_data(tactile_data, camera_data, num_envs, nrows, ncols):
    """Collect frame data for video creation."""
    frame_data = {}
    
    if tactile_data.tactile_shear_force is not None and tactile_data.tactile_normal_force is not None:
        # visualize tactile forces
        tactile_normal_force = tactile_data.tactile_normal_force.view((num_envs, nrows, ncols))
        tactile_shear_force = tactile_data.tactile_shear_force.view((num_envs, nrows, ncols, 2))

        tactile_image = visualize_tactile_shear_image(
            tactile_normal_force[0, :, :].detach().cpu().numpy(), tactile_shear_force[0, :, :].detach().cpu().numpy()
        )

        if tactile_normal_force.shape[0] > 1:
            tactile_image_1 = visualize_tactile_shear_image(
                tactile_normal_force[1, :, :].detach().cpu().numpy(),
                tactile_shear_force[1, :, :].detach().cpu().numpy(),
            )
            combined_tactile_image = np.vstack([tactile_image, tactile_image_1])
        else:
            combined_tactile_image = tactile_image
            
        frame_data['tactile_force_field'] = (combined_tactile_image * 255).astype(np.uint8)

    if tactile_data.taxim_tactile is not None:
        taxim_data = tactile_data.taxim_tactile.cpu().numpy()
        taxim_data = np.transpose(taxim_data, axes=(0, 2, 1, 3))
        taxim_data_first_2 = taxim_data[:2] if len(taxim_data) >= 2 else taxim_data
        taxim_tiled = np.concatenate(taxim_data_first_2, axis=0)
        frame_data['tactile_taxim'] = taxim_tiled
    
    # Add RGB camera data
    if camera_data is not None and 'rgb' in camera_data.output:
        rgb_data = camera_data.output['rgb'].cpu().numpy()
        # Take the first environment's RGB data
        rgb_frame = rgb_data[0]  # Shape: (H, W, C)
        frame_data['rgb_camera'] = rgb_frame
        
    return frame_data


def create_episode_video(episode_frames, output_dir, episode_num):
    """Create video from episode frames using save_numpy_video."""
    if not episode_frames:
        return
        
    # Get frame dimensions from first frame
    first_frame = episode_frames[0]
    has_force_field = 'tactile_force_field' in first_frame
    has_taxim = 'tactile_taxim' in first_frame
    has_rgb = 'rgb_camera' in first_frame
    
    if not (has_force_field or has_taxim or has_rgb):
        return
    
    # Prepare frames for save_numpy_video
    video_frames = []
    
    for frame_data in episode_frames:
        frame_components = []
        
        # Add RGB camera if available
        if has_rgb:
            rgb_img = frame_data['rgb_camera']
            # Convert from RGB to BGR for OpenCV compatibility
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            frame_components.append(rgb_img)
        
        # Add tactile force field if available
        if has_force_field:
            force_field_img = frame_data['tactile_force_field']
            frame_components.append(force_field_img)
        
        # Add tactile taxim if available
        if has_taxim:
            taxim_img = frame_data['tactile_taxim']
            frame_components.append(taxim_img)
        
        if len(frame_components) == 1:
            combined_frame = frame_components[0]
        elif len(frame_components) == 2:
            # Resize images to have the same height
            target_height = max(frame_components[0].shape[0], frame_components[1].shape[0])
            img1_resized = cv2.resize(frame_components[0], 
                                    (int(frame_components[0].shape[1] * target_height / frame_components[0].shape[0]), 
                                     target_height))
            img2_resized = cv2.resize(frame_components[1], 
                                    (int(frame_components[1].shape[1] * target_height / frame_components[1].shape[0]), 
                                     target_height))
            combined_frame = np.hstack([img1_resized, img2_resized])
        else:  # 3 components
            # Resize all images to have the same height
            target_height = max(comp.shape[0] for comp in frame_components)
            resized_components = []
            for comp in frame_components:
                resized = cv2.resize(comp, 
                                   (int(comp.shape[1] * target_height / comp.shape[0]), 
                                    target_height))
                resized_components.append(resized)
            combined_frame = np.hstack(resized_components)
        
        video_frames.append(combined_frame)
    
    # Convert to numpy array with shape (T, H, W, C) for save_numpy_video
    video_array = np.array(video_frames)
    
    # Create video using save_numpy_video
    video_path = os.path.join(output_dir, f"episode_{episode_num:03d}")
    # save_numpy_video(video_array, video_path, fps=20, format='mp4', draw_idx=True)
    print(f"[INFO]: Saved episode video: {video_path}.mp4")


def run_simulator(sim, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Assign different masses to indenters in different environments
    num_envs = scene.num_envs

    if args_cli.save_viz:
        # Create output directory for tactile videos
        output_dir = mkdir_helper(args_cli.save_viz_dir)
        episode_frames = []  # Store frames for current episode
        episode_num = 0

    # Create constant downward force
    force_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    torque_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    force_tensor[:, 0, 2] = -1.0

    nrows = scene["tactile_sensor"].cfg.num_tactile_rows
    ncols = scene["tactile_sensor"].cfg.num_tactile_cols

    physics_timer = Timer()
    physics_total_time = 0.0
    physics_total_count = 0

    scene.update(sim_dt)

    entity_list = ["robot"]
    if "indenter" in scene.keys():
        entity_list.append("indenter")

    while simulation_app.is_running():

        if count == 200:
            print(scene["tactile_sensor"].get_timing_summary())
            
            # Save current episode video if we have frames
            if args_cli.save_viz and episode_frames:
                create_episode_video(episode_frames, output_dir, episode_num)
                episode_frames = []  # Clear frames for next episode
                episode_num += 1
            
            # Reset robot and indenter positions
            count = 0
            for entity in entity_list:
                root_state = scene[entity].data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                scene[entity].write_root_state_to_sim(root_state)

            scene.reset()
            print("[INFO]: Resetting robot and indenter state...")

        if "indenter" in scene.keys():
            # rotation
            if count > 30:
                env_indices = torch.arange(scene.num_envs, device=sim.device)
                odd_mask = env_indices % 2 == 1
                even_mask = env_indices % 2 == 0
                # torque_tensor[odd_mask, 0, 2] = 0.005 # rotation for odd environments
                # torque_tensor[even_mask, 0, 2] = -0.005  # rotation for even environments
                force_tensor[odd_mask, 0, 0] = 0.4
                force_tensor[even_mask, 0, 0] = -0.4
                scene["indenter"].set_external_force_and_torque(force_tensor, torque_tensor)

        # Step simulation
        scene.write_data_to_sim()
        physics_timer.start()
        sim.step()
        physics_timer.stop()
        physics_total_time += physics_timer.total_run_time
        physics_total_count += 1
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        # Access tactile sensor data
        tactile_data = scene["tactile_sensor"].data
        
        # Access camera data
        camera_data = scene["nut_camera"].data if "nut_camera" in scene.keys() else None

        if args_cli.save_viz:
            # Collect frame data for video creation
            frame_data = collect_frame_data(tactile_data, camera_data, num_envs, nrows, ncols)
            if frame_data:  # Only add if we have data
                episode_frames.append(frame_data)

    # Save final episode video if we have frames
    if args_cli.save_viz and episode_frames:
        create_episode_video(episode_frames, output_dir, episode_num)
        print(f"[INFO]: Saved final episode video (episode {episode_num})")

    # Get timing summary from sensor and add physics timing
    timing_summary = scene["tactile_sensor"].get_timing_summary()

    # Add physics timing to the summary
    physics_avg = physics_total_time / (physics_total_count * scene.num_envs) if physics_total_count > 0 else 0.0
    timing_summary["physics_total"] = physics_total_time
    timing_summary["physics_average"] = physics_avg
    timing_summary["physics_fps"] = 1 / physics_avg if physics_avg > 0 else 0.0

    print(timing_summary)


def main():
    """Main function."""
    # Initialize simulation
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args_cli.device,
        physx=sim_utils.PhysxCfg(
            gpu_collision_stack_size=2
            ** 30,  # Important to prevent collisionStackSize buffer overflow in contact-rich environments.
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.0])

    # Create scene based on indenter type
    if args_cli.indenter_type == "cube":
        scene_cfg = CubeTactileSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2)
        # disabled force field for cube indenter because a SDF collision mesh cannot be created for the Shape Prims
        scene_cfg.tactile_sensor.enable_force_field = False
        # Update tactile sensor configuration for cube
        scene_cfg.tactile_sensor.indenter_rigid_body = "indenter"
        scene_cfg.tactile_sensor.indenter_sdf_mesh = None
    elif args_cli.indenter_type == "nut":
        scene_cfg = NutTactileSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2)
        # Update tactile sensor configuration for nut
        scene_cfg.tactile_sensor.indenter_rigid_body = "indenter/factory_nut_loose"
        scene_cfg.tactile_sensor.indenter_sdf_mesh = "indenter/factory_nut_loose/collisions"
    elif args_cli.indenter_type == "none":
        scene_cfg = TactileSensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2)
        # this flag is to visualize the tactile sensor points
        scene_cfg.tactile_sensor.debug_vis = True

    scene = InteractiveScene(scene_cfg)

    # Setup compliant materials (required after scene initialization, can be skipped if materials are pre-configured and not needed to be changed)
    scene["tactile_sensor"].setup_compliant_materials()

    # Initialize simulation
    sim.reset()
    print("[INFO]: Setup complete...")

    # Juana: this should be manually called before running any simulation ?
    scene["tactile_sensor"].get_initial_render()

    # Run simulation
    run_simulator(sim, scene)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
