"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


DOF control methods example
---------------------------
An example that demonstrates various DOF control methods:
- Load cartpole asset from an urdf
- Get/set DOF properties
- Set DOF position and velocity targets
- Get DOF positions
- Apply DOF efforts
"""

import math
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import yaml
import numpy as np

# For Generating Observation Like RL script
import torch
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *

import time
import matplotlib.pyplot as plt
import numpy as np

def lerp(val1, val2, ratio):
    if(ratio>1):
        ratio = 1
    if(ratio<0):
        ratio = 0
    return (1-ratio)*val1 + (ratio)*val2

# load configuration data
with open("../../training/cfg/task/CubeBot.yaml", "r") as cfg:
    try:
        cfg = yaml.safe_load(cfg)
    except yaml.YAMLError as exc:
        print(exc)

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity.x = 0
sim_params.gravity.y = 0
sim_params.gravity.z = -9.81

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# set up the env grid
num_envs = 1
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)

# add cartpole urdf asset
asset_root = "../../assets"
asset_file = "urdf/CubeBot.urdf"

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.angular_damping = cfg["env"]["angularDamping"]
asset_options.max_angular_velocity = cfg["env"]["angularVelocity"]
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
cubebot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# initial root pose for cartpole actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
initial_pose.r = gymapi.Quat(0, 0.0, 0.0, 1.0)
start_rotation = torch.tensor([initial_pose.r.x, initial_pose.r.y, initial_pose.r.z, initial_pose.r.w])
# Create environment 0
# Cart held steady using position target mode.
# Pole held at a 45 degree angle using position target mode.
env0 = gym.create_env(sim, env_lower, env_upper, 2)
cubebot0 = gym.create_actor(env0, cubebot_asset, initial_pose, 'CubeBot', 0, 1)
# Configure DOF properties
props = gym.get_actor_dof_properties(env0, cubebot0)
props["driveMode"][:] = gymapi.DOF_MODE_VEL
props["stiffness"] = cfg["env"]["stiffness"]
props['damping'][:] = cfg["env"]["damping"]
props['velocity'][:] = cfg["env"]["maxSpeed"]
props['effort'][:] = cfg["env"]["maxTorque"]
props['friction'][:] = cfg["env"]["friction"]

gym.set_actor_dof_properties(env0, cubebot0, props)
# Set DOF drive targets
dof_dict = gym.get_actor_dof_dict(env0, cubebot0)
dof_keys = list(dof_dict.keys())

dof_handles = []
for key in dof_keys:
    dof_handles.append(gym.find_actor_dof_handle(env0, cubebot0, key))

# targets = torch.tensor([1000, 0, 0, 0, 0, 0])
# gym.set_dof_velocity_target_tensor(env0, gymtorch.unwrap_tensor(targets))

# Look at the first env
cam_pos = gymapi.Vec3(8, 4, 1.5)
cam_target = gymapi.Vec3(0, 2, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Some Variables to control whats happening
loop_count = 1
control_idx = 0
target_speed = 0
pair_idx = 0
update_period = 100

obs_buf = np.zeros(19)
actor_root_state = gym.acquire_actor_root_state_tensor(sim)
root_states = gymtorch.wrap_tensor(actor_root_state)
print(root_states)
root_pos = root_states.view(1, 13)[0, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
root_ori = root_states.view(1, 13)[0, 3:7] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
root_linvel = root_states.view(1, 13)[0, 7:10] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
root_angvel = root_states.view(1, 13)[0, 10:13] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)

dof_state_tensor = gym.acquire_dof_state_tensor(sim)
num_dof = 6
dof_state = gymtorch.wrap_tensor(dof_state_tensor)
print(dof_state)
dof_pos = dof_state.view(num_dof, 2)[:, 0]
dof_vel = dof_state.view(num_dof, 2)[:, 1]

# Measurements for rewards
potentials = torch.zeros((3), device='cpu')
up_axis_idx = 2 # Set z to up so this should be 2
up_vec = to_torch(get_axis_params(1., up_axis_idx), device='cpu').repeat((num_envs, 1))
heading_vec = to_torch([0, 1, 0], device='cpu').repeat((num_envs, 1))
inv_start_rot = quat_conjugate(start_rotation).repeat((num_envs, 1))
basis_vec0 = heading_vec.clone()
basis_vec1 = up_vec.clone()   


# Simulate
task='force'
plot_linvel = []
plot_angvel = []
plot_linvel_loc = []
plot_angvel_loc = []
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    if(task == 'force'):
        if(loop_count % update_period == 0):
            forces = torch.zeros((21, 3), device='cpu', dtype=torch.float)
            torques = torch.zeros((21, 3), device='cpu', dtype=torch.float)
            forces[0, 2] = 250
            torques[0, 2] = 10
            
            if(control_idx == 0):
                forces[0, 0] = 250
            if(control_idx == 1):
                forces[0, 1] = 250
            if(control_idx == 2):
                forces[0, 0] = -250
            if(control_idx == 3):
                forces[0, 1] = -250
            control_idx += 1
            if(control_idx > 3):
                control_idx = 0
                # Some Plotting
                f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
                plt_array = np.array(plot_linvel_loc)
                ax1.plot(plt_array[:,0], "-r", label="loc_vel x")
                ax1.plot(plt_array[:,1], "-g", label="loc_vel y")
                ax1.plot(plt_array[:,2], "-b", label="loc_vel z")
                ax1.legend(loc="upper right")
                ax1.set_title('plot_linvel_loc')

                plt_array = np.array(plot_angvel_loc)
                ax2.plot(plt_array[:,0], "-r", label="loc_vel x")
                ax2.plot(plt_array[:,1], "-g", label="loc_vel y")
                ax2.plot(plt_array[:,2], "-b", label="loc_vel z")
                ax2.legend(loc="upper right")
                ax2.set_title('plot_angvel_loc')

                plt_array = np.array(plot_linvel)
                ax3.plot(plt_array[:,0], "-r", label="loc_vel x")
                ax3.plot(plt_array[:,1], "-g", label="loc_vel y")
                ax3.plot(plt_array[:,2], "-b", label="loc_vel z")
                ax3.legend(loc="upper right")
                ax3.set_title('plot_linvel')

                plt_array = np.array(plot_angvel)
                ax4.plot(plt_array[:,0], "-r", label="loc_vel x")
                ax4.plot(plt_array[:,1], "-g", label="loc_vel y")
                ax4.plot(plt_array[:,2], "-b", label="loc_vel z")
                ax4.legend(loc="upper right")
                ax4.set_title('plot_angvel')
                plt.show()
                time.sleep(10)
                plot_linvel_loc = []
                plot_angvel_loc = []
                plot_linvel = []
                plot_angvel = []

            gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)
        loop_count += 1 

    if(task == 'step'):
        # Every 100 steps, incriment the control_idx variable
        if(loop_count % update_period == 0):
            control_idx += 1
            if(control_idx>1):
                control_idx = 0
        
        if(control_idx == 0):
            target_speed = lerp(0, -cfg["env"]["maxSpeed"], (loop_count % update_period)/update_period)
        if(control_idx == 1):
            target_speed = 0

        # Set the DOF target velocities
        gym.set_dof_target_velocity(env0, dof_handles[2*pair_idx], target_speed)
        gym.set_dof_target_velocity(env0, dof_handles[2*pair_idx+1], target_speed)

        loop_count += 1 

        
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    goal_pos = torch.zeros((num_envs, 3))
    goal_pos[:, 1] = 100
    to_target = goal_pos - root_pos
    to_target[:, 2] = 0.0

    prev_potentials= potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / sim_params.dt

    # print('!!!!')
    # print(root_ori.repeat((num_envs, 1)))
    # print(inv_start_rot)
    # print(to_target)
    # print(basis_vec0)
    # print(basis_vec1)
    # print('!!!!')
    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
            root_ori.repeat((num_envs, 1)), inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    # print('!!!!')
    # print(torso_quat)
    # print(root_linvel.repeat((num_envs, 1)))
    # print(root_angvel.repeat((num_envs, 1)))
    # print(goal_pos)
    # print(root_pos.repeat((num_envs, 1)))
    # print('!!!!')
    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
            torso_quat, root_linvel.repeat((num_envs, 1)), root_angvel.repeat((num_envs, 1)), goal_pos, root_pos.repeat((num_envs, 1)))

    dof_vel_scaled = dof_vel[0:6]/cfg["env"]["maxSpeed"]

    actions = torch.zeros((6))
    # print('vel_loc : {}'.format(vel_loc))
    # print('angvel_loc : {}'.format(angvel_loc))
    # print('yaw.unsqueeze(-1) : {}'.format(yaw.unsqueeze(-1)))
    # print('roll.unsqueeze(-1) : {}'.format(roll.unsqueeze(-1)))
    # print('pitch.unsqueeze(-1) : {}'.format(pitch.unsqueeze(-1)))
    # print('angle_to_target.unsqueeze(-1) : {}'.format(angle_to_target.unsqueeze(-1)))
    # print('up_proj.unsqueeze(-1) : {}'.format(up_proj.unsqueeze(-1)))
    # print('heading_proj.unsqueeze(-1) : {}'.format(heading_proj.unsqueeze(-1)))
    # print('dof_vel_scaled.repeat((num_envs, 1)) : {}'.format(dof_vel_scaled.repeat((num_envs, 1))))
    # print('actions.repeat((num_envs, 1)) : {}'.format(actions.repeat((num_envs, 1))))
    plot_linvel_loc.append([vel_loc[0,0], vel_loc[0,1], vel_loc[0,2]])
    plot_angvel_loc.append([angvel_loc[0,0], angvel_loc[0,1], angvel_loc[0,2]])
    plot_linvel.append([root_linvel[0].clone(), root_linvel[1].clone(), root_linvel[2].clone()])
    plot_angvel.append([root_angvel[0].clone(), root_angvel[1].clone(), root_angvel[2].clone()])

    obs = torch.cat((root_pos, root_ori, vel_loc, angvel_loc, 
                            angle_to_target.unsqueeze(-1), up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1), 
                            dof_vel_scaled.repeat((num_envs, 1)), actions.repeat((num_envs, 1))), dim=-1)

    
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
