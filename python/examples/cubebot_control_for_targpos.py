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
import torch
from isaacgym.torch_utils import *
import time
# load configuration data
with open("../../training/cfg/task/CubeBot_TargPos.yaml", "r") as cfg:
    try:
        cfg = yaml.safe_load(cfg)
    except yaml.YAMLError as exc:
        print(exc)

def lerp(val1, val2, ratio):
    if(ratio>1):
        ratio = 1
    if(ratio<0):
        ratio = 0
    return (1-ratio)*val1 + (ratio)*val2

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

# sim_params.gravity = gymapi.Vec3(0.0, -9.81/2, 0.0)

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

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
gym.add_ground(sim, gymapi.PlaneParams())

# set up the env grid
num_envs = 4
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
initial_pose.p = gymapi.Vec3(0.0, 0.5, 0.0)
initial_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# Create environment 0
# Cart held steady using position target mode.
# Pole held at a 45 degree angle using position target mode.
env0 = gym.create_env(sim, env_lower, env_upper, 2)
cubebot0 = gym.create_actor(env0, cubebot_asset, initial_pose, 'CubeBot', 0, 1)
# Configure DOF properties
props = gym.get_actor_dof_properties(env0, cubebot0)
control_mode = cfg["env"]["controlMode"]
if(control_mode):
    props["driveMode"][:] = gymapi.DOF_MODE_VEL
else:
    props["driveMode"][:] = gymapi.DOF_MODE_EFFORT

props["stiffness"] = cfg["env"]["stiffness"] * control_mode
props['damping'][:] = cfg["env"]["damping"] * control_mode
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
# Simulate
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Every 100 steps, incriment the control_idx variable
    if(loop_count % update_period == 0):
        control_idx += 1
        if(control_idx>5):
            control_idx = 0
            pair_idx += 1
            if(pair_idx > 2):
                pair_idx = 0
        print('new_control_idx, new_pair_idx = {}, {}'.format(control_idx, pair_idx))
        # time.sleep(1)

    torque_tensor = to_torch([0, 0, 0, 0, 0, 0], dtype=torch.float, device='cpu')
    action = 0
    bangbang=True
    if(control_idx == 0):
        if(bangbang):
            action = 1
        else:
            action = lerp(0, 1, (loop_count % update_period)/update_period)
    elif(control_idx == 1):
        action = 1
    elif(control_idx == 2):
        if(bangbang):
            action = -1
        else:
            action = lerp(1, -1, (loop_count % update_period)/update_period)
    elif(control_idx == 3):
        action = -1
    elif(control_idx == 4):
        if(bangbang):
            action = 0
        else:
            action = lerp(-1, 0, (loop_count % update_period)/update_period)

    target_speed = action*cfg["env"]["maxSpeed"]
    target_torque = action*cfg["env"]["maxTorque"] 
    

        
    if(control_mode):
        # Set the DOF target velocities
        gym.set_dof_target_velocity(env0, dof_handles[2*pair_idx], target_speed)
        gym.set_dof_target_velocity(env0, dof_handles[2*pair_idx+1], target_speed)
    else:
        offset = 2
        max_available_torque = torch.clip(cfg["env"]["maxTorque"] - (offset*dof_vel/cfg["env"]["maxSpeed"] + (1-offset))*cfg["env"]["maxTorque"] , -cfg["env"]["maxTorque"], cfg["env"]["maxTorque"])
        min_available_torque = torch.clip(-cfg["env"]["maxTorque"] - (offset*dof_vel/cfg["env"]["maxSpeed"] - (1-offset))*cfg["env"]["maxTorque"], -cfg["env"]["maxTorque"], cfg["env"]["maxTorque"])

        torque_tensor[2*pair_idx] = np.clip(target_torque, min_available_torque[2*pair_idx], max_available_torque[2*pair_idx])
        torque_tensor[2*pair_idx+1] = np.clip(target_torque, min_available_torque[2*pair_idx+1], max_available_torque[2*pair_idx+1]) 

        print('target_torque = {}'.format(target_torque))
        print('torque_tensor = {}'.format(torque_tensor))
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torque_tensor))
    
    

    loop_count += 1 
        
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    print('root_state')
    print(root_states)
    print('dof_state')
    print(dof_state)
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
