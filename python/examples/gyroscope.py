import math
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import yaml
import torch
import time

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 5
sim_params.dt = 1.0 / 60.0
sim_params.gravity=gymapi.Vec3(0.0, -9.81, 0.0)

sim_params.physx.solver_type = 2
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
device = args.sim_device if sim_params.use_gpu_pipeline else 'cpu'
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


# add t_handle urdf asset
asset_root = "../../assets"
asset_file = "urdf/gyroscope.urdf"

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.angular_damping = 0.005
asset_options.linear_damping = 0.0
asset_options.max_angular_velocity = 100
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
gyroscope_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# initial root pose for actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 2.0, 0.0)
initial_pose.r = gymapi.Quat(-0.70710678118, 0.0, 0.0, 0.70710678118)

# set up the env grid
num_envs = 2
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)
# Create environments

env0 = gym.create_env(sim, env_lower, env_upper, num_envs)
gyroscope0 = gym.create_actor(env0, gyroscope_asset, initial_pose, 'Gyroscope', 0, 1)
env1 = gym.create_env(sim, env_lower, env_upper, num_envs)
gyroscope1 = gym.create_actor(env1, gyroscope_asset, initial_pose, 'Gyroscope', 0, 1)

# Look at the first env
cam_pos = gymapi.Vec3(8, 4, 1.5)
cam_target = gymapi.Vec3(0, 2, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# prepare tensor access
gym.prepare_sim(sim)

# Configure DOF properties
props = gym.get_actor_dof_properties(env0, gyroscope0)
props["driveMode"].fill(gymapi.DOF_MODE_NONE)
props["stiffness"].fill(0.0)
props['damping'][:].fill(0.0)
props['velocity'][:].fill(10000.0)
props['effort'][:].fill(0.0)
props['friction'][:].fill(0.0)
gym.set_actor_dof_properties(env0, gyroscope0, props)

props = gym.get_actor_dof_properties(env1, gyroscope1)
props["driveMode"].fill(gymapi.DOF_MODE_NONE)
props["stiffness"].fill(0.0)
props['damping'][:].fill(0.0)
props['velocity'][:].fill(10000.0)
props['effort'][:].fill(0.0)
props['friction'][:].fill(0.0)
gym.set_actor_dof_properties(env1, gyroscope1, props)

dof_dict = gym.get_actor_dof_dict(env0, gyroscope0)
dof_keys = list(dof_dict.keys())
print(dof_keys)

# Get DOF State Tensor
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_state = gymtorch.wrap_tensor(dof_state_tensor)
dof_pos = dof_state.view(num_envs, 3, 2)[..., 0]
dof_vel = dof_state.view(num_envs, 3, 2)[..., 1]
gym.refresh_dof_state_tensor(sim)

dof_pos[:, 0] = 1.57075
dof_pos[:, 1] = 1.57075/2

dof_vel[0, 2]=1

print(dof_state)
actor_indicies = torch.tensor([0, 1], dtype=torch.int32, device=device)
gym.set_dof_state_tensor_indexed(sim, 
                                gymtorch.unwrap_tensor(dof_state),
                                gymtorch.unwrap_tensor(actor_indicies), 
                                len(actor_indicies))


# Simulate
idx=0
while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    
    if(idx % 20 == 0):
        print(idx)
        # update the viewer
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
    
        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)
    idx += 1

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
