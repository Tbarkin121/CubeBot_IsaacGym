import math
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import yaml

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 10
sim_params.dt = 1.0 / 60.0
sim_params.gravity=gymapi.Vec3(0.0, 0.0, 0.0)

sim_params.physx.solver_type = 2
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

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


# add t_handle urdf asset
asset_root = "../../assets"
asset_file = "urdf/THandle.urdf"

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.angular_damping = 0.0
asset_options.linear_damping = 0.0
asset_options.max_angular_velocity = 1000
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
thandle_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# initial root pose for actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 2.0, 0.0)
initial_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# set up the env grid
num_envs = 1
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)
# Create environment 0
env0 = gym.create_env(sim, env_lower, env_upper, num_envs)
thandle0 = gym.create_actor(env0, thandle_asset, initial_pose, 'THandle', 0, 1)

# Set Angular Velocity
gym.set_rigid_angular_velocity(env0, thandle0, gymapi.Vec3(0.0, 10.0, 0.0))

# Look at the first env
cam_pos = gymapi.Vec3(8, 4, 1.5)
cam_target = gymapi.Vec3(0, 2, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Get Root State Tensor
actor_root_state = gym.acquire_actor_root_state_tensor(sim)
root_states = gymtorch.wrap_tensor(actor_root_state)
root_pos = root_states.view(num_envs, 1, 13)[..., 0, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
root_ori = root_states.view(num_envs, 1, 13)[..., 0, 3:7] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
root_lin_vel = root_states.view(num_envs, 1, 13)[..., 0, 7:10] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
root_ang_vel = root_states.view(num_envs, 1, 13)[..., 0, 10:13] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)



# Simulate
while not gym.query_viewer_has_closed(viewer):
    gym.refresh_actor_root_state_tensor(sim)
    print(root_ang_vel)
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
   
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
