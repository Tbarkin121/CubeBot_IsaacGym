import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from tasks.base.vec_task import VecTask
import matplotlib.pyplot as plt
import time

class CubeBot_WheelVel(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.stiffness = self.cfg["env"]["stiffness"]
        self.damping = self.cfg["env"]["damping"]
        self.maxSpeed = self.cfg["env"]["maxSpeed"]
        self.maxTorque = self.cfg["env"]["maxTorque"]
        self.friction = self.cfg["env"]["friction"]
        self.angularDamping = self.cfg["env"]["angularDamping"]
        # 3 reaction-wheel velocities
        # cube orientation
        # cube velocity
        self.cfg["env"]["numObservations"] = 19
        # Drive signal for each of the three primary axis reaction wheels
        self.cfg["env"]["numActions"] = 6

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_pos = self.root_states.view(self.num_envs, 1, 13)[..., 0, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.root_ori = self.root_states.view(self.num_envs, 1, 13)[..., 0, 3:7] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.root_linvel = self.root_states.view(self.num_envs, 1, 13)[..., 0, 7:10] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.root_angvel = self.root_states.view(self.num_envs, 1, 13)[..., 0, 10:13] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)

        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state = gymtorch.wrap_tensor(rb_state_tensor)
        self.rb_pos = self.rb_state.view(self.num_envs, self.num_bodies, 13)[..., 0:self.num_bodies, 0:3] #num_envs, num_rigid_bodies, 13 (pos,ori,Lvel,Avel)
        self.corner1_pos = self.rb_state.view(self.num_envs, self.num_bodies, 13)[..., self.body_dict['CornerBumper_1'], 0:3] #num_envs, num_rigid_bodies, 13 (pos,ori,Lvel,Avel)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.plot_buffer = []
        self.penelty_accumulation = 0
        
    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        # self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/CubeBot.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = self.angularDamping
        asset_options.max_angular_velocity = self.maxSpeed

        cubebot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(cubebot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(cubebot_asset)
        # self.num_actor = get_sim_actor_count
        # self.num_rb = get_actor_rigid_body_count(cubebot_asset)
        target_asset = self.gym.create_sphere(self.sim, 0.05)

        pose = gymapi.Transform()
        pose.p.z = 1.0
        # asset is rotated z-up by default, no additional rotations needed
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.cubebot_handles = []
        # self.target_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            cubebot_handle = self.gym.create_actor(env_ptr, cubebot_asset, pose, "cubebot", i, 1, 0)
            dof_props = self.gym.get_actor_dof_properties(env_ptr, cubebot_handle)
    
            dof_props['driveMode'][:] = gymapi.DOF_MODE_VEL
            dof_props['stiffness'][:] = self.stiffness
            dof_props['damping'][:] = self.damping
            dof_props['velocity'][:] = self.maxSpeed
            dof_props['effort'][:] = self.maxTorque
            dof_props['friction'][:] = self.friction


            self.gym.set_actor_dof_properties(env_ptr, cubebot_handle, dof_props)

            self.envs.append(env_ptr)
            self.cubebot_handles.append(cubebot_handle)

            # target_pose = gymapi.Transform()
            # target_handle = self.gym.create_actor(env_ptr, target_asset, target_pose, "target", i, 1, 0)
            # self.gym.set_rigid_body_color(env_ptr, target_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.8, 0.2))
            # self.target_handles.append(target_handle)

        self.body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, cubebot_handle)
        for b in self.body_dict:
                print(b)
                
    def compute_reward(self):
        # retrieve environment observations from buffer

        # box_pos = self.obs_buf[:, 0:2]
        # box_ori = self.obs_buf[:, 3:7]
        # box_lin_vel = self.obs_buf[:, 7:10]
        # box_ang_vel = self.obs_buf[:, 10:13]
        # print(self.corner1_pos)
        # print(self.corner1_pos.shape)

        self.rew_buf[:], self.reset_buf[:] = compute_cubebot_reward(
            self.corner1_pos[:, 2], self.obs_buf[:,13:19], self.reset_buf, self.progress_buf, self.max_episode_length
        )
        penelty = torch.sum(torch.square(self.obs_buf[:,13:19]), dim=1)/6 # Wheel velocity observation is scaled between -1 and 1 
        self.penelty_accumulation += penelty.cpu().detach().numpy()
        # print('penelty = {}'.format(penelty))

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # print(self.root_states)
        # print(self.root_states.shape)
        self.obs_buf[env_ids, 0:13] = self.root_states[env_ids, :] #Position(3), Orientation(4), Linear Vel(3), Angular Vel(3)
        self.obs_buf[env_ids, 0:3] /= 100
        self.obs_buf[env_ids, 7:10] /= 100
        self.obs_buf[env_ids, 10:13] /= 100
        self.obs_buf[env_ids, 13] = self.dof_vel[env_ids, 0].squeeze()/self.maxSpeed #Wheels 1 and 2 should be driven with the same signal
        self.obs_buf[env_ids, 14] = self.dof_vel[env_ids, 1].squeeze()/self.maxSpeed #Wheels 3 and 4 should be driven with the same signal
        self.obs_buf[env_ids, 15] = self.dof_vel[env_ids, 2].squeeze()/self.maxSpeed #Wheels 5 and 6 should be driven with the same signal 
        self.obs_buf[env_ids, 16] = self.dof_vel[env_ids, 3].squeeze()/self.maxSpeed #Wheels 1 and 2 should be driven with the same signal
        self.obs_buf[env_ids, 17] = self.dof_vel[env_ids, 4].squeeze()/self.maxSpeed #Wheels 3 and 4 should be driven with the same signal
        self.obs_buf[env_ids, 18] = self.dof_vel[env_ids, 5].squeeze()/self.maxSpeed #Wheels 5 and 6 should be driven with the same signal 
        self.plot_buffer.append(self.obs_buf[0:3, 13:19].cpu().detach().numpy())
        return self.obs_buf

    def reset_idx(self, env_ids):
        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        root_pos_update = torch.zeros((len(env_ids), 3), device=self.device)
        root_pos_update[:,2] = 0.3

        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        root_ori_update = quat_mul(quat_from_angle_axis(rand_floats[:,0] * np.pi, self.x_unit_tensor[env_ids]),
                    quat_from_angle_axis(rand_floats[:,1] * np.pi, self.y_unit_tensor[env_ids]))

        root_linvel_update = torch.zeros((len(env_ids), 3), device=self.device)
        root_angvel_update = torch.zeros((len(env_ids), 3), device=self.device)
        self.root_pos[env_ids, :] = root_pos_update
        self.root_ori[env_ids, :] = root_ori_update
        self.root_linvel[env_ids, :] = root_linvel_update
        self.root_angvel[env_ids, :] = root_angvel_update
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.root_states),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        plt.plot([0,0,0])
        plt.show()
        cam_pos = gymapi.Vec3(10, 8, 1.5)
        cam_target = gymapi.Vec3(0, 2, 1.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        if(self.plot_buffer):
            # print('Total Penelty Accumulated = {}'.format(self.penelty_accumulation))
            print('Mean Penelty Accumulated for n envs = {}'.format(np.mean(np.array(self.penelty_accumulation))))
            plot_data = np.array(self.plot_buffer)
            # fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            # fig.subplots_adjust(hspace=0.5)
            # ax1.plot(plot_data[:,0,:])
            # ax1.set_xlim(0,500)
            # ax1.set_ylim(-1,1)
            # ax1.set_xlabel('Steps')
            # ax1.set_ylabel('Scaled Velocity')
            # ax1.grid(True)

            # ax2.plot(plot_data[:,1,:])
            # ax2.set_xlim(0,500)
            # ax2.set_ylim(-1,1)
            # ax2.set_xlabel('Steps')
            # ax2.set_ylabel('Scaled Velocity')
            # ax2.grid(True)

            # ax3.plot(plot_data[:,2,:])
            # ax3.set_xlim(0,500)
            # ax3.set_ylim(-1,1)
            # ax3.set_xlabel('Steps')
            # ax3.set_ylabel('Scaled Velocity')
            # ax3.grid(True)
            # plt.show()
            for n in range(3):
                plt.plot(plot_data[:,n,:])
                plt.ylabel('Scaled Wheel Velocity')
                plt.xlabel('Steps')
                plt.grid()
                plt.xlim([0, 500])
                plt.ylim([-1.1, 1.1])
                plt.show()

            self.plot_buffer = []
            self.penelty_accumulation = 0

    def pre_physics_step(self, actions):
        # print('actions')
        # print(actions)
        # print(actions.shape)
        # print(actions.to(self.device).squeeze() * self.max_push_effort)
        # print(actions.to(self.device).squeeze().shape())
        self.actions = actions.clone().to(self.device)
        targets = self.actions*self.maxSpeed
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_cubebot_reward(corner_height, wheel_speeds,
                                reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    penelty = torch.sum(torch.square(wheel_speeds), dim=1)/6 # Wheel velocity observation is scaled between -1 and 1 
    reward = corner_height - penelty*0.5
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return reward, reset
