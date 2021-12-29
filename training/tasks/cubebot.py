import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from tasks.base.vec_task import VecTask


class CubeBot(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]

        # 3 reaction-wheel velocities
        # cube orientation
        # cube velocity
        self.cfg["env"]["numObservations"] = 22
        # Drive signal for each of the three primary axis reaction wheels
        self.cfg["env"]["numActions"] = 6

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_pos = self.root_states.view(self.num_envs, 2, 13)[..., 0, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)

        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state = gymtorch.wrap_tensor(rb_state_tensor)
        self.rb_pos = self.rb_state.view(self.num_envs, 16, 13)[..., 0:16, 0:3] #num_envs, num_rigid_bodies, 13 (pos,ori,Lvel,Avel)
        self.corner1_pos = self.rb_state.view(self.num_envs, 16, 13)[..., self.body_dict['CornerBumper_1'], 0:3] #num_envs, num_rigid_bodies, 13 (pos,ori,Lvel,Avel)
        
        self.target_pos = torch.tensor([0, 0, 0])
        self.target_idx = torch.tensor(0.0)

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')

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
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 10000

        cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(cartpole_asset)
        # self.num_actor = get_sim_actor_count
        # self.num_rb = get_actor_rigid_body_count(cartpole_asset)
        target_asset = self.gym.create_sphere(self.sim, 0.05)

        pose = gymapi.Transform()
        pose.p.z = 1.0
        # asset is rotated z-up by default, no additional rotations needed
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.cartpole_handles = []
        self.target_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            cartpole_handle = self.gym.create_actor(env_ptr, cartpole_asset, pose, "cubebot", i, 1, 0)
            dof_props = self.gym.get_actor_dof_properties(env_ptr, cartpole_handle)
    
            dof_props['driveMode'][:] = gymapi.DOF_MODE_VEL
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 10000.0
            dof_props['velocity'][:] = 10000.0
            dof_props['effort'][:] = 2.5
            dof_props['friction'][:] = 0.0

            self.gym.set_actor_dof_properties(env_ptr, cartpole_handle, dof_props)

            self.envs.append(env_ptr)
            self.cartpole_handles.append(cartpole_handle)

            target_pose = gymapi.Transform()
            target_handle = self.gym.create_actor(env_ptr, target_asset, target_pose, "target", i, 1, 0)
            self.gym.set_rigid_body_color(env_ptr, target_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.8, 0.2))
            self.target_handles.append(target_handle)

        self.body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, cartpole_handle)
        for b in self.body_dict:
                print(b)
                
    def compute_reward(self):
        # retrieve environment observations from buffer
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        box_pos = self.obs_buf[:, 0:2]
        # box_ori = self.obs_buf[:, 3:7]
        # box_lin_vel = self.obs_buf[:, 7:10]
        # box_ang_vel = self.obs_buf[:, 10:13]
        # print(self.corner1_pos)
        # print(self.corner1_pos.shape)
        self.rew_buf[:], self.reset_buf[:] = compute_cartpole_reward(
            self.corner1_pos[:, 2], box_pos,
            self.reset_dist, self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        # print(self.root_states)
        # print(self.root_states.shape)
        self.obs_buf[env_ids, 0:13] = self.root_states[env_ids, :] #Position(3), Orientation(4), Linear Vel(3), Angular Vel(3)
        self.obs_buf[env_ids, 13] = self.dof_vel[env_ids, 0].squeeze() #Wheels 1 and 2 should be driven with the same signal
        self.obs_buf[env_ids, 14] = self.dof_vel[env_ids, 1].squeeze() #Wheels 3 and 4 should be driven with the same signal
        self.obs_buf[env_ids, 15] = self.dof_vel[env_ids, 2].squeeze() #Wheels 5 and 6 should be driven with the same signal 
        self.obs_buf[env_ids, 16] = self.dof_vel[env_ids, 3].squeeze() #Wheels 1 and 2 should be driven with the same signal
        self.obs_buf[env_ids, 17] = self.dof_vel[env_ids, 4].squeeze() #Wheels 3 and 4 should be driven with the same signal
        self.obs_buf[env_ids, 18] = self.dof_vel[env_ids, 5].squeeze() #Wheels 5 and 6 should be driven with the same signal 
        
        # Target Position
        # self.target_pos = torch.tensor([torch.sin(self.target_idx/100),
        #                                 torch.cos(self.target_idx/100),
        #                                 0], 
        #                                 device=self.device)
        # self.target_idx += 1
        target_pos_envs = self.target_pos.repeat(len(env_ids),1)
        self.obs_buf[env_ids, 19:22] = target_pos_envs

        # Update the position of the target ball (Not working yet)
        # target_pose = gymapi.Transform()
        # self.gym.set_rigid_body_state_tensor(env_ids, self.target_handles, target_pose)

        # Not to start... Needs extra work.
        # Need Target Orientation
        # Need Target Velocity?

        return self.obs_buf

    def reset_idx(self, env_ids):
        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        # env_ids_int32 = env_ids.to(dtype=torch.int32, device=self.device)
        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                       gymtorch.unwrap_tensor(self.dof_state),
        #                                       gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
  
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        self.target_pos = torch.cat(torch.rand( 2 , device=self.device ), torch.tensor(0))

    def pre_physics_step(self, actions):
        # print('actions')
        # print(actions)
        # print(actions.shape)
        # print(actions.to(self.device).squeeze() * self.max_push_effort)
        # print(actions.to(self.device).squeeze().shape())
        self.actions = actions.clone().to(self.device)
        targets = self.actions*200
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(targets))
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
def compute_cartpole_reward(corner_height, root_pos,
                            reset_dist, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    reward = corner_height - torch.linalg.norm(root_pos, dim=1)/2.0
    # reward = 1 - (pole1_angle * pole1_angle)/2 - (pole2_angle * pole2_angle)/2 - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole1_vel) - 0.005 * torch.abs(pole2_vel)

    # adjust reward for reset agents
    # reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)
    # reward = torch.where(torch.abs(pole1_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

    # reset = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reset_buf), reset_buf)
    # reset = torch.where(torch.abs(pole1_angle) > np.pi / 2, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return reward, reset
