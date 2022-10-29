import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from tasks.base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import *
import time

class CubeBot_TargPos(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self.dt = self.cfg["sim"]["dt"]

        self.reset_dist = self.cfg["env"]["resetDist"]
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.control_mode = self.cfg["env"]["controlMode"]
        self.stiffness = self.cfg["env"]["stiffness"] * self.control_mode
        self.damping = self.cfg["env"]["damping"] * self.control_mode
        self.maxSpeed = self.cfg["env"]["maxSpeed"]
        self.maxTorque = self.cfg["env"]["maxTorque"]
        self.friction = self.cfg["env"]["friction"]
        self.angularDamping = self.cfg["env"]["angularDamping"]
        self.angularVelocity = self.cfg["env"]["angularVelocity"]
        self.goal_dist = self.cfg["env"]["goalDist"]
       
        # cube root state (13) pos(3),ori(4),linvel(3),angvel(3)
        # wheel velocities (6)
        # Drive signal for each of the three primary axis reaction wheels
        self.cfg["env"]["numActions"] = 3
        # goal position (3) probably will ignore the z though... keep it in for now
        self.cfg["env"]["numObservations"] = 31+self.cfg["env"]["numActions"]
        

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.cube_pos = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.cube_ori = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0, 3:7] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.cube_linvel = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0, 7:10] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.cube_angvel = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0, 10:13] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.goal_pos = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 1, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)


        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state = gymtorch.wrap_tensor(rb_state_tensor)
        self.corner1_pos = self.rb_state.view(self.num_envs, self.num_bodies, 13)[..., self.body_dict['CornerBumper_1'], 0:3] #num_envs, num_rigid_bodies, 13 (pos,ori,Lvel,Avel)
       
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        print('self.dof_pos')
        print(self.dof_pos)
        print(self.dof_pos.shape)
        print('self.dof_vel')
        print(self.dof_vel)
        print(self.dof_vel.shape)


        print('self.cube_pos')
        print(self.cube_pos)
        print(self.cube_pos.shape)
        print('self.cube_ori')
        print(self.cube_ori)
        print(self.cube_ori.shape)
        print('self.cube_linvel')
        print(self.cube_linvel)
        print(self.cube_linvel.shape)
        print('self.cube_angvel')
        print(self.cube_angvel)
        print(self.cube_angvel.shape)

        print('self.corner1_pos')
        print(self.corner1_pos)
        print(self.corner1_pos.shape)

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        
        # Used for rewarding moving towards a target
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        to_target = self.goal_pos - self.cube_pos
        to_target[:, 2] = 0.0
        self.potentials = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.prev_potentials = self.potentials.clone()

        self.goal_reset = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        goal_ids = self.goal_reset.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_ids) > 0:
            self.reset_goal(goal_ids)
        
        # Measurements for rewards
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()   

        self.reward_total = torch.zeros((self.num_envs), device=self.device)
        self.torques = torch.zeros((self.num_envs, self.num_dof), device=self.device)


    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        # self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81/2)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        # plane_params.static_friction = 0.0
        # plane_params.dynamic_friction = 0.0
        # plane_params.restitution = 0.1
        # print('{} : {} : {}'.format(plane_params.static_friction, plane_params.dynamic_friction, plane_params.restitution))
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
        asset_options.max_angular_velocity = self.angularVelocity

        cubebot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(cubebot_asset)
        
        # self.num_actor = get_sim_actor_count
        goal_asset = self.gym.create_sphere(self.sim, 0.05)
        self.num_bodies = self.gym.get_asset_rigid_body_count(cubebot_asset) + self.gym.get_asset_rigid_body_count(goal_asset)
        

        pose = gymapi.Transform()
        pose.p.z = 1.0
        # asset is rotated z-up by default, no additional rotations needed
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.start_rotation = torch.tensor([pose.r.x, pose.r.y, pose.r.z, pose.r.w], device=self.device)

        self.cubebot_handles = []
        self.goal_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            cubebot_handle = self.gym.create_actor(env_ptr, cubebot_asset, pose, "cubebot", 0, 0, 0)
            dof_props = self.gym.get_actor_dof_properties(env_ptr, cubebot_handle)

            if(self.control_mode):
                dof_props['driveMode'][:] = gymapi.DOF_MODE_VEL
            else:
                dof_props['driveMode'][:] = gymapi.DOF_MODE_EFFORT

            dof_props['stiffness'][:] = self.stiffness
            dof_props['damping'][:] = self.damping
            dof_props['velocity'][:] = self.maxSpeed
            dof_props['effort'][:] = self.maxTorque
            dof_props['friction'][:] = self.friction


            self.gym.set_actor_dof_properties(env_ptr, cubebot_handle, dof_props)

            self.envs.append(env_ptr)
            self.cubebot_handles.append(cubebot_handle)

            goal_pose = gymapi.Transform()
            goal_pose.p.y = self.goal_dist
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_pose, "goal", 0, 0, 1)
            self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.8, 0.2))
            self.goal_handles.append(goal_handle)

        self.num_actors = self.gym.get_actor_count(self.envs[0])
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
        # distance_to_goal = torch.norm(self.cube_pos - self.goal_pos, dim=-1)
        # goal_reward = torch.where(distance_to_goal<1, 1, 0)
        # print(goal_reward)
        # progress_reward = self.potentials - self.prev_potentials
        # print(progress_reward)
        # print('progress_reward')
        # # print(progress_reward)
        # self.reward_total += progress_reward
        # print('self.reward_total')
        # print(self.reward_total)

        self.rew_buf[:], self.reset_buf[:], self.goal_reset = compute_cubebot_reward(
            self.corner1_pos[:, 2], 
            self.obs_buf[:,17:23], #obs_old [13:19] obs_new[17:23] 
            self.cube_pos,
            self.goal_pos,
            self.potentials,
            self.prev_potentials,
            self.reset_buf, 
            self.progress_buf, 
            self.max_episode_length
        )
        # print('{} : {} '.format(self.potentials, self.prev_potentials))
        # if(torch.abs(self.rew_buf[0]) > 2 or torch.abs(self.rew_buf[1]) > 2):
        # print('self.rew_buf')
        # print(self.rew_buf)
        #     print(self.rew_buf.shape)
        #     time.sleep(1)

        # if(torch.abs(self.reset_buf[0]) == 1 or torch.abs(self.reset_buf[1]) == 1):
        #     print('self.reset_buf')
        #     print(self.reset_buf)
        #     print(self.reset_buf.shape)
        #     time.sleep(1)

        # if(torch.abs(self.goal_reset[0]) == 1 or torch.abs(self.goal_reset[1]) == 1):
        #     print('self.goal_reset')
        #     print(self.goal_reset)
        #     print(self.goal_reset.shape)
        #     time.sleep(1)


    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.obs_buf, self.potentials, self.prev_potentials = compute_cubebot_observations(
            self.cube_pos, 
            self.cube_ori, 
            self.cube_linvel, 
            self.cube_angvel, 
            self.dof_vel, 
            self.goal_pos,
            self.potentials,
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.actions,
            self.torques,
            self.maxSpeed,
            self.dt)

        # print('Potential = {}. Previous_Potential = {}. Diff = {}'.format(self.potentials[0], self.prev_potentials[0], self.potentials[0] - self.prev_potentials[0]))

        # print('actions = {}'.format(self.obs_buf[:, 25:28]))
        # print('torques = {}'.format(self.obs_buf[:, 28:34]))
        # print('dof_vel = {}'.format(self.obs_buf[:, 19:25]))
        # print()
        return self.obs_buf

    def reset_idx(self, env_ids):
        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]
        env_ids_int32 = env_ids.to(dtype=torch.int32)*self.num_actors
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        cube_pos_update = torch.zeros((len(env_ids), 3), device=self.device)
        cube_pos_update[:,2] = 0.3

        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        cube_ori_update = quat_mul(quat_from_angle_axis(rand_floats[:,0] * np.pi, self.x_unit_tensor[env_ids]),
                    quat_from_angle_axis(rand_floats[:,1] * np.pi, self.y_unit_tensor[env_ids]))

        cube_linvel_update = torch.zeros((len(env_ids), 3), device=self.device)
        cube_angvel_update = torch.zeros((len(env_ids), 3), device=self.device)
        self.cube_pos[env_ids, :] = cube_pos_update
        self.cube_ori[env_ids, :] = cube_ori_update
        self.cube_linvel[env_ids, :] = cube_linvel_update
        self.cube_angvel[env_ids, :] = cube_angvel_update
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.root_states),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_goal(env_ids)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def reset_goal(self, env_ids):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # print('reset_goals')
        # print('Old Goal Position = {}'.format(self.goal_pos))
        env_ids_int32 = env_ids.to(dtype=torch.int32)*self.num_actors
        goal_pos_update = torch_rand_float(-10.0, 10.0, (len(env_ids), 3), device=self.device)
        # goal_pos_update[:,0] = 0
        # goal_pos_update[:,1] = self.goal_dist
        goal_pos_update[:,2] = 0.1
        self.goal_pos[env_ids, :] = goal_pos_update
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.root_states),
                                              gymtorch.unwrap_tensor(env_ids_int32+1), len(env_ids_int32))

        self.gym.refresh_actor_root_state_tensor(self.sim)
        to_target = self.goal_pos[env_ids] - self.cube_pos[env_ids]
        to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.goal_reset[env_ids] = 0

        # print('New Goal Position = {}'.format(self.goal_pos))

    def pre_physics_step(self, actions):
        # print(actions)
        # print(actions.shape)
        # print(actions.to(self.device).squeeze() * self.max_push_effort)
        # print(actions.to(self.device).squeeze().shape())
        self.actions = actions.clone().to(self.device)
        if(self.control_mode):
            # Vel Control
            self.set_motor_velocitys(self.actions)
        else:
            # Torque Control
            self.set_motor_torques(self.actions)
        

    def set_motor_velocitys(self, targets):
        target_vels = torch.zeros((self.num_envs, self.num_dof))
        target_vels[0:2] = targets[0]*self.maxSpeed
        target_vels[2:4] = targets[1]*self.maxSpeed
        target_vels[2:6] = targets[2]*self.maxSpeed
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(target_vels))
    
    def set_motor_torques(self, targets):
        target_torques = torch.zeros((self.num_envs, self.num_dof), device=self.device)

        target_torques[:, 0] = targets[:, 0]*self.maxTorque
        target_torques[:, 1] = targets[:, 0]*self.maxTorque
        target_torques[:, 2] = targets[:, 1]*self.maxTorque
        target_torques[:, 3] = targets[:, 1]*self.maxTorque
        target_torques[:, 4] = targets[:, 2]*self.maxTorque
        target_torques[:, 5] = targets[:, 2]*self.maxTorque

        # print('target_torques = {}'.format(target_torques))
        offset = 2
        
        max_available_torque = torch.clip(self.maxTorque - (offset*self.dof_vel/self.maxSpeed + (1-offset))*self.maxTorque, -self.maxTorque, self.maxTorque)
        min_available_torque = torch.clip(-self.maxTorque - (offset*self.dof_vel/self.maxSpeed - (1-offset))*self.maxTorque, -self.maxTorque, self.maxTorque)
        self.torques = torch.clip(target_torques, min_available_torque, max_available_torque)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

    def post_physics_step(self):
        self.progress_buf += 1
        
        goal_ids = self.goal_reset.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_ids) > 0:
            self.reset_goal(goal_ids)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_cubebot_reward(corner_height, wheel_speeds, cube_pos, goal_pos, potentials, prev_potentials,
                                reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor]

    penelty = torch.square((torch.sum(torch.abs(wheel_speeds), dim=1)/6)) # Wheel velocity observation is scaled between -1 and 1
    progress_reward = potentials - prev_potentials
    distance_to_goal = torch.norm(cube_pos - goal_pos, dim=-1)
    goal_reached = torch.where(distance_to_goal < 0.5, 1, 0)
    # reward = corner_height + goal_reward - torch.square(distance_to_goal/5.0)
    reward = progress_reward+goal_reached
    # reward = corner_height
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    goal_reset = torch.where(goal_reached==1, 1, 0)
    # goal_reset = torch.zeros_like(reset)

    return reward, reset, goal_reset


@torch.jit.script
def compute_cubebot_observations(cube_pos, cube_ori, cube_linvel, cube_angvel, dof_vel, goal_pos, 
                                    potentials, inv_start_rot, basis_vec0, basis_vec1, actions, torques, maxSpeed, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float) -> Tuple[Tensor, Tensor, Tensor]

    to_target = goal_pos - cube_pos
    to_target[:, 2] = 0.0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        cube_ori, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, cube_linvel, cube_angvel, goal_pos, cube_pos)

    dof_vel_scaled = dof_vel[:, 0:6]/maxSpeed
     

    # obs_buf shapes: 3, 4, 3, 3, 6, 3  = 22
    # obs = torch.cat((cube_pos/100,
    #                 cube_ori,
    #                 cube_linvel/100,
    #                 cube_angvel/100,
    #                 dof_vel_scaled,
    #                 goal_pos/100), dim=-1)
    

    # obs_buf shapes: 3, 4, 3, 3, (cube_pos, cube_ori, vel_loc, angvel_loc) 
    #                 3, 1, 1, 1 (goal_pos, angle_to_target, up_proj, heading_proj)
    #                 6, 3, 6 (dof_vel_scaled, actions, torques)
    #         total = 34
    # obs = torch.cat((cube_pos, cube_ori, vel_loc, angvel_loc, 
    #                     dof_vel_scaled, actions), dim=-1)
    obs = torch.cat((cube_pos, cube_ori, vel_loc, angvel_loc, goal_pos, 
                        angle_to_target.unsqueeze(-1), up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1),
                        dof_vel_scaled, actions, torques), dim=-1)

    return obs, potentials, prev_potentials_new

     