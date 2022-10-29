#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 14:19:24 2022

@author: tyler
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import imageio
#%%

maxSpeed = 1
maxPosErr = 1
maxTorque = 1
Stiffness = 2 #Scales posError
Damping = 0   #Scales velError
num_points = 50
ones = np.ones((num_points, 1))

posError = np.linspace(-maxPosErr, maxPosErr, num=num_points, endpoint=True, retstep=False, dtype=None, axis=0)
posError = np.expand_dims(posError, 1)
posError = np.transpose(np.dot(posError, np.transpose(ones)))
posTorque = posError*Stiffness

velError = np.linspace(-maxSpeed, maxSpeed, num=num_points, endpoint=True, retstep=False, dtype=None, axis=0)
velError = np.expand_dims(velError, 1)
velError = np.dot(velError, np.transpose(ones))
velTorque = velError*Damping


Torque = np.clip(posTorque + velTorque, -maxTorque, maxTorque)

azim_setpoints = np.linspace(0, 360, num=100, endpoint=True, retstep=False, dtype=None, axis=0)
with imageio.get_writer('TorqueEquation_S{}_D{}.gif'.format(Stiffness, Damping), mode='I') as writer:
    for azim in azim_setpoints:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.view_init(elev=45, azim=azim)
        ax.dist = 10
        surf = ax.plot_surface(posError, velError, Torque, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf)
        ax.set_xlabel('Pos Error')
        ax.set_ylabel('Vel Error')
        ax.set_zlabel('Torque')
        plt.savefig('tmp_img.png')
        plt.show()


        # Build GIF
        image = imageio.imread('tmp_img.png')
        writer.append_data(image)



#%%
plt.plot(posError, Torque)
plt.ylabel('Torque')
plt.xlabel('Error')
plt.show()

#%%
maxSpeed = 1
maxTorque = 1
num_points = 50
ones = np.ones((num_points, 1))

dof_vel = np.linspace(-maxSpeed, maxSpeed, num=num_points, endpoint=True, retstep=False, dtype=None, axis=0)
dof_vel = np.expand_dims(dof_vel, 1)
dof_vel = np.transpose(np.dot(dof_vel, np.transpose(ones)))

action = np.linspace(-1, 1, num=num_points, endpoint=True, retstep=False, dtype=None, axis=0)
action = np.expand_dims(action, 1)
action = np.dot(action, np.transpose(ones))

TorqueTarget = action*maxTorque
# max_available_torque = np.clip(maxTorque - dof_vel*maxTorque/maxSpeed, -maxTorque, maxTorque)
# min_available_torque = np.clip(-maxTorque - dof_vel*maxTorque/maxSpeed, -maxTorque, maxTorque)
k = maxTorque/maxSpeed
max_available_torque = maxTorque - dof_vel*k
min_available_torque = -maxTorque - dof_vel*k
tmp_avaialbe_torque = maxTorque - np.abs(dof_vel*k)

Torque = np.clip(TorqueTarget, min_available_torque, max_available_torque)

azim_setpoints = np.linspace(0, 360, num=100, endpoint=True, retstep=False, dtype=None, axis=0)
with imageio.get_writer('MaxTorqueRight.gif', mode='I') as writer:
    for azim in azim_setpoints:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.view_init(elev=45, azim=azim)
        ax.dist = 10
        surf = ax.plot_surface(dof_vel, TorqueTarget, Torque, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf)
        ax.set_xlabel('DOF Vel')
        ax.set_ylabel('Torque Target')
        ax.set_zlabel('Torque Actual')
        plt.savefig('tmp_img.png')
        plt.show()
        
        # Build GIF
        image = imageio.imread('tmp_img.png')
        writer.append_data(image)

#%%
maxSpeed = 1
maxTorque = 1
num_points = 50
ones = np.ones((num_points, 1))

dof_vel = np.linspace(-maxSpeed, maxSpeed, num=num_points, endpoint=True, retstep=False, dtype=None, axis=0)
dof_vel = np.expand_dims(dof_vel, 1)
dof_vel = np.transpose(np.dot(dof_vel, np.transpose(ones)))

action = np.linspace(-1, 1, num=num_points, endpoint=True, retstep=False, dtype=None, axis=0)
action = np.expand_dims(action, 1)
action = np.dot(action, np.transpose(ones))

TorqueTarget = action*maxTorque

offset = 0.25

k = maxTorque/maxSpeed
max_available_torque = maxTorque - (offset*dof_vel*k + (1-offset)*maxTorque)
min_available_torque = -maxTorque - (offset*dof_vel*k - (1-offset)*maxTorque)


Torque = np.clip(TorqueTarget, min_available_torque, max_available_torque)

azim_setpoints = np.linspace(0, 360, num=100, endpoint=True, retstep=False, dtype=None, axis=0)
with imageio.get_writer('TorqueOffset_{}.gif'.format(offset), mode='I') as writer:
    for azim in azim_setpoints:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.view_init(elev=45, azim=azim)
        ax.dist = 10
        surf = ax.plot_surface(dof_vel, TorqueTarget, Torque, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf)
        ax.set_xlabel('DOF Vel')
        ax.set_ylabel('Torque Target')
        ax.set_zlabel('Torque Actual')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        ax.set_zlim(-1,1)
        plt.savefig('tmp_img.png')
        plt.show()
        
        # Build GIF
        image = imageio.imread('tmp_img.png')
        writer.append_data(image)

#%%
offset*dof_vel*maxTorque/maxSpeed + (1-offset)*maxTorque

(offset*dof_vel/maxSpeed + (1-offset))*maxTorque

offset*dof_vel/maxSpeed + (1-offset)
