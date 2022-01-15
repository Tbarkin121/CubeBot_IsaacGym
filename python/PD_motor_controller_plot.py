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

#%%

maxSpeed = 200
maxTorque = 2
Stiffness = 0 #Scales posError
Damping = 0.1   #Scales velError
num_points = 25
ones = np.ones((num_points, 1))

posError = np.linspace(-10, 10, num=num_points, endpoint=True, retstep=False, dtype=None, axis=0)
posError = np.expand_dims(posError, 1)
posError = np.transpose(np.dot(posError, np.transpose(ones)))
posTorque = posError*Stiffness

velError = np.linspace(-maxSpeed, maxSpeed, num=num_points, endpoint=True, retstep=False, dtype=None, axis=0)
velError = np.expand_dims(velError, 1)
velError = np.dot(velError, np.transpose(ones))
velTorque = velError*Damping


Torque = np.clip(posTorque + velTorque, -maxTorque, maxTorque)

azim_setpoints = np.linspace(0, 360, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
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
    plt.show()

#%%
plt.plot(posError, Torque)
plt.ylabel('Torque')
plt.xlabel('Error')
plt.show()

#%%
maxSpeed = 200
maxTorque = 2
Stiffness = 0 #Scales posError
Damping = 0.01   #Scales velError
num_points = 25
ones = np.ones((num_points, 1))

dof_vel = np.linspace(-maxSpeed, maxSpeed, num=num_points, endpoint=True, retstep=False, dtype=None, axis=0)
dof_vel = np.expand_dims(dof_vel, 1)
dof_vel = np.transpose(np.dot(dof_vel, np.transpose(ones)))

action = np.linspace(-1, 1, num=num_points, endpoint=True, retstep=False, dtype=None, axis=0)
action = np.expand_dims(action, 1)
action = np.dot(action, np.transpose(ones))

TorqueTarget = action*maxTorque
max_available_torque = np.clip(maxTorque - dof_vel*maxTorque/maxSpeed, -maxTorque, maxTorque)
min_available_torque = np.clip(-maxTorque - dof_vel*maxTorque/maxSpeed, -maxTorque, maxTorque)


Torque = np.clip(TorqueTarget, min_available_torque, max_available_torque)

azim_setpoints = np.linspace(0, 45, num=100, endpoint=True, retstep=False, dtype=None, axis=0)
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
    plt.show()

#%%
maxSpeed = 200
maxTorque = 2
Stiffness = 0 #Scales posError
Damping = 0.01   #Scales velError
num_points = 25
ones = np.ones((num_points, 1))

dof_vel = np.linspace(-maxSpeed, maxSpeed, num=num_points, endpoint=True, retstep=False, dtype=None, axis=0)
dof_vel = np.expand_dims(dof_vel, 1)
dof_vel = np.transpose(np.dot(dof_vel, np.transpose(ones)))

action = np.linspace(-1, 1, num=num_points, endpoint=True, retstep=False, dtype=None, axis=0)
action = np.expand_dims(action, 1)
action = np.dot(action, np.transpose(ones))

TorqueTarget = action*maxTorque
offset = 2
max_available_torque = np.clip(maxTorque - (offset*dof_vel/maxSpeed + (1-offset))*maxTorque, -maxTorque, maxTorque)
min_available_torque = np.clip(-maxTorque - (offset*dof_vel/maxSpeed - (1-offset))*maxTorque, -maxTorque, maxTorque)


Torque = np.clip(TorqueTarget, min_available_torque, max_available_torque)

azim_setpoints = np.linspace(0, 45, num=100, endpoint=True, retstep=False, dtype=None, axis=0)
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
    plt.show()

