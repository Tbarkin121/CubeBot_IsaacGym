# CubeBot_IsaacGym

The assets directory contains the xacro and urdf robot descriptions for this project. A shell script named compile_xacro.sh sources ROS and compiles the urdf file from the xacro macro. 

The python director contains some test code that was useful for getting training started and examining the environment

The training directory contains the script that launches the Isaac Gym training session. To launch run : 
python train.py task=CubeBot