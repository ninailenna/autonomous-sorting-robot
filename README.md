# autonomous-sorting-robot
A 3D-printed robotic arm for sorting colored cubes using computer vision and inverse kinematics.
Developed as a team project (from Nina-Ilenna Müller and Maksym Poizdnyk) for the Master's program at TH Köln.

## Overview
The system uses an overhead camera to detect colored cubes
and determine their random positions. Inverse kinematics is used
to calculate the joint positions. These positions are needed to pick up and
sort the cubes.

## Key Features
- 3D printed robotic arm
- Camera based detection of colored cubes
- ArUco markers for camera-to-workspace calibration (for orientation: the main camera or rather the top-view camera uses the ArUco markers for orientation)
- Inverse kinematics for positioning the gripper
- Integration of mechanical components, electronics and software

## Project Photo
<img width="620" height="776" alt="image" src="https://github.com/user-attachments/assets/8e5d67ae-008b-4d06-995f-4ae42e4b019e" />
<img width="638" height="479" alt="image" src="https://github.com/user-attachments/assets/25a306dc-93da-4c24-8e64-0616b24e845d" />


## Video
I made a Youtube Video describing the project and showing the final work: https://youtu.be/ObwNQjSZZhc?si=5rtt6Ly3HG1N77sa 

Source code is available in this repository.
