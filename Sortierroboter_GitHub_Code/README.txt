Vision-Based Robotic Sorting of Colored Cubes with Dual-Camera Perception and Kinematic Motion Planning

This repository contains the software developed for the TH Köln Forschungsprojekt on autonomous sorting of colored cubes with an SO-101 robotic arm. The system combines a fixed top camera, a wrist/gripper camera, ArUco-based workspace registration, YOLO object detection, oriented bounding box (OBB) orientation estimation from the wrist camera, inverse kinematics, several calibration layers, and a modular Tkinter control interface.

The application is intentionally divided into modules. main_gui.py starts one shared RobotWorker and opens Module 6 as the main Control Center. The other modules are launched from Module 6 and exchange scene, calibration, and task state through the shared application context.

Important: hand detection is an interaction and recovery feature, not a certified safety system. An emergency stop or the ability to physically disconnect robot power should remain available. Computer vision alone must not be relied upon as a protective measure around the robot.


1) Repository Structure

- main_gui.py - application entry point
- core/ - robot commands and the shared RobotWorker
- robotics/ - calibration helpers, forward kinematics, and inverse kinematics
- modules/3_camera_panel.py - top camera, ArUco, YOLO, and workspace registration
- modules/6_robot_3d_panel.py - main Control Center and virtual robot
- modules/7_ik_target_panel.py - IK Target and robot calibration
- modules/8_cube_pick.py - cube calibration and pick-and-place
- modules/9_wrist_vision.py - wrist camera and OBB orientation estimation
- modules/10_autonomous_sort_loop.py - autonomous task orchestration
- data/ - saved calibration and pose data
- YOLO/best.pt - top-camera detector
- YOLO_OBB/best.pt - wrist-camera OBB detector
- _aruco_board_3x3_ids10-18_A4.pdf - board for intrinsic camera calibration
- aruco_tag_0_1_2_3_4_5_6_7_8_19.pdf - workspace, robot, container, and test-target markers
- 3D_Print/ - project 3D-printing files


2) Hardware Setup

The software was developed around the following setup:

- SO-101 / SO-ARM100-style 6-DOF robotic arm
- Feetech STS3215 serial bus servos
- Waveshare Serial Bus Servo Driver / Adapter
- external servo power supply
- fixed top USB camera
- wrist/gripper USB camera
- printed ArUco markers for workspace, base, containers, and calibration
- colored cubes
- four sorting containers

2.1) Marker IDs

- ID0-ID3 - workspace corners / workspace registration
- ID4 - robot base position and heading
- ID5 - red-cube container
- ID6 - yellow-cube container
- ID7 - green-cube container
- ID8 - movable IK / test target
- ID19 - blue-cube container
- ID10-ID18 - 3x3 intrinsic camera calibration board

Color-to-container mapping:

red    -> ID5
yellow -> ID6
green  -> ID7
blue   -> ID19

The 3x3 calibration board uses DICT_4X4_50, a marker size of 30.0 mm, a gap of 14.5 mm, a center-to-center spacing of 44.5 mm, and an overall board size of 119 x 119 mm.


3) Software Requirements and Virtual Environment

3.1) Recommended Environment

The project is intended to run inside a normal Python virtual environment. A Windows PC is the most straightforward setup because the robot is typically connected through a COM port and the wrist camera uses DirectShow when available.

Python 3.12, 64-bit, is recommended.

The source code requires the following third-party packages:

- numpy
- matplotlib
- Pillow
- OpenCV with cv2.aruco support
- ultralytics
- lerobot

Tkinter is also required. It is normally included with the standard Windows Python installation from python.org. On some Linux distributions it must be installed separately as python3-tk.

3.2) Creating the Virtual Environment - Windows PowerShell

Open PowerShell in the project directory:

py -3.12 -m venv .venv

Activate the environment:

.\.venv\Scripts\Activate.ps1

If PowerShell blocks local activation scripts, either use Command Prompt or allow script execution only for the current process:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

Upgrade the package installation tools:

python -m pip install --upgrade pip setuptools wheel

Install the project dependencies:

pip install numpy matplotlib pillow ultralytics lerobot opencv-contrib-python

3.3) Checking the Installation

Check the main imports:

python -c "import numpy, matplotlib, PIL, cv2; from ultralytics import YOLO; print('OpenCV:', cv2.__version__); print('ArUco:', hasattr(cv2, 'aruco'))"

The final line should report:

ArUco: True

If cv2.aruco is missing, the installed OpenCV build does not contain the required ArUco module. Remove conflicting OpenCV packages and install the contrib build:

pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install opencv-contrib-python

Then repeat the import check.

Before connecting hardware, the project can also be checked for Python syntax errors:

python -m compileall -q .

No output means that Python did not detect syntax errors.


4) First Start and Robot Connection

4.1) Starting the Application

From the activated virtual environment:

python main_gui.py

The application opens the SO-101 Control Center (Module 6). One RobotWorker is started in the background and shared by all modules.

4.2) Connecting the Robot

1. Power the servo bus from the external power supply.
2. Connect the Waveshare bus-servo adapter to the computer.
3. Identify the robot serial port using:

lerobot-find-port

Follow the instructions in the terminal. Disconnect and reconnect the USB adapter when requested. LeRobot will report the corresponding COM port.

If lerobot-find-port does not detect the adapter, the COM port can be checked manually in the Windows Device Manager.

4. Enter the detected COM port in Module 6.
5. Click Connect.
6. Watch the terminal. Depending on the local LeRobot configuration, the connection/calibration procedure may request confirmation by pressing ENTER.
7. Verify that the Control Center reports the robot as connected and that the encoder values are updating.

The application creates only one robot connection. Do not run a second copy of the application or another servo-control script at the same time.


5) Recommended Module Order

For a complete demonstration, use the following order:

1. Module 3 - Top Camera
2. Module 9 - Wrist Vision
3. Module 7 - IK Target / Robot Calibration, when calibration or testing is required
4. Module 8 - Cube Pick + READY + Test Auto
5. Module 10 - Autonomous Sorting

Module 10 expects an already-open, working instance of Module 8. It does not create a second hidden motion engine.


6) Complete Calibration Procedure

The repository contains calibration files produced for the project setup. If the same robot, cameras, markers, and mechanics are used without changes, recalibration is normally not required.

Recalibration is required after moving a camera mount, changing the gripper or arm mechanics, changing physical marker dimensions, replacing servos, moving the robot base relative to the scene, or creating a new installation.

Recommended order:

Top-camera intrinsic calibration
-> Top-camera workspace / marker registration
-> Robot joint limits
-> Robot frame calibration (XY + heading)
-> Robot Z compensation
-> Wrist-camera intrinsic calibration
-> Cube approach calibration
-> Grasp calibration
-> Container calibration
-> Gripper calibration
-> HOME + READY poses
-> Manual one-cube test
-> Autonomous sorting


6.1) Top Camera - Module 3

6.1.1) Starting the Camera

The default top-camera index is normally 0.

1. Open Module 3.
2. Set camera index 0 or the correct device index.
3. Start the camera.
4. Verify that the workspace markers are visible.

If the wrong camera opens, stop it, change the index, and start it again.

6.1.2) Loading the Top-Camera YOLO Model

Use:

YOLO/best.pt

Load the model in Module 3, enable detection, and verify that colored cubes and hands are detected.

During development, a confidence value around 0.40 and an inference limit around 10 FPS were used. These are working settings, not measured end-to-end system performance values.

6.1.3) Top-Camera Intrinsic Calibration

Use:

_aruco_board_3x3_ids10-18_A4.pdf

The calibration dialog uses the raw camera image and the 3x3 board with IDs 10-18.

Procedure:

1. Start the top camera.
2. Open Camera Calibration.
3. Place the 3x3 board in view.
4. At least 5 of the 9 markers must be visible for a useful sample.
5. Capture samples in different positions:
   - image center;
   - near each edge;
   - near the corners;
   - at several distances;
   - with several board tilts.
6. Capture at least 8 different samples; around 20 diverse samples are recommended by the interface.
7. Click Calculate + Save Calibration.
8. Verify that camera matrix, distortion coefficients, RMS, and mean reprojection error are displayed.
9. Keep lens correction / undistortion enabled.

The result is stored in:

data/camera_calibration.npz

Calibration RMS is a pixel reprojection-error diagnostic. It is not the physical positioning error of the robot.

6.1.4) Workspace Marker Placement

Place the markers so that:

- ID0-ID3 define the workspace;
- ID4 is located on or near the robot base and defines the base heading;
- ID5, ID6, ID7, and ID19 identify the four containers;
- ID8 is used as a movable test / IK target.

Module 3 uses ID0 as the metric scene reference. In the project setup, the physical black-square size of ID0 is 31.5 mm. If a newly printed ID0 has a different physical size, the corresponding constant must be changed before building the metric map.

6.1.5) Fixing Scene Coordinates

After the workspace, base, and containers are detected correctly:

1. Select Live markers.
2. Check ID0-ID7/ID19 positions and the workspace polygon.
3. Click Fix coordinates.
4. Use Fixed markers for normal operation.

Fixed mode prevents the workspace, base, and container reference positions from moving when markers are temporarily occluded by the robot.

ID8 intentionally remains live because it is a movable target.

If the physical workspace, robot base, or containers are moved, fix the coordinates again.


6.2) Robot Connection and Joint Limits - Module 7

Before task calibration, verify that the motor encoder values are stable and that the physical robot is approximately represented by the virtual robot in Module 6.

6.2.1) Joint-Limit Calibration

Repeat this procedure only when the mechanical or servo configuration changes.

Use the Joint Limit Calibration controls in Module 7 to save the allowed minimum and maximum motor positions for each joint. These limits are used when validating IK candidates and converting model angles back to motor positions.

Do not increase the limits only to force an unreachable IK target to work. First verify geometry, frame calibration, and target position.


6.3) Robot Frame Calibration - Module 7

This calibration corrects residual XY offset and heading between camera coordinates and the real robot coordinate system.

The movable ArUco ID8 is used.

For each sample:

1. Place ID8 at a reachable point in the workspace.
2. Select ID8 as the target in Module 7.
3. LOCK the current ID8 position.
4. Move the real TCP / gripper tip exactly to the center of the ID8 target.
5. Click Capture.
6. Repeat at several workspace positions, preferably with different X and Y values.
7. Fit / solve the frame calibration.
8. Check the resulting X offset, Y offset, heading correction, and RMS.

After this calibration, camera/world coordinates are transformed into robot-local coordinates using the saved frame correction.

Do not calibrate the frame from only one point.


6.4) Z Compensation - Module 7

Z compensation corrects the systematic distance-dependent height error observed on the physical robot.

Use the measurement mode in which the robot first moves to a requested target and the actual vertical error is then measured.

For each sample:

1. Select a reachable XY point.
2. Send the robot to a known Z target.
3. Measure the actual TCP height relative to the desired surface or target.
4. Save the sample.
5. Repeat at different distances from the robot base.
6. Fit the Z compensation.

Do not mix older samples captured with a different measurement method with the current calibration.


6.5) Wrist Camera - Module 9

6.5.1) Starting the Camera

The wrist camera normally uses index 1.

1. Open Module 9.
2. Select the correct camera index.
3. Start the camera.
4. Verify the image orientation.

6.5.2) Wrist-Camera Intrinsic Calibration

Use the same board:

_aruco_board_3x3_ids10-18_A4.pdf

The procedure is similar to the top camera:

1. Open Wrist Camera Calibration.
2. Show the board at different angles and in different image regions.
3. Save enough diverse samples.
4. Click Calculate + Save Calibration.
5. Verify that undistortion works.

The result is stored in:

data/wrist_camera_calibration.npz

6.5.3) OBB Model

Load:

YOLO_OBB/best.pt

Enable OBB detection and verify that the cube top face is detected correctly.

Module 9 only observes and estimates orientation. It does not send robot motion commands and does not rotate wrist_roll by itself.


6.6) Cube Approach Calibration - Module 8

This calibration corrects residual XY/Z error when approaching a cube.

Before starting, the following should already be correct:

- top-camera calibration;
- workspace registration;
- robot frame calibration;
- Z compensation.

Procedure:

1. Place a cube at a reachable position.
2. Select / LOCK the cube in Module 8.
3. Run GO ABOVE CUBE.
4. Use the calibration trims to position the TCP exactly above the cube.
5. Save the sample.
6. Repeat for several cube positions.
7. Fit the Cube XYZ correction.
8. Keep the learned Cube XYZ correction enabled for normal operation.

This correction is a separate task-level calibration layer. It should not be used to compensate for a globally incorrect robot-frame calibration.


6.7) Grasp XYZ Calibration - Module 8

Grasp calibration separately corrects the final grasp position. It does not replace Cube Approach Calibration.

For each sample:

1. LOCK a cube.
2. Run the calibrated approach above the cube.
3. Move to the grasp pose.
4. Use Grasp X/Y/Z trims to physically center the gripper around the cube.
5. Save the sample.
6. Repeat at several positions.
7. Fit the Grasp XYZ correction.

After a successful fit, residual manual trims should be reset.


6.8) Container XYZ Calibration - Module 8

The container correction is position-dependent rather than hard-coded separately for each marker ID. This allows a container to be moved to another part of the workspace while preserving its color assignment.

For each sample:

1. Select container marker ID 5, 6, 7, or 19.
2. Move the container to a suitable workspace position.
3. Click GO CONTAINER CAL.
4. Use Container X/Y/Z trim until the TCP is safely and accurately above the container center.
5. Click CAPTURE SET.
6. Move the container to another position and repeat.
7. Save at least 3 different positions.
8. Click FIT CONTAINER XYZ.
9. Keep Use learned Container XYZ correction enabled.

The model fits position-dependent affine corrections for X, Y, and Z.


6.9) Gripper Calibration - Module 8

The project does not use direct servo load/current measurement for grasp detection. Instead, it uses a position-error resistance proxy: resistance is estimated from the difference between commanded and measured gripper position during closing.

Calibration:

1. Click Free Move.
2. Place the empty gripper in the fully open physical position.
3. Click CAPTURE FULL OPEN.
4. Place the empty gripper in the fully closed physical position.
5. Click CAPTURE FULL CLOSED.
6. Return to torque / hold mode.
7. If required, adjust Grip strength [%], 100% resistance [deg], and the required number of consecutive resistance hits.
8. Test CLOSE GRIPPER before a complete pick-and-place run.

The displayed resistance value is only a proxy and must not be interpreted as measured gripping force in newtons.


6.10) HOME and READY - Module 8

6.10.1) Capture HOME

HOME is the pose used to clear the workspace during scene changes, recovery, and rescanning.

1. Click Free Move.
2. Place the arm in a safe folded / retracted pose.
3. Click CAPTURE HOME.

The pose is stored in data/ik_calibration.json.

6.10.2) Capture READY

READY is the working pose used between cube tasks.

1. Place the robot in the desired ready pose.
2. Click CAPTURE READY.
3. Test GO READY several times before autonomous operation.

The pose is stored in:

data/module8_ready_pose.json

HOME and READY intentionally serve different purposes:

- READY - fast working pose between cube tasks.
- HOME - retracted pose for recovery and rescanning.


7) Manual Validation Before Autonomous Mode

Do not start Module 10 immediately after recalibration. First validate one cube manually.

Recommended sequence in Module 8:

LOCK CUBE
-> GO ABOVE CUBE
-> GO TO GRASP POSE
-> OBSERVE / wrist orientation
-> ALIGN WRIST
-> CLOSE GRIPPER
-> TEST LIFT
-> GO CONTAINER
-> RELEASE

Then test START PICK + PLACE for one cube.

Proceed to autonomous sorting only when:

- the top camera provides stable cube world coordinates;
- base and container positions are correct;
- Module 7 reaches ID8 with sufficient accuracy;
- wrist OBB orientation becomes stable;
- the gripper reliably detects cube resistance;
- lifting and container transport do not collide with the table, camera, or other objects;
- HOME and READY are safe.


8) Autonomous Sorting - Module 10

Module 10 is the task coordinator. It does not compute its own low-level robot trajectories. It uses the already-open Module 8 for calibrated pick-and-place execution.

8.1) Before START

Verify that:

- the robot is connected;
- Module 3 is running;
- top-camera YOLO is enabled;
- workspace/base/container coordinates are available;
- Module 9 is running;
- the wrist OBB model is loaded;
- Module 8 is open;
- HOME is saved;
- READY is saved;
- cube/grasp/container calibration is enabled;
- gripper calibration has been checked.

8.2) Starting Autonomous Sorting

1. Open Module 10 - Autonomous Sorting.
2. Select ALL or a specific color.
3. Set the after-grasp transport speed if required.
4. Set hand-clear debounce and rescan delay if required.
5. Click START.

Normal high-level sequence:

START
-> HOME
-> SCAN / freeze scene
-> select cube
-> READY
-> calibrated pick-and-place through Module 8
-> READY
-> next cube
-> when no cubes remain: wait for scene change / hand interaction
-> HOLD while the hand is present
-> after the hand leaves: HOME
-> wait for the configured rescan delay
-> discard the old scene memory
-> new SCAN
-> continue

If a hand is detected inside the workspace during operation, the application requests HOLD and, after the hand has been stably absent, performs the recovery/rescan logic. This is an interaction feature, not a certified protective stop.


9) Cameras and Models

9.1) Camera Indices

Typical project configuration:

Top camera:   index 0
Wrist camera: index 1

USB enumeration can change after reconnecting cameras or changing USB ports. If the wrong camera opens, change the index in the corresponding module.

Using both cameras through a weak USB hub can cause dropped frames. Direct PC USB ports are preferred when possible.

9.2) YOLO Models

Top camera:

YOLO/best.pt

Wrist camera:

YOLO_OBB/best.pt

The top detector provides cube and hand positions in the fixed-camera image. Module 3 transforms image positions into workspace coordinates using the calibrated image geometry and planar scene mapping.

The wrist detector uses oriented bounding boxes to estimate local cube orientation. It is intentionally not used as a second global coordinate system.


10) Data Files

Do not delete the data/ directory before a demonstration unless a complete recalibration is intended.

- data/main.json - basic servo calibration and limits used by RobotWorker
- data/camera_calibration.npz - top-camera intrinsic matrix and distortion
- data/wrist_camera_calibration.npz - wrist-camera intrinsic matrix and distortion
- data/ik_target_calibration.json - robot geometry, mapping, frame calibration, joint limits, and Z compensation
- data/ik_calibration.json - saved robot poses including HOME
- data/cube_target_calibration.json - cube approach, grasp, container, and gripper calibration
- data/module8_ready_pose.json - READY motor pose
- data/camera_fixed_scene.json - saved fixed-scene data from the top camera
- data/object_geometry.json - virtual-scene object dimensions and TCP settings

After a successful physical calibration, backing up the entire data/ directory is recommended.

Example:

Copy-Item -Recurse data data_backup_working


11) Troubleshooting

11.1) ModuleNotFoundError: lerobot

Activate the correct virtual environment and run:

pip install lerobot

11.2) No module named ultralytics

Run:

pip install ultralytics

11.3) AttributeError: module 'cv2' has no attribute 'aruco'

Install OpenCV with ArUco support:

pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install opencv-contrib-python

11.4) Robot Does Not Connect

- check the external servo power supply;
- run lerobot-find-port and verify the reported COM port;
- close other applications using the serial adapter;
- check the terminal for LeRobot confirmation/calibration prompts;
- if the COM device disappears, reconnect the adapter and restart the application.

11.5) Wrong Camera Opens

Stop the camera and try another integer index. The project normally used top camera 0 and wrist camera 1, but Windows enumeration can change.

11.6) OBB / YOLO Model Not Found

Select the models manually:

YOLO/best.pt
YOLO_OBB/best.pt

11.7) Wrist Orientation Never Becomes Stable

Check:

- whether the requested cube class is visible;
- whether the wrist camera is close enough to the cube;
- whether the image orientation / rotation is correct;
- whether lighting is sufficient;
- whether confidence is set too high;
- whether center / angle jitter thresholds are too strict.

11.8) Robot Moves to the Wrong X/Y Position

Check in this order:

1. the physical ID0 size constant matches the printed marker;
2. workspace ID0-ID3 is defined correctly;
3. ID4 position and heading are correct;
4. fixed scene coordinates were updated after moving equipment;
5. robot-frame calibration in Module 7 is current;
6. cube calibration is enabled.

11.9) Z Is Correct Near the Base but Wrong Farther Away

Repeat the empirical Z calibration in Module 7 at several distances and refit the linear Z compensation.

11.10) Cube Approach Is Correct but the Final Grasp Is Offset

Do not change the global robot frame first. Check and, if necessary, repeat the separate Grasp XYZ Calibration in Module 8.

11.11) Robot Reaches Containers Incorrectly After They Were Moved

Update the fixed container coordinates in Module 3 if required and verify Container XYZ Calibration in Module 8.

11.12) Autonomous Sorting Reports That Module 8 Is Not Open

Open Module 8 first. Module 10 intentionally searches for an already-open live Module 8 and does not create a second instance.


12) Operational Notes and Known Limitations

- The system is a research prototype, not an industrial robot controller.
- Hand detection is not a safety-rated system.
- Motion accuracy depends on mechanical compliance, servo behavior, camera calibration, marker placement, and empirical calibration layers.
- The displayed gripper "force" is a commanded-versus-measured position resistance proxy, not a direct force or load measurement.
- Camera intrinsic calibration RMS is an image reprojection diagnostic, not end-effector accuracy.
- The virtual robot in Module 6 is a visualization / reconstruction tool, not a separately validated physical simulator.
- Detector validation metrics refer to the corresponding detector datasets and must not be interpreted as the overall autonomous sorting success rate.
- USB camera indices can change between computers or USB ports.


13) Quick Demonstration Checklist

- Activate .venv.
- Run python main_gui.py.
- Power the robot.
- Use lerobot-find-port to identify the serial port.
- Connect using the correct COM port in Module 6.
- Open Module 3 and start the top camera.
- Load YOLO/best.pt and enable detection.
- Check the workspace, ID4, and containers.
- Update / fix scene coordinates if anything was physically moved.
- Open Module 9 and start the wrist camera.
- Load YOLO_OBB/best.pt and enable detection.
- Open Module 8.
- Verify HOME and READY.
- If the setup was transported or rebuilt, manually test one cube.
- Open Module 10.
- Make sure the physical workspace is clear.
- Click START.


Authors

Nina-Ilenna Müller
Maksym Poizdnyk

TH Köln - Forschungsprojekt, 2026
