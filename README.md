# MCTR 
<table>
  <tr>
    <td align="center" valign="top">
      <img src="./first_vision.gif" height="300"><br>
      <sub>first_vision</sub>
    </td>
    <td align="center" valign="top">
      <img src="./third_vision.gif" height="300"><br>
      <sub>third_vision</sub>
    </td>
    <td align="center" valign="top">
      <img src="./digital_vision.gif" height="300"><br>
      <sub>digital_vision</sub>
    </td>
  </tr>
</table>

More details please see ：

>bilibili:

>youtube:

<div id="Main"></div>

## Main project

This project consists of two main modules:
<nav>
  <ul>
    <li><a href="#module1">F1tenth Simulator && Physical Controller</a></li>
    <li><a href="#module2">Carla Simulator Controller</a></li>
    <li><a href="#module3">Create Digital Twin Map</a></li>
    <li><a href="#module4">Custom Vehicle Import</a></li>
  </ul>
</nav>

### Project Struct

```bash
  project_mctr/
  │
  ├──controller/                      
  │   ├── controller_manager.py        
  │   ├── mctr/                        
  ├──requirements.txt   #For carla                 
  ├──launch_carla.sh                 
  ├──carla-ros-bridge/
     └── carla_ws/
         └── src/
             └── carla_stack/  
  ├──rcarla_example/
  │    ├── digital_twin.ply 
  │    ├── map.obj                
  │    └── tmp.ply                   
```

## Prerequisites 
To run the MCTR race stack make sure that you have assembled and setup a car following the [official F1TENTH instructions](https://f1tenth.org/build). 

Please install strictly in the following order:
  - Ubuntu 20.04: [Double System](https://zhuanlan.zhihu.com/p/363640824) or [WSL with Windows 11](https://zhuanlan.zhihu.com/p/466001838)
  - [ROS Noetic](https://blog.csdn.net/qq_44339029/article/details/120579608)
  - [Catkin Build Tools (Installing on Ubuntu with apt-get)](https://catkin-tools.readthedocs.io/en/latest/installing.html)
  - [FORZA_ETH RaceStack](https://github.com/ForzaETH/race_stack)
  - [CCMA](https://github.com/UniBwTAS/ccma)


**Note**:

Add the following aliases to your `~/.bashrc` for convenience:
```
alias sros1="source /opt/ros/noetic/setup.bash"
alias sf1="sros1; source ~/catkin_ws/devel/setup.bash" # for convenience, no need>
alias mctr="sf1"
```
Now, you can simply type `mctr` in the terminal to activate the environment.
<!-- F1tenth Simulator && Physical Controller -->
<h2 id="module1"></h2>
<a href="#Main">

# **F1tenth Simulator && Physical Controller**</a>

## How to use F1tenth Simulator
MCTR race stack runs on the Forza ETH stack. Merge or replace folders `controller` with our files.

Ensure it subscribes to the following ROS topics: 

| Topic Name                          | Message Type                 | Description                                                   |
| ----------------------------------- | ---------------------------- | ------------------------------------------------------------- |
| `/scan`                             | `sensor_msgs/LaserScan`      | LiDAR scan data for environment and obstacle perception       |
| `/car_state/odom`                   | `nav_msgs/Odometry`          | Current vehicle velocity                                      |
| `/car_state/pose`                   | `geometry_msgs/PoseStamped`  | Vehicle pose in map frame (position + orientation)            |
| `/car_state/odom_frenet`            | `nav_msgs/Odometry`          | Vehicle position in Frenet frame (currently unused)           |
| `/vesc/sensors/imu/raw`             | `sensor_msgs/Imu`            | Raw IMU acceleration data (for steering adjustment)           |
| `/local_waypoints`                  | `f110_msgs/WpntArray`        | Locally planned waypoints (short-term trajectory)             |
| `/global_waypoints`                 | `f110_msgs/WpntArray`        | Global trajectory for initialization                          |
| `/perception/obstacles`             | `f110_msgs/ObstacleArray`    | Detected obstacles (position, velocity, static/dynamic)       |
| `/state_machine`                    | `std_msgs/String`            | Current vehicle control mode (e.g., FTGONLY, TRAILING)        |
| `/l1_param_tuner/parameter_updates` | `dynamic_reconfigure/Config` | Real-time parameter tuning (e.g., L1 controller PID settings) |


After setup, you can launch the base system and our algorithm in the F1tenth Simulator with:
```
launch stack_master base_system.launch sim:=true map_name:=test_map
```
Another terminal:
```
roslaunch stack_master time_trials.launch ctrl_algo:=MCTR racecar_version:=NUC2
```
If you have a F1tenth car, choose the sim to `false`:
```
roslaunch stack_master base_system.launch sim:=false map_name:=test_map
```
<!--Carla Simulator Controller-->
<h2 id="module2"></h2> <a href="#Main">

# **Carla Simulator Controller**</a>
## Environment Setup
```
# 1. Create a virtual environment (recommended: Python 2.7.18)
python2.7 -m venv carla_env

# 2. Activate the environment
source carla_env/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

```
Add this alias to your `~/.bashrc` to quickly source the Carla environment:
```
alias carla='source ~/carla-ros-bridge/catkin_ws/devel/setup.bash'
```
Now, you can simply type `carla` in the terminal to activate the environment.
## Pre-Installation

Install CARLA and CARLA-ROS-BRIDGE:
 - [CARLA](https://github.com/carla-simulator/carla)
 - [CARLA-ROS-BRIDGE](https://github.com/carla-simulator/ros-bridge)

For your convenience, you can directly use our integrated `Carla-ros-bridge`.
```
carla-ros-bridge/
  └── carla_ws/
      └── src/
          └── carla_stack/
              ├── carla_spawn_objects/   
              ├── carla_controller/ 
              └── spawn_f1tenth/             
```
If you are using the official carla-ros-bridge, please replace or add our package to the location below.
```
carla-ros-bridge/
  └── catkin_ws/
      └── src/
          ├── vision_opencv/
          └── ros-bridge/
              ├── carla_spawn_objects/   
              ├── carla_controller/ 
              └── spawn_f1tenth/             
```

## How to use Carla Simulator
All startup commands are designed as scripts to facilitate program startup. 
You can run the script `launch_carla.sh` anywhere (recommended in the `carla` folder).
Start carla:
```
./launch_carla.sh
```
Click `Play` in the Carla after launching.

Start CARLA-ROS-Bridge (in a new terminal):
```
# Source Carla environment
carla

# Launch ROS bridge with ego vehicle
roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch

```
Launch the MCTR 3D Algorithm (in another terminal):

```
roslaunch carla_controller controller_manager.launch ctrl_algo:=DELAUNAY_FTG

```
<!-- Digital Twin Map Creation-->
<h2 id="module3"></h2><a href="#Main">

# **Digital Twin Map Creation**</a>
## Pre-Installation
We used `rcarla`, developed by ETH, to build the digital twin mesh in Carla simulator:
- Install [rcarla](https://github.com/ForzaETH/rcarla)

## Obtain Point Cloud Data
Use 2D/3D LiDAR to capture point cloud data (.pcd/.ply files). Example track point cloud data `digital_twin.ply` is provided for mesh generation.

You need to use a point cloud processing tool (such as `CloudCompare`) to delete point clouds that are irrelevant to the track and obtain a pre-processed point cloud file.

Please place the point cloud files in the `rcarla/digital_twin_creator/data` folder.

## Generate Mesh Files

```
cd rcarla/digital_twin_creator
./setup.sh
source venv/bin/activate
python3 digital_twin_creator.py data/<YOUR_POINTCLOUD_FILE>
```
This will generate `tmp.ply` and `map.obj` files. You can rename them as needed. Example track point cloud data `tmp.ply` and `map.obj` generated by `digital_twin.ply` are provided.
## Create Digital Twin Map
Start Carla:
```
cd ~/carla
./launch_carla.sh
```
Import the `.obj` file in UE4, create a new level to build the map, or open example maps like `Test_map` or `Digital_Twin_map`:
```
cp PATH/TO/Digital_Twin_map/Carla/Maps/* PATH/TO/carla/Unreal/CarlaUE4/Content/Carla/Maps
```
Launch New Map:

Modify the `carla_ros_bridge_with_example_ego_vehicle.launch` file in `/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_ros_bridge/launch`:
```
<arg name="town" default='<YOUR_MAP_NAME>'/>
```

Modify vehicle spawn point:
```
<arg name="spawn_point" default="<X,Y,Z,ROLL,PITCH,YAW>"/>
```
Click Play in the UE4 window, then in a new terminal, launch the vehicle:

```
cd ~/carla-ros-bridge/catkin_ws
./launch_ros_bridge_with_ego.sh
```
<!-- Custom Vehicle Import-->
<h2 id="module4"></h2><a href="#Main">

# Custom Vehicle Import</a>

## Vehicle Specifications:
Wheelbase: `37cm`
Axle track: `50cm` <br>
CARLA Vehicle Asset (Length: 77cm, Width: 36cm) Usage

## Import Vehicle Assets

1. Open CARLA.
2. Create a new folder in the **Content Browser**.
3. Right-click the folder and select **Open in File Explorer**.

## Configure Wheel Assets

1. In the **Content Browser**, navigate to `Content/Carla/Blueprints/Vehicles`.
2. Place the `Wheel_shape_small.uasset` in this folder.

## Register Vehicle in VehicleFactory

1. In `Content/Carla/Blueprints/Vehicles`, double-click `VehicleFactory`.
2. Click on **Vehicle**.
3. In the right panel, click the plus icon to add a new element.
4. Fill in the configuration as shown in the documentation.

## Spawn Custom Vehicle

- When spawning vehicles, use the vehicle name `rgv`.

