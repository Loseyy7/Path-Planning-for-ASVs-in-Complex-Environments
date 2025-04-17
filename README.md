# Path-Planning-for-ASVs-in-Complex-Environments
This repository contains the source code for the EPS6 4th year project in Heriot Watt University titled

**"Path Planning for Autonomous Surface Vehicles in Complex Environments"**.

This project uses Blueboat to realize path planning in Stonefish simulation environment.

# Setup Instructions
The specific simulation world and robot configuration used in this project are adapted from:

[markusbuchholz/stonefish_ros2_marine_robotics](https://github.com/Michele1996/stonefish_resources_pub.git)

# Overview
The navigation process is divided into the following key stages:
## 1. Grid Map Creation
A grid-based map of the environment is generated using create_map.py.

Output file:
1. **map.pgm** – grayscale occupancy grid mapmap.pgm
2. **map.yaml** – metadata file describing the map resolution

## 2. Path Planning
Path planners load the generated map and compute an optimal path from the robot's current position to the specified goal.

Output file:
**mission_path.txt** – a list of waypoints representing the planned path

## 3. Navigation
Using sensor data and the pre-computed path (mission_path.txt), the robot navigates through the environment.

# Usage
```bash
# Terminal 1: Launch the simulation environment
ros2 launch cola2_stonefish target_blueboat_alpha_launch.py

# Terminal 2: Create map
python3 create_map.py

# Terminal 2: Path Planning (For example, using A* algorithm)
python3 astar_planner.py

# Terminal 2: Navigation
python3 planner_replan.py
```

## Demo videos can be found on YouTube.
  A*: https://youtu.be/KpTtjEfi-70
  
  RRT*: https://youtu.be/cR-bUWz6gMA
  
  Q-Learning:https://youtu.be/lFhJiqGW9pQ

