import os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
import numpy as np
import cv2
import yaml

class GridMapPublisher(Node):
    def __init__(self, save_dir="map"):  # Add save path parameter
        super().__init__('grid_map_publisher')
        self.publisher = self.create_publisher(OccupancyGrid, '/map', 10)
        self.timer = self.create_timer(1.0, self.publish_map)
        self.map_saved = False  
        self.save_dir = save_dir  
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def publish_map(self):
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = "map"

        map_msg.info.resolution = 1.0  # Each grid 1m
        map_msg.info.width = 11
        map_msg.info.height = 11
        map_msg.info.origin = Pose()  # Map origin

        # Initialize map data
        grid_data = np.full((map_msg.info.width * map_msg.info.height), 0)  # -1 = Unknown region
        
        # Add known obstacles
        obstacles = [(2, 3, 1, 5), (3, 3, 1, 1), (6, 6, 2, 2), (10, 0, 1, 1)]
        for x, y, w, h in obstacles:
            for i in range(int(x), int(x + w)):
                for j in range(int(y), int(y + h)):
                    index = j * map_msg.info.width + i
                    if 0 <= index < len(grid_data):
                        grid_data[index] = 100  # 100 = obs

        map_msg.data = grid_data.tolist()
        self.publisher.publish(map_msg)

        if not self.map_saved:
            self.save_map(grid_data, map_msg.info)
            self.map_saved = True 

    def save_map(self, grid_data, map_info):
        # Convert to 2D numpy array
        grid_2d = np.array(grid_data, dtype=np.int8).reshape(map_info.height, map_info.width)
        
        # Processing data (in ROS, 100 = obstacle, 0 = clearing, -1 = unknown)
        image = np.zeros((map_info.height, map_info.width), dtype=np.uint8)
        image[grid_2d == 100] = 0   # Black = Obstacle
        image[grid_2d == 0] = 255   # White = accessible area
        # image[grid_2d == -1] = 205  # Gray = Unknown area

        # save `.pgm` 
        pgm_filename = os.path.join(self.save_dir, "map.pgm")
        cv2.imwrite(pgm_filename, image)
        self.get_logger().info(f"Map saved as {pgm_filename}")

        # save `.yaml` 
        yaml_data = {
            "image": "map.pgm",
            "resolution": map_info.resolution,
            "origin": [map_info.origin.position.x, map_info.origin.position.y, 0.0],
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.196
        }
        yaml_filename = os.path.join(self.save_dir, "map.yaml")
        with open(yaml_filename, "w") as yaml_file:
            yaml.dump(yaml_data, yaml_file)
        
        self.get_logger().info(f"Map metadata saved as {yaml_filename}")

def main(args=None):
    rclpy.init(args=args)
    node = GridMapPublisher( )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
