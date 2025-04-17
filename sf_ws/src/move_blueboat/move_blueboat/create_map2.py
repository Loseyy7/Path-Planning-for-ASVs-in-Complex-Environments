import os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
import numpy as np
import cv2
import yaml

class GridMapPublisher(Node):
    def __init__(self, save_dir="map"):
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

        map_msg.info.resolution = 1.0
        map_msg.info.width = 10
        map_msg.info.height = 10
        map_msg.info.origin = Pose()

        grid = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
        ])

        grid_data = (grid * 100).flatten()
        map_msg.data = grid_data.tolist()
        self.publisher.publish(map_msg)

        if not self.map_saved:
            self.save_map(grid, map_msg.info)
            self.map_saved = True 

    def save_map(self, grid, map_info):
        image = np.zeros((map_info.height, map_info.width), dtype=np.uint8)
        image[grid == 1] = 0   # Black = Obstacle
        image[grid == 0] = 255 # White = Accessible

        pgm_filename = os.path.join(self.save_dir, "map.pgm")
        cv2.imwrite(pgm_filename, image)
        self.get_logger().info(f"Map saved as {pgm_filename}")

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
    node = GridMapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
