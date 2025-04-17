import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import NavSatFix, Imu, LaserScan
import math
import time
import random
import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt

EPISODES = 2000
ALPHA = 0.2
GAMMA = 0.9
EPSILON = 0.5
MIN_EPSILON = 0.01
DECAY_RATE = 0.995
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)] 

class ASVController(Node):
    def __init__(self):
        super().__init__('asv_controller')

        self.time_start = time.time()

        # Publisher and subscribers
        self.motor_publisher = self.create_publisher(Float64MultiArray, '/blueboat/controller/thruster_setpoints_sim', 10)
        self.odometry_subscription = self.create_subscription(Odometry, '/blueboat/navigator/odometry', self.odometry_callback, 10)
        self.imu_subscription = self.create_subscription(Imu, '/blueboat/navigator/imu', self.imu_callback, 10)
        self.profiler_subscription = self.create_subscription(LaserScan, '/blueboat/profiler', self.profiler_callback, 10)

        # Waypoints for navigation
        self.waypoints = self.read_mission_path("mission_path.txt")
        print(self.waypoints)
        self.current_position = (0, 0)
        self.current_yaw = 0.0
        self.state = 'rotate_to_waypoint'
        self.current_waypoint_index = 0

        self.v_x = 0.0
        self.omega_z = 0.0 

        # EKF Initialization
        self.x_ekf = np.zeros(3)
        self.P_ekf = np.eye(3) * 1000
        self.Q_ekf = np.eye(3) * 0.1
        self.R_ekf = np.eye(1) * 0.5

        self.previous_time = time.time()


        # PID parameters for control
        self.linear_kP = 0.2
        self.linear_kI = 0.01
        self.linear_kD = 0.05
        self.angular_kP = 0.5
        self.angular_kI = 0.03
        self.angular_kD = 0.02
        # self.linear_kP = 0.2
        # self.linear_kI = 0.05
        # self.linear_kD = 0.01
        # self.angular_kP = 0.3
        # self.angular_kI = 0.15
        # self.angular_kD = 0.05

        self.previous_angular_error = 0
        self.angular_integral = 0
        self.previous_linear_error = 0
        self.linear_integral = 0

        self.grid_map = self.load_pgm_map("map/map.pgm")

        self.last_replan_time = 0.0  # Record the last replanning time
        self.replan_interval = 5.0  # Set the minimum interval (in seconds) for replanning


        self.path_history = []  
        self.fig, self.ax = plt.subplots()
        self.initialize_plot()


    def initialize_plot(self):
        self.ax.imshow(self.grid_map, cmap="gray", origin="upper")
        self.robot_marker, = self.ax.plot([], [], 'ro', markersize=6, label="Robot")  
        self.path_line, = self.ax.plot([], [], 'b-', linewidth=2.5, label="Path")  
        self.ax.legend()
        plt.ion()  # Allow Matplotlib to update dynamically
        plt.show()

    def update_plot(self):
        x, y = self.current_position
        self.path_history.append((x, y))

        path_x, path_y = zip(*self.path_history)
        self.path_line.set_data(path_y, path_x)

        self.robot_marker.set_data(y, x)

        plt.draw()
        plt.pause(0.1)  # Pause briefly to refresh the window



    def profiler_callback(self, msg):
        current_time = time.time()
        if current_time - self.last_replan_time >= self.replan_interval:
            for i, intensity in enumerate(msg.intensities):
                if intensity == 1.0:
                    self.get_logger().info(f"Obstacle detected at angle {msg.angle_min + i * msg.angle_increment}")
                    
                    x, y = self.get_obstacle_position()
                    if self.grid_map[x][y] == 0:
                        self.grid_map[x][y] = 1
                        self.replan_path()  # Trigger re-planning path
                        self.last_replan_time = current_time  # Update the time of last replanning
                    break

    def replan_path(self):      
        # # Update the map and mark obstacles
        # self.update_map_with_obstacle()
        
        # Take the current coordinate, and round it and convert it into an integer
        rounded_position = (round(self.current_position[0]), round(self.current_position[1]))
        print(rounded_position)

        # self.current_waypoint_index += 1
        env = GridEnvironment(self.grid_map, rounded_position, self.waypoints[-1])
        agent = QLearningAgent(env)
        agent.q_learning()
        new_path = agent.extract_path()
        print("Path:", new_path)

        if new_path:
            self.waypoints = new_path  # Update the new path
            self.current_waypoint_index = 0  # Starting from the first waypoint
            self.get_logger().info("New path planned successfully!")
        else:
            self.get_logger().warn("Failed to replan path.")

    # def update_map_with_obstacle(self):
    #     # Update the obstacles in the map (it is assumed that the obstacle is 1 here)
    #     # If the sensor detects an obstacle, mark the obstacle according to the currently detected position
    #     x, y = self.get_obstacle_position() # Change from right-handed frame to left-handed frame
    #     self.grid_map[x][y] = 1  # Mark the location as an obstacle
    #     # print(self.grid_map)


    def get_obstacle_position(self):
        x, y = self.current_position
        yaw = self.current_yaw

        distance = 1.0 

        # Calculate the position of the obstacle, the orientation is a true north (0 degree) vector
        obstacle_x = x + distance * round(math.cos(yaw))
        obstacle_y = y + distance * round(math.sin(yaw))
        print(obstacle_x)
        print(obstacle_y)

        # Rounding the obstacle position to the nearest grid point (assuming the map resolution is 1m)
        obstacle_x = int(round(obstacle_x))
        obstacle_y = int(round(obstacle_y))

        return obstacle_x, obstacle_y


    def load_pgm_map(self, filename):
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        grid_map = (image < 255 * 0.65).astype(np.int8)  # 1 = obstacle, 0 = free
        print(grid_map)
        return grid_map


    def navigate_to_waypoint(self):
        if self.current_waypoint_index >= len(self.waypoints):
            self.stop_asv()
            return

        waypoint = self.waypoints[self.current_waypoint_index]
        distance_to_waypoint = self.calculate_distance(self.current_position, waypoint)
        bearing_to_waypoint = self.calculate_bearing(self.current_position, waypoint)
        heading_error = self.normalize_angle(bearing_to_waypoint - self.current_yaw)

        self.get_logger().info(f"State: {self.state}, Current Position: {self.current_position}, Target Waypoint: {waypoint}, Distance Left: {distance_to_waypoint:.2f} meters, Heading Error: {heading_error:.2f}, Current Yaw: {self.current_yaw:.2f}")

        current_time = time.time()
        delta_time = current_time - self.previous_time

        if self.state == 'rotate_to_waypoint':
            if abs(heading_error) < 0.4:
                self.state = 'move_to_waypoint'
                self.get_logger().info("Transition to state: move_to_waypoint")
            else:
                # self.angular_integral += heading_error * delta_time  # update self.angular_integral
                angular_velocity = self.calculate_pid(self.angular_kP, self.angular_kI, self.angular_kD, heading_error, self.previous_angular_error, self.angular_integral, delta_time)
                # self.previous_angular_error = heading_error  # update previous_angular_error        
                self.publish_twist(0.0, angular_velocity)
        elif self.state == 'move_to_waypoint':
            if distance_to_waypoint < 0.5: # big 1.0 small 0.5
                self.state = 'stop_at_waypoint'
                self.stop_asv()
                self.get_logger().info("Transition to state: stop_at_waypoint")
            elif abs(heading_error) > 0.4:
                self.state = 'rotate_to_waypoint'
                self.get_logger().info("Transition to state: rotate_to_waypoint")
            else:
                # self.linear_integral += heading_error * delta_time 
                linear_velocity = self.calculate_pid(self.linear_kP, self.linear_kI, self.linear_kD, distance_to_waypoint, self.previous_linear_error, self.linear_integral, delta_time)
                # self.previous_linear_error = heading_error       
                self.publish_twist(linear_velocity, 0.0)
        elif self.state == 'stop_at_waypoint':
            self.stop_asv()
            self.current_waypoint_index += 1
            if self.current_waypoint_index < len(self.waypoints):
                self.state = 'rotate_to_waypoint'
                self.get_logger().info("Transition to state: rotate_to_waypoint")
            else:
                self.state = 'idle'
                self.get_logger().info("All waypoints achieved, state: idle")
                print(time.time() - self.time_start)


        self.previous_time = current_time

    def publish_twist(self, linear_x, angular_z):
        thrust_port = linear_x - angular_z
        thrust_stbd = linear_x + angular_z
        max_thrust = 0.3
        thrust_port = max(min(thrust_port, max_thrust), -max_thrust)
        thrust_stbd = max(min(thrust_stbd, max_thrust), -max_thrust)
        
        right_thrust = 0.0
        left_thrust = 0.0

        if self.omega_z > 0:
            right_thrust = self.omega_z * 2.2
            thrust_stbd = thrust_stbd * (1-self.omega_z) * 0.8
        
        if self.omega_z < -0:
            left_thrust = self.omega_z * (-2.2)
            thrust_port = thrust_port * (1+self.omega_z) * 0.8


        thruster_msg = Float64MultiArray()
        thruster_msg.data = [thrust_port, thrust_stbd, left_thrust, right_thrust]
        self.motor_publisher.publish(thruster_msg)
        # print(self.omega_z)
        # print("thrust", thrust_port, thrust_stbd)



    # Remaining methods stay the same (ekf_update, calculate_pid, publish_twist, etc.)
    def imu_callback(self, msg):
        imu_q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        _, _, yaw = self.euler_from_quaternion(imu_q)
        self.ekf_update(yaw)

    def odometry_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        self.v_x = msg.twist.twist.linear.x  
        self.omega_z = msg.twist.twist.angular.z  
        
        self.current_position = (x, y)

        self.update_plot()  # Visualize the path every time the location is updated

        self.navigate_to_waypoint()

    def ekf_update(self, yaw_measurement):
        # Prediction Step
        dt = time.time() - self.previous_time
        self.previous_time = time.time()

        F = np.eye(3)
        F[0, 2] = -dt * self.v_x * np.sin(self.x_ekf[2])
        F[1, 2] = dt * self.v_x * np.cos(self.x_ekf[2])
        self.x_ekf[0] += self.v_x * dt * np.cos(self.x_ekf[2])
        self.x_ekf[1] += self.v_x * dt * np.sin(self.x_ekf[2])
        self.x_ekf[2] += self.omega_z * dt  # Angle update

        # Process noise update (this is where the motion model could be refined)
        self.P_ekf = F @ self.P_ekf @ F.T + self.Q_ekf

        # Update Step (Measurement)
        H = np.array([[0, 0, 1]])  # Only yaw is observed
        z = np.array([yaw_measurement])
        y = z - H @ self.x_ekf  # Innovation
        S = H @ self.P_ekf @ H.T + self.R_ekf
        K = self.P_ekf @ H.T @ np.linalg.inv(S)  # Kalman gain

        # Update state and covariance
        self.x_ekf += K @ y
        self.P_ekf = (np.eye(3) - K @ H) @ self.P_ekf

        # Update yaw with fused result
        self.current_yaw = self.normalize_angle(self.x_ekf[2])

    def calculate_pid(self, kP, kI, kD, error, previous_error, integral, delta_time):
        integral += error * delta_time
        derivative = (error - previous_error) / delta_time
        output = kP * error + kI * integral + kD * derivative
        return output


    def stop_asv(self):
        self.publish_twist(0.0, 0.0)


    def read_mission_path(self, file_path):
        waypoints = []
        with open(file_path, 'r') as file:
            for line in file:
                x, y = map(int, line.strip().split(',')) 
                waypoints.append((x, y))  
        return waypoints

    @staticmethod
    def euler_from_quaternion(quat):
        x, y, z, w = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return roll, pitch, yaw

    @staticmethod
    def calculate_distance(pointA, pointB):
        return math.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)

    @staticmethod
    def calculate_bearing(pointA, pointB):
        x1, y1 = pointA
        x2, y2 = pointB
        angle = math.atan2(y2 - y1, x2 - x1)
        return angle

    @staticmethod
    def normalize_angle(theta):
        return (theta + math.pi) % (2 * math.pi) - math.pi

class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((env.rows, env.cols, 4))

    def q_learning(self):
        epsilon = EPSILON

        for _ in range(EPISODES):
            state = list(self.env.start)
            done = False

            while not done:
                if random.random() < epsilon:
                    action = random.choice(range(4))
                else:
                    action = np.argmax(self.q_table[state[0], state[1], :])

                new_state = (state[0] + ACTIONS[action][0], state[1] + ACTIONS[action][1])

                if not self.env.is_valid(new_state):
                    new_state = tuple(state)

                reward = self.env.get_reward(new_state)
                done = (new_state == self.env.goal) or (self.env.grid[new_state] == 1)

                current_q = self.q_table[state[0], state[1], action]
                max_next_q = np.max(self.q_table[new_state[0], new_state[1], :]) if not done else 0
                new_q = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)

                self.q_table[state[0], state[1], action] = new_q
                state = list(new_state)

            epsilon = max(MIN_EPSILON, epsilon * DECAY_RATE)

    def extract_path(self, visualize=True):
        path = [self.env.start]
        current = list(self.env.start)
        steps = 0
        max_steps = 200

        if visualize:
            fig, ax = plt.subplots()
            ax.set_xticks(np.arange(self.env.cols))
            ax.set_yticks(np.arange(self.env.rows))
            ax.set_xticklabels(np.arange(self.env.cols))
            ax.set_yticklabels(np.arange(self.env.rows))

            ax.set_xticks(np.arange(-0.5, self.env.cols, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.env.rows, 1), minor=True)
            ax.grid(which="minor", color="black", linestyle='-', linewidth=1)

            cmap = plt.cm.binary
            ax.imshow(self.env.grid, cmap=cmap, origin="upper")

        while current != list(self.env.goal) and steps < max_steps:
            action = np.argmax(self.q_table[current[0], current[1], :])
            new_r = current[0] + ACTIONS[action][0]
            new_c = current[1] + ACTIONS[action][1]

            if self.env.is_valid((new_r, new_c)):
                current = [new_r, new_c]
                path.append(tuple(current))

                if visualize:
                    ax.add_patch(plt.Rectangle((new_c - 0.5, new_r - 0.5), 1, 1, color="green", alpha=0.6))
                    plt.pause(0.1)

            steps += 1

        if visualize:
            plt.pause(1)
            plt.show()

        return path if current == list(self.env.goal) else []



class GridEnvironment:
    def __init__(self, grid_map, start, goal):
        self.grid = grid_map
        self.rows, self.cols = grid_map.shape
        self.start = start
        self.goal = goal

    def is_valid(self, state):
        r, c = state
        return (0 <= r < self.rows) and (0 <= c < self.cols) and (self.grid[r, c] == 0)

    def get_reward(self, state):
        if state == self.goal:
            return 100
        if self.grid[state] == 1:
            return -100
        return -1


def main(args=None):
    rclpy.init(args=args)
    node = ASVController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
