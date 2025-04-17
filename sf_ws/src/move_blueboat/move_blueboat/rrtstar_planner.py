import numpy as np
import cv2
import random

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.cost = 0  # Cost from start to this node

def load_pgm_map(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    grid_map = (image < 255 * 0.65).astype(np.int8)  # 1 = obstacle, 0 = free
    return grid_map

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def get_random_point(grid):
    rows, cols = grid.shape
    while True:
        point = (random.randint(0, rows - 1), random.randint(0, cols - 1))
        if grid[point] == 0:
            return point

def get_nearest_node(tree, random_point):
    return min(tree, key=lambda node: manhattan_distance(node.position, random_point))

def get_neighbors(tree, new_node, radius):
    return [node for node in tree if manhattan_distance(node.position, new_node.position) < radius]

def is_valid_path(grid, p1, p2):
    num_checks = max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))
    for i in range(1, num_checks + 1):
        interp_point = (
            p1[0] + (p2[0] - p1[0]) * i // num_checks,
            p1[1] + (p2[1] - p1[1]) * i // num_checks
        )
        if not (0 <= interp_point[0] < grid.shape[0] and 0 <= interp_point[1] < grid.shape[1]) or grid[interp_point] == 1:
            return False
    return True

def rrt_star_search(grid, start, goal, max_iterations=500, step_size=1, radius=3):
    tree = [Node(start)]
    for _ in range(max_iterations):
        rand_point = get_random_point(grid)
        nearest_node = get_nearest_node(tree, rand_point)
        new_point = (
            nearest_node.position[0] + np.sign(rand_point[0] - nearest_node.position[0]) * step_size,
            nearest_node.position[1] + np.sign(rand_point[1] - nearest_node.position[1]) * step_size
        )
        
        if 0 <= new_point[0] < grid.shape[0] and 0 <= new_point[1] < grid.shape[1] and grid[new_point] == 0:
            if is_valid_path(grid, nearest_node.position, new_point):
                new_node = Node(new_point, nearest_node)
                new_node.cost = nearest_node.cost + manhattan_distance(nearest_node.position, new_point)
                
                neighbors = get_neighbors(tree, new_node, radius)
                for neighbor in neighbors:
                    new_cost = neighbor.cost + manhattan_distance(neighbor.position, new_point)
                    if new_cost < new_node.cost and is_valid_path(grid, neighbor.position, new_point):
                        new_node.parent = neighbor
                        new_node.cost = new_cost
                
                tree.append(new_node)
                if manhattan_distance(new_point, goal) < step_size:
                    goal_node = Node(goal, new_node)
                    tree.append(goal_node)
                    path = []
                    while goal_node:
                        path.append(goal_node.position)
                        goal_node = goal_node.parent
                    path.reverse()
                    return path
    return None

def save_path(path, filename):
    with open(filename, "w") as file:
        for point in path:
            file.write(f"{point[0]}, {point[1]}\n")

if __name__ == "__main__":
    grid_map = load_pgm_map("map/map.pgm")
    start, goal = (0, 0), (4, 5)
    path = rrt_star_search(grid_map, start, goal)
    
    if path:
        save_path(path, "mission_path.txt")
        print("Path found and saved to mission_path.txt")
    else:
        print("No path found")
