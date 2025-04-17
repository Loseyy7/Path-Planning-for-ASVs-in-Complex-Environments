import numpy as np
import matplotlib.pyplot as plt
import random

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent

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

def is_valid_path(grid, p1, p2):
    if p1[0] != p2[0] and p1[1] != p2[1]:
        return False  # Prevent diagonal movement
    
    num_checks = abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])
    for i in range(1, num_checks + 1):
        interp_point = (
            p1[0] + (p2[0] - p1[0]) * i // num_checks,
            p1[1] + (p2[1] - p1[1]) * i // num_checks
        )
        if not (0 <= interp_point[0] < grid.shape[0] and 0 <= interp_point[1] < grid.shape[1]) or grid[interp_point] == 1:
            return False
    return True

def rrt_search(grid, start, goal, max_iterations=500, step_size=1):
    rows, cols = grid.shape
    tree = [Node(start)]
    
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap=plt.cm.binary, origin="upper")
    
    for _ in range(max_iterations):
        rand_point = get_random_point(grid)
        nearest_node = get_nearest_node(tree, rand_point)
        
        if abs(rand_point[0] - nearest_node.position[0]) > abs(rand_point[1] - nearest_node.position[1]):
            new_point = (nearest_node.position[0] + np.sign(rand_point[0] - nearest_node.position[0]) * step_size, nearest_node.position[1])
        else:
            new_point = (nearest_node.position[0], nearest_node.position[1] + np.sign(rand_point[1] - nearest_node.position[1]) * step_size)
        
        if 0 <= new_point[0] < rows and 0 <= new_point[1] < cols and grid[new_point] == 0:
            if is_valid_path(grid, nearest_node.position, new_point):
                new_node = Node(new_point, nearest_node)
                tree.append(new_node)
                ax.plot([nearest_node.position[1], new_point[1]], [nearest_node.position[0], new_point[0]], 'bo-', markersize=2)
                
                if manhattan_distance(new_point, goal) < step_size:
                    goal_node = Node(goal, new_node)
                    tree.append(goal_node)
                    
                    path = []
                    while goal_node:
                        path.append(goal_node.position)
                        goal_node = goal_node.parent
                    path.reverse()
                    
                    for i in range(len(path) - 1):
                        ax.plot([path[i][1], path[i+1][1]], [path[i][0], path[i+1][0]], 'go-', markersize=2)
                    
                    plt.show()
                    return path
    
    plt.show()
    return None

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

start = (0, 0)
end = (9, 7)

path = rrt_search(grid, start, end)
print("Path:", path)
