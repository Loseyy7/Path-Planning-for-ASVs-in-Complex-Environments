import numpy as np
import matplotlib.pyplot as plt
import heapq
import time

class Node:
    def __init__(self, position, parent=None, g=0, h=0):
        self.position = position  # (row, col)
        self.parent = parent
        self.g = g 
        self.h = h  
        self.f = g + h 

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_search(grid, start, end, visualize=True):
    rows, cols = grid.shape
    open_set = []
    closed_set = set()
    
    start_node = Node(start, None, 0, heuristic(start, end))
    heapq.heappush(open_set, start_node)

    came_from = {}
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    if visualize:
        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.set_xticklabels(np.arange(cols))
        ax.set_yticklabels(np.arange(rows))
        
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)

        cmap = plt.cm.binary
        ax.imshow(grid, cmap=cmap, origin="upper")

    while open_set:
        current_node = heapq.heappop(open_set)
        current_position = current_node.position

        if current_position in closed_set:
            continue
        closed_set.add(current_position)

        if current_position == end:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            path.reverse()

            if visualize:
                for pos in path:
                    ax.add_patch(plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, color="green", alpha=0.6))
                plt.pause(1)
                plt.show()
            return path

        for d in directions:
            new_row, new_col = current_position[0] + d[0], current_position[1] + d[1]

            if 0 <= new_row < rows and 0 <= new_col < cols and grid[new_row, new_col] == 0:
                new_position = (new_row, new_col)
                if new_position in closed_set:
                    continue

                new_g = current_node.g + 1
                new_h = heuristic(new_position, end)
                new_node = Node(new_position, current_node, new_g, new_h)

                heapq.heappush(open_set, new_node)
                came_from[new_position] = current_position

                if visualize:
                    ax.add_patch(plt.Rectangle((new_col - 0.5, new_row - 0.5), 1, 1, color="blue", alpha=0.5))
                    plt.pause(0.1)

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
# (4, 5)

path = astar_search(grid, start, end)
print("Path:", path)
