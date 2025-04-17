import numpy as np
import matplotlib.pyplot as plt
import heapq

def dijkstra_search(grid, start, end, visualize=True):
    rows, cols = grid.shape
    open_set = []
    closed_set = set()
    
    heapq.heappush(open_set, (0, start))  # (cost, position)
    came_from = {}
    cost_so_far = {start: 0}
    
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
        current_cost, current_position = heapq.heappop(open_set)
        
        if current_position in closed_set:
            continue
        closed_set.add(current_position)
        
        if current_position == end:
            path = []
            while current_position in came_from:
                path.append(current_position)
                current_position = came_from[current_position]
            path.append(start)
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
                new_cost = current_cost + 1
                
                if new_position not in cost_so_far or new_cost < cost_so_far[new_position]:
                    cost_so_far[new_position] = new_cost
                    heapq.heappush(open_set, (new_cost, new_position))
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

path = dijkstra_search(grid, start, end)
print("Path:", path)
