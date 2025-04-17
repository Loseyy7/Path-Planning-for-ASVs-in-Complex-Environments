import heapq
import cv2
import numpy as np

# Load map from PGM file
def load_pgm_map(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    grid_map = (image < 255 * 0.65).astype(np.int8)  # 1 = obstacle, 0 = free
    return grid_map

def dijkstra(grid, start, goal):
    if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
        print("Error: Start or goal is inside an obstacle.")
        return []

    neighbors = [(1, 0), (0, 1), (0, -1), (-1, 0)]  # 4-directional movement
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        cost, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0], neighbor[1]] == 1:  
                    continue  # Obstacle
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(open_set, (new_cost, neighbor))
                    came_from[neighbor] = current

    return []  # No path found

def save_path(path, filename):
    with open(filename, "w") as file:
        for point in path:
            file.write(f"{point[0]}, {point[1]}\n")

if __name__ == "__main__":
    grid_map = load_pgm_map("map/map.pgm")
    start, goal = (0, 0), (4, 5)
    path = dijkstra(grid_map, start, goal)

    if path:
        save_path(path, "mission_path.txt")
        print("Path found and saved to mission_path.txt")
    else:
        print("No path found")
