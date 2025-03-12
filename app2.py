import numpy as np
import heapq
import matplotlib.pyplot as plt
import streamlit as st

# Define Grid Size
GRID_WIDTH, GRID_HEIGHT = 10, 10

# Define Obstacles (randomly placed for now)
def generate_random_map():
    grid = np.zeros((GRID_WIDTH, GRID_HEIGHT))
    num_obstacles = 15  # Number of obstacles
    for _ in range(num_obstacles):
        x, y = np.random.randint(0, GRID_WIDTH), np.random.randint(0, GRID_HEIGHT)
        grid[x, y] = 1  # Mark obstacle
    return grid

# A* Search Algorithm for shortest path
def astar_search(grid, start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < GRID_WIDTH and 0 <= neighbor[1] < GRID_HEIGHT:
                if grid[neighbor] == 1:  # Obstacle
                    continue
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

# Visualization
def visualize_map(grid, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid.T, cmap="gray_r", origin="lower")

    # Plot Path
    for (x, y) in path:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color='blue', alpha=0.5))

    # Mark Start and Goal
    ax.scatter(start[0], start[1], color='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], color='red', s=100, label='Goal')

    ax.set_xticks(range(GRID_WIDTH))
    ax.set_yticks(range(GRID_HEIGHT))
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

# Streamlit Web App
st.title("AI Road Network Planning")
grid = generate_random_map()
start, goal = (0, 0), (GRID_WIDTH - 1, GRID_HEIGHT - 1)
path = astar_search(grid, start, goal)
visualize_map(grid, path)
