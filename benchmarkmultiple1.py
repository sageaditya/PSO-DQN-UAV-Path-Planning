import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Tuple, Optional, Any
import importlib.util
import sys
import inspect
import os


def calculate_path_length(path: np.ndarray) -> float:
    """Calculate the total length of a path."""
    if path is None or len(path) < 2:
        return 0.0
    return np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1))


def calculate_path_clearance(path: np.ndarray, obstacles: List[Any]) -> float:
    """Calculate the minimum clearance from obstacles."""

    def distance_to_obstacle(point: np.ndarray, obstacle: Any) -> float:
        """Calculate the distance from a point to an obstacle."""
        if isinstance(obstacle, np.ndarray):  # Box obstacle
            x, y, z = point
            ox_min, oy_min, oz_min, ox_max, oy_max, oz_max = obstacle
            dx = max(ox_min - x, 0, x - ox_max)
            dy = max(oy_min - y, 0, y - oy_max)
            dz = max(oz_min - z, 0, z - oz_max)
            return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        elif isinstance(obstacle, dict):  # Dynamic obstacle
            return np.linalg.norm(point - obstacle['position']) - obstacle['radius']
        else:  # Static spherical obstacle
            center, radius = obstacle
            return np.linalg.norm(point - center) - radius

    if path is None:
        return 0.0

    clearances = []
    for point in path:
        min_clearance = float('inf')
        for obs in obstacles:
            min_clearance = min(min_clearance, distance_to_obstacle(point, obs))
        clearances.append(min_clearance)
    return np.mean(clearances)


def calculate_path_smoothness(path: np.ndarray) -> float:
    """Calculate the smoothness of the path as a percentage."""
    if path is None or len(path) < 3:
        return 0.0

    # Calculate angles between consecutive segments
    angles = []
    for i in range(1, len(path) - 1):
        v1 = path[i] - path[i - 1]
        v2 = path[i + 1] - path[i]
        # Normalize vectors
        v1 = v1 / (np.linalg.norm(v1) + 1e-6)
        v2 = v2 / (np.linalg.norm(v2) + 1e-6)
        # Calculate angle in radians
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angles.append(angle)

    # Convert angles to smoothness percentage
    # 0 degrees (parallel segments) = 100% smooth
    # 180 degrees (complete reversal) = 0% smooth
    angles = np.array(angles)
    smoothness = 100 * (1 - np.mean(angles) / np.pi)

    return smoothness


def calculate_cpu_usage(process: psutil.Process, start_time: float, end_time: float) -> float:
    """Calculate the average CPU usage percentage during execution."""
    try:
        # Get CPU times for the process
        cpu_times = process.cpu_times()
        total_cpu_time = cpu_times.user + cpu_times.system

        # Calculate actual CPU percentage based on total time and number of cores
        elapsed_time = end_time - start_time
        cpu_count = psutil.cpu_count()

        if elapsed_time > 0:
            cpu_percent = (total_cpu_time / elapsed_time / cpu_count) * 100
            return min(cpu_percent, 100.0)  # Cap at 100%
        return 0.0
    except Exception:
        return 0.0


def calculate_unsafe_points(path: np.ndarray, obstacles: List[Any], safety_threshold: float = 0.5) -> int:
    """Calculate number of points that are too close to obstacles."""
    if path is None:
        return 0

    unsafe_count = 0
    for point in path:
        min_clearance = float('inf')
        for obs in obstacles:
            if isinstance(obs, np.ndarray):  # Box obstacle
                x, y, z = point
                ox_min, oy_min, oz_min, ox_max, oy_max, oz_max = obs
                dx = max(ox_min - x, 0, x - ox_max)
                dy = max(oy_min - y, 0, y - oy_max)
                dz = max(oz_min - z, 0, z - oz_max)
                dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            elif isinstance(obs, dict):  # Dynamic obstacle
                dist = np.linalg.norm(point - obs['position']) - obs['radius']
            else:  # Static spherical obstacle
                center, radius = obs
                dist = np.linalg.norm(point - np.array(center)) - radius
            min_clearance = min(min_clearance, dist)

        if min_clearance < safety_threshold:
            unsafe_count += 1

    return unsafe_count


def visualize_path(bounds: np.ndarray, obstacles: List[Any], path: np.ndarray,
                   start: np.ndarray, goal: np.ndarray) -> None:
    """Visualize the path in a 3D environment."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set bounds
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])

    # Plot obstacles
    for obs in obstacles:
        if isinstance(obs, np.ndarray):  # Box obstacle
            ox_min, oy_min, oz_min, ox_max, oy_max, oz_max = obs
            vertices = [
                # Bottom face
                [(ox_min, oy_min, oz_min), (ox_max, oy_min, oz_min),
                 (ox_max, oy_max, oz_min), (ox_min, oy_max, oz_min)],
                # Top face
                [(ox_min, oy_min, oz_max), (ox_max, oy_min, oz_max),
                 (ox_max, oy_max, oz_max), (ox_min, oy_max, oz_max)],
                # Side faces
                [(ox_min, oy_min, oz_min), (ox_max, oy_min, oz_min),
                 (ox_max, oy_min, oz_max), (ox_min, oy_min, oz_max)],
                [(ox_max, oy_min, oz_min), (ox_max, oy_max, oz_min),
                 (ox_max, oy_max, oz_max), (ox_max, oy_min, oz_max)],
                [(ox_max, oy_max, oz_min), (ox_min, oy_max, oz_min),
                 (ox_min, oy_max, oz_max), (ox_max, oy_max, oz_max)],
                [(ox_min, oy_max, oz_min), (ox_min, oy_min, oz_min),
                 (ox_min, oy_min, oz_max), (ox_min, oy_max, oz_max)]
            ]
            ax.add_collection3d(Poly3DCollection(vertices,
                                                 facecolors='gray',
                                                 alpha=0.3,
                                                 linewidth=1,
                                                 edgecolor='black'))

    # Plot path
    if path is not None and len(path) > 0:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'b-',
                linewidth=2, label='Path')

    # Plot start and goal points
    ax.scatter(start[0], start[1], start[2],
               c='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], goal[2],
               c='red', s=100, label='Goal')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('UAV Path Planning in 3D Environment')
    ax.legend()
    plt.show()


def visualize_multiple_paths(bounds: np.ndarray, obstacles: List[Any],
                             paths_dict: dict, start: np.ndarray, goal: np.ndarray) -> None:
    """Visualize multiple paths in the 3D environment with different colors."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set bounds
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])

    # Plot obstacles
    for obs in obstacles:
        if isinstance(obs, np.ndarray):  # Box obstacle
            ox_min, oy_min, oz_min, ox_max, oy_max, oz_max = obs
            vertices = [
                # Bottom face
                [(ox_min, oy_min, oz_min), (ox_max, oy_min, oz_min),
                 (ox_max, oy_max, oz_min), (ox_min, oy_max, oz_min)],
                # Top face
                [(ox_min, oy_min, oz_max), (ox_max, oy_min, oz_max),
                 (ox_max, oy_max, oz_max), (ox_min, oy_max, oz_max)],
                # Side faces
                [(ox_min, oy_min, oz_min), (ox_max, oy_min, oz_min),
                 (ox_max, oy_min, oz_max), (ox_min, oy_min, oz_max)],
                [(ox_max, oy_min, oz_min), (ox_max, oy_max, oz_min),
                 (ox_max, oy_max, oz_max), (ox_max, oy_min, oz_max)],
                [(ox_max, oy_max, oz_min), (ox_min, oy_max, oz_min),
                 (ox_min, oy_max, oz_max), (ox_max, oy_max, oz_max)],
                [(ox_min, oy_max, oz_min), (ox_min, oy_min, oz_min),
                 (ox_min, oy_min, oz_max), (ox_min, oy_max, oz_max)]
            ]
            ax.add_collection3d(Poly3DCollection(vertices,
                                                 facecolors='gray',
                                                 alpha=0.3,
                                                 linewidth=1,
                                                 edgecolor='black'))

    # Colors for different paths
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown', 'pink']

    # Plot paths with different colors
    for i, (name, path) in enumerate(paths_dict.items()):
        if path is not None and len(path) > 0:
            path = np.array(path)
            color = colors[i % len(colors)]
            ax.plot(path[:, 0], path[:, 1], path[:, 2],
                    c=color, linewidth=2, label=f'Path ({name})')

    # Plot start and goal points
    ax.scatter(start[0], start[1], start[2],
               c='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], goal[2],
               c='red', s=100, label='Goal')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('UAV Path Planning in 3D Environment - Algorithm Comparison')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def load_algorithm(file_path: str) -> Any:
    """Dynamically load the algorithm module and find the planner class."""
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location("algorithm", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["algorithm"] = module
        spec.loader.exec_module(module)

        # Find the first class that has plan_path method
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and hasattr(obj, 'plan_path'):
                return obj

        raise AttributeError("No suitable path planning class found in the module")
    except Exception as e:
        print(f"Error loading algorithm: {str(e)}")
        return None


def get_algorithm_files() -> List[str]:
    """Get multiple algorithm files from user input."""
    files = []
    while True:
        file_input = input("Enter algorithm file path (or press Enter to finish): ").strip()
        if not file_input:
            break

        # Check if file exists in current directory
        if not os.path.exists(file_input):
            local_file = os.path.join(os.path.dirname(__file__), file_input)
            if os.path.exists(local_file):
                file_input = local_file
            else:
                print(f"Warning: File {file_input} not found, skipping...")
                continue

        files.append(file_input)

    return files


def calculate_path_efficiency(path: np.ndarray) -> float:
    """Calculate path efficiency as percentage (direct distance / path length)"""
    if path is None or len(path) < 2:
        return 0.0

    # Calculate direct distance from start to goal
    direct_distance = np.linalg.norm(path[-1] - path[0])

    # Calculate actual path length
    path_length = calculate_path_length(path)

    if path_length == 0:
        return 0.0

    # Calculate efficiency as percentage
    efficiency = (direct_distance / path_length) * 100

    return min(efficiency, 100.0)  # Cap at 100%


if __name__ == "__main__":
    # Get multiple algorithm files
    algorithm_files = get_algorithm_files()
    if not algorithm_files:
        print("No algorithm files provided")
        exit(1)

    # Environment setup
    bounds = np.array([[0, 50], [0, 50], [0, 10]])
    obstacles = np.array([
        [20, 20, 0, 25, 25, 10],  # Central obstacle
        [30, 15, 0, 35, 20, 10],  # Original obstacle 2
        [10, 30, 0, 15, 35, 10],  # Original obstacle 3
        [5, 15, 0, 8, 40, 8],  # Long wall
        [25, 35, 2, 30, 45, 7],  # Vertical barrier
        [35, 25, 0, 45, 28, 6],  # Horizontal barrier
        [15, 5, 0, 35, 8, 5],  # Lower wall
        [40, 35, 0, 43, 45, 9],  # Upper right corner
        [18, 18, 6, 22, 22, 10],  # Floating obstacle
        [28, 28, 4, 32, 32, 8],  # Another floating obstacle
        [12, 25, 3, 16, 28, 7],  # Mid-left obstacle
        [38, 12, 2, 42, 16, 6]  # Mid-right obstacle
    ])
    start = np.array([2., 2., 2.])
    goal = np.array([47., 33., 2.])

    # Store results for each algorithm
    results = {}
    paths = {}

    # Test each algorithm
    for algo_file in algorithm_files:
        algo_name = os.path.splitext(os.path.basename(algo_file))[0]
        print(f"\nTesting algorithm: {algo_name}")

        # Load the algorithm
        PlannerClass = load_algorithm(algo_file)
        if PlannerClass is None:
            print(f"Failed to load algorithm: {algo_name}")
            continue

        # Record initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)

        # Start timing
        start_time = time.time()

        try:
            # Create planner instance
            planner = PlannerClass(
                start=start,
                goal=goal,
                bounds=bounds,
                obstacles=obstacles
            )

            # Run path planning
            path = planner.plan_path()
            success = path is not None and len(path) > 0

            # End timing
            end_time = time.time()
            computational_time = end_time - start_time

            # Calculate memory usage
            final_memory = process.memory_info().rss / (1024 * 1024)
            memory_usage = final_memory - initial_memory

            # Calculate CPU usage
            cpu_usage = calculate_cpu_usage(process, start_time, end_time)

            # Calculate path efficiency
            path_efficiency = calculate_path_efficiency(path) if success else None

            # Store results
            results[algo_name] = {
                'success': success,
                'path_length': calculate_path_length(path) if success else None,
                'path_clearance': calculate_path_clearance(path, obstacles) if success else None,
                'path_smoothness': calculate_path_smoothness(path) if success else None,
                'path_efficiency': path_efficiency,
                'cpu_usage': cpu_usage if success else None,
                'computational_time': computational_time,
                'memory_usage': memory_usage,
                'iterations': len(planner.nodes) if hasattr(planner, 'nodes') else None
            }

            if success:
                paths[algo_name] = path

        except Exception as e:
            print(f"Error testing {algo_name}: {str(e)}")
            results[algo_name] = {
                'success': False,
                'error': str(e)
            }

    # Print comparison results
    print("\n=== Algorithm Comparison Results ===")
    for algo_name, result in results.items():
        print(f"\n{algo_name}:")
        if result.get('success', False):
            print(f"Path Length (m): {result['path_length']:.2f}")
            print(f"Path Clearance (m): {result['path_clearance']:.2f}")
            print(f"Path Smoothness (%): {result['path_smoothness']:.2f}")
            print(f"Path Efficiency (%): {result['path_efficiency']:.2f}")
            print(f"CPU Usage (%): {result['cpu_usage']:.2f}")
            print(f"Computational Time (s): {result['computational_time']:.3f}")
            print(f"Memory Usage (MB): {result['memory_usage']:.2f}")
            if result['iterations'] is not None:
                print(f"Iterations: {result['iterations']}")
            print("Success Rate: Yes")
        else:
            print("Success Rate: No")
            if 'error' in result:
                print(f"Error: {result['error']}")

    # Visualize all successful paths
    if paths:
        visualize_multiple_paths(bounds, obstacles, paths, start, goal)
    else:
        print("\nNo successful paths to visualize.")

        # Must have a plan_path() method that returns the path
# Must accept start, goal, bounds, and obstacles in its constructor in algorithm