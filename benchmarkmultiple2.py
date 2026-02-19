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

# Import all the calculation functions from ftest.py
def calculate_path_length(path: np.ndarray) -> float:
    if path is None or len(path) < 2:
        return 0.0
    return np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1))

def calculate_path_clearance(path: np.ndarray, obstacles: List[Any]) -> float:
    def distance_to_obstacle(point: np.ndarray, obstacle: Any) -> float:
        if isinstance(obstacle, np.ndarray):
            x, y, z = point
            ox_min, oy_min, oz_min, ox_max, oy_max, oz_max = obstacle
            dx = max(ox_min - x, 0, x - ox_max)
            dy = max(oy_min - y, 0, y - oy_max)
            dz = max(oz_min - z, 0, z - oz_max)
            return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        elif isinstance(obstacle, dict):
            return np.linalg.norm(point - obstacle['position']) - obstacle['radius']
        else:
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
    if path is None or len(path) < 3:
        return 0.0
    
    angles = []
    for i in range(1, len(path) - 1):
        v1 = path[i] - path[i-1]
        v2 = path[i+1] - path[i]
        v1 = v1 / (np.linalg.norm(v1) + 1e-6)
        v2 = v2 / (np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angles.append(angle)
    
    angles = np.array(angles)
    smoothness = 100 * (1 - np.mean(angles) / np.pi)
    return smoothness

def calculate_cpu_usage(process: psutil.Process, start_time: float, end_time: float) -> float:
    try:
        cpu_times = process.cpu_times()
        total_cpu_time = cpu_times.user + cpu_times.system
        elapsed_time = end_time - start_time
        cpu_count = psutil.cpu_count()
        
        if elapsed_time > 0:
            cpu_percent = (total_cpu_time / elapsed_time / cpu_count) * 100
            return min(cpu_percent, 100.0)
        return 0.0
    except Exception:
        return 0.0

def calculate_path_efficiency(path: np.ndarray) -> float:
    if path is None or len(path) < 2:
        return 0.0
    
    direct_distance = np.linalg.norm(path[-1] - path[0])
    path_length = calculate_path_length(path)
    
    if path_length == 0:
        return 0.0
    
    efficiency = (direct_distance / path_length) * 100
    return min(efficiency, 100.0)

def visualize_multiple_paths(bounds: np.ndarray, obstacles: List[Any], 
                           paths_dict: dict, start: np.ndarray, goal: np.ndarray) -> None:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    
    # Plot obstacles
    for obs in obstacles:
        ox_min, oy_min, oz_min, ox_max, oy_max, oz_max = obs
        vertices = [
            [(ox_min, oy_min, oz_min), (ox_max, oy_min, oz_min),
             (ox_max, oy_max, oz_min), (ox_min, oy_max, oz_min)],
            [(ox_min, oy_min, oz_max), (ox_max, oy_min, oz_max),
             (ox_max, oy_max, oz_max), (ox_min, oy_max, oz_max)],
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
                                           alpha=0.6,
                                           linewidth=1, 
                                           edgecolor='black'))
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown', 'pink']
    
    for i, (name, path) in enumerate(paths_dict.items()):
        if path is not None and len(path) > 0:
            path = np.array(path)
            color = colors[i % len(colors)]
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                   c=color, linewidth=2, label=f'Path ({name})')
    
    ax.scatter(start[0], start[1], start[2], 
              c='green', s=200, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], 
              c='red', s=200, label='Goal')
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('UAV Path Planning in Urban Environment')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def load_algorithm(file_path: str) -> Any:
    try:
        spec = importlib.util.spec_from_file_location("algorithm", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["algorithm"] = module
        spec.loader.exec_module(module)
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and hasattr(obj, 'plan_path'):
                return obj
        
        raise AttributeError("No suitable path planning class found in the module")
    except Exception as e:
        print(f"Error loading algorithm: {str(e)}")
        return None

def get_algorithm_files() -> List[str]:
    files = []
    while True:
        file_input = input("Enter algorithm file path (or press Enter to finish): ").strip()
        if not file_input:
            break
        
        if not os.path.exists(file_input):
            local_file = os.path.join(os.path.dirname(__file__), file_input)
            if os.path.exists(local_file):
                file_input = local_file
            else:
                print(f"Warning: File {file_input} not found, skipping...")
                continue
        
        files.append(file_input)
    
    return files

if __name__ == "__main__":
    algorithm_files = get_algorithm_files()
    if not algorithm_files:
        print("No algorithm files provided")
        exit(1)

    # Environment setup from dtest2.py
    bounds = np.array([[-15, 15], [-15, 15], [0, 15]])
    obstacles = np.array([
        [-8, -8, 0, -6, -6, 9],      # Building 1
        [4, 4, 0, 6, 6, 10],         # Building 2
        [-3, 3, 0, -1, 5, 8],        # Building 3
        [0, -5, 0, 2, -3, 9.5],      # Building 4
        [-4, -2, 0, -2, 0, 8.5],     # Building 5
        [8, 8, 0, 10, 10, 9.8],      # Building 6
        [-10, 5, 0, -8, 7, 8.7],     # Building 7
        [6, -8, 0, 8, -6, 9.2],      # Building 8
        [7, -3, 0, 9, -1, 8.8],      # Building 9
        [-7, -4, 0, -5, -2, 9.7],    # Building 10
        [1, 1, 0, 3, 3, 9.4],        # Building 11
        [-9, 0, 0, -7, 2, 8.9],      # Building 12
        [5, -7, 0, 7, -5, 9.1],      # Building 13
        [-2, -7, 0, 0, -5, 8.6],     # Building 14
        [3, 6, 0, 5, 8, 9.3]         # Building 15
    ])
    start = np.array([-12., -12., 1.])
    goal = np.array([12., 12., 5.])

    results = {}
    paths = {}

    for algo_file in algorithm_files:
        algo_name = os.path.splitext(os.path.basename(algo_file))[0]
        print(f"\nTesting algorithm: {algo_name}")
        
        PlannerClass = load_algorithm(algo_file)
        if PlannerClass is None:
            print(f"Failed to load algorithm: {algo_name}")
            continue

        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)
        start_time = time.time()

        try:
            planner = PlannerClass(
                start=start,
                goal=goal,
                bounds=bounds,
                obstacles=obstacles
            )
            
            path = planner.plan_path()
            success = path is not None and len(path) > 0
            
            end_time = time.time()
            computational_time = end_time - start_time
            
            final_memory = process.memory_info().rss / (1024 * 1024)
            memory_usage = final_memory - initial_memory
            
            cpu_usage = calculate_cpu_usage(process, start_time, end_time)
            
            results[algo_name] = {
                'success': success,
                'path_length': calculate_path_length(path) if success else None,
                'path_clearance': calculate_path_clearance(path, obstacles) if success else None,
                'path_smoothness': calculate_path_smoothness(path) if success else None,
                'path_efficiency': calculate_path_efficiency(path) if success else None,
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

    if paths:
        visualize_multiple_paths(bounds, obstacles, paths, start, goal)
    else:
        print("\nNo successful paths to visualize.")
