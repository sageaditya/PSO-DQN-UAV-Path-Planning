import numpy as np
import time
from typing import Optional, Type
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import importlib
import sys

class TestPlanner:
    def __init__(self, planner_class: Type = None):
        # Define environment bounds
        self.bounds = np.array([
            [-15, 15],  # x bounds
            [-15, 15],  # y bounds
            [0, 15]     # z bounds
        ])
        
        # Define static box obstacles (taller buildings 8-10 units high)
        self.static_obstacles = [
            np.array([-8, -8, 0, -6, -6, 9]),     # Building 1 (9 units tall)
            np.array([4, 4, 0, 6, 6, 10]),        # Building 2 (10 units tall)
            np.array([-3, 3, 0, -1, 5, 8]),       # Building 3 (8 units tall)
            np.array([0, -5, 0, 2, -3, 9.5]),     # Building 4 (9.5 units tall)
            np.array([-4, -2, 0, -2, 0, 8.5]),    # Building 5 (8.5 units tall)
        ]
        
        # Define dynamic spherical obstacles with velocity
        self.dynamic_obstacles = [
            {
                'position': np.array([2.0, 2.0, 4.0]),
                'radius': 1.8,
                'velocity': np.array([0.4, 0.0, 0.2])
            },
            {
                'position': np.array([-4.0, 4.0, 6.0]),
                'radius': 1.8,
                'velocity': np.array([0.3, -0.3, 0.0])
            },
            {
                'position': np.array([3.0, -3.0, 5.0]),
                'radius': 1.8,
                'velocity': np.array([-0.2, -0.2, 0.3])
            },
            {
                'position': np.array([-2.0, 0.0, 7.0]),
                'radius': 1.8,
                'velocity': np.array([0.0, 0.4, -0.2])
            }
        ]
        
        # Store original obstacle positions
        self.original_positions = [obs['position'].copy() for obs in self.dynamic_obstacles]
        
        # Start and goal positions (more challenging positions)
        self.start = np.array([-12, -12, 1])
        self.goal = np.array([12, 12, 12])
        
        # Store metrics
        self.planning_time = 0
        self.path_length = 0
        self.path_smoothness = 0
        self.success = False
        
        # Store planner class and path
        self.planner_class = planner_class
        self.path = None
        self.current_path = None
        
        # Animation parameters
        self.fig = None
        self.ax = None
        self.anim = None
        
        # Planner specific parameters
        self.is_rrt_dqn = planner_class.__name__ == 'EnhancedRRTStarDQN' if planner_class else False
        
        # Additional metrics tracking
        self.replan_times = []
        self.collision_count = 0
        self.total_replans = 0
        self.initial_path = None
        self.optimal_path_length = None

    def get_current_obstacles(self):
        """Get current obstacles in the format expected by the planner"""
        if self.is_rrt_dqn:
            # Convert dynamic obstacles to the format expected by RRT*DQN
            dynamic_obs = []
            for obs in self.dynamic_obstacles:
                pos = obs['position']
                r = obs['radius']
                dynamic_obs.append((pos, r))
            return self.static_obstacles + dynamic_obs
        else:
            # Return obstacles in original format for other planners
            return self.static_obstacles + self.dynamic_obstacles

    def reset_obstacles(self):
        """Reset dynamic obstacles to their original positions"""
        for obs, orig_pos in zip(self.dynamic_obstacles, self.original_positions):
            obs['position'] = orig_pos.copy()

    def update_dynamic_obstacles(self):
        """Update positions of dynamic obstacles"""
        for obs in self.dynamic_obstacles:
            new_pos = obs['position'] + obs['velocity']
            
            # Bounce off bounds
            for i in range(3):
                if new_pos[i] < self.bounds[i][0] or new_pos[i] > self.bounds[i][1]:
                    obs['velocity'][i] *= -1
                    new_pos = obs['position'] + obs['velocity']
            
            obs['position'] = new_pos

    def check_path_collision(self) -> bool:
        """Check if current path collides with dynamic obstacles"""
        if self.current_path is None:
            return False
            
        for i in range(len(self.current_path) - 1):
            point = self.current_path[i]
            next_point = self.current_path[i + 1]
            
            for obs in self.dynamic_obstacles:
                center = obs['position']
                radius = obs['radius'] + 1.0  # Collision margin
                
                # Check line segment intersection with sphere
                segment = next_point - point
                segment_length = np.linalg.norm(segment)
                
                if segment_length == 0:
                    continue
                    
                segment_direction = segment / segment_length
                point_to_center = center - point
                
                projection_length = np.dot(point_to_center, segment_direction)
                
                if 0 <= projection_length <= segment_length:
                    closest_point = point + segment_direction * projection_length
                    if np.linalg.norm(closest_point - center) <= radius:
                        return True
        
        return False

    def animate(self, frame):
        """Animation update function"""
        self.ax.cla()
        
        # Update dynamic obstacles
        self.update_dynamic_obstacles()
        
        # Check for collisions and replan if needed
        if frame % 10 == 0 and self.check_path_collision():
            print(f"\nCollision detected! Replanning... (Collision #{self.collision_count + 1})")
            self.collision_count += 1
            self.replan_path()
        
        # Plot static obstacles
        for obs in self.static_obstacles:
            x, y, z = obs[:3]
            dx, dy, dz = obs[3:] - obs[:3]
            self.ax.bar3d(x, y, z, dx, dy, dz, color='gray', alpha=0.5)
        
        # Plot dynamic obstacles
        for obs in self.dynamic_obstacles:
            center = obs['position']
            radius = obs['radius']
            
            # Create sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax.plot_surface(x, y, z, color='red', alpha=0.4)
        
        # Plot current path
        if self.current_path is not None:
            self.ax.plot(self.current_path[:, 0], self.current_path[:, 1], self.current_path[:, 2], 
                        'b-', linewidth=2, label='Path')
        
        # Plot start and goal
        self.ax.scatter(self.start[0], self.start[1], self.start[2], 
                       c='green', s=100, label='Start')
        self.ax.scatter(self.goal[0], self.goal[1], self.goal[2], 
                       c='red', s=100, label='Goal')
        
        # Set plot parameters
        self.ax.set_xlim(self.bounds[0])
        self.ax.set_ylim(self.bounds[1])
        self.ax.set_zlim(self.bounds[2])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()

    def replan_path(self):
        """Replan path from current position to goal"""
        if self.current_path is None or len(self.current_path) < 2:
            return
            
        current_pos = self.current_path[1]
        
        # Start timing for replanning
        replan_start_time = time.time()
        
        planner = self.planner_class(
            start=current_pos,
            goal=self.goal,
            bounds=self.bounds,
            obstacles=self.get_current_obstacles()
        )
        
        new_path = planner.plan_path()
        
        # Record replanning time
        replan_time = time.time() - replan_start_time
        self.replan_times.append(replan_time)
        self.total_replans += 1
        
        if new_path is not None:
            self.current_path = np.vstack([current_pos, new_path])
            # Calculate metrics for the new path
            self.calculate_metrics(self.current_path, is_replan=True)

    def visualize_path(self, path: np.ndarray):
        """Visualize the path with animated dynamic obstacles"""
        self.current_path = path
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Reset obstacles to original positions
        self.reset_obstacles()
        
        # Create animation
        self.anim = animation.FuncAnimation(self.fig, self.animate, 
                                          frames=200, interval=50, 
                                          blit=False)
        plt.show()

    def calculate_metrics(self, path: np.ndarray, is_replan: bool = False) -> dict:
        """Calculate path metrics"""
        if path is None or len(path) < 2:
            return {
                'success': False,
                'planning_time': self.planning_time,
                'path_length': 0,
                'path_smoothness': 0,
                'path_deviation': 0,
                'goal_reached': False
            }

        # Calculate path length
        path_length = sum(np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path)))
        
        # Calculate smoothness as a percentage (0-100%)
        angles = []
        for i in range(1, len(path)-1):
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            # Add check for zero vectors
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 > 1e-10 and norm_v2 > 1e-10:  # Check if vectors are non-zero
                cos_angle = np.dot(v1, v2)/(norm_v1 * norm_v2)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(angle)
        
        # Convert smoothness to percentage (0° = 100% smooth, 180° = 0% smooth)
        smoothness = 100 * (1 - np.mean(angles) / np.pi) if angles else 100
        
        # Calculate path deviation from optimal (always positive)
        if self.initial_path is not None:
            optimal_length = sum(np.linalg.norm(self.initial_path[i] - self.initial_path[i-1]) 
                               for i in range(1, len(self.initial_path)))
            path_deviation = abs((path_length - optimal_length) / optimal_length) * 100
        else:
            path_deviation = 0
        
        goal_reached = np.linalg.norm(path[-1] - self.goal) < 1.0

        # Print metrics if this is a replan
        if is_replan:
            print("\nReplan Metrics:")
            print(f"Path Length: {path_length:.2f} units")
            print(f"Replan Time: {self.replan_times[-1]:.3f} seconds")
            print(f"Path Deviation: {path_deviation:.1f}%")

        return {
            'success': True,
            'planning_time': self.planning_time,
            'path_length': path_length,
            'path_smoothness': smoothness,
            'path_deviation': path_deviation,
            'goal_reached': goal_reached,
            'total_replans': self.total_replans
        }

    def plan_path(self) -> Optional[np.ndarray]:
        """Plan initial path"""
        if self.planner_class is None:
            print("No planner specified!")
            return None
        
        # Reset metrics
        self.replan_times = []
        self.total_replans = 0
        
        # Reset obstacles to original positions
        self.reset_obstacles()
        
        # Initialize planner with appropriate obstacle format
        planner = self.planner_class(
            start=self.start,
            goal=self.goal,
            bounds=self.bounds,
            obstacles=self.get_current_obstacles()
        )
        
        # Start timing
        start_time = time.time()
        
        # Plan initial path
        path = planner.plan_path()
        self.current_path = path
        self.initial_path = path.copy() if path is not None else None
        
        # End timing
        self.planning_time = time.time() - start_time
        
        # Calculate and display metrics
        metrics = self.calculate_metrics(path)
        
        print("\nInitial Path Metrics:")
        print(f"Success: {metrics['success']}")
        print(f"Planning Time: {metrics['planning_time']:.3f} seconds")
        print(f"Path Length: {metrics['path_length']:.2f} units")
        print(f"Path Smoothness: {metrics['path_smoothness']:.1f}%")
        print(f"Goal Reached: {metrics['goal_reached']}")
        
        # Visualize path with animation
        self.visualize_path(path)
        
        return path

def get_planner_class():
    """Get planner class from file path"""
    while True:
        try:
            print("\nEnter planner file path/name (without .py extension)")
            planner_path = input("File path: ").strip()
            
            # Handle path separators
            module_path = planner_path.replace('/', '.').replace('\\', '.')
            
            # Remove .py extension if provided
            if module_path.endswith('.py'):
                module_path = module_path[:-3]
            
            # Import the module
            try:
                planner_module = importlib.import_module(module_path)
                
                # Find the first class that has a plan_path method
                for name, obj in vars(planner_module).items():
                    if isinstance(obj, type) and hasattr(obj, 'plan_path'):
                        print(f"\nUsing planner: {name}")
                        return obj
                
                print(f"No valid planner class found in {planner_path}")
                print("Make sure the file contains a class with a plan_path method")
                
            except ImportError as e:
                print(f"Error importing planner: {e}")
                print("Make sure the file exists and is in the Python path")
            except Exception as e:
                print(f"Error: {e}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again")

def main():
    """Main function to run the planner"""
    try:
        # Get planner selection from user
        planner_class = get_planner_class()
        if planner_class:
            # Create and run test planner
            test_planner = TestPlanner(planner_class)
            path = test_planner.plan_path()
            
            if path is None:
                print("No path found!")
                
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error running planner: {e}")

if __name__ == "__main__":
    main()