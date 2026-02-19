import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PrioritizedReplayBuffer:
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling weight
        self.beta_increment = 0.001
        self.epsilon = 1e-6  # Small constant to avoid zero priorities

    def add(self, experience, error=None):
        priority = max(self.priorities) if self.priorities else 1.0
        if error is not None:
            priority = (abs(error) + self.epsilon) ** self.alpha
        self.memory.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size):
        total_priority = sum(self.priorities)
        probabilities = [p/total_priority for p in self.priorities]
        
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = [(len(self.memory) * p) ** (-self.beta) for p in probabilities]
        max_weight = max(weights)
        weights = [w/max_weight for w in weights]
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            if idx < len(self.priorities):
                self.priorities[idx] = (abs(error) + self.epsilon) ** self.alpha

class ChimpDQNPathPlanner:
    def __init__(self, start, goal, bounds, obstacles):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.bounds = np.array(bounds)
        self.obstacles = obstacles
        
        # DQN parameters
        self.state_size = 9  # current_pos(3) + goal(3) + nearest_obstacle_vector(3)
        self.action_size = 27  # 3x3x3 possible movements
        self.memory = PrioritizedReplayBuffer(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize DQN
        self.dqn = DQN(self.state_size, self.action_size).to(self.device)
        self.target_dqn = DQN(self.state_size, self.action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        
        # Chimp parameters
        self.population_size = 50
        self.max_iterations = 100
        self.nodes = []  # Store explored nodes
        
        # Enhanced DQN parameters
        self.batch_size = 32
        self.update_target_freq = 10
        self.min_replay_size = 1000
        self.path_history = []  # Store successful paths
        self.training_steps = 0
        
        # Boltzmann exploration parameters
        self.temperature = 1.0
        self.min_temperature = 0.1
        self.temperature_decay = 0.995
        
        # Enhanced reward parameters
        self.distance_weight = 1.0
        self.clearance_weight = 2.0
        self.smoothness_weight = 1.5
        self.goal_weight = 3.0
        self.prev_actions = deque(maxlen=3)  # Store recent actions for smoothness
        self.prev_min_clearance = float('inf')  # Add this line
        
    def get_state(self, position):
        # Get nearest obstacle vector
        nearest_obs_vector = np.zeros(3)
        min_dist = float('inf')
        
        for obs in self.obstacles:
            if isinstance(obs, np.ndarray):  # Box obstacle
                # Calculate center of obstacle
                center = np.array([
                    (obs[0] + obs[3]) / 2,
                    (obs[1] + obs[4]) / 2,
                    (obs[2] + obs[5]) / 2
                ])
                dist = np.linalg.norm(position - center)
                if dist < min_dist:
                    min_dist = dist
                    nearest_obs_vector = center - position
        
        state = np.concatenate([
            position,
            self.goal - position,
            nearest_obs_vector
        ])
        return state
    
    def is_collision(self, position):
        for obs in self.obstacles:
            if isinstance(obs, np.ndarray):  # Box obstacle
                if (position[0] >= obs[0] and position[0] <= obs[3] and
                    position[1] >= obs[1] and position[1] <= obs[4] and
                    position[2] >= obs[2] and position[2] <= obs[5]):
                    return True
        return False
    
    def get_valid_action(self, position):
        while True:
            action_idx = np.random.randint(0, self.action_size)
            x_change = (action_idx // 9) - 1
            y_change = ((action_idx % 9) // 3) - 1
            z_change = (action_idx % 3) - 1
            
            new_pos = position + np.array([x_change, y_change, z_change])
            
            # Check bounds
            if not all(new_pos >= [self.bounds[i][0] for i in range(3)]) or \
               not all(new_pos <= [self.bounds[i][1] for i in range(3)]):
                continue
                
            # Check collision
            if self.is_collision(new_pos):
                continue
                
            return action_idx, new_pos
    
    def chimp_update(self, chimps):
        # Sort chimps by fitness (distance to goal)
        chimps.sort(key=lambda x: np.linalg.norm(x - self.goal))
        
        # Update positions based on the best chimp
        best_chimp = chimps[0]
        updated_chimps = []
        
        for i in range(len(chimps)):
            if i == 0:  # Keep the best chimp
                updated_chimps.append(best_chimp)
                continue
                
            # Generate new position using ChOA
            r1 = np.random.random()
            r2 = np.random.random()
            
            new_pos = chimps[i] + r1 * (best_chimp - chimps[i]) + \
                     r2 * (np.random.random(3) * 2 - 1)
                     
            # Ensure new position is within bounds
            new_pos = np.clip(new_pos, 
                            [self.bounds[i][0] for i in range(3)],
                            [self.bounds[i][1] for i in range(3)])
            
            if not self.is_collision(new_pos):
                updated_chimps.append(new_pos)
            else:
                updated_chimps.append(chimps[i])
        
        return updated_chimps
    
    def get_enhanced_reward(self, current_pos, new_pos, action_idx):
        """Enhanced reward function with multiple components"""
        reward = 0
        
        # Distance component
        prev_dist = np.linalg.norm(current_pos - self.goal)
        new_dist = np.linalg.norm(new_pos - self.goal)
        distance_reward = (prev_dist - new_dist) * self.distance_weight
        reward += distance_reward * 1.5

        # Enhanced Clearance component with even more aggressive penalties
        min_clearance = float('inf')
        for obs in self.obstacles:
            if isinstance(obs, np.ndarray):
                # Calculate minimum distance to obstacle faces and corners
                clearances = []
                for i in range(3):
                    clearances.append(max(new_pos[i] - obs[i], 0))
                    clearances.append(max(obs[i+3] - new_pos[i], 0))
                
                # Add corner distances with increased safety margin
                corners = [
                    [obs[0], obs[1], obs[2]],
                    [obs[0], obs[1], obs[5]],
                    [obs[0], obs[4], obs[2]],
                    [obs[0], obs[4], obs[5]],
                    [obs[3], obs[1], obs[2]],
                    [obs[3], obs[1], obs[5]],
                    [obs[3], obs[4], obs[2]],
                    [obs[3], obs[4], obs[5]]
                ]
                for corner in corners:
                    corner_dist = np.linalg.norm(new_pos - corner)
                    clearances.append(corner_dist)
                    # Even more aggressive corner avoidance
                    if corner_dist < 4.0:  # Increased safety distance further
                        reward -= (4.0 - corner_dist) * 30  # Increased penalty

                min_obs_clearance = min(clearances)
                min_clearance = min(min_clearance, min_obs_clearance)
        
        # Even more aggressive clearance reward/penalty
        safety_threshold = 4.0  # Further increased safety margin
        if min_clearance < safety_threshold:
            # Steeper exponential penalty
            clearance_reward = -self.clearance_weight * np.exp(safety_threshold - min_clearance) * 3
        else:
            clearance_reward = self.clearance_weight * np.log(min_clearance + 1)
        reward += clearance_reward * 4  # Quadrupled the impact of clearance

        # Enhanced goal reaching reward
        if new_dist < 1.5:
            goal_reward = self.goal_weight * (30.0 / (new_dist + 0.1))
            reward += goal_reward

        # Even higher collision penalties
        if self.is_collision(new_pos):
            reward -= 300.0  # Tripled collision penalty
        if not all(new_pos >= [self.bounds[i][0] for i in range(3)]) or \
           not all(new_pos <= [self.bounds[i][1] for i in range(3)]):
            reward -= 300.0  # Tripled boundary penalty

        self.prev_actions.append(action_idx)
        return reward

    def boltzmann_action_selection(self, state, current_pos):
        """Action selection using Boltzmann exploration"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.dqn(state_tensor).squeeze()
            
        # Apply temperature scaling
        scaled_q_values = q_values / self.temperature
        
        # Calculate Boltzmann probabilities
        probabilities = F.softmax(scaled_q_values, dim=0).cpu().numpy()
        
        # Sample action based on probabilities
        action_idx = np.random.choice(self.action_size, p=probabilities)
        
        # Calculate new position
        x_change = (action_idx // 9) - 1
        y_change = ((action_idx % 9) // 3) - 1
        z_change = (action_idx % 3) - 1
        new_pos = current_pos + np.array([x_change, y_change, z_change])
        
        return action_idx, new_pos

    def train_dqn(self):
        """Enhanced DQN training with PER"""
        if len(self.memory.memory) < self.min_replay_size:
            return
            
        # Sample batch with priorities
        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q_values = self.dqn(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_dqn(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # Compute TD errors and weighted loss
        td_errors = target_q_values - current_q_values.squeeze()
        loss = (weights * F.mse_loss(current_q_values.squeeze(), target_q_values, reduction='none')).mean()
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors.abs().cpu().numpy())
        
        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.update_target_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

    def smooth_path(self, path, smoothing_factor=0.5, iterations=50):
        """Apply path smoothing using weighted averaging"""
        if path is None or len(path) <= 2:
            return path
            
        smoothed_path = np.copy(path)
        
        for _ in range(iterations):
            temp_path = np.copy(smoothed_path)
            
            # Skip first and last points to preserve start and goal
            for i in range(1, len(smoothed_path) - 1):
                # Get neighboring points
                prev_point = smoothed_path[i - 1]
                current_point = smoothed_path[i]
                next_point = smoothed_path[i + 1]
                
                # Calculate weighted average
                weighted_avg = current_point + smoothing_factor * (
                    (prev_point + next_point) / 2 - current_point
                )
                
                # Check if new position is collision-free
                if not self.is_collision(weighted_avg):
                    # Interpolate between points to ensure no obstacles in between
                    steps = 5
                    safe = True
                    for t in np.linspace(0, 1, steps):
                        interp_prev = current_point + t * (weighted_avg - current_point)
                        if self.is_collision(interp_prev):
                            safe = False
                            break
                    
                    if safe:
                        temp_path[i] = weighted_avg
            
            smoothed_path = temp_path
            
        return smoothed_path

    def plan_path(self):
        # Initialize population of chimps
        chimps = [self.start.copy()]
        for _ in range(self.population_size - 1):
            valid_pos = False
            while not valid_pos:
                pos = np.array([
                    np.random.uniform(self.bounds[0][0], self.bounds[0][1]),
                    np.random.uniform(self.bounds[1][0], self.bounds[1][1]),
                    np.random.uniform(self.bounds[2][0], self.bounds[2][1])
                ])
                if not self.is_collision(pos):
                    chimps.append(pos)
                    valid_pos = True
        
        best_path = None
        min_distance = float('inf')
        
        for iteration in range(self.max_iterations):
            # Update chimps using ChOA
            chimps = self.chimp_update(chimps)
            
            for i in range(len(chimps)):
                current_pos = chimps[i]
                state = self.get_state(current_pos)
                
                # Use Boltzmann exploration for action selection
                action_idx, new_pos = self.boltzmann_action_selection(state, current_pos)
                
                # Get enhanced reward
                reward = self.get_enhanced_reward(current_pos, new_pos, action_idx)
                next_state = self.get_state(new_pos)
                done = np.linalg.norm(new_pos - self.goal) < 1.0
                
                # Store experience with initial high priority
                self.memory.add((state, action_idx, reward, next_state, done))
                
                # Train DQN
                self.train_dqn()
                
                # Store the node for visualization
                self.nodes.append(current_pos)
                
                # Update position if valid
                if not self.is_collision(new_pos) and \
                   all(new_pos >= [self.bounds[i][0] for i in range(3)]) and \
                   all(new_pos <= [self.bounds[i][1] for i in range(3)]):
                    chimps[i] = new_pos
                
                # Check if we found a better path
                distance = np.linalg.norm(new_pos - self.goal)
                if distance < min_distance:
                    min_distance = distance
                    best_path = self.reconstruct_path(chimps[i])
                    if best_path is not None:
                        self.path_history.append(best_path)  # Store successful path
                
                # Check if we reached the goal
                if distance < 1.5:  # Increased goal detection radius
                    final_path = self.reconstruct_path(chimps[i])
                    if final_path is not None:
                        # Add goal point to final path
                        final_path = np.vstack((final_path, self.goal))
                        # Apply path smoothing
                        smoothed_path = self.smooth_path(final_path)
                        self.path_history.append(smoothed_path)
                        return smoothed_path
            
            # Decay temperature for Boltzmann exploration
            self.temperature = max(self.min_temperature, 
                                 self.temperature * self.temperature_decay)
        
        if best_path is not None:
            # Apply path smoothing to best path if goal not reached
            smoothed_best_path = self.smooth_path(best_path)
            return smoothed_best_path
        return None
    
    def reconstruct_path(self, final_pos):
        path = [final_pos]
        current_pos = final_pos
        
        while np.linalg.norm(current_pos - self.start) > 1.0:
            # Find the nearest node that's closer to start
            nearest_node = None
            min_dist = float('inf')
            
            for node in self.nodes:
                dist_to_current = np.linalg.norm(node - current_pos)
                dist_to_start = np.linalg.norm(node - self.start)
                
                if dist_to_current < min_dist and dist_to_start < np.linalg.norm(current_pos - self.start):
                    min_dist = dist_to_current
                    nearest_node = node
            
            if nearest_node is None:
                break
                
            path.append(nearest_node)
            current_pos = nearest_node
        
        path.append(self.start)
        return np.array(path[::-1])
