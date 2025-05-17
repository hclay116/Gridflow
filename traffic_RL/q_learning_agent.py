import numpy as np  # type: ignore
import os
import json
from collections import defaultdict

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, q_table_file="q_table.json"):
        """
        Initialize the Q-Learning Agent.
        :param state_size: Dimension of the state vector.
        :param action_size: Number of possible actions.
        :param alpha: Learning rate.
        :param gamma: Discount factor.
        :param epsilon: Initial exploration rate.
        :param epsilon_min: Minimum exploration rate.
        :param epsilon_decay: Decay factor for epsilon per episode.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table_file = q_table_file

        # We'll store Q-values in a dictionary since states are multi-dimensional.
        # Key = (discretized state tuple), Value = Q-value array for each action
        self.q_table = defaultdict(lambda: np.zeros(action_size))
       
        self.load_q_table()

    def discretize_state(self, state):
        """
        Convert continuous state to a discrete tuple (e.g., bin the queue lengths).
        If your state is already discrete, you can skip or modify this step.
        """
        # Example: Round queue lengths to nearest integer
        return tuple(map(int, state))

    def get_action(self, state):
        """
        Epsilon-greedy policy to choose action.
        """
        discrete_state = self.discretize_state(state)
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, self.action_size)
            return action
        else:
            action = np.argmax(self.q_table[discrete_state])
            return action

    def update(self, state, action, reward, next_state, done):
        """
        Update Q-table based on experience.
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        current_q = self.q_table[discrete_state][action]
        next_max_q = np.max(self.q_table[discrete_next_state])

        # Q-learning update
        new_q = current_q + self.alpha * (reward + (0 if done else self.gamma * next_max_q) - current_q)
        self.q_table[discrete_state][action] = new_q

        # Decay epsilon after each update or after each episode
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.save_q_table()
        
    def save_q_table(self):
        """
        Save the Q-table to a file in JSON format.
        """
        q_table_serializable = {str(k): list(v) for k, v in self.q_table.items()}  # Convert keys to strings and arrays to lists
        with open(self.q_table_file, "w") as f:
            json.dump(q_table_serializable, f)

    def load_q_table(self):
        """
        Load the Q-table from a file if it exists.
        """
        if os.path.exists(self.q_table_file):
            with open(self.q_table_file, "r") as f:
                q_table_serializable = json.load(f)
                self.q_table = defaultdict(lambda: np.zeros(self.action_size), 
                                          {eval(k): np.array(v) for k, v in q_table_serializable.items()})
