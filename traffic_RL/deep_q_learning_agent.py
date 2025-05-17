import random
import os
import torch                    # type: ignore    
import torch.nn as nn           # type: ignore
import torch.optim as optim     # type: ignore
from collections import deque


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, 80)
        self.fc2 = nn.Linear(80, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=1e-3, batch_size=64, memory_size=10000, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Q-network and target network
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()
    
    def update(self, state, action, reward, next_state, done):
        if len(self.memory) < self.batch_size:
            self.store(state, action, reward, next_state, done)
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q target values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute Q values for selected actions
        q_values = self.q_network(states).gather(1, actions).squeeze()
        loss = self.criterion(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.store(state, action, reward, next_state, done)
        
    def sync_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, filename="dqn_model.pth"):
        """Save the Q-network weights to a file."""
        torch.save(self.q_network.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename="dqn_model.pth"):
        """Load the Q-network weights from a file if it exists."""
        if os.path.exists(filename):
            self.q_network.load_state_dict(torch.load(filename))
            self.target_network.load_state_dict(self.q_network.state_dict())  # Sync target network
            print(f"Model loaded from {filename}")
        else:
            print("No saved model found, starting fresh.")