from traffic_env import SumoEnvironment
from q_learning_agent import QLearningAgent
from deep_q_learning_agent import DQNAgent
from baseline_agents.fixed_timer_agent import FixedTimerAgent

def train_sumo_rl(sumo_cfg_path, tl_id="J2", n_episodes=100, max_steps=5000, use_gui=False, agent_type="q_learning"):
    """
    Train an RL agent to control a traffic light in SUMO.
    
    Args:
        sumo_cfg_path: Path to SUMO configuration file
        tl_id: ID of the traffic light to control
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        use_gui: Whether to use the SUMO GUI
        agent_type: Type of agent to use (q_learning, dqn, or fixed_timer)
        
    Returns:
        The trained agent
    """
    # Create SUMO environment
    env = SumoEnvironment(sumo_cfg_path, tl_id, use_gui)
    
    # Get the state size
    state_size = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else len(env.reset())
    
    # Get the action size (number of possible actions)
    action_size = env.action_space.n if hasattr(env.action_space, 'n') else 4  # Default to 4 if not specified
    
    # Initialize appropriate agent
    if agent_type == "q_learning":
        agent = QLearningAgent(state_size, action_size, epsilon=0, epsilon_min=0)
    elif agent_type == "dqn":
        agent = DQNAgent(state_size=state_size, action_size=action_size, epsilon=0, epsilon_min=0)
    elif agent_type == "fixed_timer":
        print('using fixed timer')
        agent = FixedTimerAgent(action_size=action_size)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Training loop
    best_reward = float('-inf')
    best_episode = 0
    
    overall_avg_wait_time = 0
    overall_avg_trip_loss = 0
    avg_total_reward = 0

    for episode in range(n_episodes):
        state = env.reset()
        
        total_reward = 0
        total_wait_time = 0
        total_trip_loss = 0
        num_steps = 0

        for step in range(max_steps):
            # Agent selects an action
            action = agent.get_action(state)

            # Perform step
            next_state, reward, done, info = env.step(action)

            # Update Q-values
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            total_wait_time += info.get("average_wait_time", 0)
            total_trip_loss += info.get("average_trip_loss", 0)
            num_steps += 1

            if done:
                break
        
        # Track best episode
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode
        
        # calculate and report other metrics 
        avg_wait_time = total_wait_time / num_steps if num_steps > 0 else 0
        avg_trip_loss = total_trip_loss / num_steps if num_steps > 0 else 0

        overall_avg_wait_time += avg_wait_time
        overall_avg_trip_loss += avg_trip_loss
        avg_total_reward += total_reward

        if agent_type == "dqn" and episode % 10 == 0:
            agent.sync_target_network()
            agent.save_model()

        # Print progress
        if episode % 1 == 0:
            print(f"\n Episode: {episode}, Total Reward: {total_reward}, Best: {best_reward} (ep {best_episode})")
            print(f"  - Average Wait Time: {avg_wait_time:.2f}")
            print(f"  - Average Trip Loss: {avg_trip_loss:.2f}")
    
    print(f"Training completed with {agent_type}. Best episode: {best_episode} with reward: {best_reward}")
    print(f"Overall average wait time: {overall_avg_wait_time / n_episodes:.2f}")
    print(f"Overall average trip loss: {overall_avg_trip_loss / n_episodes:.2f}")
    print(f"Overall average total reward: {avg_total_reward / n_episodes:.2f}")
    return agent

if __name__ == "__main__":
    # Example usage:
    sumo_cfg_path = "data/sumoconfig.sumocfg"
    tl_id = "J2"  # The ID of the traffic light you want to control
    trained_agent = train_sumo_rl(sumo_cfg_path, tl_id, n_episodes=10, max_steps=5000, use_gui=True, agent_type="q_learning")
    #trained_agent = train_sumo_rl(sumo_cfg_path, tl_id, n_episodes=100, max_steps=5000, use_gui=True, agent_type="dqn")
    #trained_agent = train_sumo_rl(sumo_cfg_path, tl_id, n_episodes=100, max_steps=5000, use_gui=True, agent_type="fixed_timer")
