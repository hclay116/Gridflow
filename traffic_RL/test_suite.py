import csv
from traffic_env import SumoEnvironment
from q_learning_agent import QLearningAgent
from deep_q_learning_agent import DQNAgent
from baseline_agents.fixed_timer_agent import FixedTimerAgent


def run_test_suite(sumo_cfg_path, tl_ids, n_episodes=50, max_steps=200, use_gui=False):
    results = []

    for tl_id in tl_ids:
        # Create environment
        env = SumoEnvironment(sumo_cfg_path, tl_id, use_gui=use_gui)

        # Create agents
        agents = {
            'q_learning': QLearningAgent(state_size=env.state_size, action_size=len(env.action_space)),
            'dqn': DQNAgent(state_size=env.state_size, action_size=len(env.action_space)),
            'fixed_timer': FixedTimerAgent(action_size=len(env.action_space))
        }

        for agent_name, agent in agents.items():
            if agent_name == 'dqn':
                agent.load_model()  

            overall_avg_wait_time = 0
            overall_avg_trip_loss = 0
            avg_total_reward = 0

            # Run episodes
            for episode in range(n_episodes):
                state = env.reset()
                total_reward = 0
                total_wait_time = 0
                total_trip_loss = 0

                for step in range(max_steps):
                    action = agent.get_action(state)
                    next_state, reward, done, info = env.step(action)

                    total_reward += reward
                    total_wait_time += info.get('wait_time', 0)
                    total_trip_loss += info.get('trip_loss', 0)

                    state = next_state

                    if done:
                        break

                avg_total_reward += total_reward / max_steps
                overall_avg_wait_time += total_wait_time / max_steps
                overall_avg_trip_loss += total_trip_loss / max_steps

            results.append({
                'tl_id': tl_id,
                'agent': agent_name,
                'avg_reward': avg_total_reward / n_episodes,
                'avg_wait_time': overall_avg_wait_time / n_episodes,
                'avg_trip_loss': overall_avg_trip_loss / n_episodes
            })

    # Write results to CSV
    with open('test_results.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['tl_id', 'agent', 'avg_reward', 'avg_wait_time', 'avg_trip_loss'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)


if __name__ == "__main__":
    sumo_cfg_path = "data/sumoconfig.sumocfg"
    tl_ids = ["0", "1", "2"] 
    run_test_suite(sumo_cfg_path, tl_ids)
