import time

class FixedTimerAgent:
    def __init__(self, action_size, cycle_time=90):
        """
        Initialize a fixed-time traffic agent.
        :param action_size: Number of possible actions (e.g., traffic light phases).
        :param cycle_time: Time in seconds before switching to the next action.
        """
        self.action_size = action_size
        self.cycle_time = cycle_time  # Duration before switching lights
        self.last_switch_time = time.time()
        self.current_action = 0

    def get_action(self, state):
        """
        Return the action based on a fixed time schedule.
        :param state: (Ignored) The current state of traffic.
        :return: The current action (traffic light phase).
        """
        current_time = time.time()
        if current_time - self.last_switch_time >= self.cycle_time:
            self.last_switch_time = current_time
            #print('time diff:', current_time - )
            print('last time:', self.last_switch_time)
            return 1

        return 0

    def update(self, state, action, reward, next_state, done):
        pass # No learning or updates needed