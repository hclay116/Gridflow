import os
import sys
import numpy as np  # type: ignore
import random

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv      # type: ignore
    load_dotenv()  # Load environment variables from .env file
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed, trying to set SUMO_HOME manually")

# If SUMO_HOME is still not in environment, try to set it based on common locations
if 'SUMO_HOME' not in os.environ:
    # Try common installation paths
    possible_paths = [
        "/usr/local/opt/sumo/share/sumo",  # macOS Homebrew
        "/usr/share/sumo",                 # Linux
        "C:\\Program Files (x86)\\Eclipse\\Sumo"  # Windows
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            os.environ['SUMO_HOME'] = path
            print(f"Set SUMO_HOME to {path}")
            break

# Now check if SUMO_HOME is properly set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print(f"Using SUMO tools from: {tools}")
else:
    sys.exit("Please declare environment variable 'SUMO_HOME' or install python-dotenv")

import traci
import math

from sumolib import checkBinary
from collections import defaultdict

def generate_routefile():
    """Generate a route file with vehicles that have realistic colors and different car shapes."""
    # print(f"Current working directory: {os.getcwd()}")
    route_file_path = "data/sumoconfig.rou.xml"
    # print(f"Writing route file to: {os.path.abspath(route_file_path)}")
    
    # Common car colors with realistic distribution
    car_colors = [
        "1.00,1.00,1.00",  # White (most common)
        "1.00,1.00,1.00",  # White (duplicate to increase probability)
        "0.00,0.00,0.00",  # Black
        "0.00,0.00,0.00",  # Black (duplicate)
        "0.75,0.75,0.75",  # Silver/Gray
        "0.75,0.75,0.75",  # Silver/Gray (duplicate)
        "0.80,0.00,0.00",  # Red
        "0.00,0.00,0.80",  # Blue
        "0.60,0.40,0.20",  # Brown/Beige
        "0.00,0.50,0.00",  # Green
        "1.00,0.80,0.00",  # Yellow
        "1.00,0.50,0.00",  # Orange
        "0.50,0.00,0.50",  # Purple (least common)
    ]
    
    # Car shapes as requested
    car_shapes = [
        "passenger",
        "passenger/sedan",
        "passenger/hatchback",
        "passenger/wagon",
        "passenger/van"
    ]
    
    # Use fixed generation pattern
    with open(route_file_path, "w") as routes:
        # Define all vehicle types with different shapes
        print("""<routes>
        <!-- Multiple vehicle types with different shapes -->
        <vType id="standard_car" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
        <vType id="sedan" accel="0.8" decel="4.5" sigma="0.5" length="4.8" minGap="2.5" maxSpeed="16.67" guiShape="passenger/sedan"/>
        <vType id="hatchback" accel="0.8" decel="4.5" sigma="0.5" length="4.3" minGap="2.3" maxSpeed="16.67" guiShape="passenger/hatchback"/>
        <vType id="wagon" accel="0.8" decel="4.5" sigma="0.5" length="5.1" minGap="2.5" maxSpeed="16.67" guiShape="passenger/wagon"/>
        <vType id="van" accel="0.7" decel="4.0" sigma="0.5" length="5.5" minGap="2.8" maxSpeed="15.67" guiShape="passenger/van"/>

        <!-- Straight routes -->
        <route id="west_to_east" edges="E0 -E1 -E2 E3" />
        <route id="east_to_west" edges="-E3 E2 E1 -E0" />
        <route id="north_to_south" edges="-E4 -E5" />
        <route id="south_to_north" edges="E5 E4" />
        
        <!-- Right turn routes -->
        <route id="west_to_north" edges="E0 -E1 E4" />
        <route id="north_to_east" edges="-E4 -E2 E3" />
        <route id="east_to_south" edges="-E3 E2 -E5" />
        <route id="south_to_west" edges="E5 E1 -E0" />
        
        <!-- Left turn routes -->
        <route id="west_to_south" edges="E0 -E1 -E5" />
        <route id="north_to_west" edges="-E4 E1 -E0" />
        <route id="east_to_north" edges="-E3 E2 E4" />
        <route id="south_to_east" edges="E5 -E2 E3" />""", file=routes)
        
        # Map shape names to vehicle type IDs
        shape_to_type = {
            "passenger": "standard_car",
            "passenger/sedan": "sedan",
            "passenger/hatchback": "hatchback",
            "passenger/wagon": "wagon",
            "passenger/van": "van"
        }
        
        # All possible route directions
        directions = [
            "west_to_east", "east_to_west", "north_to_south", "south_to_north",  # straight
            "west_to_north", "north_to_east", "east_to_south", "south_to_west",  # right turns
            "west_to_south", "north_to_west", "east_to_north", "south_to_east"   # left turns
        ]
        
        # Generate mixed traffic for the entire simulation time
        vehicle_count = 0
        
        # Higher frequency during rush hour (0-40s)
        for i in range(0, 40):
            # Generate 1-3 vehicles per second during peak time
            num_vehicles = random.randint(1, 3)
            for _ in range(num_vehicles):
                direction = random.choice(directions)
                color = random.choice(car_colors)
                car_shape = random.choice(car_shapes)
                veh_type = shape_to_type[car_shape]
                
                print(f'    <vehicle id="veh_{vehicle_count}" type="{veh_type}" route="{direction}" depart="{i}" color="{color}" />', file=routes)
                vehicle_count += 1
        
        # Lower frequency after rush hour (40-100s)
        for i in range(40, 100):
            # 50% chance of generating a vehicle each second during off-peak
            if random.random() < 0.5:
                direction = random.choice(directions)
                color = random.choice(car_colors)
                car_shape = random.choice(car_shapes)
                veh_type = shape_to_type[car_shape]
                
                print(f'    <vehicle id="veh_{vehicle_count}" type="{veh_type}" route="{direction}" depart="{i}" color="{color}" />', file=routes)
                vehicle_count += 1
            
        print("</routes>", file=routes)
        
    # print(f"Route file created with {vehicle_count} vehicles with varied colors and shapes")


class SumoEnvironment:
    def __init__(self, sumo_cfg_path, tl_id="J2", use_gui=False):
        """
        Initialize the SUMO environment.
        :param sumo_cfg_path: Path to .sumocfg file.
        :param tl_id: ID of the traffic light to be controlled. Default is "J2" based on the network file.
        :param use_gui: Whether to run with the SUMO GUI or not.
        """
        self.sumo_cfg_path = sumo_cfg_path
        self.tl_id = tl_id
        self.use_gui = use_gui
        
        # Define action space
        # Assuming 4 phases (0=NS green, 1=NSL green, 2=EW green, 3=EWL green)
        self.action_space = type('obj', (object,), {
            'n': 4,  # Number of possible actions
            'sample': lambda: random.randint(0, 3)  # Method to sample random action
        })
        
        # Define observation space
        # This will be determined by the number of lanes + 1 (for current phase)
        # Will be populated in reset()
        self.observation_space = None
        
        # state space = queue in either direction + traffic light phase
        self.state_size = 5         # 4 lanes + 1 traffic light phase
        self.last_phase = None      # Track previous phase to infer yellow states
        self.yellow_duration = 3    # Assume yellow light lasts 3 steps
        self.yellow_counter = 0
        
        # Reward parameters
        self.waiting_time_scale = 1.0

        # Car speed tracking
        self.car_speeds = defaultdict(lambda: 0.0)

    def start_sumo(self):
        """
        Start a SUMO simulation instance (TraCI) with or without GUI.
        """
        if self.use_gui:
            sumo_binary = checkBinary('sumo-gui')
            gui_options = ["--start", "--delay", "100"]  # Add delay and start automatically
        else:
            sumo_binary = checkBinary('sumo')
            gui_options = []
        
        # First make sure we generate the route file
        # print("Generating route file...")
        generate_routefile()
        
        # Now use the provided configuration path
        # print(f"Starting SUMO with config: {self.sumo_cfg_path}")
        
        # Add verbose output for debugging
        traci.start([sumo_binary, "-c", self.sumo_cfg_path,
                     "--tripinfo-output", "tripinfo.xml",
                     #"--verbose",  # Add verbose output
                     "--no-step-log", "false"] + gui_options)  # Show step information

    def reset(self):
        """
        Reset the environment (start a new simulation).
        Returns initial state.
        """
        # Try to close any existing connection, ignoring errors if no connection exists
        try:
            traci.close()
        except:
            print("No connection to SUMO")
            pass
        
        # Start a new simulation
        self.start_sumo()
        
        # Reset any state variables
        self.last_phase = None
        self.yellow_counter = 0
        self.car_speeds = defaultdict(lambda: 0.0)

        # Perform an initial step to get initial observation
        traci.simulationStep()

        state = self.get_state()
        return state

    def get_state(self):
        """
        Retrieve the current state from the simulation.
        In this example, we measure the queue length on each lane 
        connected to the traffic light.

        THIS IDENTITY NEEDS TO BE RETHOUGHT OUT BECAUSE WE NEED MUCH MORE INFORMATION GIVEN TO THE MODEL HERE
        """
        lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        queue_lengths = [traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in lanes]
        current_phase = traci.trafficlight.getPhase(self.tl_id)

        if self.last_phase is None:  # First call, initialize
            self.last_phase = current_phase

        if current_phase != self.last_phase:  # Yellow phase
            self.yellow_counter = self.yellow_duration
            self.last_phase = current_phase

        if self.yellow_counter > 0:
            self.yellow_counter -= 1
            paper_phase = 1 if self.last_phase == 0 else 3  
        else:
            paper_phase = 0 if current_phase == 0 else 2  

        # Construct state representation: queue lengths + inferred phase
        state = np.array(queue_lengths + [paper_phase], dtype=np.float32)

        # print(f"State Retrieved: {state}")  # Debugging statement
        return state

    def switch_traffic_lights(self):
        """
        Switch the traffic lights at the intersection.
        """
        current_traffic_light_id = self.tl_id
        cur = traci.trafficlight.getPhase(current_traffic_light_id) 
        traci.trafficlight.setPhase(current_traffic_light_id, (cur + 1) % 4)

    def set_tl_phase(self, action):
        """
        Set the traffic light to a phase corresponding to the chosen action.
        :param action: The index in self.action_space
        """
        if action == 1:
            self.switch_traffic_lights()
    
    def get_average_wait_time(self):
        """
        Calculate the average waiting time of all vehicles in the network.
        """
        vehicle_ids = traci.vehicle.getIDList()  # Get all vehicle IDs
        if not vehicle_ids:
            return 0.0  # No vehicles, no waiting time

        total_wait_time = sum(traci.vehicle.getWaitingTime(veh_id) for veh_id in vehicle_ids)
        return total_wait_time / len(vehicle_ids)

    def get_average_trip_loss(self):
        """
        Calculate the average trip loss (deviation from ideal travel time).
        """
        vehicle_ids = traci.vehicle.getIDList()  # Get all vehicle IDs
        if not vehicle_ids:
            return 0.0  # No vehicles, no trip loss

        total_trip_loss = sum(traci.vehicle.getTimeLoss(veh_id) for veh_id in vehicle_ids)
        return total_trip_loss / len(vehicle_ids)

    def step(self, action):
        """
        Take one step in the environment using the given action.
        Returns: next_state, reward, done, info
        """
        # Set the phase
        self.set_tl_phase(action)

        # Advance the simulation
        step_length = 1
        for _ in range(step_length):
            traci.simulationStep()
            # Print how many vehicles are in the simulation
            vehicle_count = len(traci.vehicle.getIDList())
            #print(f"Vehicles in simulation: {vehicle_count}")

        # Compute reward
        reward = self.compute_reward()

        # Get new state
        next_state = self.get_state()

        # End simulation if no vehicles remain
        done = len(traci.vehicle.getIDList()) == 0 or traci.simulation.getMinExpectedNumber() <= 0
        
        # Additional information for monitoring
        info = {
            "average_wait_time": self.get_average_wait_time(),
            "average_trip_loss": self.get_average_trip_loss(),
            "vehicles_remaining": len(traci.vehicle.getIDList())
        }
        return next_state, reward, done, info

    def compute_reward(self):
        """
        Compute a reward for the current traffic situation.
        The reward is based on the total waiting time of vehicles in the queue,
        weighted exponentially by the queue length.
        """
        weighted_queue_time = 0
        lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        
        for lane_id in lanes:
            # eponentially weight by wait time of first stopped car 
            # postively weight cars with positive speed
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            wait_time = 0
            speed_reward = 0
            num_vehicles = len(vehicles)
            for i in range(num_vehicles):
                vehicle = vehicles[i]  
                wait_time += traci.vehicle.getWaitingTime(vehicle) * (1 - i / num_vehicles)
                cur_speed = traci.vehicle.getSpeed(vehicle)
                if self.car_speeds[vehicle] <= cur_speed:
                    speed_reward += 100
                self.car_speeds[vehicle] = cur_speed
            
            wait_time_weight = 1.25 ** wait_time
            weighted_queue_time += wait_time

        reward = -weighted_queue_time + speed_reward
        # print(f"Computed Reward: {reward}")  # Debugging output
        return reward


    def close(self):
        """
        Close the SUMO simulation.
        """
        try:
            traci.close()
        except:
            pass

