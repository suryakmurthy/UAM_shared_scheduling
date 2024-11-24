import numpy as np
import ray
import numba as nb
from bluesky.tools import geo
from shapely.geometry import LineString, Point
from shapely.geometry.multilinestring import MultiLineString
from shapely.ops import nearest_points
import time
import math
from gym.spaces import Discrete, Box, Tuple
import os
import random
import json
import utm
from pyproj import Transformer
import glob
import gin
from copy import copy
import yaml
from itertools import groupby
from math import radians, sin, cos, sqrt, atan2, degrees


from inspect import currentframe
from timeit import default_timer as timer

# # Auto ATC Setup items
from D2MAV_A.qatc import TrafficManager, VehicleHelper, load_routes, BadLogic, VLS

# Load traffic manager configuration file
FILE_PREFIX = str(os.path.dirname(__file__))
TOWER_CONFIG_FILE = FILE_PREFIX + "/DFW_towers.yaml"
with open(TOWER_CONFIG_FILE, "r") as f:
    tower_config = yaml.load(f, Loader=yaml.Loader)
# Load some route data
import pickle

# with open("linestring_dict.pkl", 'rb') as file:
#     route_linestrings = pickle.load(file)

with open("route_data.pkl", "rb") as file:
    route_data = pickle.load(file)


## Limit GPU usage
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


## TODO: Move to a different file like "util/" or something similar
@nb.njit()
def discount(r, discounted_r, v, done, gae):
    """Compute the gamma-discounted rewards over an episode."""
    for t in range(len(r) - 1, -1, -1):
        if done[t] or t == (len(r) - 1):
            delta = r[t] - v[t][0]
            gae[t] = delta

        else:
            delta = r[t] + 0.95 * v[t + 1][0] - v[t][0]
            gae[t] = delta + 0.95 * 0.95 * gae[t + 1]

        discounted_r[t] = gae[t] + v[t][0]

    return discounted_r


## Checks the feasibility of the generated route based on
## specified threshold
def checkPoint(x1, y1, x2, y2, ls, threshold):
    for j in range(len(ls)):
        old_ls = ls[j]
        start, end = list(old_ls.coords)

        x_1, y_1 = start
        xe_1, ye_1 = end

        # distance start to old LS start
        dx = np.sqrt((x_1 - x1) ** 2 + (y_1 - y1) ** 2) / geo.nm

        # distance start to old LS end
        dx_1 = np.sqrt((xe_1 - x1) ** 2 + (ye_1 - y1) ** 2) / geo.nm

        # distance end to old LS end
        dx_2 = np.sqrt((xe_1 - x2) ** 2 + (ye_1 - y2) ** 2) / geo.nm

        # distance end to old LS start
        dx_3 = np.sqrt((x_1 - x2) ** 2 + (y_1 - y2) ** 2) / geo.nm

        dist = np.array([dx, dx_1, dx_2, dx_3])

        # feasible route
        if any(dist <= threshold):
            return False
    return True


@ray.remote
class Runner(object):
    import tensorflow as tf
    import bluesky as bs

    """
        Worker agent. Runs the BlueSky sim within its own process. This agent
        collects the experience and sends to the scheduler/trainer Worker

    """

    def __init__(
            self,
            actor_id,
            agent,
            max_steps=1024,
            speeds=[5, 0, 220],
            simdt=1,
            bsperf="openap",
            scenario_file=None,
            working_directory=None,
            LOS=10,
            dGoal=100,
            maxRewardDistance=100,
            intruderThreshold=750,
            rewardBeta=[0.001],
            rewardAlpha=[0.1],
            speedChangePenalty=[0.001],
            rewardLOS=[-1],
            shieldPenalty=[0.1],
            stepPenalty=[0],
            clearancePenalty=0.005,
            gamma = 0.01,
            alpha = 0,
            config_file=None,
            gui=False,
            non_coop_tag=0,
            traffic_manager_active=True,
            d2mav_active=True,
            vls_active=False,
            run_type="train",
    ):
        self.id = actor_id

        self.tf.config.threading.set_intra_op_parallelism_threads(1)
        self.tf.config.threading.set_inter_op_parallelism_threads(1)
        self.tf.compat.v1.logging.set_verbosity(self.tf.compat.v1.logging.ERROR)

        self.agents = {}
        self.agent = agent
        self.scen_file = scenario_file
        self.working_directory = working_directory
        self.speeds = np.array(speeds)
        self.altitudes = np.array([-500,0,500])
        self.max_steps = max_steps
        self.simdt = simdt
        self.bsperf = bsperf
        self.step_counter = 0
        self.LOS = LOS
        self.dGoal = dGoal
        self.maxRewardDistance = maxRewardDistance
        self.intruderThreshold = intruderThreshold
        self.rewardBeta = rewardBeta
        self.rewardAlpha = rewardAlpha
        self.speedChangePenalty = speedChangePenalty

        ## ALT Modification Params
        self.altChangePenalty = speedChangePenalty
        self.speed_change_counter = 0
        self.alt_change_counter = 0
        self.min_alt = 1000                    # 800
        self.max_alt = 3000                   # 2500 - 3000
        
        self.noise_const_a = -14.3088
        self.noise_const_b = 103.4767

        ### ADD TO CONFIG LATER
        self.alpha = alpha
        self.beta = 1 - self.alpha
        self.gamma = gamma
        self.safety_train = True
        self.noise_train = False
        print("Noise Reward Params: ", self.alpha, self.beta, self.gamma)
        self.max_noise_increase = 0
        self.average_noise_increase = []

        ## Noise Calculation Params
        self.a_0 = 88.09
        self.a_1 = 3.21
        self.a_2 = -2.62

        with open('/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/D2MAV_A/ambient_noise_dict.json', 'r') as file:
            self.ambient_noise_level = json.load(file)

        self.rewardLOS = rewardLOS
        self.shieldPenalty = shieldPenalty[0] # add [0] if testing
        self.stepPenalty = stepPenalty
        self.clearancePenalty = clearancePenalty
        self.gui = gui
        self.traffic_manager_active = traffic_manager_active
        self.d2mav_active = d2mav_active
        self.vls_active = vls_active
        self.route_active = True
        self.intersection_active = True
        self.run_type = run_type
        self.los_counter = 0
        if not "SIMDT" in os.environ.keys():
            os.environ["SIMDT"] = "{}".format(self.simdt)

        ## building episode specific parameters not configured by config.gin
        self.dones = []

        self.episode_done = True  ## initialization

        self.epsg_proj = "epsg:2163"
        self.epsg_from = "epsg:4326"
        self.transformer = Transformer.from_crs(
            self.epsg_from, self.epsg_proj, always_xy=True
        )

        self.timer = 0
        self.num_ac = 0
        self.counter = 0

        self.action_key = {}

        self.min_x = 281134.8350222109  # 686785.5111184405 #np.inf
        self.max_x = 332359.3446274982  # 737690.7518448773 #-np.inf
        self.min_y = -1352500.1522055818  # 3627144.8191298996 #np.inf
        self.max_y = -1306410.5905290868  # 3673125.271272543 #-np.inf
        self.tas_min = np.round(
            self.speeds[0] * geo.nm / 3600, 4
        )  # converting knots to m/s
        self.tas_max = np.round(
            self.speeds[2] * geo.nm / 3600, 4
        )  # converting knots to m/s
        self.ax_min = -3.5
        self.ax_max = 3.5
        self.max_d = 46726.453433800954  # 0

        # Non-cooperative
        self.non_coop_tag = non_coop_tag  # 0 for cooperative. 1 for Loos of Control. 2 for loss of communication.
        self.LControl_lst = ["PNHWL0"]
        self.LComm_lst = ["PI30L0"]
        # Format of actions: [slow down, maintain, speed up] x [decend, maintain, ascend]
        # speed action =  action / 3
        # alt action  = action % 3
        # Modified to account for altitude
        self.action_dim = 9 # (speed up, ascend), (speed up, maintain) ...
        self.speed_dim = 3 # speed up, maintain, slow down
        self.alt_dim = 3 # ascend, maintain, descend
        self.ownship_obs_dim = 8
        self.intruder_obs_dim = 9

        self.action_space = Discrete(self.action_dim)
        print("Shield Parameters: ", self.vls_active, self.intersection_active, self.route_active)
        # self.new_action_space = Tuple((self.action_space, self.alt_space))


        ### VLS Variables
        self.nmac_offset = 3 * 86
        self.nmac_distance = self.LOS + self.nmac_offset
        self.max_halt_time = 0
        self.halting_times = []
        self.halt_start = {}
        self.full_travel = {}
        self.travel_start = {}
        print(self.nmac_distance)
        print("Reward_params", self.shieldPenalty, self.rewardLOS)

        self.communication_radius = self.intruderThreshold
        
        # Initialize Traffic Manager
        self.create_traffic_manager()

        if self.traffic_manager_active:
            print("This is the flag that I should be missing")
            self.ownship_obs_dim += 2

        if self.gui:
            self.bs.init(
                mode="sim", configfile=self.working_directory + "/" + config_file
            )
            self.bs.net.connect()

        else:
            self.bs.init(
                mode="sim",
                detached=True,
                configfile=self.working_directory + "/" + config_file,
            )

        self.agent.initialize(
            self.tf, self.ownship_obs_dim, self.intruder_obs_dim, self.action_dim
        )

    def create_traffic_manager(self):
        route_linestrings = {}
        for route_id, gps_wp_list in route_data.items():
            rtemp = []
            for item in gps_wp_list:  # item is a tuple of (lon, lat)
                x, y = self.transformer.transform(item[0], item[1])
                rtemp.append((x, y))
            route_linestrings[route_id] = LineString(rtemp)
        self.traffic_manager = TrafficManager(tower_config)
        self.vehicle_helpers = {}  # Store vehicle helpers
        self.routes_loaded = load_routes(
            tower_config, self.traffic_manager, route_linestrings
        )  # Odd name to make sure it doesn't clash
        self.pending_requests = (
            []
        )  # TODO: Make this a dict and somehow store number of requests made
        self.pending_initial_requests = (
            []
        )  # TODO: Make this a dict and somehow store number of requests made
        self.exiting_vehicles = []

    def reset(self):
        """
        Beginning of the episode. In this function, all variables need to be reset to default.
        """

        self.agent.reset()

        self.timer = 0
        self.num_ac = 0
        self.counter = 0
        self.episode_done = False
        self.step_counter = 0
        self.time_without_traffic = 0
        self.dones = []
        self.acInfo = {}
        self.file_keeper = []
        collected_responses = {}
        self.los_counter = 0
        self.los_events = 0
        self.prev_LOS_pairs = []

        ## Noise Params
        self.max_noise_increase = 0
        self.average_noise_increase = {}

        ### VLS Objects ###
        self.agent_to_id ={}
        self.id_to_group = {}
        self.group_to_i_r = {}
        self.num_groups = 0
        self.vls = {}
        self.current_movement = {}
        self.vls_modifications_slow = []
        self.shield_counter = 0
        self.shield_counter_intersect = 0
        self.speed_change_counter = 0
        self.alt_change_counter = 0
        self.shield_counter_route = 0
        self.vls_modifications_halt = []
        self.full_travel = {}
        self.travel_start = {}
        self.wait_time = {}

        #### ADDITIONAL METRIC ####
        self.intersection_radius = 2700
        #### ADDITIONAL METRIC ####

        # randomly sample
        if ".scn" not in self.scen_file:
            scenario_files = glob.glob(f"{self.scen_file}" + "/*.scn")
            scenario_file = np.random.choice(scenario_files, 1)[0]
        else:
            scenario_file = self.scen_file
        self.scen_file_temp = scenario_file
        print("New Scenario: ", scenario_file)
        # Reset Traffic Manager
        self.create_traffic_manager()  # easier to just create a new one for now
        # TODO: Implement proper reset
        # self.traffic_manager.reset()
        # self.traffic_manager.reset()

        # Starting the bluesky and sim
        print("resetting")
        self.bs.stack.stack("IC " + self.working_directory + "/" + scenario_file)
        self.bs.stack.stack("FF")
        self.bs.sim.step()  # bs.sim.simt = 0.0 AFTER the call to bs.sim.step()
        self.bs.stack.stack("FF")

        # alt, clearance denied, cleared, no clearance request
        if self.traffic_manager_active:
            self.observation_space = Box(
                np.array([0, self.tas_min, self.ax_min, 0, 0, 40, 0, 0, 0, 0]),
                np.array(
                    [
                        self.max_d,
                        self.tas_max,
                        self.ax_max,
                        360,
                        self.max_alt,
                        60,
                        self.max_d,
                        2,
                        2,
                        self.max_d,
                    ]
                ),
                dtype=np.float64,
            )

        else:
            ## utm position, dist goal, speed, acceleration, heading, alt, LOS distance
            self.observation_space = Box(
                np.array([0, self.tas_min, self.ax_min, 0, 0, 40, 0, 0]),
                np.array([self.max_d, self.tas_max, self.ax_max, 360, self.max_alt, 60,self.max_d, 2]),
                dtype=np.float64,
            )
        ## rel utm position, dist goal, speed, acceleration, heading, distance ownship to intruder, distance intruder intersection, distance ownship to intersection
        self.context_space = Box(
            np.array(
                [
                    self.min_x - self.max_x,
                    self.min_y - self.max_y,
                    0,
                    self.tas_min,
                    self.ax_min,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ),
            np.array(
                [
                    self.max_x - self.min_x,
                    self.max_y - self.min_y,
                    self.max_d,
                    self.tas_max,
                    self.ax_max,
                    360,
                    self.max_alt,
                    self.max_d,
                    self.max_d,
                    self.max_d,
                    2,
                ]
            ),
            dtype=np.float64,
        )

        ## This is a catch to make sure the time between bluesky sim steps is 1 seconds
        # Should 1 second be changed to something smaller like 0.1?
        before = self.bs.sim.simt
        self.bs.sim.step()
        after = self.bs.sim.simt
        if (after - before) == 0:
            return self.reset()

        assert (after - before) == self.simdt

        ## The first self.bs.sim.step() spawns in the initial aircraft. Sim time should be at t = 12 seconds now
        self.step_counter += 1

        response = {}
        if self.traffic_manager_active:
            # First time the auto atc system is pinged
            initial_requests = []
            for i in range(self.bs.traf.lat.shape[0]):
                id_ = self.bs.traf.id[i]  # ownship ID

                # Check if the current ID exists. If not then create a new vehicle helper
                if not id_ in self.vehicle_helpers.keys():
                    # Get and reformat the route name coming from Bluesky
                    route_name = self.bs.traf.ap.route[i].wpname[0][0:-1]
                    self.vehicle_helpers[id_] = VehicleHelper(
                        id_, self.routes_loaded[route_name]
                    )
                    # Add initial request to enter the system
                    initial_requests.append(id_)

            # Pass collected requests to the Traffic Manager and process them
            for id_ in initial_requests:
                formatted_request = self.vehicle_helpers[id_].format_request()  # tuple
                #print("request_1: ", formatted_request)
                self.traffic_manager.add_request(id_, formatted_request)

            if initial_requests:
                initial_request_response = self.traffic_manager.process_requests()
            else:
                response = None
                initial_request_response = {}
                print("No initial requests")

            collected_responses = {}
            for id_, response in initial_request_response.items():
                k_idx = self.bs.traf.id2idx(id_)
                collected_responses[id_] = [
                    response,
                    self.vehicle_helpers[id_].distance_to_next_boundary(
                        [self.bs.traf.lon[k_idx], self.bs.traf.lat[k_idx]]
                    ),
                ]
                if response and not self.within_LOS(id_):
                    self.vehicle_helpers[id_].enter_request_status = True
                    self.vehicle_helpers[id_].initial_request_granted = True
                    self.travel_start[id_] = self.bs.sim.simt
                    self.wait_time[id_] = 0
                else:
                    self.vehicle_helpers[id_].enter_request_status = False
                    self.pending_initial_requests.append(id_)
                    k_idx = self.bs.traf.id2idx(id_)
                    self.bs.traf.ap.setclrcmd(
                        k_idx, False
                    )  # set the clearance to False (i.e., denied and hold on ground)
        self.prev_id_copy  = self.bs.traf.id.copy()
        state, _, _, _ = self.state_update(
            self.bs.traf, init=True, tm_response=collected_responses
        )

        if self.gui:
            self.bs.net.update()
        return state

    def step(self, actions, policy, value):
        """
        Update the environment with the actions from the agents
        """

        collected_responses = {}
        ## Update Actions
        self.vls_modifications_slow = []
        self.vls_modifications_halt = []
        speed_dict = {}
        alt_dict = {}

        coord_transform = self.transformer.transform(self.bs.traf.lon, self.bs.traf.lat)
        geometries = MultiLineString(
            [
                [(coord_transform[0][i], coord_transform[1][i])]
                + [
                    self.transformer.transform(
                        self.bs.traf.ap.route[i].wplon[j],
                        self.bs.traf.ap.route[i].wplat[j],
                    )
                    for j in range(
                        self.bs.traf.ap.route[i].iactwp,
                        len(self.bs.traf.ap.route[i].wplon),
                    )
                ]
                for i in range(self.bs.traf.lat.shape[0])
            ]
        ) 

        for ac_id in actions.keys():
            if actions[ac_id] == -1:
                """Uncomment if running unequiped"""
                continue
            # Check if the D2MAV-A is disabled
            if self.d2mav_active == False:
                # check if traffic manager is active
                if self.traffic_manager_active == False:
                    continue
                else:
                    print("D2MAV Inactive")
                    if ac_id in self.action_override:
                        speed = 0
                    else:
                        speed = 40
                    self.bs.stack.stack("{} SPD {}".format(ac_id, speed))
                    continue
            speed_action = int(actions[ac_id]/3)
            alt_action = actions[ac_id]%3

            # Max Speed and Action Baseline
            # alt_action = 0
            # speed_action = 2

            # print("action_val: ", speed_action)
            speed = self.speeds[speed_action]
            # New Altitude Changes
            alt_change = self.altitudes[alt_action]
            # print("Update Altitude: ", self.bs.traf.alt[self.bs.traf.id2idx(ac_id)])
            current_alt = self.meters_to_feet(self.bs.traf.alt[self.bs.traf.id2idx(ac_id)])
            new_alt = current_alt + alt_change
            if speed_action == 1:  # hold/maintain
                # Convert current speed in m/s to knots
                speed = int(
                    np.round(
                        (self.bs.traf.cas[self.bs.traf.id2idx(ac_id)] / geo.nm) * 3600
                    )
                )
            ### VLS CASES
            if self.vls_active:
                n_ac = self.bs.traf.lat.shape[0]
                d = (
                geo.kwikdist_matrix(
                    np.repeat(self.bs.traf.lat, n_ac),
                    np.repeat(self.bs.traf.lon, n_ac),
                    np.tile(self.bs.traf.lat, n_ac),
                    np.tile(self.bs.traf.lon, n_ac),
                ).reshape(n_ac, n_ac)
                * geo.nm)
                
                nmac_flag = False
                if self.wait_time[ac_id] != 0:
                    self.wait_time[ac_id] -= 1
                    self.action_override.append(ac_id)
                else:
                    for other_id in self.bs.traf.id:
                        if ac_id == other_id:
                            continue
                        i_idx = self.bs.traf.id2idx(ac_id)
                        j_idx = self.bs.traf.id2idx(other_id)
                        if d[i_idx][j_idx] > self.intersection_radius:
                            continue
                        ownship_obj = geometries.geoms[i_idx]
                        intruder_obj = geometries.geoms[j_idx]
                        # if not ownship_obj.intersects(intruder_obj) and not intruder_obj.intersects(ownship_obj):
                        #     continue
                        # Intersection Shielding Logic
                        if self.intersection_active:
                            if self.vehicle_helpers[other_id].route.route_id[0:3] != self.vehicle_helpers[ac_id].route.route_id[0:3]:
                                if self.vehicle_helpers[other_id].current_intersection != None and self.vehicle_helpers[ac_id].current_intersection == None:
                                    # if ac_id == 'PBUSL0':
                                        # print("Checking Potential Conflict Between: ", ac_id, other_id, d[i_idx][j_idx], self.intersection_radius, self.vehicle_helpers[ac_id].next_intersection, self.vehicle_helpers[other_id].current_intersection)
                                    if self.vehicle_helpers[ac_id].next_intersection == self.vehicle_helpers[other_id].current_intersection:
                                        if self.bs.traf.tas[j_idx] != 0:
                                            self.action_override.append(ac_id)
                                else:
                                    if self.vehicle_helpers[other_id].current_intersection != None:
                                        if self.bs.traf.tas[j_idx] != 0:
                                            if self.bs.traf.tas[i_idx] == 0:
                                                self.action_override.append(ac_id)
                                            else:
                                                self.action_override.append(ac_id)
                                                self.wait_time[ac_id] = random.randint(1, 100)

                        # Same Route Shielding Logic
                        if self.route_active:
                            if self.vehicle_helpers[other_id].route.route_id[0:3] == self.vehicle_helpers[ac_id].route.route_id[0:3]:
                                if self.bs.traf.distflown[i_idx] < self.bs.traf.distflown[j_idx]:
                                    if d[i_idx][j_idx] < self.nmac_distance:
                                        if self.bs.traf.tas[j_idx] == 0:
                                            self.action_override.append(ac_id)
                                        else:
                                            speed = self.speeds[0]
                                            self.vls_modifications_slow.append(ac_id)
                                        self.shield_counter += 1
                                        self.shield_counter_route += 1

                            # if self.vehicle_helpers[other_id].route.route_id[0:3] != self.vehicle_helpers[ac_id].route.route_id[0:3]:
                            #     if self.vehicle_helpers[other_id].current_intersection != None and self.vehicle_helpers[ac_id].current_intersection == None:
                            #         if self.bs.traf.tas[j_idx] != 0:
                            #             self.action_override.append(ac_id)
                              
            if ac_id in self.action_override:
                # print("Action Override: ", ac_id)
                speed = 0
                if ac_id not in self.halt_start:
                    self.halt_start[ac_id] = self.bs.sim.simt
            
            if ac_id not in self.action_override:
                i_idx = self.bs.traf.id2idx(ac_id)
                if self.bs.traf.tas[i_idx] == 0:
                    speed = self.speeds[2]
                if ac_id in self.halt_start.keys():
                    halting_time = self.bs.sim.simt - self.halt_start[ac_id]
                    del self.halt_start[ac_id]
                    self.halting_times.append(halting_time)
                    
            speed_dict[ac_id] = speed
            alt_dict[ac_id] = new_alt

        for ac_id in speed_dict.keys():
            self.bs.stack.stack("{} SPD {}".format(ac_id, speed_dict[ac_id]))
            # print("Checking Alt: ", ac_id, alt_dict[ac_id], self.min_alt, self.max_alt)
            if alt_dict[ac_id] >= self.min_alt and alt_dict[ac_id] <= self.max_alt: # min flight alt
                self.bs.stack.stack("{} ALT {}".format(ac_id, alt_dict[ac_id]))
            else:
                if alt_dict[ac_id] < self.min_alt:
                    self.bs.stack.stack("{} ALT {}".format(ac_id, self.min_alt))
                if alt_dict[ac_id] > self.max_alt:
                    self.bs.stack.stack("{} ALT {}".format(ac_id, self.max_alt))
        # time.sleep(10)
        # updates the bluesky environment by 1 simulation timestep (1 seconds)
        self.bs.sim.step()
        self.step_counter += 1

        if self.gui:
            self.bs.net.update()

        ### Start of traffic manager code ###
        if self.traffic_manager_active:
            new_requests = []  # requests from vehicles that are in the system
            initial_requests = []  # requests from vehicles that are not in the system

            """ Use the traffic object to determine the state of all intersections and route sections.
                1) Determine which vehicles are currently inside which intersections
                2) Determine which vehicles have transitioned:
                    a) into an intersection
                    b) out of an intersection  """

            for i in range(self.bs.traf.lat.shape[0]):
                id_ = self.bs.traf.id[i]  # ownship ID

                # Skip GA aircraft
                if id_[0:2] == "GA":
                    continue

                # Skip vehicles who are waiting outside the system
                if id_ in self.pending_initial_requests:
                    continue

                """ Vehicles that have spawned during the current step might be removed if space is not available,
                    therefore we need to skip them until they are authorized to enter the system.
                    Vehicles who were in self.vehicle_helpers at the beginning of the step have already entered the system """
                if id_ not in self.vehicle_helpers.keys():
                    continue

                curr_gps = [self.bs.traf.lon[i], self.bs.traf.lat[i]]
                # Determine which vehicles are inside which intersections
                for intersection in self.traffic_manager.intersections.values():
                    if self.traffic_manager.check_if_within_intersection(
                            curr_gps, intersection.tower_ID
                    ):
                        # Determine if a vehicle is already accepted into the intersection
                        if (
                                id_ in intersection.accepted
                                or id_ in intersection.recently_left
                        ):
                            break
                        # Determine if authorized vehicle has transitioned into intersection
                        elif id_ in intersection.authorized:
                            if id_ not in intersection.accepted:
                                intersection.accepted.append(id_)
                            intersection.authorized.remove(id_)
                            # Update vehicle helper
                            self.vehicle_helpers[id_].within_intersection = True
                            self.vehicle_helpers[
                                id_
                            ].current_intersection = intersection.tower_ID
                            # Update the route section that the vehicle left if not an initial request
                            crs = self.vehicle_helpers[
                                id_
                            ].current_route_section  # Will be None if initial request
                            if crs:
                                self.traffic_manager.towers[crs].accepted.remove(id_)
                            break
                        # Determine if vehicle has entered illegally
                        elif (
                                id_ not in intersection.authorized
                                and id_ not in intersection.accepted
                                and id_ not in intersection.recently_left
                        ):
                            if id_ not in intersection.illegal:
                                intersection.illegal.append(id_)
                                # print(
                                #     id_,
                                #     "has entered",
                                #     intersection.tower_ID,
                                #     "illegally",
                                # )
                                # Update vehicle helper
                                self.vehicle_helpers[id_].within_intersection = True
                                self.vehicle_helpers[
                                    id_
                                ].current_intersection = intersection.tower_ID
                                break
                            else:
                                break
                        else:  # Raise error because it should not be possible to pass through all of the above checks
                            raise BadLogic(
                                "Bad logic when checking if vehicle is within intersection. id_: ",
                                id_,
                                "intersection: ",
                                intersection.tower_ID,
                            )
                    else:
                        # Determine if accepted vehicle has transitioned out of intersection
                        if id_ in intersection.accepted:
                            intersection.accepted.remove(id_)
                            intersection.recently_left.append(id_)
                            # Update vehicle helper to reflect that this vehicle is no longer in the intersection
                            self.vehicle_helpers[id_].within_intersection = False
                            self.vehicle_helpers[id_].current_intersection = None
                            self.vehicle_helpers[
                                id_
                            ].enter_request_status = False  # reset
                            # Update the route section that the vehicle entered if not exiting the system

                            """ THE FOLLOWING IF STATEMENT SHOULD BE REMOVABLE """
                            nrs = self.vehicle_helpers[
                                id_
                            ].next_route_section  # Will be None if exiting the system
                            if nrs:
                                self.traffic_manager.towers[nrs].accepted.append(id_)
                                self.traffic_manager.towers[nrs].authorized.remove(id_)
                                self.vehicle_helpers[id_].change_route_section()
                                self.vehicle_helpers[id_].next_intersection = self.traffic_manager.search_for_intersection(self.vehicle_helpers[id_].current_route_section, "inbound").tower_ID

                            break
                        # Determine if illegal vehicle has transitioned out of intersection
                        elif id_ in intersection.illegal:
                            intersection.illegal.remove(id_)
                            # Update vehicle helper to reflect that this vehicle is no longer in the intersection
                            self.vehicle_helpers[id_].within_intersection = False
                            self.vehicle_helpers[id_].current_intersection = None
                            self.vehicle_helpers[id_].enter_request_status = False
                            # print(
                            #     "Illegal vehicle",
                            #     id_,
                            #     "has exited",
                            #     intersection.tower_ID,
                            # )
                else:
                    # Set vehicle helper to reflect that this vehicle is not in an intersection
                    self.vehicle_helpers[id_].within_intersection = False
                    self.vehicle_helpers[id_].current_intersection = None

            # Update the volume of each intersection
            for intersection in self.traffic_manager.intersections.values():
                intersection.set_volume()
            # Update the volume of each route section
            for tower in self.traffic_manager.towers.values():
                tower.set_volume()

            """ Collect new vehicle requests for processing """
            for i in range(self.bs.traf.lat.shape[0]):
                id_ = self.bs.traf.id[i]  # ownship ID
                if id_[0:2] == "GA":
                    continue

                # Check if the current ID exists. If not then create a new vehicle helper
                if not id_ in self.vehicle_helpers.keys():
                    # Get and reformat the route name coming from Bluesky
                    route_name = self.bs.traf.ap.route[i].wpname[0][0:-1]
                    # print("Checking Route Name: ", id_, self.bs.traf.ap.route[i].wpname[0], route_name)
                    self.vehicle_helpers[id_] = VehicleHelper(
                        id_, self.routes_loaded[route_name]
                    )
                    # Add initial request to enter the system
                    initial_requests.append(id_)
                else:
                    a = id_ not in self.pending_requests
                    b = id_ not in self.pending_initial_requests
                    c = (
                            id_ not in self.exiting_vehicles
                    )  # TODO: Check if this is necessary
                    d = not self.vehicle_helpers[id_].enter_request_status
                    if a and b and c and d:
                        request_eligibility = self.vehicle_helpers[
                            id_
                        ].check_if_request_eligible(
                            [self.bs.traf.lon[i], self.bs.traf.lat[i]]
                        )
                        if request_eligibility:
                            new_requests.append(id_)

            """ Process requests """
            # First process pending requests
            for id_ in self.pending_requests:
                formatted_request = self.vehicle_helpers[id_].format_request()  # tuple
                #print("request_1: ", formatted_request)
                self.traffic_manager.add_request(id_, formatted_request)
            pending_request_response = self.traffic_manager.process_requests()

            # Second process new requests
            for id_ in new_requests:
                formatted_request = self.vehicle_helpers[id_].format_request()
                #print("request_1: ", formatted_request)
                self.traffic_manager.add_request(id_, formatted_request)
            new_request_response = self.traffic_manager.process_requests()

            # Third process pending initial requests
            for id_ in self.pending_initial_requests:
                formatted_request = self.vehicle_helpers[id_].format_request()
                #print("request_1: ", formatted_request)
                self.traffic_manager.add_request(id_, formatted_request)
            # print("Updating Initial Requests")
            pending_initial_request_response = self.traffic_manager.process_requests()

            # Fourth process initial requests
            for id_ in initial_requests:
                # Add takeoff shield
                formatted_request = self.vehicle_helpers[id_].format_request()
                #print("request_1: ", formatted_request)
                self.traffic_manager.add_request(id_, formatted_request)
            initial_request_response = self.traffic_manager.process_requests()

            """ Collect responses and update request lists based on the response """
            collected_responses = {}
            # Pending in system responses
            for id_, response in pending_request_response.items():
                k_idx = self.bs.traf.id2idx(id_)
                collected_responses[id_] = [
                    response,
                    self.vehicle_helpers[id_].distance_to_next_boundary(
                        [self.bs.traf.lon[k_idx], self.bs.traf.lat[k_idx]]
                    ),
                ]
                if response:
                    self.vehicle_helpers[id_].enter_request_status = True
                    self.pending_requests.remove(id_)
                    if self.vehicle_helpers[id_].final_route_segment:
                        self.exiting_vehicles.append(id_)
                        left_time = self.bs.sim.simt
                        self.full_travel[id_] = left_time - self.travel_start[id_]
                        # print(f"Aircraft {id_} is exiting with exit time {self.full_travel}")
                else:
                    self.vehicle_helpers[id_].enter_request_status = False
            # New in system responses
            for id_, response in new_request_response.items():
                k_idx = self.bs.traf.id2idx(id_)
                collected_responses[id_] = [
                    response,
                    self.vehicle_helpers[id_].distance_to_next_boundary(
                        [self.bs.traf.lon[k_idx], self.bs.traf.lat[k_idx]]
                    ),
                ]
                if response:
                    self.vehicle_helpers[id_].enter_request_status = True
                    if self.vehicle_helpers[id_].final_route_segment:
                        self.exiting_vehicles.append(id_)
                        left_time = self.bs.sim.simt
                        self.full_travel[id_] = left_time - self.travel_start[id_]
                        # print(f"Aircraft {id_} is exiting with exit time {self.full_travel}")
                else:
                    self.pending_requests.append(id_)
                    self.vehicle_helpers[id_].enter_request_status = False
            # Pending initial request responses
            for id_, response in pending_initial_request_response.items():
                k_idx = self.bs.traf.id2idx(id_)
                collected_responses[id_] = [
                    response,
                    self.vehicle_helpers[id_].distance_to_next_boundary(
                        [self.bs.traf.lon[k_idx], self.bs.traf.lat[k_idx]]
                    ),
                ]
                # Add Shield for takeoff
                if response and not self.within_LOS(id_):
                    self.vehicle_helpers[id_].enter_request_status = True
                    self.vehicle_helpers[id_].initial_request_granted = True
                    self.pending_initial_requests.remove(id_)
                    self.travel_start[id_] = self.bs.sim.simt
                    # set the clearance to True. Optional fields are ALT, SPD
                    k_idx = self.bs.traf.id2idx(id_)
                    self.bs.traf.ap.setclrcmd(k_idx, True, self.min_alt, 30)
                    # print(f"{id_} cleared for departure")
                    self.wait_time[id_] = 0
                else:
                    # print(f"{id_} denied departure")
                    self.vehicle_helpers[id_].enter_request_status = False
                    # self.bs.traf.ap.setclrcmd(k_idx, False)
            # Initial request responses
            for id_, response in initial_request_response.items():
                # print("new initial request: ", id_, response)
                k_idx = self.bs.traf.id2idx(id_)
                collected_responses[id_] = [
                    response,
                    self.vehicle_helpers[id_].distance_to_next_boundary(
                        [self.bs.traf.lon[k_idx], self.bs.traf.lat[k_idx]]
                    ),
                ]
                # print(f"Checking {id_} for takeoff", response)
                # Add Shield for takeoff
                if response and not self.within_LOS(id_):
                    # print(f"{id_} approved departure")
                    self.travel_start[id_] = self.bs.sim.simt
                    self.vehicle_helpers[id_].initial_request_granted = True
                    self.vehicle_helpers[id_].enter_request_status = True
                    self.wait_time[id_] = 0
                else:
                    self.vehicle_helpers[id_].enter_request_status = False
                    self.pending_initial_requests.append(id_)
                    k_idx = self.bs.traf.id2idx(id_)
                    # print(f"{id_} denied departure")
                    self.bs.traf.ap.setclrcmd(k_idx, False)

        ### End of traffic manager code ###
        
        obs, reward, done, info = self.state_update(
            self.bs.traf,
            a=actions,
            policy=policy,
            value=value,
            init=False,
            tm_response=collected_responses,
        )

        if len(self.bs.traf.id) == 0:
            self.time_without_traffic += self.bs.sim.simdt
        else:
            self.time_without_traffic = 0

        if self.time_without_traffic > 1800:  # 0.5 hours
            done["__all__"] = True
        else:
            done["__all__"] = False

        return obs, reward, done, info

    ### VLS Helper Functions ###
    def meters_to_feet(self, meters):
        feet = meters * 3.28084
        return feet

    def update_communication_groups(self):
        n_ac = self.bs.traf.lat.shape[0]
        d = (
                geo.kwikdist_matrix(
                    np.repeat(self.bs.traf.lat, n_ac),
                    np.repeat(self.bs.traf.lon, n_ac),
                    np.tile(self.bs.traf.lat, n_ac),
                    np.tile(self.bs.traf.lon, n_ac),
                ).reshape(n_ac, n_ac)
                * geo.nm
        )
        coord_transform = self.transformer.transform(self.bs.traf.lon, self.bs.traf.lat)
        geometries = MultiLineString(
            [
                [(coord_transform[0][i], coord_transform[1][i])]
                + [
                    self.transformer.transform(
                        self.bs.traf.ap.route[i].wplon[j],
                        self.bs.traf.ap.route[i].wplat[j],
                    )
                    for j in range(
                        self.bs.traf.ap.route[i].iactwp,
                        len(self.bs.traf.ap.route[i].wplon),
                    )
                ]
                for i in range(self.bs.traf.lat.shape[0])
            ]
        )
        # Do a Check to see if any vehicles in any groups have left the airspace
        removal_list = []
        for g in self.id_to_group.keys():
            group_i = self.id_to_group[g].copy()
            for v_left in group_i:
                id_left = v_left.vehicle_ID
                if id_left not in self.bs.traf.id:
                    self.id_to_group[g].remove(v_left)
                    if self.id_to_group[g] == []:
                        removal_list.append(g)
                    else:
                        if self.current_movement[g] == id_left:
                            self.current_movement[g] = self.vls[g].get_next_vehicle()
        for r in removal_list:
            del self.id_to_group[r]
            del self.vls[r]
        for i in range(self.bs.traf.lat.shape[0]):
            id_i = self.bs.traf.id[i]
            ## Potential Issue can occur if intersections are close to one another
            ### COME BACK HERE
            # if not self.check_in_intersection([self.bs.traf.lon[i], self.bs.traf.lat[i]]):
            if id_i in self.agent_to_id:
                group_i = self.agent_to_id[id_i]
                if self.vehicle_helpers[id_i].current_route_section != self.group_to_i_r[group_i] and self.vehicle_helpers[id_i].current_intersection != self.group_to_i_r[group_i]:
                    # print("Case 1")
                    del self.agent_to_id[id_i]
                    self.id_to_group[group_i].remove(self.vehicle_helpers[id_i])
                    
                    if self.id_to_group[group_i] != []:
                        if self.current_movement[group_i] == id_i:
                            self.current_movement[group_i] = self.vls[group_i].get_next_vehicle()
                        else:
                            self.vls[group_i].remove_vehicle(self.vehicle_helpers[id_i])
                    else:
                        del self.id_to_group[group_i]
                        del self.current_movement[group_i]
                    continue
                ### New Removal Logic
                ### Remove if closer to end of segment than all the other agents
                idx = self.bs.traf.id2idx(id_i)
                out_of_range = True
                for a in self.id_to_group[group_i]:
                    a_idx = self.bs.traf.id2idx(a.vehicle_ID)
                    if a_idx != idx:
                        ownship_obj = geometries.geoms[idx]
                        other_obj = geometries.geoms[a_idx]
                        dist = np.abs(ownship_obj.length - other_obj.length)
                        d_2 = d[idx][a_idx]
                        if dist <= self.communication_radius or d_2 <= self.communication_radius:
                            out_of_range = False
                if out_of_range == True:
                    # print("removing due to out of range: ", id_i)
                    del self.agent_to_id[id_i]
                    self.id_to_group[group_i].remove(self.vehicle_helpers[id_i])
                    if self.id_to_group[group_i] != []:
                        if self.current_movement[group_i] == id_i:
                            self.current_movement[group_i] = self.vls[group_i].get_next_vehicle()
                        else:
                            self.vls[group_i].remove_vehicle(self.vehicle_helpers[id_i])
                    else:
                        del self.id_to_group[group_i]
                        del self.current_movement[group_i]
                    continue

            for j in range(self.bs.traf.lat.shape[0]):
                if i != j:
                    id_j = self.bs.traf.id[j]
                    if self.vehicle_helpers[id_i].current_route_section != self.vehicle_helpers[id_j].current_route_section and self.vehicle_helpers[id_i].current_intersection != self.vehicle_helpers[id_j].current_intersection:
                        continue
                    if self.vehicle_helpers[id_i].current_route_section == None and self.vehicle_helpers[id_j].current_route_section == None and self.vehicle_helpers[id_i].current_intersection != self.vehicle_helpers[id_j].current_intersection:
                        continue
                    if self.vehicle_helpers[id_i].current_intersection == None and self.vehicle_helpers[id_j].current_intersection == None and self.vehicle_helpers[id_i].current_route_section != self.vehicle_helpers[id_j].current_route_section:
                        continue
                    if self.vehicle_helpers[id_i].current_intersection == None and self.vehicle_helpers[id_j].current_intersection == None and self.vehicle_helpers[id_i].current_route_section == None and self.vehicle_helpers[id_j].current_route_section == None:
                        continue
                    ownship_obj = geometries.geoms[i]
                    other_obj = geometries.geoms[j]
                    dist = np.abs(ownship_obj.length - other_obj.length)
                    d_2 = d[i][j]
                    # Check if the agents are in communication distance
                    # p_flag = False
                    # if id_i == 'PTWYR0':
                    #     p_flag = True
                    #     print("potential group: ", id_i, id_j, self.vehicle_helpers[id_i].current_route_section, self.vehicle_helpers[id_j].current_route_section, self.vehicle_helpers[id_i].current_intersection, self.vehicle_helpers[id_j].current_intersection, dist, d_2, self.communication_radius)
                    if dist <= self.communication_radius or d_2 <= self.communication_radius:
                        # Check if the agents belong to a communication group
                        if id_i in self.agent_to_id.keys():
                            # if p_flag:
                            #     print("case 1")
                            group_i = self.agent_to_id[id_i]
                            if id_j in self.agent_to_id.keys():
                                # if p_flag:
                                #     print("case 1.2")
                                group_j = self.agent_to_id[id_j]
                                # If the agents do not belong to the same group. Merge the two groups
                                # print(id_i, id_j, dist, self.communication_radius, self.vehicle_helpers[id_i].current_intersection, self.vehicle_helpers[id_j].current_intersection, self.group_to_i_r[group_i], self.group_to_i_r[group_j])
                                ## CURRENT ERROR self.group_to_i_r[group_i] == self.group_to_i_r[group_j]
                                if group_i != group_j:
                                    # if p_flag:
                                    #     print("case 1.3")
                                    if self.group_to_i_r[group_i] == self.group_to_i_r[group_j]:
                                        for agent in self.id_to_group[group_j]:
                                            self.id_to_group[group_i].append(agent)
                                            self.agent_to_id[agent.vehicle_ID] = group_i
                                            self.vls[group_i].add_vehicle(agent)
                                        del self.id_to_group[group_j]
                                        del self.vls[group_j]
                                    else:
                                        if len(self.id_to_group[group_i]) >= len(self.id_to_group[group_j]):
                                            for agent in self.id_to_group[group_j]:
                                                if self.vehicle_helpers[id_i].current_route_section == self.vehicle_helpers[id_j].current_route_section:
                                                    if agent.current_route_section == self.vehicle_helpers[id_i].current_route_section and agent.current_route_section == self.group_to_i_r[group_i]:
                                                        self.id_to_group[group_i].append(agent)
                                                        self.agent_to_id[agent.vehicle_ID] = group_i
                                                        self.vls[group_i].add_vehicle(agent)
                                                        self.id_to_group[group_j].remove(self.vehicle_helpers[agent.vehicle_ID])
                                                        if self.id_to_group[group_j] != []:
                                                            if self.current_movement[group_j] == agent.vehicle_ID:
                                                                self.current_movement[group_j] = self.vls[group_j].get_next_vehicle()
                                                            else:
                                                                self.vls[group_j].remove_vehicle(agent)
                                                        else:
                                                            del self.id_to_group[group_j]
                                                            del self.vls[group_j]
                                                if self.vehicle_helpers[id_i].current_intersection == self.vehicle_helpers[id_j].current_intersection:
                                                    if agent.current_intersection == self.vehicle_helpers[id_i].current_intersection and agent.current_intersection != None and self.vehicle_helpers[id_i].current_intersection != None and self.agent_to_id[agent.vehicle_ID] != group_i:
                                                        self.id_to_group[group_i].append(agent)
                                                        self.agent_to_id[agent.vehicle_ID] = group_i
                                                        self.vls[group_i].add_vehicle(agent)
                                                        self.id_to_group[group_j].remove(self.vehicle_helpers[agent.vehicle_ID])
                                                        if self.id_to_group[group_j] != []:
                                                            if self.current_movement[group_j] == agent.vehicle_ID:
                                                                self.current_movement[group_j] = self.vls[group_j].get_next_vehicle()
                                                            else:
                                                                self.vls[group_j].remove_vehicle(agent)
                                                        else:
                                                            del self.id_to_group[group_j]
                                                            del self.vls[group_j]
                                            if self.group_to_i_r[group_i] != self.vehicle_helpers[id_i].current_intersection and self.vehicle_helpers[id_i].current_intersection != None:
                                                self.group_to_i_r[group_i] = self.vehicle_helpers[id_i].current_intersection
                                            # if p_flag:
                                            #     print("case 1.7")
                                            #     for i in self.id_to_group.keys():
                                            #         print(i, [element.vehicle_ID for element in self.id_to_group[i]], self.group_to_i_r[group_i]) 
                                            # print("deleting issue?: ", group_j)
                                        else:
                                            # if p_flag:
                                            #     print("case 1.4")
                                            # Error to fix tomorrow:
                                            # One element in a group has switched route segments. How do we account for this before reconstructing the groups.
                                            for agent in self.id_to_group[group_i]:
                                                if self.vehicle_helpers[id_i].current_route_section == self.vehicle_helpers[id_j].current_route_section:
                                                    if agent.current_route_section == self.vehicle_helpers[id_j].current_route_section and agent.current_route_section == self.group_to_i_r[group_j]:
                                                        # if p_flag:
                                                        #     print("case 1.4.1", agent.vehicle_ID, agent.current_route_section, self.group_to_i_r[group_j])
                                                        self.id_to_group[group_j].append(agent)
                                                        self.agent_to_id[agent.vehicle_ID] = group_j
                                                        self.vls[group_j].add_vehicle(agent)
                                                        self.id_to_group[group_i].remove(self.vehicle_helpers[agent.vehicle_ID])
                                                        if self.id_to_group[group_i] != []:
                                                            if self.current_movement[group_i] == agent.vehicle_ID:
                                                                self.current_movement[group_i] = self.vls[group_i].get_next_vehicle()
                                                            else:
                                                                self.vls[group_i].remove_vehicle(agent)
                                                        else:
                                                            del self.id_to_group[group_i]
                                                            del self.vls[group_i]
                                                if self.vehicle_helpers[id_i].current_intersection == self.vehicle_helpers[id_j].current_intersection:
                                                    if agent.current_intersection == self.vehicle_helpers[id_j].current_intersection and agent.current_intersection !=  None and self.vehicle_helpers[id_j].current_intersection != None and self.agent_to_id[agent.vehicle_ID] != group_j:
                                                        # if p_flag:
                                                        #     print("case 1.4.2", agent.vehicle_ID, id_j, agent.current_intersection, self.vehicle_helpers[id_j].current_intersection)
                                                        self.id_to_group[group_j].append(agent)
                                                        self.agent_to_id[agent.vehicle_ID] = group_j
                                                        self.vls[group_j].add_vehicle(agent)
                                                        self.id_to_group[group_i].remove(self.vehicle_helpers[agent.vehicle_ID])
                                                        if self.id_to_group[group_i] != []:
                                                            if self.current_movement[group_i] == agent.vehicle_ID:
                                                                self.current_movement[group_i] = self.vls[group_i].get_next_vehicle()
                                                            else:
                                                                self.vls[group_i].remove_vehicle(agent)
                                                        else:
                                                            del self.id_to_group[group_i]
                                                            del self.vls[group_i]
                                            if self.group_to_i_r[group_j] != self.vehicle_helpers[id_j].current_intersection and self.vehicle_helpers[id_j].current_intersection != None:
                                                self.group_to_i_r[group_j] = self.vehicle_helpers[id_j].current_intersection
                                            # print("deleting issue 2?", group_i)
                            # If j is not any group, add it to i's group
                            else:
                                self.agent_to_id[id_j] = group_i
                                self.id_to_group[group_i].append(self.vehicle_helpers[id_j])
                                self.vls[group_i].add_vehicle(self.vehicle_helpers[id_j])
                                if self.vehicle_helpers[id_i].current_route_section != self.vehicle_helpers[id_j].current_route_section:
                                    self.group_to_i_r[group_i] = self.vehicle_helpers[id_j].current_intersection
                                # if p_flag:
                                #     print("case 1.5")
                                    # for i in self.id_to_group.keys():
                                    #     print(i, [element.vehicle_ID for element in self.id_to_group[i]], self.group_to_i_r[i])
                                
                        # If neither are in a communication group, create a new group with the agents in it
                        else:
                            # if p_flag:
                            #     print("case 2")
                            if id_j in self.agent_to_id.keys():
                                group_j = self.agent_to_id[id_j]
                                if  self.vehicle_helpers[id_i].current_intersection == self.group_to_i_r[group_j] or self.vehicle_helpers[id_i].current_route_section == self.group_to_i_r[group_j]:
                                    self.agent_to_id[id_i] = group_j
                                    self.id_to_group[group_j].append(self.vehicle_helpers[id_i])
                                    self.vls[group_j].add_vehicle(self.vehicle_helpers[id_i])
                                else:
                                    ## Remove from old group and add to new one
                                    self.agent_to_id[id_j] = self.num_groups
                                    self.id_to_group[group_j].remove(self.vehicle_helpers[id_j])
                                    
                                    if self.id_to_group[group_j] != []:
                                        if self.current_movement[group_j] == id_j:
                                            self.current_movement[group_j] = self.vls[group_j].get_next_vehicle()
                                        else:
                                            self.vls[group_j].remove_vehicle(self.vehicle_helpers[id_j])
                                    else:
                                        del self.id_to_group[group_j]
                                        del self.current_movement[group_j]
                                    # Make new Group
                                    self.id_to_group[self.num_groups] = []
                                    self.id_to_group[self.num_groups].append(self.vehicle_helpers[id_i])
                                    self.id_to_group[self.num_groups].append(self.vehicle_helpers[id_j])
                                    self.agent_to_id[id_i] = self.num_groups
                                    self.agent_to_id[id_j] = self.num_groups
                                    if self.num_groups not in self.vls.keys():
                                        self.vls[self.num_groups] = VLS(self.id_to_group[self.num_groups])
                                    if self.num_groups not in self.current_movement:
                                        self.current_movement[self.num_groups] = self.vls[self.num_groups].get_next_vehicle()
                                    if self.vehicle_helpers[id_i].current_intersection != None:
                                        self.group_to_i_r[self.num_groups] = self.vehicle_helpers[id_i].current_intersection
                                    else:
                                        self.group_to_i_r[self.num_groups] = self.vehicle_helpers[id_i].current_route_section
                                    self.num_groups += 1
                            else:
                                # self.group_to_i_r[group_i] == self.group_to_i_r[group_j]
                                self.id_to_group[self.num_groups] = []
                                self.id_to_group[self.num_groups].append(self.vehicle_helpers[id_i])
                                self.id_to_group[self.num_groups].append(self.vehicle_helpers[id_j])
                                self.agent_to_id[id_i] = self.num_groups
                                self.agent_to_id[id_j] = self.num_groups
                                if self.num_groups not in self.vls.keys():
                                    self.vls[self.num_groups] = VLS(self.id_to_group[self.num_groups])
                                if self.num_groups not in self.current_movement:
                                    self.current_movement[self.num_groups] = self.vls[self.num_groups].get_next_vehicle()
                                if self.vehicle_helpers[id_i].current_intersection != None:
                                    self.group_to_i_r[self.num_groups] = self.vehicle_helpers[id_i].current_intersection
                                else:
                                    self.group_to_i_r[self.num_groups] = self.vehicle_helpers[id_i].current_route_section
                                self.num_groups += 1

    def check_in_intersection(self, curr_gps):
        # Determine which vehicles are inside which intersections
        for intersection in self.traffic_manager.intersections.values():
            if self.traffic_manager.check_if_within_intersection(
                    curr_gps, intersection.tower_ID
            ):
                return True
        return False
    
    def within_LOS(self, id_i):
        n_ac = self.bs.traf.lat.shape[0]
        d = (
            geo.kwikdist_matrix(
                np.repeat(self.bs.traf.lat, n_ac),
                np.repeat(self.bs.traf.lon, n_ac),
                np.tile(self.bs.traf.lat, n_ac),
                np.tile(self.bs.traf.lon, n_ac),
            ).reshape(n_ac, n_ac)
            * geo.nm
        )
        i = self.bs.traf.id2idx(id_i)
        for j in range(self.bs.traf.lat.shape[0]):
            id_j = self.bs.traf.id[j]
            if id_i == id_j:
                continue
            if (not self.bs.traf.active[j]) or (self.vehicle_helpers[id_j].initial_request_granted == False):  # self.vehicle_helpers[id_j].initial_request_granted == False:
                continue
            dist = d[i, j]
            if dist <= self.nmac_distance:
                # print("within NMAC: ", id_i, id_j)
                return True
        return False
                        
    
    def distance_function(self, lat1, lon1, lat2, lon2):
        radius = 6371 # km

        dlat = math.radians(lat2-lat1)
        dlon = math.radians(lon2-lon1)
        a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = radius * c

        return d
    
    def distance_and_direction(self, aircraft_lat, aircraft_lon, aircraft_hdg, intersection):
        location = self.traffic_manager.intersections[intersection].location
        if intersection == 'I1':
            intersection_lat = location[0][0]
            intersection_lon = location[0][1]
        else:
            intersection_lat = location[0]
            intersection_lon = location[1]

        R = 6371.0  # Radius of the Earth in kilometers

        d_lat = radians(intersection_lat - aircraft_lat)
        d_lon = radians(intersection_lon - aircraft_lon)

        a = sin(d_lat / 2)**2 + cos(radians(aircraft_lat)) * cos(radians(intersection_lat)) * sin(d_lon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c

        d_lon = radians(intersection_lon - aircraft_lon)
        bearing = atan2(sin(d_lon) * cos(radians(intersection_lat)),
                        cos(radians(aircraft_lat)) * sin(radians(intersection_lat)) -
                        sin(radians(aircraft_lat)) * cos(radians(intersection_lat)) * cos(d_lon))
        bearing = (bearing + 2 * 3.14159) % (2 * 3.14159)
        bearing = degrees(bearing)

        if abs(bearing - aircraft_hdg) < 90 or abs(bearing - aircraft_hdg) > 270:
            direction = 1
        else:
            direction = -1

        return direction, distance
                        
    

    def state_update(
            self,
            traf,
            a=None,
            policy=None,
            value=None,
            pp=None,
            tp=None,
            init=False,
            tm_response: dict = None,
    ):
        # current number of a/c in bluesky sim
        n_ac = traf.lat.shape[0]
        rew = {}
        state = {}
        done = {}
        info = {}

        #
        self.action_override = []

        ## creating an index for the unique aircraft
        index = np.arange(n_ac).reshape(-1, 1)

        ## calculating the distance from each aircraft to all others. Will result in a n_ac x n_ac matrix
        # d = geo.latlondist_matrix(np.repeat(traf.lat,n_ac), np.repeat(traf.lon,n_ac), np.tile(traf.lat,n_ac), np.tile(traf.lon,n_ac)).reshape(n_ac,n_ac)*geo.nm ## conver to meters
        # argsort = np.array(np.argsort(d, axis=1))
        d = (
                geo.kwikdist_matrix(
                    np.repeat(traf.lat, n_ac),
                    np.repeat(traf.lon, n_ac),
                    np.tile(traf.lat, n_ac),
                    np.tile(traf.lon, n_ac),
                ).reshape(n_ac, n_ac)
                * geo.nm
        )
        argsort = np.array(np.argsort(d, axis=1))

        # transform all aircraft lon/lat positions
        coord_transform = self.transformer.transform(traf.lon, traf.lat)

        geometries = MultiLineString(
            [
                [(coord_transform[0][i], coord_transform[1][i])]
                + [
                    self.transformer.transform(
                        traf.ap.route[i].wplon[j],
                        traf.ap.route[i].wplat[j],
                    )
                    for j in range(
                        traf.ap.route[i].iactwp,
                        len(traf.ap.route[i].wplon),
                    )
                ]
                for i in range(traf.lat.shape[0])
            ]
        )
        # geometries = MultiLineString(
        #     [
        #         [[traf.lon[i], traf.lat[i]], [traf.ap.route[i].wplon[-1], traf.ap.route[i].wplat[-1]]]
        #         for i in range(traf.lat.shape[0])
        #     ]
        # )
        avg_noise_vals = {}
        noise_vals = {}
        for id_val in self.bs.traf.id:
            idx_val = self.bs.traf.id2idx(id_val)
            ac_alt = self.meters_to_feet(self.bs.traf.alt[idx_val])
            
            if ac_alt <= self.min_alt:
                ac_noise = 0
            else:
                ac_noise = self.a_0 +  (self.a_1 * math.log10(ac_alt)) + (self.a_2 * ((math.log10(ac_alt))**2))
            if self.vehicle_helpers[id_val].current_route_section != None:
                if self.vehicle_helpers[id_val].current_route_section not in noise_vals.keys():
                    noise_vals[self.vehicle_helpers[id_val].current_route_section] = []
                noise_vals[self.vehicle_helpers[id_val].current_route_section].append(10**(ac_noise / 10))
            if self.vehicle_helpers[id_val].current_intersection != None:
                if self.vehicle_helpers[id_val].current_intersection not in noise_vals.keys():
                    noise_vals[self.vehicle_helpers[id_val].current_intersection] = []
                noise_vals[self.vehicle_helpers[id_val].current_intersection].append(10**(ac_noise / 10))
        # print("Noise Values: ", noise_vals)
        for route_id in noise_vals.keys():
            if np.sum(noise_vals[route_id]) == 0:
                total_noise_impact = 0
                noise_increase = 0
            else:
                total_noise_impact = 10 * math.log10(np.sum(noise_vals[route_id]))
                if route_id not in self.ambient_noise_level.keys():
                    ambient_noise_val = 40
                else:
                    ambient_noise_val = self.ambient_noise_level[route_id]
                noise_increase = total_noise_impact - ambient_noise_val
            if noise_increase < 0:
                noise_increase = 0
            if noise_increase != 0:
                if route_id not in self.average_noise_increase.keys():
                    self.average_noise_increase[route_id] = []
                self.average_noise_increase[route_id].append(noise_increase)
                avg_noise_vals[route_id] = noise_increase
            if noise_increase >= self.max_noise_increase:
                self.max_noise_increase = noise_increase
        # if len(avg_noise_vals) > 0:
        #     self.average_noise_increase.append(avg_noise_vals)
        # self.average_noise_increase.append(np.mean(avg_noise_vals))
        


        self.prev_id_copy = self.bs.traf.id.copy()
        new_LOS_pairs = []
        # looping over the ownships
        for i in range(d.shape[0]):
            # ownship ID
            id_ = traf.id[i]

            if id_ not in self.acInfo:
                self.acInfo[id_] = {
                    "NMAC": [],
                    "Lat": [],
                    "Lon": [],
                    "Spd": [],
                    "Action": [],
                    "time": [],
                }

            # if the aircraft has not taken off, skip
            if not traf.active[i]:
                continue
            ### PATH LENGTH INFO ###
            ownship_obj = geometries.geoms[i]

            dist = ownship_obj.length

            """ Apply the result of the vehicle request WHERE? """

            ## Converting ownship lat/lon to UTM coords
            xEast_own, yNorth_own = (
                coord_transform[0][i],
                coord_transform[1][i],
            )  # self.transformer.transform(traf.lon[i], traf.lat[i])

            prev_action_own = 1  # maintain
            if a is not None:
                if id_ in a:
                    prev_action_own = a[id_]

            rew[id_] = 0
            ### Start of VLS Code ###
                           # Change this with further testing
            # print(self.current_movement)
            # self.vehicle_helpers[id_].enter_request_status = False
            # print(self.id_to_group)
            if self.vls_active:
                if id_ in self.vls_modifications_halt:
                    rew[id_] += self.rewardLOS
                if id_ in self.vls_modifications_slow:
                    # print("speed change penalty")
                    rew[id_] -= self.shieldPenalty
                    


            ## Hold the vehicles that do not move
            # Step 1. Create a VLS for each communication group
            # Step 2. Only change the trajectories for agents inside intersections
            # for i in range(self.bs.traf.lat.shape[0]):
                # Step 1, Determine Communication Radius
            ### END of VLS Code ###
            # own_glat, own_glon = traf.ap.route[i].wplat[-1], traf.ap.route[i].wplon[-1]
            # dist = geo.kwikdist(traf.lat[i], traf.lon[i], own_glat, own_glon) * geo.nm  ##meters
            if self.traffic_manager_active:
                # TODO: The following check will not work for vehicles that exist but arent currently requesting.
                # TODO: tm_response is only populated with vehicles that are requesting
                # TODO: the other option is to set the distance value that indicates no request
                if id_ in tm_response:
                    # try:
                    response, distance = tm_response[id_]  # 0 = denied, 1 = approved
                    # here
                    # except:
                    #    response = tm_response[id_]
                    #    distance = -1
                    response = int(response)

                    # if response == 0:
                    #    rew[id_] += -self.clearancePenalty

                    # hard action for when aircraft is too close to the boundary and SA did not slow down
                    # TODO: Store a flag of the vehicle that needs to be overwritten
                    if distance != -1 and distance < 250 and response == 0:
                        # self.bs.stack.stack(f"SPD {id_} 0")
                        print("Clear Penalty")
                        rew[id_] += -self.clearancePenalty
                        self.action_override.append(id_)
                else:
                    response = 2  # no request
                    distance = 0
                # TODO: Set a dist value that indicates the vehicle is not requesting
                ambient_noise = 40
                current_route_section_name = self.vehicle_helpers[id_].current_route_section
                current_intersection_name = self.vehicle_helpers[id_].current_intersection
                if current_route_section_name != None and current_route_section_name in self.ambient_noise_level.keys():
                    ambient_noise = self.ambient_noise_level[current_route_section_name]
                else:
                    if current_intersection_name != None and current_intersection_name in self.ambient_noise_level.keys():
                        ambient_noise = self.ambient_noise_level[current_intersection_name]
                own_state = np.array(
                    [
                        dist,
                        traf.cas[i],
                        traf.ax[i],
                        traf.hdg[i],
                        traf.alt[i],
                        ambient_noise,
                        self.LOS,
                        prev_action_own,
                        response,
                        distance,
                    ]
                ).reshape(1, self.observation_space.shape[0])
                

            else:
                ambient_noise = 40
                current_route_section_name = self.vehicle_helpers[id_].current_route_section
                current_intersection_name = self.vehicle_helpers[id_].current_intersection
                if current_route_section_name != None:
                    ambient_noise = self.ambient_noise_level[current_route_section_name]
                else:
                    if current_intersection_name != None:
                        ambient_noise = self.ambient_noise_level[current_intersection_name]
                
                own_state = np.array(
                    [
                        dist,
                        traf.cas[i],
                        traf.ax[i],
                        traf.hdg[i],
                        traf.alt[i],
                        ambient_noise,
                        self.LOS,
                        prev_action_own,
                    ]
                ).reshape(1, self.observation_space.shape[0])

            ## check normalization values
            self.normalization_check(x=xEast_own, y=yNorth_own, d=dist)
            own_state = (own_state - self.observation_space.low) / (
                    self.observation_space.high - self.observation_space.low
            )

            self.acInfo[id_]["Lat"].append(traf.lat[i])
            self.acInfo[id_]["Lon"].append(traf.lon[i])
            self.acInfo[id_]["Spd"].append(traf.cas[i])
            self.acInfo[id_]["NMAC"].append(
                0
            )  # place holder 0 that is overwritten later if NMAC occurred
            self.acInfo[id_]["time"].append(self.bs.sim.simt)
            if a is not None and id_ in a:
                self.acInfo[id_]["Action"].append(a[id_])

            else:
                self.acInfo[id_]["Action"].append(1)  # "hold"

            done[id_] = False
            info[id_] = None

            # if self.traffic_manager_active:
            #     if id_ in self.exiting_vehicles:
            #         done[id_] = True
            # else:
            ## made it to the goal
            if dist < self.dGoal:
                # if self.traffic_manager_active:
                #    if id_ in self.exiting_vehicles:
                #        done[id_] = True
                #
                # else:

                if self.traffic_manager_active:
                    intersection = self.vehicle_helpers[id_].current_intersection
                    if intersection != None:
                        if id_ in self.traffic_manager.intersections[
                            intersection
                        ].accepted:
                            self.traffic_manager.intersections[
                                intersection
                            ].accepted.remove(id_)
                            self.traffic_manager.intersections[intersection].set_volume()
                left_time = self.bs.sim.simt
                self.full_travel[id_] = left_time - self.travel_start[id_]
                # print(f"Aircraft {id_} is exiting with exit time {self.full_travel}")
                
                done[id_] = True


            # is this a GA aircraft?
            if id_[0:2] == "GA":
                if done[id_]:
                    self.bs.stack.stack("DEL {}".format(id_))
                # should prevent non-coop from formining a state
                # but state info of non-coop will still be available
                # to the remaining coop aircraft
                continue

            reward_count = False
            closest_count = self.agent.max_agents
            intruder_state = None
            alt_own = self.bs.traf.alt[i]
            alt_own_ft = self.meters_to_feet(alt_own)
            intruder_noise_vals = {} 
            los_flag = False
            intruder_flag = False
            for j in range(len(argsort[i])):
                index = int(argsort[i][j])

                # The first entry will be: intruder == ownship so we need to skip
                if i == index:
                    continue

                if not traf.active[index]:
                    continue

                # -1 index so that it is the true goal location even with multiple waypoints
                # glat, glon = traf.ap.route[index].wplat[-1], traf.ap.route[index].wplon[-1]
                # dist = geo.latlondist(traf.lat[index],traf.lon[index],glat,glon) # meters
                # dist = geo.kwikdist(traf.lat[index], traf.lon[index], glat, glon) * geo.nm
                intruder_obj = geometries.geoms[index]
                dist = intruder_obj.length
                alt_intruder = self.bs.traf.alt[j]
                alt_intruder_ft = self.meters_to_feet(alt_intruder)
                id_j = self.bs.traf.id[index]
                dist_with_alt = np.sqrt((d[i, index]**2) + ((alt_own - alt_intruder)**2)) # Replace d[i, index] with this value
                # intruder to be removed
                if dist < self.dGoal and dist_with_alt > self.LOS:
                    continue

                # if the intruder is > 750 meters (0.5 nm) away, skip it.
                if (
                        dist_with_alt > self.intruderThreshold
                ):  # TODO: This is a hyperparameter that needs to be moved to a config file
                    continue
                if alt_intruder != 0:
                    intruder_noise_vals[j] = self.a_0 +  (self.a_1 * math.log10(alt_intruder_ft)) + (self.a_2 * ((math.log10(alt_intruder_ft))**2))
                else:
                    intruder_noise_vals[j] = 0
                # if the ownship and intruder do not intersect and they are not on the same route
                
                origin_i = id_[1:3]
                dest_i = id_[3:5]

                origin_j = id_j[1:3]
                dest_j = id_j[3:5]

                

                if not ownship_obj.intersects(intruder_obj):
                    continue
                
                ## At this point. The intruder is only considered if the routes intersect
                ## Now I need to take care of tracks on the same route/lane

                # ilon, ilat = list(ownship_obj.intersection(intruder_obj).coords)[0]
                #
                # dist_int_inter = geo.kwikdist(traf.lat[index], traf.lon[index], ilat, ilon)  # nautical miles
                # dist_own_inter = geo.kwikdist(traf.lat[i], traf.lon[i], ilat, ilon)  # nautical miles

                if dist_with_alt < self.LOS and not reward_count:
                    if self.safety_train:
                        rew[id_] += self.rewardLOS
                    self.los_counter += 1
                    los_flag = True
                    # if self.vehicle_helpers[id_].within_intersection == True:
                    curr_m = None
                    if id_ in self.agent_to_id.keys():
                        v_group = self.agent_to_id[id_]
                        curr_m = self.current_movement[v_group]
                    id_j = self.bs.traf.id[index]
                    if (id_, id_j) not in self.prev_LOS_pairs and (id_j, id_) not in self.prev_LOS_pairs and (id_, id_j) not in new_LOS_pairs and (id_j, id_) not in new_LOS_pairs:
                        self.los_events += 1
                    new_LOS_pairs.append((id_, id_j))
                    reward_count = True
                    info[id_] = 1
                    # print("Id check: ", id_, a)
                    if id_ in a:
                        self.acInfo[id_]["NMAC"].append(1)
                    print("LOS has occurred: ", id_, id_j)
                else:
                    self.acInfo[id_]["NMAC"].append(0)

                if self.safety_train:
                    if not reward_count:
                        if dist_with_alt < self.maxRewardDistance and dist_with_alt > self.LOS:
                            intruder_flag = True
                            rew[id_] += -self.rewardAlpha + self.rewardBeta * (dist_with_alt)
                            reward_count = True

                ## Converting intruder lat/lon to UTM coords
                xEast_int, yNorth_int = (
                    coord_transform[0][index],
                    coord_transform[1][index],
                )

                distIntGoal = dist

                relX = xEast_int - xEast_own
                relY = yNorth_int - yNorth_own
                prev_action_int = 1
                if a is not None:
                    if traf.id[index] in a:
                        prev_action_int = a[traf.id[index]]

                int_state = np.array(
                    [
                        relX,
                        relY,
                        distIntGoal,
                        traf.cas[index],
                        traf.ax[index],
                        traf.hdg[index],
                        traf.alt[index],
                        d[i, index],
                        prev_action_int,
                    ]
                ).reshape(1, self.context_space.shape[0])

                self.normalization_check(x=xEast_int, y=yNorth_int, d=distIntGoal)

                int_state = (int_state - self.context_space.low) / (
                        self.context_space.high - self.context_space.low
                )

                if intruder_state is None:
                    intruder_state = int_state

                else:
                    intruder_state = np.append(intruder_state, int_state, axis=0)

                closest_count -= 1

                if closest_count == 0:
                    break

            if closest_count != 0:
                remaining = np.zeros((closest_count, self.intruder_obs_dim))
                if intruder_state is None:
                    intruder_state = remaining

                else:
                    intruder_state = np.append(intruder_state, remaining, axis=0)

                state[id_] = {
                    "ownship_obs": own_state.reshape(1, self.ownship_obs_dim),
                    "intruder_obs": intruder_state.reshape(
                        1, self.agent.max_agents, self.intruder_obs_dim
                    ),
                }

            else:
                state[id_] = {
                    "ownship_obs": own_state.reshape(1, self.ownship_obs_dim),
                    "intruder_obs": intruder_state.reshape(
                        1, self.agent.max_agents, self.intruder_obs_dim
                    ),
                }
            ### NOISE REWARD IMPLEMENTATION ###
            ## 1: Single Event Noise Calculation
            next_route_section_name = self.vehicle_helpers[id_].next_route_section
            current_route_section_name = self.vehicle_helpers[id_].current_route_section
            current_intersection_name = self.vehicle_helpers[id_].current_intersection
            ambient_noise = 40
            if current_route_section_name != None and current_route_section_name in self.ambient_noise_level.keys():
                ambient_noise = self.ambient_noise_level[current_route_section_name]
            else:
                if current_intersection_name != None and current_intersection_name in self.ambient_noise_level.keys():
                    ambient_noise = self.ambient_noise_level[current_intersection_name]
            if alt_own > 0:
                current_aircraft_noise = self.a_0 +  (self.a_1 * math.log10(alt_own_ft)) + (self.a_2 * ((math.log10(alt_own_ft))**2))
                # current_aircraft_noise = (self.noise_const_a * math.log(alt_own)) + self.noise_const_b
            else:
                current_aircraft_noise = 0
            ## 2: Group Event Noise
            # inner_log_val = (10**(ambient_noise / 10))
            inner_log_val = 0 # (10**(current_aircraft_noise / 10))
            for single_noise_val in intruder_noise_vals.values():
                inner_log_val += (10**(single_noise_val / 10))
            if inner_log_val != 0:
                total_noise_impact = 10 * math.log10(inner_log_val)
            else:
                total_noise_impact = 0
            # This is a test to see if this is a viable reward formulation
            # total_noise_factored = current_aircraft_noise - total_noise_impact

            ## Old Version:
            total_noise_factored_own = ambient_noise - current_aircraft_noise
            best_case_noise = self.a_0 +  (self.a_1 * math.log10(self.max_alt)) + (self.a_2 * ((math.log10(self.max_alt))**2))
            best_case_noise_increase = ambient_noise - best_case_noise
            noise_increase_over_best_case = total_noise_factored_own # - best_case_noise_increase
            total_noise_factored_int = (ambient_noise - current_aircraft_noise - (10 * math.log10(10**((total_noise_impact - current_aircraft_noise)/10)+1)))
            # Ambient noise needs to be included
            # Total noise is around ~ -26per time step. Max negative reward per time step is 
            total_noise_factored = (self.alpha * noise_increase_over_best_case) + (self.beta * total_noise_factored_int)
            if total_noise_factored >= 0:
                total_noise_factored = 0
            
            
            # print("Total Noise Value: ", total_noise_factored, current_aircraft_noise, total_noise_impact, ambient_noise)
            # 10 * math.log10(sum((10**(single_noise_val / 10)))) - ambient noise
            # log((scaling factor 1) * pov_aircraft_noise + (scaling factor 2) * other_aircraft_noise(sum of all individual noise values)) - ambient noise
            # 
            # aircraft_noise
            # single event noise
            # ambient noise level, cumulative noise of the other aircraft
            # R(pov_aircraft) = (scaling factor 1 * own_noise) - ((scaling factor 2) * other_aircraft_noise)) - ambient_noise
            # POV Aircraft | Ambient Noise + Noise of Other Aircraft
            if total_noise_factored >= 0:
                total_noise_factored = 0
            if self.noise_train:
                rew[id_] += (self.gamma * total_noise_factored)
            # if alt_own_ft > 800  and (-1 * (self.gamma * total_noise_factored) < self.altChangePenalty):
            #     print("noise reward: ", ambient_noise, current_aircraft_noise, (self.gamma * total_noise_factored), self.altChangePenalty)
            # print("Checking New Noise Formulation: ", alt_own, total_noise_factored_own, best_case_noise_increase, noise_increase_over_best_case, total_noise_factored, self.gamma * total_noise_factored)


            ### NOISE METRIC: Maximum Increase of single time-step Noise (over Ambient Value) over all Areas in the Environment
            ### Combine Average and Maximum
            ### END OF NOISE REWARD IMPLEMENTATION ###
            speed_change_flag = False
            alt_change_flag = False
            if not init and not done[id_]:
                if a is not None:
                    if id_ in a.keys():
                        a_spd_val = int(a[id_]/3)
                        a_alt_val = int(a[id_])%3
                        if a_spd_val == 0 or a_spd_val == 2:
                            rew[id_] += -self.speedChangePenalty
                            speed_change_flag = True
                            self.speed_change_counter += 1
                        if a_alt_val == 2:
                            rew[id_] += -self.altChangePenalty
                            alt_change_flag = True
                            self.alt_change_counter += 1
                        rew[id_] += -self.stepPenalty * (((18 - self.bs.traf.tas[i])/18))## step penalty
                        # print("Step Penalty: ", -self.stepPenalty * (((18 - self.bs.traf.tas[i])/18) + ((self.max_alt - alt_own_ft) / self.max_alt)), self.bs.traf.tas[i], alt_own_ft)
            # print("End of Reward Calculation: ", id_, los_flag, intruder_flag, speed_change_flag, alt_change_flag, rew[id_])
        self.prev_LOS_pairs = new_LOS_pairs.copy()
        return state, rew, done, info

    def run_one_iteration(self, weights):
        """
        2022/11/1 modify the policy implementation and introduce the non-cooperative behaviors
        """
        # print("Running one iteration")
        if self.agent.equipped:
            self.agent.model.set_weights(weights)
        # self.agent.data = {}
        self.agent.reset()
        self.step_counter = 0
        # self.shield_counter = 0
        if self.episode_done:
            obs = self.reset()
            self.nmacs = 0
            self.total_ac = 0
        else:
            obs = self.last_obs

        while True:
            if len(obs) > 0:
                action, policy, value = self.agent.predict(
                    obs, self.non_coop_tag, self.LControl_lst, self.LComm_lst
                )

            else:
                action, policy, value = {}, {}, {}
            next_obs, rew, term, term_type = self.step(action, policy, value)

            next_obs = self.store_data(
                obs, action, rew, next_obs, term, term_type, policy, value
            )

            obs = next_obs

            if term["__all__"] or self.step_counter >= self.max_steps:
                self.last_obs = next_obs

                if term["__all__"]:
                    self.episode_done = True

                if self.step_counter >= self.max_steps:
                    self.step_counter = 0

                #     # Need to process remaining entries in self.memory to self.data
                for id_ in self.agent.memory.keys():
                    # if the id_ has already been processed then skip it
                    if id_ in self.agent.data.keys():
                        continue
                    self.agent.process_memory(id_)

                # # TODO: is this necessary?
                for key in self.agent.data.keys():
                    if type(self.agent.data[key]) == list:
                        self.agent.data[key] = np.concatenate(
                            self.agent.data[key], axis=0
                        )

                self.agent.data["nmacs"] = self.nmacs
                self.agent.data["total_ac"] = self.total_ac

                if self.episode_done:  # self.run_type == "eval":
                    self.agent.data["aircraft"] = self.acInfo

                # # TODO: Do someething ike this
                # self.agent.data["intersection_metrics"] = self.intersection_metrics
                # ************

                # if not "raw_reward" in self.agent.data:
                #     self.agent.data["raw_reward"] = np.array([0.0])

                self.agent.data["environment_done"] = self.episode_done
                self.agent.data["los_counter"] = self.los_counter
                self.agent.data["los_events"] = self.los_events
                self.agent.data["shield_events"] = self.shield_counter
                self.agent.data["shield_events_i"] = self.shield_counter_intersect
                self.agent.data["shield_events_r"] = self.shield_counter_route
                self.agent.data["max_halting_time"] = self.max_halt_time
                self.agent.data["halting_time_list"] = self.halting_times
                self.agent.data["max_noise_increase"] = self.max_noise_increase
                # print("average_noise_increase: ", self.average_noise_increase)
                self.agent.data["avg_noise_increase"] = self.average_noise_increase
                self.agent.data["speed_change_counter"] = self.speed_change_counter
                self.agent.data["alt_change_counter"] = self.alt_change_counter
                self.agent.data["scenario_file"] = self.scen_file_temp
                if len(self.full_travel) == 0:
                    self.agent.data["max_travel_time"] = 0
                else:
                    # print("Returning Full Travel: ", self.full_travel, max(self.full_travel.values()))
                    self.agent.data["max_travel_time"] = max(self.full_travel.values())
                self.agent.data["full_travel_times"] = self.full_travel
                data_ID = ray.put([self.agent.data, self.id])

                return data_ID

            # if self.step_counter >= self.max_steps:

            #     self.last_obs = next_obs

            #     # Need to process remaining entries in self.memory to self.data
            #     for id_ in self.memory.keys():

            #         # if the id_ has already been processed then skip it
            #         if id_ in self.data.keys():
            #             continue

            #         self.process_memory(id_)

            #     # convert to array from list of arrays
            #     for model in self.data.keys():
            #         for key in self.data[model].keys():
            #             if type(self.data[model][key]) == list:
            #                 self.data[model][key] = np.concatenate(self.data[model][key], axis=0)

            #     self.data_ID = ray.put([self.data, self.id])
            #     del self.data

            #     return self.data_ID  # self.data, self.id

    def store_data(self, obs, action, rew, next_obs, term, term_type, policy, value):
        obs_updated = copy(next_obs)
        for ac_id in obs.keys():
            self.agent.store_step(
                ac_id, obs, action, rew, next_obs, term, policy, value
            )

            if term[ac_id]:
                self.total_ac += 1

                # did an NMAC occur
                if 1 in self.acInfo[ac_id]["NMAC"]:
                    # if term_type[ac_id] == 1:
                    group = groupby(self.acInfo[ac_id]["NMAC"])
                    group = np.array([x[0] for x in group])
                    self.nmacs += sum(group)
                self.bs.stack.stack("DEL {}".format(ac_id))
                # TODO: MIGHT NEED TO REMOVE VEHICLES HERE....
                # if ac_id in self.exiting_vehicles: # TODO: NOT SUSTAINABLE. THIS WILL RESULT IN AIRSPACE VOLUME NEVER BEING REDUCED

                # if self.traffic_manager_active:

                # if self.traffic_manager_active:

                #     try: # ?????
                #         self.exiting_vehicles.remove(ac_id)

                #     except:
                #         import ipdb;ipdb.set_trace()
                del obs_updated[ac_id]

        return obs_updated

    def normalization_check(self, x=None, y=None, d=None):
        traf = self.bs.traf

        if len(traf.lat) == 0:
            return

        if traf.cas.max() > self.tas_max:
            self.tas_max = traf.cas.max()

        if traf.cas.min() < self.tas_min:
            self.tas_min = traf.cas.min()

        if traf.ax.max() > self.ax_max:
            self.ax_max = traf.ax.max()
        if traf.ax.min() < self.ax_min:
            self.ax_min = traf.ax.min()

        if x is not None:
            if x < self.min_x:
                self.min_x = x

            elif x > self.max_x:
                self.max_x = x

            # prevent numerical normalization errors
            if self.max_x == self.min_x:
                self.max_x += 1e-4

        if y is not None:
            if y < self.min_y:
                self.min_y = y

            elif y > self.max_y:
                self.max_y = y


            # prevent numerical normalization errors
            if self.max_y == self.min_y:
                self.max_y += 1e-4

        if d is not None:
            if d > self.max_d:
                self.max_d = d

        ## utm position, dist goal, speed, acceleration, heading, LOS distance, transiting from utm position, transiting heading, transiting to utm position
        # clearance denied, cleared, no clearance request
        if self.traffic_manager_active:
            self.observation_space = Box(
                np.array([0, self.tas_min, self.ax_min, 0, 0, 40, 0, 0, 0, 0]),
                np.array(
                    [
                        self.max_d,
                        self.tas_max,
                        self.ax_max,
                        360,
                        self.max_alt,
                        60,
                        self.max_d,
                        2,
                        2,
                        self.max_d,
                    ]
                ),
                dtype=np.float64,
            )

        else:
            ## utm position, dist goal, speed, acceleration, heading, LOS distance
            self.observation_space = Box(
                np.array([0, self.tas_min, self.ax_min, 0, 0, 40, 0, 0]),
                np.array([self.max_d, self.tas_max, self.ax_max, 360, self.max_alt, 60,self.max_d, 2]),
                dtype=np.float64,
            )
        ## utm position, dist goal, speed, acceleration, heading, distance ownship to intruder, distance intruder intersection, distance ownship to intersection, transiting from utm position, transiting heading, transiting to utm position
        self.context_space = Box(
            np.array(
                [
                    -(self.max_x - self.min_x),
                    -(self.max_y - self.min_y),
                    0,
                    self.tas_min,
                    self.ax_min,
                    0,
                    0,
                    0,
                    0,
                ]
            ),
            np.array(
                [
                    self.max_x - self.min_x,
                    self.max_y - self.min_y,
                    self.max_d,
                    self.tas_max,
                    self.ax_max,
                    360,
                    self.max_alt,
                    self.max_d,
                    2,
                ]
            ),
            dtype=np.float64,
        )
