import ray
import os
import gin
import argparse
from D2MAV_A.agent import Agent
from D2MAV_A.runner import Runner
from bluesky.tools import geo
from copy import deepcopy
import pandas as pd
import random
import time
import platform
import numpy as np
import logging

class Communication_Node():
    import tensorflow as tf
    import bluesky as bs
    def __init__(self, scenario_file):
        self.bs.init(mode="sim", configfile=f'/home/suryamurthy/UT_Autonomous_Group/vehicle_level_shielding/settings.cfg')
        self.bs.net.connect()
        self.reset(scenario_file)
    def reset(self, scenario_file):
        self.bs.stack.stack(r'IC ' + scenario_file)
        self.bs.stack.stack("FF")
        self.bs.sim.step()  # bs.sim.simt = 0.0 AFTER the call to bs.sim.step()
        self.bs.stack.stack("FF")
    def send_command(self, cmd):
        self.bs.stack.stack(cmd)
        self.bs.net.update()
    def update(self):
        self.bs.sim.step()
        self.bs.net.update()

def generate_scenario(path, num_scenarios, num_aircraft, dep_interval):
    for n_s in range(0, num_scenarios):
        print(path + f"/aircraft_"+str(num_aircraft)+"/test_case_"+str(n_s)+".scn")
        folder_path = path + f"/aircraft_"+str(num_aircraft)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        f = open(path + f"/aircraft_"+str(num_aircraft)+"/test_case_"+str(n_s)+".scn", "w")

        f.write("00:00:00.00>TRAILS ON \n")
        f.write("\n")
        f.write("00:00:00.00>PAN 32.77173371	-96.83678249 \n")
        f.write("\n")
        paths_of_interest = ["I30L", "I30R", "BUSR", "BUSL", "CENR", "CENL", "TWYL", "TWYR", "635R", "635L", "H12L", "H12R", "NHWL", "NHWR", "COLL", "COLR", "I35L", "I35R"]
        start_offset_times = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        # Load Route Names for Generation
        waypoint_data = pd.read_csv(
            f'/home/suryamurthy/UT_Autonomous_Group/vehicle_level_shielding/bluesky/resources/navdata/nav.dat',
            delimiter='\t')
        waypoint_names = waypoint_data.iloc[:, 7]
        route_waypoints = {}
        for i in waypoint_names:
            route_id = i[0:4]
            if route_id not in route_waypoints.keys():
                route_waypoints[route_id] = []
            route_waypoints[route_id].append(i)
        route_dict = {}
        starting_time_dict  = {}
        for path_idx in range(len(paths_of_interest)):
            path = paths_of_interest[path_idx]
            route_dict[path] = num_aircraft
            starting_time_dict[path] = start_offset_times[path_idx]
        # for i in range(num_aircraft):
        #     # plane="A"+str(i)
        #     route_id = random.choice(list(paths_of_interest))
        #     while "NHW" in route_id:
        #         route_id = random.choice(list(route_waypoints.keys()))
        #     route_length = len(route_waypoints[route_id])
        #     if route_id not in route_dict:
        #         route_dict[route_id] = 0
        #     route_dict[route_id]+= 1
        print(route_dict)
        offset_counter = 0
        while route_dict != {}:
            copy_dict = route_dict.copy()
            for route_id in copy_dict.keys():
                plane = "P" + route_id + str(offset_counter)
                time = "00:00:" + str(starting_time_dict[route_id] + offset_counter * dep_interval) + ".00"
                f.write(time + ">CRE " + plane + ",Mavic," + route_id + "1,0,0" + "\n")
                f.write(time + ">ORIG " + plane + " " + route_waypoints[route_id][0] + "\n")
                f.write(time + ">DEST " + plane + " " + route_waypoints[route_id][-1] + "\n")
                f.write(time + ">SPD " + plane + " 30" + "\n")
                f.write(time + ">ALT " + plane + " 400" + "\n")
                for wpt in route_waypoints[route_id]:
                    f.write(time + ">ADDWPT " + plane + " " + wpt + " 400 40" + "\n")
                f.write(time + ">" + plane + " VNAV on \n")
                f.write("\n")
                route_dict[route_id] -= 1
                if route_dict[route_id] == 0:
                    del route_dict[route_id]
            offset_counter += 1

        f.close()

def generate_scenario_routes(path, num_scenarios, num_aircraft, dep_interval, num_routes = 4):
    for n_s in range(0, num_scenarios):
        route_list = ["BUSR", "BUSL", "TWYL", "TWYR", "CENR", "CENL", "I30L", "I30R", "635L", "635R", "COLL", "COLR", "I35L", "I35R"]
        start_offset_times_list = [0, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        paths_of_interest = route_list[0:num_routes]
        start_offset_times = start_offset_times_list[0:num_routes]
        print(path + f"/routes_"+str(num_routes)+"/test_case_"+str(n_s)+".scn")
        folder_path = path + f"/route_"+str(num_routes)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        f = open(path + f"/route_"+str(num_routes)+"/test_case_"+str(n_s)+".scn", "w")

        f.write("00:00:00.00>TRAILS ON \n")
        f.write("\n")
        f.write("00:00:00.00>PAN 32.77173371	-96.83678249 \n")
        f.write("\n")
        
        # Load Route Names for Generation
        waypoint_data = pd.read_csv(
            f'/home/suryamurthy/UT_Autonomous_Group/vehicle_level_shielding/bluesky/resources/navdata/nav.dat',
            delimiter='\t')
        waypoint_names = waypoint_data.iloc[:, 7]
        route_waypoints = {}
        for i in waypoint_names:
            route_id = i[0:4]
            if route_id not in route_waypoints.keys():
                route_waypoints[route_id] = []
            route_waypoints[route_id].append(i)
        route_dict = {}
        starting_time_dict  = {}
        for path_idx in range(len(paths_of_interest)):
            path = paths_of_interest[path_idx]
            route_dict[path] = num_aircraft
            starting_time_dict[path] = start_offset_times[path_idx]
        # for i in range(num_aircraft):
        #     # plane="A"+str(i)
        #     route_id = random.choice(list(paths_of_interest))
        #     while "NHW" in route_id:
        #         route_id = random.choice(list(route_waypoints.keys()))
        #     route_length = len(route_waypoints[route_id])
        #     if route_id not in route_dict:
        #         route_dict[route_id] = 0
        #     route_dict[route_id]+= 1
        print(route_dict)
        offset_counter = 0
        while route_dict != {}:
            copy_dict = route_dict.copy()
            for route_id in copy_dict.keys():
                plane = "P" + route_id + str(offset_counter)
                time = "00:00:" + str(starting_time_dict[route_id] + offset_counter * dep_interval) + ".00"
                f.write(time + ">CRE " + plane + ",Mavic," + route_id + "1,0,0" + "\n")
                f.write(time + ">ORIG " + plane + " " + route_waypoints[route_id][0] + "\n")
                f.write(time + ">DEST " + plane + " " + route_waypoints[route_id][-1] + "\n")
                f.write(time + ">SPD " + plane + " 30" + "\n")
                f.write(time + ">ALT " + plane + " 400" + "\n")
                for wpt in route_waypoints[route_id]:
                    f.write(time + ">ADDWPT " + plane + " " + wpt + " 400 40" + "\n")
                f.write(time + ">" + plane + " VNAV on \n")
                f.write("\n")
                route_dict[route_id] -= 1
                if route_dict[route_id] == 0:
                    del route_dict[route_id]
            offset_counter += 1

        f.close()


# Test Scenario Generation:
intervals = [5, 10, 15, 20, 25]
for num_a in intervals:
    generate_scenario(f'/home/suryamurthy/UT_Autonomous_Group/vehicle_level_shielding/scenarios/generated_scenarios_full', 1, num_a, 15)

# scenario_file = f'C:\Users\surya\PycharmProjects\ISMS_39\ILASMS_func3a-update-routes\scenarios\generated_scenarios\test_case_0.scn'
# # scenario_file = r'C:\Users\surya\PycharmProjects\ISMS_39\ILASMS_func3a-update-routes\scenarios\basic_env.scn'
# # https://github.com/TUDelft-CNS-ATM/bluesky/wiki/navdb
# node_1 = Communication_Node(scenario_file)

# interval_1 = 1000
# interval_2 = 100000
# counter = 0
# counter_2 = 0
# counter_3 = 0
# # Simulation Update Loop: reset and load a new scenario once all vehicles have exited the simulation.
# while 1:
#     # time.sleep(0.01)
#     node_1.update()
























# counter += 1
#     if counter % interval_1 == 0:
#         counter_2 +=1
#         if counter_2 % 2 == 0:
#             print("setting speed to 20")
#             for id in node_1.bs.traf.id:
#                 node_1.send_command(r'SPD ' + id + ' 20')
#                 # node_1.send_command(r'PAN '+ id)
#         else:
#             print("setting speed to 30")
#             for id in node_1.bs.traf.id:
#                 node_1.send_command(r'SPD ' + id + ' 30')
#                 # node_1.send_command(r'PAN ' + id)