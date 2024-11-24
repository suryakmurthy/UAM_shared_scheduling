import ray
import os
import gin
import argparse
from D2MAV_A.agent import Agent
from D2MAV_A.runner import Runner
from copy import deepcopy
import time
import platform
import json
import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter

os.environ["PYTHONPATH"] = os.getcwd()

import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser()
parser.add_argument("--cluster", action="store_true")
parser.add_argument("--learn_action", action="store_true")
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()


@gin.configurable
class Driver:
    def __init__(
            self,
            cluster=False,
            run_name=None,
            scenario_file=None,
            config_file=None,
            num_workers=1,
            iterations=1000,
            simdt=1,
            max_steps=1024,
            speeds=[0, 0, 84],
            LOS=10,
            dGoal=100,
            maxRewardDistance=100,
            intruderThreshold=750,
            rewardBeta=0.001,
            rewardAlpha=0.1,
            speedChangePenalty=0.001,
            shieldPenalty = 0.1,
            rewardLOS=-1,
            stepPenalty=0.002,
            clearancePenalty=0.005,
            gui=False,
            non_coop_tag=0,
            weights_file=None,
            run_type='train',
            traffic_manager_active=True,
            d2mav_active=True,
            protocol_active=False,
            csma_cd_active=False,
            srtf_active=False,
            round_robin_active=False,
            non_compliant_percentage=0,
            testing_scenarios = 100
    ):

        self.cluster = cluster
        self.run_name = run_name
        self.run_type = run_type
        self.num_workers = num_workers
        self.simdt = simdt
        self.iterations = iterations
        self.max_steps = max_steps
        self.speeds = speeds
        self.LOS = LOS
        self.dGoal = dGoal
        self.maxRewardDistance = maxRewardDistance
        self.intruderThreshold = intruderThreshold
        self.rewardBeta = rewardBeta
        self.rewardAlpha = rewardAlpha
        self.speedChangePenalty = speedChangePenalty
        self.shieldPenalty = shieldPenalty
        self.rewardLOS = rewardLOS
        self.stepPenalty = stepPenalty
        self.clearancePenalty = clearancePenalty
        self.scenario_file = scenario_file
        self.config_file = config_file
        self.weights_file = weights_file
        self.gui = gui
        self.action_dim = 4
        self.observation_dim = 6
        self.context_dim = 8
        self.agent = Agent()
        self.agent_template = deepcopy(self.agent)
        self.working_directory = os.getcwd()
        self.non_coop_tag = non_coop_tag
        self.traffic_manager_active = traffic_manager_active
        self.d2mav_active = d2mav_active

        self.protocol_active = protocol_active
        self.csma_cd_active = csma_cd_active
        self.srtf_active = srtf_active
        self.round_robin_active = round_robin_active
        self.testing_scenarios = testing_scenarios
        self.non_compliant_percentage = non_compliant_percentage

        if self.traffic_manager_active:
            self.observation_dim += 2

        self.agent.initialize(tf, self.observation_dim, self.context_dim, self.action_dim)

        if self.run_name is None:
            path_results = "results"
            path_models = "models"
        else:
            path_results = f"results/{self.run_name}"
            path_models = f"models/{self.run_name}"

        os.makedirs(path_results, exist_ok=True)
        os.makedirs(path_models, exist_ok=True)

        self.path_models = path_models
        self.path_results = path_results

    def train(self):

        # Start simulations on actors
        workers = {
            i: Runner.remote(
                i,
                self.agent_template,
                scenario_file=self.scenario_file,
                config_file=self.config_file,
                working_directory=self.working_directory,
                max_steps=self.max_steps,
                simdt=self.simdt,
                speeds=self.speeds,
                LOS=self.LOS,
                dGoal=self.dGoal,
                maxRewardDistance=self.maxRewardDistance,
                intruderThreshold=self.intruderThreshold,
                rewardBeta=self.rewardBeta,
                rewardAlpha=self.rewardAlpha,
                speedChangePenalty=self.speedChangePenalty,
                rewardLOS=self.rewardLOS,
                shieldPenalty=self.shieldPenalty,
                stepPenalty=self.stepPenalty,
                clearancePenalty=self.clearancePenalty,
                gui=self.gui,
                non_coop_tag=self.non_coop_tag,
                traffic_manager_active=self.traffic_manager_active,
                d2mav_active=self.d2mav_active,
                protocol_active=self.protocol_active,
                csma_cd_active = self.csma_cd_active,
                srtf_active = self.srtf_active,
                round_robin_active = self.round_robin_active,
                non_compliant_percentage = self.non_compliant_percentage
            )
            for i in range(self.num_workers)
        }
        # datetimestr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = f"logs/tensorboard/speed_changes_ver_2"
        os.makedirs(save_dir, exist_ok=True)
        logger = SummaryWriter(save_dir)
        rewards = []
        total_nmacs = []
        total_LOS = []
        max_travel_times = []
        full_travel_times = []
        iteration_record = []
        total_transitions = 0
        best_reward = -np.inf

        if self.agent.equipped:
            if self.weights_file is not None:
                self.agent.model.save_weights(self.weights_file)

            weights = self.agent.model.get_weights()
        else:
            weights = []

        runner_sims = [workers[agent_id].run_one_iteration.remote(weights) for agent_id in workers.keys()]
        scenario = 0
        metric_list = []
        speed_change_list = []
        for i in range(self.iterations):

            done_id, runner_sims = ray.wait(runner_sims, num_returns=self.num_workers)
            results = ray.get(done_id)
            # Uncomment this when running with trained model
            transitions, workers_to_remove = self.agent.update_weights(results)
            # transitions = 0

            if self.agent.equipped:
                weights = self.agent.model.get_weights()

            total_reward = []
            mean_total_reward = None
            nmacs = []
            total_ac = []
            LOS_total = 0
            shield_total = 0
            shield_total_intersect = 0 
            shield_total_route = 0
            scenario_file = None
            for result in results:
                data = ray.get(result)

                try:
                    total_reward.append(float(np.sum(data[0]["raw_reward"])))
                except:
                    pass

                if data[0]['environment_done']:
                    nmacs.append(data[0]['nmacs'])
                    total_ac.append(data[0]['total_ac'])

                LOS_total += data[0]['los_events']
                shield_total += data[0]['shield_events']
                shield_total_intersect += data[0]['shield_events_i']
                shield_total_route += data[0]['shield_events_r']
                halting_list = data[0]['halting_time_list']
                full_travel_times_temp = data[0]['full_travel_times'].values()
                max_travel_time = data[0]['max_travel_time']
                scenario_file = data[0]['scenario_file']
                speed_change_list.append(data[0]['speed_change_counter'])

            if total_reward:
                mean_total_reward = np.mean(total_reward)

            for j, nmac in enumerate(nmacs):
                scenario += 1
                print(f"     Scenario Complete     ")
                print("|------------------------------|")
                print(f"| Scenario File:      {scenario_file}      |")
                print(f"| Total Aircraft:   {total_ac[j]}  |")
                roll_mean = np.mean(rewards[-150:])
                # print(f"| Raw Reward: {total_reward[-1:]}  |")
                print(f"| Rolling Mean Reward: {np.round(roll_mean, 1)}  |")
                print(f"| Max Travel Time: {max_travel_time}  |")
                print(f"| Max Halting Time: {max(halting_list)}  |")
                print(f"| Average Halting Time: {np.mean(halting_list)}  |")
                print(f"| Number of LOS Events: {LOS_total}  |")
                print(f"| Number of Shield Events: {shield_total}  |")
                print(f"| Number of Intersection Shield Events: {shield_total_intersect}  |")
                print(f"| Number of Route Shield Events: {shield_total_route}  |")
                print(f"| Number of Speed Changes: {np.mean(speed_change_list)}  |")
                print("|------------------------------|")
                print(" ")
                metric_dict = {}
                metric_dict['reward_value'] = roll_mean
                metric_dict['scenario_num'] = scenario
                metric_dict['shield_total'] = shield_total
                metric_dict['shield_total_intersection'] = shield_total_intersect
                metric_dict['shield_total_route'] = shield_total_route
                metric_dict['max_travel_time'] = max_travel_time
                metric_dict['halting_list'] = halting_list
                metric_dict['full_travel_times'] = full_travel_times_temp
                metric_dict['los'] = LOS_total
                metric_dict['scenario_name'] = scenario_file
                metric_list.append(metric_dict)
                total_nmacs.append(nmac)
                max_travel_times.append(max_travel_time)
                total_LOS.append(LOS_total)
                iteration_record.append(i)
                logger.add_scalar("train/speed_changes", np.mean(speed_change_list), i)

            if mean_total_reward:
                rewards.append(mean_total_reward)
                np.save("{}/reward.npy".format(self.path_results), np.array(rewards))

            if len(nmacs) > 0:
                np.save("{}/nmacs.npy".format(self.path_results), np.array(total_nmacs))
                np.save("{}/iteration_record.npy".format(self.path_results), np.array(iteration_record))

            total_transitions += transitions

            if not mean_total_reward:
                mean_total_reward = 0

            print(f"     Iteration {i} Complete     ")
            print("|------------------------------|")
            print(f"| Mean Total Reward:   {np.round(mean_total_reward, 1)}  |")
            roll_mean = np.mean(rewards[-150:])
            print(f"| Rolling Mean Reward: {np.round(roll_mean, 1)}  |")
            print(f"| Number of LOS Events: {LOS_total}  |")
            print(f"| Number of Shield Events: {shield_total}  |")
            print("|------------------------------|")
            print(" ")

            if self.agent.equipped:
                # print("checking_vals: ", len(rewards), rewards)
                if len(rewards) > 150:
                    if np.mean(rewards[-150:]) > best_reward:
                        best_reward = np.mean(rewards[-150:])
                        self.agent.model.save_weights("{}/best_model.h5".format(self.path_models, i))
                if i%100 == 0:
                    self.agent.model.save_weights("{}/training_model_{}.h5".format(self.path_models, i))

            runner_sims = [workers[agent_id].run_one_iteration.remote(weights) for agent_id in workers.keys()]
        print("Mean Travel Times: ", np.mean(max_travel_times))
        print("Mean number of LOS: ", np.mean(total_LOS))
        print(metric_list)
        with open('log/training/{}.json'.format(self.run_name), 'w') as file:
            json.dump(metric_list, file, indent=4)
            
    def evaluate(self):

        # Start simulations on actors
        workers = {
            i: Runner.remote(
                i,
                self.agent_template,
                scenario_file=self.scenario_file,
                config_file=self.config_file,
                working_directory=self.working_directory,
                max_steps=self.max_steps,
                simdt=self.simdt,
                speeds=self.speeds,
                LOS=self.LOS,
                dGoal=self.dGoal,
                maxRewardDistance=self.maxRewardDistance,
                intruderThreshold=self.intruderThreshold,
                rewardBeta=self.rewardBeta,
                rewardAlpha=self.rewardAlpha,
                speedChangePenalty=self.speedChangePenalty,
                rewardLOS=self.rewardLOS,
                stepPenalty=self.stepPenalty,
                gui=self.gui,
                traffic_manager_active=self.traffic_manager_active,
                protocol_active=self.protocol_active,
                csma_cd_active = self.csma_cd_active,
                srtf_active = self.srtf_active,
                round_robin_active = self.round_robin_active,
                non_compliant_percentage = self.non_compliant_percentage
            )
            for i in range(self.num_workers)
        }
        rewards = []
        total_nmacs = []
        iteration_record = []
        total_transitions = 0
        best_reward = -np.inf
        max_travel_times = []
        iteration_record = []
        total_LOS = []
        
        if self.agent.equipped:
            self.agent.model.load_weights(self.weights_file)
            weights = self.agent.model.get_weights()
        else:
            weights = []

        runner_sims = [workers[agent_id].run_one_iteration.remote(weights) for agent_id in workers.keys()]
        metric_list = []
        scenario = 0
        i = 0
        while scenario < self.testing_scenarios:

            done_id, runner_sims = ray.wait(runner_sims, num_returns=self.num_workers)
            results = ray.get(done_id)

            total_reward = []

            nmacs = []
            total_ac = []
            LOS_total = 0
            LOS_total_compliant = 0
            LOS_total_non_compliant = 0
            shield_total = 0
            shield_total_intersect = 0 
            shield_total_route = 0
            scenario_file = None
            non_compliant_count = 0
            for result in results:
                data = ray.get(result)
                try:
                    total_reward.append(float(np.sum(data[0]["raw_reward"])))
                except:
                    pass
                if data[0]['environment_done']:
                    nmacs.append(data[0]['nmacs'])
                    total_ac.append(data[0]['total_ac'])
                    non_compliant_count = data[0]['non_compliant_count']
                    scenario += 1
                LOS_total += data[0]['los_events']
                LOS_total_non_compliant += data[0]['los_events_non_compliant']
                LOS_total_compliant += data[0]['los_events_compliant']
                shield_total += data[0]['shield_events']
                shield_total_intersect += data[0]['shield_events_i']
                shield_total_route += data[0]['shield_events_r']
                halting_list = data[0]['halting_time_list']
                full_travel_times_temp = data[0]['full_travel_times']
                max_travel_time = data[0]['max_travel_time']
                scenario_file = data[0]['scenario_file']

            mean_total_reward = np.mean(total_reward)

            for j, nmac in enumerate(nmacs):
                print(f"     Scenario {scenario} Complete     ")
                print("|------------------------------|")
                print(f"| Run Name:      {self.run_name}      |")
                print(f"| Total Aircraft:   {total_ac[j]}  |")
                print(f"| Total Non-Compliant Aircraft {non_compliant_count}")
                print(f"| Max Travel Time: {max_travel_time}  |")
                print(f"| Number of LOS Events: {LOS_total}  |")
                print(f"LOS events between Non-Compliant and Compliant Aircraft {LOS_total_non_compliant}")
                print(f"LOS events between Compliant Aircraft {LOS_total_compliant}")
                print("|------------------------------|")
                print(" ")
                total_nmacs.append(nmac)
                metric_dict = {}
                metric_dict['scenario_num'] = scenario
                metric_dict['shield_total'] = shield_total
                metric_dict['shield_total_intersection'] = shield_total_intersect
                metric_dict['shield_total_route'] = shield_total_route
                metric_dict['max_travel_time'] = max_travel_time
                metric_dict['full_travel_times'] = full_travel_times_temp
                metric_dict['los_events_non_compliant'] = LOS_total_non_compliant
                metric_dict['los_events_compliant'] = LOS_total_compliant
                metric_dict['non_compliant_count'] = non_compliant_count
                metric_dict['los'] = LOS_total
                metric_dict['scenario_name'] = scenario_file
                metric_dict['halting_list'] = halting_list
                metric_list.append(metric_dict)
                total_nmacs.append(nmac)
                max_travel_times.append(max_travel_time)
                total_LOS.append(LOS_total)
                iteration_record.append(i)

                with open('log/test/{}.json'.format(self.run_name), 'w') as file:
                    json.dump(metric_list, file, indent=4)

            rewards.append(mean_total_reward)
            np.save("{}/eval_reward.npy".format(self.path_results), np.array(rewards))

            if len(nmacs) > 0:
                np.save("{}/eval_nmacs.npy".format(self.path_results), np.array(total_nmacs))
                np.save("{}/eval_iteration_record.npy".format(self.path_results), np.array(iteration_record))

            runner_sims = [workers[agent_id].run_one_iteration.remote(weights) for agent_id in workers.keys()]
            i += 1


### Main code execution
# Uncomment this for training
gin.parse_config_file("conf/config_test.gin")

if args.cluster:
    ## Initialize Ray
    ray.init(address=os.environ["ip_head"])
    print(ray.cluster_resources())
else:
    # check if running on Mac
    if platform.release() == "Darwin":
        ray.init(_node_ip_address="0.0.0.0", local_mode=args.debug)
    else:
        ray.init(local_mode=args.debug)
    print(ray.cluster_resources())

# Now initialize the trainer with 30 workers and to run for 100k episodes 3334 episodes * 30 workers = ~100k episodes
Trainer = Driver(cluster=args.cluster)
if Trainer.run_type == 'train':
    Trainer.train()
else:
    Trainer.evaluate()
