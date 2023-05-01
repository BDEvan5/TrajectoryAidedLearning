import os, shutil
import csv
import numpy as np
from matplotlib import pyplot as plt
from TrajectoryAidedLearning.Utils.utils import *
from matplotlib.ticker import MultipleLocator


SIZE = 20000


def plot_data(values, moving_avg_period=10, title="Results", figure_n=2):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    if len(values) >= moving_avg_period:
        moving_avg = true_moving_average(values, moving_avg_period)
        plt.plot(moving_avg)    
    if len(values) >= moving_avg_period*5:
        moving_avg = true_moving_average(values, moving_avg_period * 5)
        plt.plot(moving_avg)    
    # plt.pause(0.001)


class TrainHistory():
    def __init__(self, run, conf) -> None:
        self.path = conf.vehicle_path + run.path +  run.run_name 

        # training data
        self.ptr = 0
        self.lengths = np.zeros(SIZE)
        self.rewards = np.zeros(SIZE) 
        self.progresses = np.zeros(SIZE) 
        self.laptimes = np.zeros(SIZE) 
        self.t_counter = 0 # total steps
        
        # espisode data
        self.ep_counter = 0 # ep steps
        self.ep_reward = 0

    def add_step_data(self, new_r):
        self.ep_reward += new_r
        self.ep_counter += 1
        self.t_counter += 1 

    def lap_done(self, reward, progress, show_reward=False):
        self.add_step_data(reward)
        self.lengths[self.ptr] = self.ep_counter
        self.rewards[self.ptr] = self.ep_reward
        self.progresses[self.ptr] = progress
        self.ptr += 1

        if show_reward:
            plt.figure(8)
            plt.clf()
            plt.plot(self.ep_rewards)
            plt.plot(self.ep_rewards, 'x', markersize=10)
            plt.title(f"Ep rewards: total: {self.ep_reward:.4f}")
            plt.ylim([-1.1, 1.5])
            plt.pause(0.0001)

        self.ep_counter = 0
        self.ep_reward = 0
        self.ep_rewards = []

    def print_update(self, plot_reward=True):
        if self.ptr < 10:
            return
        
        mean10 = np.mean(self.rewards[self.ptr-10:self.ptr])
        mean100 = np.mean(self.rewards[max(0, self.ptr-100):self.ptr])
        # print(f"Run: {self.t_counter} --> Moving10: {mean10:.2f} --> Moving100: {mean100:.2f}  ")
        
        if plot_reward:
            # raise NotImplementedError
            plot_data(self.rewards[0:self.ptr], figure_n=2)

    def save_csv_data(self):
        data = []
        ptr = self.ptr  #exclude the last entry
        for i in range(ptr): 
            data.append([i, self.rewards[i], self.lengths[i], self.progresses[i], self.laptimes[i]])
        save_csv_array(data, self.path + "/training_data_episodes.csv")

        plot_data(self.rewards[0:ptr], figure_n=2)
        plt.figure(2)
        plt.savefig(self.path + "/training_rewards_episodes.png")

        t_steps = np.cumsum(self.lengths[0:ptr])/100
        plt.figure(3)
        plt.clf()

        plt.plot(t_steps, self.rewards[0:ptr], '.', color='darkblue', markersize=4)
        plt.plot(t_steps, true_moving_average(self.rewards[0:ptr], 20), linewidth='4', color='r')

        plt.xlabel("Training Steps (x100)")
        plt.ylabel("Reward per Episode")

        plt.tight_layout()
        plt.grid()
        plt.savefig(self.path + "/training_rewards_steps.png")

        # plt.figure(4)
        # plt.clf()
        # plt.plot(t_steps, self.progresses[0:self.ptr], '.', color='darkblue', markersize=4)
        # plt.plot(t_steps, true_moving_average(self.progresses[0:self.ptr], 20), linewidth='4', color='r')

        # plt.xlabel("Training Steps (x100)")
        # plt.ylabel("Progress")

        # plt.tight_layout()
        # plt.grid()
        # plt.savefig(self.path + "/training_progress_steps.png")

        # plt.close()



class VehicleStateHistory:
    def __init__(self, run, folder):
        self.vehicle_name = run.run_name
        self.path = "Data/Vehicles/" + run.path + run.run_name + "/" + folder
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.states = []
        self.actions = []
    

    def add_state(self, state):
        self.states.append(state)
    
    def add_action(self, action):
        self.actions.append(action)
    
    def save_history(self, lap_n=0, test_map=None):
        states = np.array(self.states)
        self.actions.append(np.array([0, 0])) # last action to equal lengths
        actions = np.array(self.actions)

        lap_history = np.concatenate((states, actions), axis=1)

        if test_map is None:
            np.save(self.path + f"Lap_{lap_n}_history_{self.vehicle_name}.npy", lap_history)
        else:
            np.save(self.path + f"Lap_{lap_n}_history_{self.vehicle_name}_{test_map}.npy", lap_history)

        self.states = []
        self.actions = []
    
    def save_history(self, lap_n=0, test_map=None):
        states = np.array(self.states)
        self.actions.append(np.array([0, 0])) # last action to equal lengths
        actions = np.array(self.actions)

        lap_history = np.concatenate((states, actions), axis=1)

        if test_map is None:
            np.save(self.path + f"Lap_{lap_n}_history_{self.vehicle_name}.npy", lap_history)
        else:
            np.save(self.path + f"Lap_{lap_n}_history_{self.vehicle_name}_{test_map}.npy", lap_history)

        self.states = []
        self.actions = []



class SafetyHistory:
    def __init__(self, run):
        self.vehicle_name = run.run_name
        self.path = "Data/Vehicles/" + run.path + self.vehicle_name + "/SafeHistory/"
        os.mkdir(self.path)

        self.planned_actions = []
        self.safe_actions = []
        self.interventions = []
        self.lap_n = 0

        self.interval_counter = 0
        self.inter_intervals = []
        self.ep_interventions = 0
        self.intervention_list = []

    def add_actions(self, planned_action, safe_action=None):
        self.planned_actions.append(planned_action)
        if safe_action is None:
            self.safe_actions.append(planned_action)
            self.interventions.append(False)
        else:
            self.safe_actions.append(safe_action)
            self.interventions.append(True)

    def add_planned_action(self, planned_action):
        self.planned_actions.append(planned_action)
        self.safe_actions.append(planned_action)
        self.interventions.append(False)
        self.interval_counter += 1

    def add_intervention(self, planned_action, safe_action):
        self.planned_actions.append(planned_action)
        self.safe_actions.append(safe_action)
        self.interventions.append(True)
        self.inter_intervals.append(self.interval_counter)
        self.interval_counter = 0
        self.ep_interventions += 1

    def train_lap_complete(self):
        self.intervention_list.append(self.ep_interventions)

        print(f"Interventions: {self.ep_interventions} --> {self.inter_intervals}")

        self.ep_interventions = 0
        self.inter_intervals = []

    def plot_safe_history(self):
        planned = np.array(self.planned_actions)
        safe = np.array(self.safe_actions)
        plt.figure(5)
        plt.clf()
        plt.title("Safe History: steering")
        plt.plot(planned[:, 0], color='blue')
        plt.plot(safe[:, 0], '-x', color='red')
        plt.legend(['Planned Actions', 'Safe Actions'])
        plt.ylim([-0.5, 0.5])
        # plt.show()
        plt.pause(0.0001)

        plt.figure(6)
        plt.clf()
        plt.title("Safe History: velocity")
        plt.plot(planned[:, 1], color='blue')
        plt.plot(safe[:, 1], '-x', color='red')
        plt.legend(['Planned Actions', 'Safe Actions'])
        plt.ylim([-0.5, 0.5])
        # plt.show()
        plt.pause(0.0001)

        self.planned_actions = []
        self.safe_actions = []

    def save_safe_history(self, training=False):
        planned_actions = np.array(self.planned_actions)
        safe_actions = np.array(self.safe_actions)
        interventions = np.array(self.interventions)
        data = np.concatenate((planned_actions, safe_actions, interventions[:, None]), axis=1)

        if training:
            np.save(self.path + f"Training_safeHistory_{self.vehicle_name}.npy", data)
        else:
            np.save(self.path + f"Lap_{self.lap_n}_safeHistory_{self.vehicle_name}.npy", data)

        self.lap_n += 1

        self.planned_actions = []
        self.safe_actions = []
        self.interventions = []
