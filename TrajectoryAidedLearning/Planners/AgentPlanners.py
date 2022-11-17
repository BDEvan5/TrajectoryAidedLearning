import numpy as np 
from TrajectoryAidedLearning.Utils.TD3 import TD3
from TrajectoryAidedLearning.Utils.HistoryStructs import TrainHistory
import torch
from numba import njit

from TrajectoryAidedLearning.Utils.utils import init_file_struct
from matplotlib import pyplot as plt


class FastArchitecture:
    def __init__(self, run, conf):
        self.state_space = conf.n_beams 
        self.range_finder_scale = conf.range_finder_scale
        self.n_beams = conf.n_beams
        self.max_speed = run.max_speed
        self.max_steer = conf.max_steer

        self.action_space = 2

        self.n_scans = run.n_scans
        self.scan_buffer = np.zeros((self.n_scans, self.n_beams))
        self.state_space *= self.n_scans

    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env

        Returns:
            nn_obs: observation vector for neural network
        """
            
        scan = np.array(obs['scan']) 

        scaled_scan = scan/self.range_finder_scale
        scan = np.clip(scaled_scan, 0, 1)

        if self.scan_buffer.all() ==0: # first reading
            for i in range(self.n_scans):
                self.scan_buffer[i, :] = scan 
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan

        nn_obs = np.reshape(self.scan_buffer, (self.n_beams * self.n_scans))

        return nn_obs

    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        #! this is a place to look if things don't work. This was max v, meaning that at lower speeds it would only limit it by clipping. That wasn't good.
        speed = (nn_action[1] + 1) * (self.max_speed  / 2 - 0.5) + 1
        speed = min(speed, self.max_speed) # cap the speed

        action = np.array([steering_angle, speed])

        return action



class AgentTrainer: 
    def __init__(self, run, conf):
        self.run, self.conf = run, conf
        self.name = run.run_name
        self.path = conf.vehicle_path + run.path + run.run_name 
        init_file_struct(self.path)

        self.v_min_plan =  conf.v_min_plan

        self.state = None
        self.nn_state = None
        self.nn_act = None
        self.action = None

        self.architecture = FastArchitecture(run, conf)

        self.agent = TD3(self.architecture.state_space, self.architecture.action_space, 1, run.run_name)
        self.agent.create_agent(conf.h_size)

        self.t_his = TrainHistory(run, conf)

        self.train = self.agent.train # alias for sss
        self.save = self.agent.save # alias for sss

    def plan(self, obs, add_mem_entry=True):
        nn_state = self.architecture.transform_obs(obs)
        if add_mem_entry:
            self.add_memory_entry(obs, nn_state)
            
        if obs['state'][3] < self.v_min_plan:
            self.action = np.array([0, 7])
            return self.action

        if np.isnan(nn_state).any():
            print(f"NAN in state: {nn_state}")

        self.nn_state = nn_state # after to prevent call before check for v_min_plan
        self.nn_act = self.agent.act(self.nn_state)

        if np.isnan(self.nn_act).any():
            print(f"NAN in act: {nn_state}")
            raise Exception("Unknown NAN in act")

        self.architecture.transform_obs(obs) # to ensure correct PP actions
        self.action = self.architecture.transform_action(self.nn_act)

        return self.action 

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.nn_state is not None:
            self.t_his.add_step_data(s_prime['reward'])

            self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, s_prime['reward'], False)

    def intervention_entry(self, s_prime):
        """
        To be called when the supervisor intervenes.
        The lap isn't complete, but it is a terminal state
        """
        nn_s_prime = self.architecture.transform_obs(s_prime)
        if self.nn_state is None:
            # print(f"Intervened on first step: RETURNING")
            return
        self.t_his.add_step_data(s_prime['reward'])

        self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, s_prime['reward'], True)

    def done_entry(self, s_prime):
        """
        To be called when ep is done.
        """
        nn_s_prime = self.architecture.transform_obs(s_prime)

        self.t_his.lap_done(s_prime['reward'], s_prime['progress'], False)
        if self.nn_state is None:
            print(f"Crashed on first step: RETURNING")
            return
        
        self.agent.save(self.path)
        if np.isnan(self.nn_act).any():
            print(f"NAN in act: {self.nn_act}")
            raise Exception("NAN in act")
        if np.isnan(nn_s_prime).any():
            print(f"NAN in state: {nn_s_prime}")
            raise Exception("NAN in state")

        self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, s_prime['reward'], True)
        self.nn_state = None

    def lap_complete(self):
        pass

    def save_training_data(self):
        self.t_his.print_update(True)
        self.t_his.save_csv_data()
        self.agent.save(self.path)

class AgentTester:
    def __init__(self, run, conf):
        """
        Testing vehicle using the reference modification navigation stack

        Args:
            agent_name: name of the agent for saving and reference
            conf: namespace with simulation parameters
            mod_conf: namespace with modification planner parameters
        """
        self.run, self.conf = run, conf
        self.v_min_plan = conf.v_min_plan
        self.path = conf.vehicle_path + run.path + run.run_name 

        self.actor = torch.load(self.path + '/' + run.run_name + "_actor.pth")

        architecture_type = select_architecture(run.architecture)
        self.architecture = architecture_type(run, conf)

        print(f"Agent loaded: {run.run_name}")

    def plan(self, obs):
        nn_obs = self.architecture.transform_obs(obs)

        if obs['state'][3] < self.v_min_plan:
            self.action = np.array([0, 7])
            return self.action

        nn_obs = torch.FloatTensor(nn_obs.reshape(1, -1))
        nn_action = self.actor(nn_obs).data.numpy().flatten()
        self.nn_act = nn_action

        self.action = self.architecture.transform_action(nn_action)

        return self.action 

    def lap_complete(self):
        pass
