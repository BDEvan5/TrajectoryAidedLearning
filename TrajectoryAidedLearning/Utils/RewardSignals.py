from TrajectoryAidedLearning.Utils.RewardUtils import *

from RacingRewards.Utils.utils import *
from TrajectoryAidedLearning.Utils.StdTrack import StdTrack

from TrajectoryAidedLearning.Planners.PurePursuit import PurePursuit
import numpy as np

# rewards functions
class ProgressReward:
    def __init__(self, track: StdTrack) -> None:
        self.track = track

    def __call__(self, observation, prev_obs, pre_action):
        if prev_obs is None: return 0

        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash
        
        
        position = observation['state'][0:2]
        prev_position = prev_obs['state'][0:2]
        theta = observation['state'][2]

        s = self.track.calculate_progress(prev_position)
        ss = self.track.calculate_progress(position)
        reward = (ss - s) / self.track.total_s
        if abs(reward) > 0.5: # happens at end of eps
            return 0.001 # assume positive progress near end

        # self.race_track.plot_vehicle(position, theta)


        reward *= 10 # remove all reward
        return reward 


class CrossTrackHeadReward:
    def __init__(self, track: StdTrack, conf):
        self.track = track
        self.r_veloctiy = 1
        self.r_distance = 1
        self.max_v = conf.max_v # used for scaling.

    def __call__(self, observation, prev_obs, pre_action):
        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash

        position = observation['state'][0:2]
        theta = observation['state'][2]
        heading, distance = self.track.get_cross_track_heading(position)
        # self.race_track.plot_vehicle(position, theta)

        d_heading = abs(robust_angle_difference_rad(heading, theta))
        r_heading  = np.cos(d_heading)  * self.r_veloctiy # velocity
        r_heading *= (observation['state'][3] / self.max_v)

        r_distance = distance * self.r_distance 

        reward = r_heading - r_distance
        reward = max(reward, 0)
        # reward *= 0.1
        return reward



class TALearningReward:
    def __init__(self, conf, run):
        run.pp_speed = "raceline"
        run.raceline = True
        self.pp = PurePursuit(conf, run, False, True) 

        self.beta_c = 0.4
        self.beta_steer_weight = 0.4
        self.beta_velocity_weight = 0.4

        self.max_steer_diff = 0.8
        self.max_velocity_diff = 2.0
        # self.max_velocity_diff = 4.0

    def __call__(self, observation, prev_obs, action):
        if prev_obs is None: return 0

        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash
        
        pp_act = self.pp.plan(prev_obs)

        steer_reward =  (abs(pp_act[0] - action[0]) / self.max_steer_diff)  * self.beta_steer_weight

        throttle_reward =   (abs(pp_act[1] - action[1]) / self.max_velocity_diff) * self.beta_velocity_weight

        reward = self.beta_c - steer_reward - throttle_reward
        reward = max(reward, 0) # limit at 0

        reward *= 0.5

        return reward


