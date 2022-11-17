from TrajectoryAidedLearning.Utils.RewardSignals import *





def select_reward_function(run, conf, std_track, race_track):
    reward = run.reward
    if reward == "Progress":
        reward_function = ProgressReward(std_track)
    elif reward == "Cth": 
        reward_function = CrossTrackHeadReward(std_track, conf)
    elif reward == "TAL":
        reward_function = TALearningReward(conf, run)
    else: raise Exception("Unknown reward function: " + reward)
        
    return reward_function