# TrajectoryAidedLearning

This repo contains the source code for the paper entitled, "High-speed Autonomous Racing using Trajectory-aided Deep Reinforcement Learning"


# Result Generation

The results in the paper are generated through a two step process of:
1. Train and test the agents
2. Process and plot the data

For every test:
- Run calculate_statistics
- Run calculate_averages

## Current formulations

### Reward Signal Comparision

- Train agents with the progress and cross-track and heading rewards 
- Use the CthVsProgress config file
- Results:
    - Training graphs: CthVsProgress_TrainingGraph
    - Lapt times and % progress bar plots: CthsVsProgress_Barplot


### Maximum Speed Investigation

- Aim: Understand how performance changes with different speeds.
- Config files: CthSpeeds, CthSpeedMaps 
- Results: 
    - Training graph: Cth_speeds_TrainingGraph
    - Lap times and % success: Cth_speeds_Barplot
        #TODO: this needs fixing and combining.

### Speed Profile Analysis

- Aim: study the trajectories, speed and slip profiles of agents wtiha max speed of 5 and 7 m/s.
- Config file: None, uses the Cth_speeds results
- Results:
    - Trajectories: GenerateVelocityProfiles, set the folder to Cth_speeds
    - Speed and slip profile: Cth_speeds_Profiles 


## Trajectory-aided Learning 

All of these results presuppose that the baseline tests have been run for comparison.

### Maximum Speed Investigation 

- Aim: Study the performance for increasing maximum speeds
- Config file: TAL_speeds
- Results:
    - Training graph: TAL_speeds_TrainingGraph
    - % progress bar plot: TAL_speeds_AvgProgress

### 6 m/s Comparison with Baseline 

- Aim: Compare the baseline and TAL on different maps
- Config file: TAL_maps
- Results:
    - Training graphs: TAL_Cth_maps_TrainingGraph
    - Lap times and success bar plot: TAL_Cth_maps_Barplot

### Speed Profile Analysis 

- Aim: Study the speed profiles
- Config file: requires the PP_speeds test
- Results:
    - Trajectories: GenerateVelocityProfiles, set the folder to TAL_speeds
    - Speed profile pp TAL: TAL_speed_profiles
    - Speed profile x3: TAL_speed_profiles 
    - Slip profile: TAL_speed_profiles

### Comparison with Literatures

- Aim: Compare our method with the literature
- Config file: Results from Bosello et al.
- Results:
    - Bar plot: LiteratureComparison


