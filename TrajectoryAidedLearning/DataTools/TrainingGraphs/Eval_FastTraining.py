import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import csv
import glob, os 
from matplotlib.ticker import MultipleLocator, PercentFormatter

from TrajectoryAidedLearning.Utils.utils import *
from TrajectoryAidedLearning.DataTools.TrainingGraphs.TrainingUtils import *



def Eval_RewardsFast_TrainingGraphs():
    p = "Data/Vehicles/Eval_RewardsFast/"

    progress_steps = []
    progress_progresses = []
    cth_steps = []
    cth_progresses = []

    set_n =1
    repeats = 5

# 
    for speed in [5, 8]:
        for map_name in ["f1_esp", "f1_mco"]:
            for i in range(repeats):
                path_progress = p + f"fast_Std_Std_Progress_{map_name}_{speed}_{set_n}_{i}/"
                path_cth = p + f"fast_Std_Std_Cth_{map_name}_{speed}_{set_n}_{i}/"

                rewards_progress, lengths_progress, progresses_progress, _ = load_csv_data(path_progress)
                rewards_cth, lengths_cth, progresses_cth, _ = load_csv_data(path_cth)

                steps_progress = np.cumsum(lengths_progress) / 100
                avg_progress_progress = true_moving_average(progresses_progress, 20)* 100
                steps_cth = np.cumsum(lengths_cth) / 100
                avg_progress_cth = true_moving_average(progresses_cth, 20) * 100

                progress_steps.append(steps_progress)
                progress_progresses.append(avg_progress_progress)
                cth_steps.append(steps_cth)
                cth_progresses.append(avg_progress_cth)


            plt.figure(1, figsize=(3.5, 2.0))
            # plt.figure(1, figsize=(6, 2.5))
            plt.clf()

            xs = np.linspace(0, 1000, 300)
            min_progress, max_progress, mean_progress = convert_to_min_max_avg(progress_steps, progress_progresses, xs)
            min_cth, max_cth, mean_cth = convert_to_min_max_avg(cth_steps, cth_progresses, xs)

            plt.plot(xs, mean_progress, '-', color=pp[0], linewidth=2, label='Progress')
            plt.gca().fill_between(xs, min_progress, max_progress, color='red', alpha=0.2)
            plt.plot(xs, mean_cth, '-', color=pp[2], linewidth=2, label='Cth')
            plt.gca().fill_between(xs, min_cth, max_cth, color='green', alpha=0.2)
            
            plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

            plt.xlabel("Training Steps (x100)")
            plt.ylabel("Track Progress %")
            plt.legend(loc='lower right', ncol=2)
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=3)
            plt.tight_layout()
            plt.grid()

            name = p + f"Eval_RewardsFast_Training_{map_name}_{speed}"
            std_img_saving(name)

    
Eval_RewardsFast_TrainingGraphs()