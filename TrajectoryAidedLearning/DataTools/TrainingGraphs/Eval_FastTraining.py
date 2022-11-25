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


    set_n =1
    repeats = 5
    speed = 5


    plt.figure(1, figsize=(4.8, 2.0))
    plt.clf()

    xs = np.linspace(0, 100, 300)

    colors_light = ["#2ECC71", "#E74C3C"]
    colors_dark = ["#1E8449", "#922B21"]

    for map_name in ["f1_esp", "f1_mco"]:
        progress_steps = []
        progress_progresses = []
        cth_steps = []
        cth_progresses = []
        for i in range(repeats):
            path_progress = p + f"fast_Std_Std_Progress_{map_name}_{speed}_{set_n}_{i}/"
            path_cth = p + f"fast_Std_Std_Cth_{map_name}_{speed}_{set_n}_{i}/"

            rewards_progress, lengths_progress, progresses_progress, _ = load_csv_data(path_progress)
            rewards_cth, lengths_cth, progresses_cth, _ = load_csv_data(path_cth)

            steps_progress = np.cumsum(lengths_progress) / 1000
            avg_progress_progress = true_moving_average(progresses_progress, 20)* 100
            steps_cth = np.cumsum(lengths_cth) / 1000
            avg_progress_cth = true_moving_average(progresses_cth, 20) * 100

            progress_steps.append(steps_progress)
            progress_progresses.append(avg_progress_progress)
            cth_steps.append(steps_cth)
            cth_progresses.append(avg_progress_cth)


        min_progress, max_progress, mean_progress = convert_to_min_max_avg(progress_steps, progress_progresses, xs)
        min_cth, max_cth, mean_cth = convert_to_min_max_avg(cth_steps, cth_progresses, xs)

        if map_name == "f1_esp": 
            plt.plot(xs, mean_progress, '-', color=colors_light[0], linewidth=2, label='Progress - ESP')
            # plt.gca().fill_between(xs, min_progress, max_progress, color='red', alpha=0.2)
            plt.plot(xs, mean_cth, '-', color=colors_light[1], linewidth=2, label='Cth - ESP')
            # plt.gca().fill_between(xs, min_cth, max_cth, color='green', alpha=0.2)
        else: 
            plt.plot(xs, mean_progress, '-', color=colors_dark[0], linewidth=2, label='Progress - MCO')
            # plt.gca().fill_between(xs, min_progress, max_progress, color='red', alpha=0.2)
            plt.plot(xs, mean_cth, '-', color=colors_dark[1], linewidth=2, label='Cth - MCO')
            # plt.gca().fill_between(xs, min_cth, max_cth, color='green', alpha=0.2)
        
    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x1000)")
    plt.ylabel("Track Progress %")
    plt.ylim(0, 100)
    if speed == 8:
        plt.legend(loc='upper right', ncol=2)
    else:
        plt.legend(loc='lower right', ncol=2)
    plt.tight_layout()
    plt.grid(True)

    name = p + f"Eval_RewardsFast_Training_{speed}"
    std_img_saving(name)

    
def CthVsProgress_TrainingGraphs():
    p = "Data/Vehicles/CthVsProgress/"

    set_n =1
    repeats = 5
    speed = 5

    # plt.figure(1, figsize=(4.8, 2.0))
    # fig, axs = plt.subplots(1, 2)
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 1.6), sharey=True)
    # plt.clf()

    xs = np.linspace(0, 100, 300)

    colors = ["#28B463", "#CB4335"]
    
    for z, map_name in enumerate(["f1_esp", "f1_mco"]):
        progress_steps = []
        progress_progresses = []
        cth_steps = []
        cth_progresses = []
        for i in range(repeats):
            path_progress = p + f"fast_Std_Std_Progress_{map_name}_{speed}_{set_n}_{i}/"
            path_cth = p + f"fast_Std_Std_Cth_{map_name}_{speed}_{set_n}_{i}/"

            rewards_progress, lengths_progress, progresses_progress, _ = load_csv_data(path_progress)
            rewards_cth, lengths_cth, progresses_cth, _ = load_csv_data(path_cth)

            steps_progress = np.cumsum(lengths_progress) / 1000
            avg_progress_progress = true_moving_average(progresses_progress, 20)* 100
            steps_cth = np.cumsum(lengths_cth) / 1000
            avg_progress_cth = true_moving_average(progresses_cth, 20) * 100

            progress_steps.append(steps_progress)
            progress_progresses.append(avg_progress_progress)
            cth_steps.append(steps_cth)
            cth_progresses.append(avg_progress_cth)


        min_progress, max_progress, mean_progress = convert_to_min_max_avg(progress_steps, progress_progresses, xs)
        min_cth, max_cth, mean_cth = convert_to_min_max_avg(cth_steps, cth_progresses, xs)

        alpha = 0.3
        if map_name == "f1_esp": 
            axs[z].set_title("ESP")
            axs[z].plot(xs, mean_progress, '-', color=colors[0], linewidth=2, label='Progress')
            axs[z].fill_between(xs, min_progress, max_progress, color=colors[0], alpha=alpha)
            axs[z].plot(xs, mean_cth, '-', color=colors[1], linewidth=2, label='Cth')
            axs[z].fill_between(xs, min_cth, max_cth, color=colors[1], alpha=alpha)
            axs[z].yaxis.set_major_locator(MultipleLocator(25))
            axs[z].grid(True)
            axs[z].set_xlabel("Training Steps (x1000)")
        else: 
            axs[z].set_title("MCO")
            axs[z].plot(xs, mean_progress, '-', color=colors[0], linewidth=2)
            axs[z].fill_between(xs, min_progress, max_progress, color=colors[0], alpha=alpha)
            axs[z].plot(xs, mean_cth, '-', color=colors[1], linewidth=2)
            axs[z].fill_between(xs, min_cth, max_cth, color=colors[1], alpha=alpha)
            axs[z].yaxis.set_major_locator(MultipleLocator(25))
            axs[z].grid(True)
            axs[z].set_xlabel("Training Steps (x1000)")
            
    plt.ylim(0, 100)
    axs[0].set_ylabel("Track Progress %")
    fig.legend(loc='center', ncol=2, bbox_to_anchor=(0.5, 0.25), framealpha=0.98)
    # if speed == 8:
    #     plt.legend(loc='upper right', ncol=2)
    # else:
    #     plt.legend(loc='lower right', ncol=2)

    name = p + f"CthVsProgress_Training_{speed}"
    std_img_saving(name)

    
    
# Eval_RewardsFast_TrainingGraphs()
CthVsProgress_TrainingGraphs()