
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from TrajectoryAidedLearning.Utils.utils import *
from TrajectoryAidedLearning.DataTools.TrainingGraphs.TrainingUtils import *
from TrajectoryAidedLearning.DataTools.plotting_utils import *


def Cth_speeds_TrainingGraph():
    p = "Data/Vehicles/Cth_speeds/"

    steps_list = []
    progresses_list = []

    n_repeats = 5
    for i, v in enumerate(range(4, 9)): 
        steps_list.append([])
        progresses_list.append([])
        for j in range(n_repeats):
            path = p + f"fast_Std_Std_Cth_f1_esp_{v}_1_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths[:-1]) / 1000
            avg_progress = true_moving_average(progresses[:-1], 20)* 100
            steps_list[i].append(steps)
            progresses_list[i].append(avg_progress)

    plt.figure(2, figsize=(4.5, 2.3))

    # labels = ["4", "5", "6", "7", "8"]
    labels = ['4 m/s', '5 m/s', '6 m/s', '7 m/s', '8 m/s']

    xs = np.linspace(0, 100, 300)
    for i in range(len(steps_list)):
        min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], xs)
        plt.plot(xs, mean, '-', color=pp[i], linewidth=2, label=labels[i])
        plt.gca().fill_between(xs, min, max, color=pp[i], alpha=0.2)


    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x1000)")
    plt.ylabel("Track Progress %")
    plt.ylim(0, 100)
    # plt.legend(loc='center', bbox_to_anchor=(1.06, 0.5), ncol=1)
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.52), ncol=3)
    plt.tight_layout()
    plt.grid()

    name = p + f"Cth_speeds_TrainingGraph"
    std_img_saving(name)

def Cth_speeds_TrainingGraph_small():
    p = "Data/Vehicles/Cth_speeds/"

    steps_list = []
    progresses_list = []

    n_repeats = 5
    for i, v in enumerate(range(4, 9)): 
        steps_list.append([])
        progresses_list.append([])
        for j in range(n_repeats):
            path = p + f"fast_Std_Std_Cth_f1_esp_{v}_1_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths[:-1]) / 1000
            avg_progress = true_moving_average(progresses[:-1], 20)* 100
            steps_list[i].append(steps)
            progresses_list[i].append(avg_progress)

    plt.figure(2, figsize=(4.5, 2.3))

    # labels = ["4", "5", "6", "7", "8"]
    labels = ['4 m/s', '5 m/s', '6 m/s', '7 m/s', '8 m/s']

    xs = np.linspace(0, 100, 300)
    for i in range(len(steps_list)):
        min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], xs)
        plt.plot(xs, mean, '-', color=pp[i], linewidth=2, label=labels[i])
        plt.gca().fill_between(xs, min, max, color=pp[i], alpha=0.2)


    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x1000)")
    plt.ylabel("Track Progress %")
    plt.ylim(0, 100)
    # plt.legend(loc='center', bbox_to_anchor=(1.06, 0.5), ncol=1)
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.52), ncol=3)
    plt.tight_layout()
    plt.grid()

    name = p + f"Cth_speeds_TrainingGraph"
    std_img_saving(name)

# Cth_speeds_TrainingGraph()
Cth_speeds_TrainingGraph_small()



