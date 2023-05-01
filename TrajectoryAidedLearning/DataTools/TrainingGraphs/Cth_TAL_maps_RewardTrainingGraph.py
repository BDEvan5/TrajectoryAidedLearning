
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from TrajectoryAidedLearning.Utils.utils import *
from TrajectoryAidedLearning.DataTools.TrainingGraphs.TrainingUtils import *
from TrajectoryAidedLearning.DataTools.plotting_utils import *




def Cth_TAL_speeds_TrainingGraph_small():
    p = "Data/Vehicles/Cth_maps/"

    steps_list = []
    progresses_list = []

    n_repeats = 5
    v = 6
    map_names = ['aut', "esp", "gbr", "mco"]
    for i, m in enumerate(map_names):
        steps_list.append([])
        progresses_list.append([])
        for j in range(n_repeats):
            path = p + f"fast_Std_Std_Cth_f1_{m}_{v}_1_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths[:-1]) / 1000
            avg_progress = true_moving_average(rewards[:-1], 20)
            steps_list[i].append(steps)
            progresses_list[i].append(avg_progress)

    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.2), sharey=False)
    plt.sca(axs[0])
    
    labels = [i.upper() for i in map_names]

    xs = np.linspace(0, 100, 300)
    for i in range(len(steps_list)):
        min, max, mean = convert_to_min_max_avg_iqm5(steps_list[i], progresses_list[i], xs)
        plt.plot(xs, mean, '-', color=pp[i], linewidth=2, label=labels[i])
        plt.gca().fill_between(xs, min, max, color=pp[i], alpha=0.2)
    plt.xlabel("Training Steps (x1000)")
    plt.grid(True)
    plt.gca().get_xaxis().set_major_locator(MultipleLocator(25))
    plt.gca().get_yaxis().set_major_locator(MultipleLocator(40))
    plt.ylabel("Ep. Reward")
    plt.title("Baseline")
    plt.ylim(0, 180)

    p = "Data/Vehicles/TAL_maps/"

    steps_list = []
    progresses_list = []

    n_repeats = 5
    for i, m in enumerate(map_names):
        steps_list.append([])
        progresses_list.append([])
        for j in range(n_repeats):
            path = p + f"fast_Std_Std_TAL_f1_{m}_{v}_1_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths[:-1]) / 1000
            avg_progress = true_moving_average(rewards[:-1], 20)
            steps_list[i].append(steps)
            progresses_list[i].append(avg_progress)

    plt.sca(axs[1])
    xs = np.linspace(0, 100, 300)
    for i in range(len(steps_list)):
        min, max, mean = convert_to_min_max_avg_iqm5(steps_list[i], progresses_list[i], xs)
        plt.plot(xs, mean, '-', color=pp[i], linewidth=2)
        # plt.plot(xs, mean, '-', color=pp[i], linewidth=2, label=labels[i])
        plt.gca().fill_between(xs, min, max, color=pp[i], alpha=0.2)
    plt.xlabel("Training Steps (x1000)")
    # plt.ylabel("Ep. Reward")
    plt.title("TAL")

    plt.gca().get_yaxis().set_major_locator(MultipleLocator(20))
    plt.gca().get_xaxis().set_major_locator(MultipleLocator(25))

    plt.ylim(0, 70)
    # plt.legend(loc='center', bbox_to_anchor=(1.06, 0.5), ncol=1)
    
    fig.legend(loc='center', bbox_to_anchor=(0.5, -0.), ncol=5)
    plt.tight_layout()
    plt.grid(True)

    name = p + f"Cth_TAL_maps_RewardTrainingGraph"
    std_img_saving(name)

Cth_TAL_speeds_TrainingGraph_small()



