
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from TrajectoryAidedLearning.Utils.utils import *
from TrajectoryAidedLearning.DataTools.TrainingGraphs.TrainingUtils import *
from TrajectoryAidedLearning.DataTools.plotting_utils import *





def TAL_Maps_TrainingProgress():
    p = "Data/Vehicles/TAL_maps8/"

    steps_list = []
    progresses_list = []
    map_names = ["f1_esp", "f1_mco", "f1_gbr", "f1_aut"]

    n_repeats = 5
    for i, map_name in enumerate(map_names): 
        steps_list.append([])
        progresses_list.append([])
        for j in range(n_repeats):
            path = p + f"fast_Std_Std_TAL_{map_name}_8_1_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths[:-1]) / 1000
            avg_progress = true_moving_average(progresses[:-1], 20)* 100
            steps_list[i].append(steps)
            progresses_list[i].append(avg_progress)

    plt.figure(2, figsize=(4.5, 2.1))

    # labels = ["4", "5", "6", "7", "8"]
    labels = ['ESP', "MCO", "GBR", "AUT"]
    # labels = ['4 m/s', '5 m/s', '6 m/s', '7 m/s', '8 m/s']

    xs = np.linspace(0, 100, 300)
    for i in range(len(steps_list)):
        min, max, mean = convert_to_min_max_avg_iqm5(steps_list[i], progresses_list[i], xs)
        # min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], xs)
        plt.plot(xs, mean, '-', color=pp[i], linewidth=2, label=labels[i])
        plt.gca().fill_between(xs, min, max, color=pp[i], alpha=0.2)


    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x1000)")
    plt.ylabel("Track Progress %")
    plt.ylim(0, 100)
    plt.legend(loc='center', bbox_to_anchor=(1.06, 0.5), ncol=1)
    # plt.legend(loc='center', bbox_to_anchor=(0.5, 1.2), ncol=5)
    plt.tight_layout()
    plt.grid()

    name = p + f"Eval_Maps_TrainingProgress"
    std_img_saving(name)


TAL_Maps_TrainingProgress()

