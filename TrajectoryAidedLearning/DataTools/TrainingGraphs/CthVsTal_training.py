import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import csv
import glob, os 
from matplotlib.ticker import MultipleLocator, PercentFormatter

from TrajectoryAidedLearning.Utils.utils import *
from TrajectoryAidedLearning.DataTools.TrainingGraphs.TrainingUtils import *



def CthVsTal_training():
    p = "Data/Vehicles/CthAndTal6/"

    set_n =1
    repeats = 5
    speed = 6

    plt.figure(1, figsize=(4.8, 2.0))
    plt.clf()

    xs = np.linspace(0, 100, 300)

    colors = ["#D68910", "#884EA0"]
    map_name = "f1_esp"

    tal_steps = []
    tal_progresses = []
    cth_steps = []
    cth_progresses = []
    for i in range(repeats):
        path_tal = p + f"fast_Std_Std_TAL_{map_name}_{speed}_{set_n}_{i}/"
        path_cth = p + f"fast_Std_Std_Cth_{map_name}_{speed}_{set_n}_{i}/"

        rewards_tal, lengths_tal, progresses_tal, _ = load_csv_data(path_tal)
        rewards_cth, lengths_cth, progresses_cth, _ = load_csv_data(path_cth)

        steps_tal = np.cumsum(lengths_tal) / 1000
        avg_progress_tal = true_moving_average(progresses_tal, 20)* 100
        steps_cth = np.cumsum(lengths_cth) / 1000
        avg_progress_cth = true_moving_average(progresses_cth, 20) * 100

        tal_steps.append(steps_tal)
        tal_progresses.append(avg_progress_tal)
        cth_steps.append(steps_cth)
        cth_progresses.append(avg_progress_cth)


    min_tal, max_tal, mean_tal = convert_to_min_max_avg(tal_steps, tal_progresses, xs)
    min_cth, max_cth, mean_cth = convert_to_min_max_avg(cth_steps, cth_progresses, xs)

    plt.plot(xs, mean_cth, '-', color=colors[1], linewidth=2.5, label='Baseline')
    plt.gca().fill_between(xs, min_cth, max_cth, color=colors[1], alpha=0.4)
    plt.plot(xs, mean_tal, '-', color=colors[0], linewidth=2.5, label='TAL')
    plt.gca().fill_between(xs, min_tal, max_tal, color=colors[0], alpha=0.4)
        
    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x1000)")
    plt.ylabel("Track Progress (%)")
    plt.ylim(0, 100)

    plt.legend(loc='lower right', ncol=2)
    plt.tight_layout()
    plt.grid(True)

    name = p + f"CthVsTal_training{speed}"
    std_img_saving(name)

    
    
CthVsTal_training()