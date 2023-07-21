
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from TrajectoryAidedLearning.Utils.utils import *
from TrajectoryAidedLearning.DataTools.TrainingGraphs.TrainingUtils import *
from TrajectoryAidedLearning.DataTools.plotting_utils import *
import matplotlib.animation as animation



def Cth_TAL_speeds_TrainingGraph_small():
    p = "Data/Vehicles/Cth_speeds/"

    cth_steps_list = []
    cth_progresses_list = []

    n_repeats = 4
    speed_range = (4, 6, 8)
    for i, v in enumerate(speed_range): 
        cth_steps_list.append([])
        cth_progresses_list.append([])
        for j in range(n_repeats):
            path = p + f"fast_Std_Std_Cth_f1_esp_{v}_1_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths[:-1]) / 1000
            avg_progress = true_moving_average(progresses[:-1], 20)* 100
            cth_steps_list[i].append(steps)
            cth_progresses_list[i].append(avg_progress)

    # plt.figure(2, figsize=(4.5, 2.3))
    
    p = "Data/Vehicles/TAL_speeds/"

    tal_steps_list = []
    tal_progresses_list = []

    n_repeats = 4
    for i, v in enumerate(speed_range): 
        tal_steps_list.append([])
        tal_progresses_list.append([])
        for j in range(n_repeats):
            path = p + f"fast_Std_Std_TAL_f1_esp_{v}_1_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths[:-1]) / 1000
            avg_progress = true_moving_average(progresses[:-1], 20)* 100
            tal_steps_list[i].append(steps)
            tal_progresses_list[i].append(avg_progress)

    fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    # fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.2), sharey=True)
    
    plt.sca(axs[0])
    plt.xlabel("Training Steps (x1000)")
    plt.grid(True)
    plt.gca().get_xaxis().set_major_locator(MultipleLocator(25))
    plt.ylabel("Track Progress %")
    plt.title("Baseline")
    plt.xlim(-4, 100)
    
    plt.sca(axs[1])
    plt.xlabel("Training Steps (x1000)")
    plt.title("Trajectory-aided Learning (ours)", fontdict={'weight': 'bold'})
    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))
    plt.gca().get_xaxis().set_major_locator(MultipleLocator(25))
    plt.ylim(0, 105)
    plt.grid(True)

    plt.xlim(-4, 100)

    labels = ['4 m/s', '6 m/s', '8 m/s']
    # labels = ['4 m/s', '5 m/s', '6 m/s', '7 m/s', '8 m/s']

    new_colors = [pp[0], pp[2], pp[4]]

    xs = np.linspace(0, 100, 300)
    line1s = []
    line2s = []
    for i in range(len(cth_steps_list)):
        plt.sca(axs[0])
        min, max, mean = convert_to_min_max_avg_iqm5(cth_steps_list[i], cth_progresses_list[i], xs)
        line1 = axs[0].plot(xs[0], mean[0], '-', color=new_colors[i], linewidth=3, label=labels[i])
        line1s.append(line1)
        # plt.gca().fill_between(xs, min, max, color=pp[i], alpha=0.2)


        plt.sca(axs[1])
        min, max, mean = convert_to_min_max_avg_iqm5(tal_steps_list[i], tal_progresses_list[i], xs)
        line2 = axs[1].plot(xs[0], mean[0], '-', color=new_colors[i], linewidth=3)
            # plt.gca().fill_between(xs, min, max, color=pp[i], alpha=0.2)
        line2s.append(line2)

    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncol=5)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    def update(frame):
        end_ind = frame * 5
        for i in range(len(cth_steps_list)):
            # plt.sca(axs[0])
            min, max, mean = convert_to_min_max_avg_iqm5(cth_steps_list[i], cth_progresses_list[i], xs)
            line1s[i][0].set_xdata(xs[:end_ind])
            line1s[i][0].set_ydata(mean[:end_ind])
            # plt.gca().fill_between(xs, min, max, color=pp[i], alpha=0.2)


            # plt.sca(axs[1])
            min, max, mean = convert_to_min_max_avg_iqm5(tal_steps_list[i], tal_progresses_list[i], xs)
            line2s[i][0].set_xdata(xs[:end_ind])
            line2s[i][0].set_ydata(mean[:end_ind])
            # plt.gca().fill_between(xs, min, max, color=pp[i], alpha=0.2)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=60, interval=50)
    # plt.show()

    ani.save("Data/animation.gif", writer='pillow')
    # ani.save("Data/animation.webp", writer='pillow')

Cth_TAL_speeds_TrainingGraph_small()



