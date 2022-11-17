import csv 
import numpy as np
from matplotlib import pyplot as plt

def load_csv_data(path):
    """loads data from a csv training file

    Args:   
        path (file_path): path to the agent

    Returns:
        rewards: ndarray of rewards
        lengths: ndarray of episode lengths
        progresses: ndarray of track progresses
        laptimes: ndarray of laptimes
    """
    rewards, lengths, progresses, laptimes = [], [], [], []
    with open(f"{path}training_data_episodes.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if float(row[2]) > 0:
                rewards.append(float(row[1]))
                lengths.append(float(row[2]))
                progresses.append(float(row[3]))
                laptimes.append(float(row[4]))

    rewards = np.array(rewards)[:-1]
    lengths = np.array(lengths)[:-1]
    progresses = np.array(progresses)[:-1]
    laptimes = np.array(laptimes)[:-1]
    
    return rewards, lengths, progresses, laptimes

def convert_to_min_max_avg(step_list, progress_list, xs):
    """Returns the 3 lines 
        - Minimum line
        - maximum line 
        - average line 
    """ 
    n = len(step_list)

    # xs = np.arange(length_xs)
    ys = np.zeros((n, len(xs)))
    # xs = np.linspace(0, x_lim, length_xs)
    # xs = np.linspace(step_list[0][0], step_list[0][-1], length_xs)
    for i in range(n):
        ys[i] = np.interp(xs, step_list[i], progress_list[i])

    min_line = np.min(ys, axis=0)
    max_line = np.max(ys, axis=0)
    avg_line = np.mean(ys, axis=0)

    return min_line, max_line, avg_line

def smooth_line(steps, progresses, length_xs=300):
    xs = np.linspace(steps[0], steps[-1], length_xs)
    smooth_line = np.interp(xs, steps, progresses)

    return xs, smooth_line


def std_img_saving(name):
    plt.savefig(name + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(name + ".svg", bbox_inches='tight', pad_inches=0)

### ----

