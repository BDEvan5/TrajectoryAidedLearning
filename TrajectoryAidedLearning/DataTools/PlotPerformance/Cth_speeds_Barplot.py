from TrajectoryAidedLearning.DataTools.plotting_utils import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


def make_speed_barplot(folder, key, ylabel):
    plt.figure(figsize=(2.5, 1.9))
    # plt.figure(figsize=(3.3, 1.9))
    xs = np.arange(4, 9)
    
    barWidth = 0.4
    w = 0.05
    br1 = xs - barWidth/2
    br2 = [x + barWidth for x in br1]

    mins, maxes, means = load_time_data(folder, "gbr")
    
    plt.bar(br1, means[key], color=pp_light[1], width=barWidth, label="GBR")
    plot_error_bars(br1, mins[key], maxes[key], pp_darkest[1], w)
    
    mins, maxes, means = load_time_data(folder, "mco")
    plt.bar(br2, means[key], color=pp_light[5], width=barWidth, label="MCO")
    plot_error_bars(br2, mins[key], maxes[key], pp_darkest[5], w)
        
    plt.gca().get_xaxis().set_major_locator(MultipleLocator(1))
    plt.xlabel("Maximum speed (m/s)")
    plt.ylabel(ylabel)
    
    plt.legend()
        
    name = folder + f"SpeedBarPlot_{key}_{folder.split('/')[-2]}"
    
    std_img_saving(name)
   
def plot_speed_barplot_series(folder):
    keys = ["time", "success", "progress"]
    ylabels = "Time (s), Success (%), Progress (%)".split(", ")
    
    for i in range(len(keys)):
        make_speed_barplot(folder, keys[i], ylabels[i])
        
        
       
#TODO: fix this 
plot_speed_barplot_series("Data/Vehicles/Cth_speedMaps/")