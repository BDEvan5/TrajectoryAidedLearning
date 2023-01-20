from TrajectoryAidedLearning.DataTools.plotting_utils import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator





def TAL_speeds_AvgProgress():
    cth_folder = "Data/Vehicles/Cth_speeds/"
    tal_folder = "Data/Vehicles/TAL_speeds/"
    
    plt.figure(figsize=(4.5, 2.6))
    xs = np.arange(4, 9)
    
    barWidth = 0.4
    w = 0.05
    br1 = xs - barWidth/2
    br2 = [x + barWidth for x in br1]
    
    key = "progress"
    ylabel = "Progress (%)"

    mins, maxes, means = load_time_data(cth_folder, "")
    
    plt.bar(br1, means[key], color=light_blue, width=barWidth, label="Baseline")
    plot_error_bars(br1, mins[key], maxes[key], dark_blue, w)
    
    mins, maxes, means = load_time_data(tal_folder, "")
    plt.bar(br2, means[key], color=light_red, width=barWidth, label="Trajectory-aided Learning (TAL)")
    plot_error_bars(br2, mins[key], maxes[key], dark_red, w)
        
    plt.gca().get_xaxis().set_major_locator(MultipleLocator(1))
    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))
    plt.xlabel("Maximum speed (m/s)")
    plt.ylabel(ylabel)
    
    plt.legend(loc="center", ncol=2, bbox_to_anchor=(0.5, -0.52))
        
    name = "Data/Images/" + f"TAL_speeds_AvgProgress"
    
    std_img_saving(name)
   
   
TAL_speeds_AvgProgress()