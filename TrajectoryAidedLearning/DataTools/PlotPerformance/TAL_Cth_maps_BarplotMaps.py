
from TrajectoryAidedLearning.DataTools.plotting_utils import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator



   
def TAL_Cth_maps_Barplot():
    cth_folder = "Data/Vehicles/Cth_maps/"
    tal_folder = "Data/Vehicles/TAL_maps/"
    
    fig, axs = plt.subplots(1, 2, figsize=(4.5, 1.8))
    xs = np.arange(4)
    
    barWidth = 0.4
    w = 0.05
    br1 = xs - barWidth/2
    br2 = [x + barWidth for x in br1]
    
    keys = ["time", "success"]
    ylabels = "Time (s), Success (%)".split(", ")

    loading_key = "gbr_6_1_test"
    for z in range(2):
        key = keys[z]
        plt.sca(axs[z])
        mins, maxes, means = load_time_data(cth_folder, loading_key)
        
        plt.bar(br1, means[key], color=light_blue, width=barWidth, label="Baseline")
        plot_error_bars(br1, mins[key], maxes[key], dark_blue, w)
        
        mins, maxes, means = load_time_data(tal_folder, loading_key)
        plt.bar(br2, means[key], color=light_red, width=barWidth, label="TAL")
        plot_error_bars(br2, mins[key], maxes[key], dark_red, w)
            
        plt.gca().get_xaxis().set_major_locator(MultipleLocator(1))
        plt.xticks([0, 1, 2, 3], ["AUT", "ESP", "GBR", "MCO"])
        plt.ylabel(ylabels[z])
        plt.grid(True)
    
    axs[0].yaxis.set_major_locator(MultipleLocator(15))
    axs[1].yaxis.set_major_locator(MultipleLocator(25))
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="center", bbox_to_anchor=(0.55, 0.01))
        
    name = "Data/Images/" + f"TAL_Cth_maps_BarplotMaps"
    
    std_img_saving(name)
   
   
TAL_Cth_maps_Barplot()
   