from TrajectoryAidedLearning.DataTools.plotting_utils import *
from matplotlib.ticker import MultipleLocator

         
def CthVsProgress_Barplot():
    folder = "Data/Vehicles/CthVsProgress/"
    fig, axs = plt.subplots(1, 2, figsize=(4.5, 2.0))
    xs = np.arange(2)
    
    barWidth = 0.4
    w = 0.05
    br1 = xs - barWidth/2
    br2 = [x + barWidth for x in br1]

    keys = ["time", "progress"]
    ylabels = "Time (s), Progress (%)".split(", ")

    for z in range(2):
        key = keys[z]
        
        mins, maxes, means = load_time_data(folder, "Progress")
        
        axs[z].bar(br1, means[key], color=pp_light[2], width=barWidth, label="Progress")
        plt.sca(axs[z])
        plot_error_bars(br1, mins[key], maxes[key], pp_darkest[2], w)
        
        mins, maxes, means = load_time_data(folder, "Cth")
        axs[z].bar(br2, means[key], color=pp_light[0], width=barWidth, label="Cross-track & Heading")
        plot_error_bars(br2, mins[key], maxes[key], pp_darkest[0], w)
            
        axs[z].xaxis.set_major_locator(MultipleLocator(1))
        axs[z].set_ylabel(ylabels[z])
        axs[z].set_xticks([0, 1], ["ESP", "MCO"])
        axs[z].grid(True)
        
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="center", bbox_to_anchor=(0.55, -0.01))
        
    name = folder + f"{folder.split('/')[-2]}_Barplot"
    
    std_img_saving(name)
   
   
   
CthVsProgress_Barplot()
 
 
      