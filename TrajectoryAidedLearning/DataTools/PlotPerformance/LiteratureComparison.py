
from TrajectoryAidedLearning.DataTools.plotting_utils import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


def LiteratureComparison_Barplot():
    brunnbauer = [32, 73, 0, 0]
    bosello = [23, 56, 48, 42]
    tal = [22, 47.5, 39, 34.6] # update these times ....
    pp = [21, 47, 38.5, 35]
    
    maps = ["AUT", "ESP", "GBR", "MCO"]
    
    plt.figure(1, figsize=(4.95, 2.2))
    barWidth = 0.2
    br1 = np.arange(len(tal)) - 0.3
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    

    
    plt.bar(br1, brunnbauer, color=light_purple, label="Brunnbauer", width=barWidth)
    plt.bar(br2, bosello, color=light_yellow, label="Bosello", width=barWidth)
    plt.bar(br3, tal, color=light_red, label="TAL", width=barWidth)
    plt.bar(br4, pp, color=light_green, label="Classic", width=barWidth)
    
    # plt.xticks([r + barWidth*2 for r in range(len(ppps))], maps)
    plt.xticks([0, 1, 2, 3], maps)
    plt.legend(loc='center', bbox_to_anchor=(0.47, -0.4), ncol=4)
    # plt.legend(loc='center', bbox_to_anchor=(1.15, 0.5))
    plt.ylabel("Lap Time (s)")
    plt.gca().get_yaxis().set_major_locator(MultipleLocator(15))
    
    plt.grid(True)
    
    name = "Data/Images/LiteratureComparison_Barplot"
    std_img_saving(name)

   
LiteratureComparison_Barplot()