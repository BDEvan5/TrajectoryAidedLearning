import numpy as np
from matplotlib import pyplot as plt
import glob 
from matplotlib.ticker import MultipleLocator

def load_time_data(folder, map_name=""):
    files = glob.glob(folder + f"Results_*{map_name}*.txt")
    files.sort()
    print(files)
    keys = ["time", "success", "progress"]
    mins, maxes, means = {}, {}, {}
    for key in keys:
        mins[key] = []
        maxes[key] = []
        means[key] = []
    
    for i in range(len(files)):
        with open(files[i], 'r') as file:
            lines = file.readlines()
            for j in range(len(keys)):
                mins[keys[j]].append(float(lines[3].split(",")[1+j]))
                maxes[keys[j]].append(float(lines[4].split(",")[1+j]))
                means[keys[j]].append(float(lines[1].split(",")[1+j]))

    return mins, maxes, means



pp_light = ["#EC7063", "#5499C7", "#58D68D", "#F4D03F", "#AF7AC5", "#F5B041", "#EB984E"]            
pp_dark = ["#943126", "#1A5276", "#1D8348", "#9A7D0A", "#633974", "#9C640C"]
pp_darkest = ["#78281F", "#154360", "#186A3B", "#7D6608", "#512E5F", "#7E5109"]


def std_img_saving(name):
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(name + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(name + ".svg", bbox_inches='tight', pad_inches=0)

def plot_error_bars(x_base, mins, maxes, dark_color, w):
    for i in range(len(x_base)):
        xs = [x_base[i], x_base[i]]
        ys = [mins[i], maxes[i]]
        plt.plot(xs, ys, color=dark_color, linewidth=2)
        xs = [x_base[i]-w, x_base[i]+w]
        y1 = [mins[i], mins[i]]
        y2 = [maxes[i], maxes[i]]
        plt.plot(xs, y1, color=dark_color, linewidth=2)
        plt.plot(xs, y2, color=dark_color, linewidth=2)

   
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
        
      
def plot_reward_barplot_series(folder):
    keys = ["time", "success", "progress"]
    ylabels = "Time (s), Success (%), Progress (%)".split(", ")
    
    for i in range(len(keys)):
        make_reward_barplot(folder, keys[i], ylabels[i])
        
   
def make_reward_barplot(folder, key, ylabel):
    plt.figure(figsize=(2.2, 2.4))
    xs = np.arange(2)
    
    barWidth = 0.4
    w = 0.05
    br1 = xs - barWidth/2
    br2 = [x + barWidth for x in br1]

    mins, maxes, means = load_time_data(folder, "Progress")
    
    plt.bar(br1, means[key], color=pp_light[2], width=barWidth, label="Prog.")
    plot_error_bars(br1, mins[key], maxes[key], pp_darkest[2], w)
    
    mins, maxes, means = load_time_data(folder, "Cth")
    plt.bar(br2, means[key], color=pp_light[0], width=barWidth, label="Cth")
    plot_error_bars(br2, mins[key], maxes[key], pp_darkest[0], w)
        
    plt.gca().get_xaxis().set_major_locator(MultipleLocator(1))
    plt.ylabel(ylabel)
    plt.xticks([0, 1], ["ESP", "MCO"])
    
    plt.legend(ncol=2, loc="center", bbox_to_anchor=(0.5, -0.25))
        
    name = folder + f"RewardBarplot_{key}_{folder.split('/')[-2]}"
    
    std_img_saving(name)
   
         
def make_reward_barplot_combined(folder):
    # plt.figure(figsize=(3.3, 1.5))
    fig, axs = plt.subplots(1, 2, figsize=(4.5, 2.0))
    # plt.figure(figsize=(2.2, 2.4))
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
    # plt.legend(ncol=2, loc="center", bbox_to_anchor=(0.5, -0.25))
    # fig.legend(ncol=2, loc="center", bbox_to_anchor=(0.5, -0.05))
    # fig.legend(["Progress", "Cth"], ncol=2, loc="center", bbox_to_anchor=(0.5, -0.05))
        
    name = folder + f"RewardBarplot_combined_{folder.split('/')[-2]}"
    
    std_img_saving(name)
   
      

def make_speed_barplot_cth_vs_tal():
    cth_folder = "Data/Vehicles/Cth_speeds/"
    tal_folder = "Data/Vehicles/TAL_speeds/"
    
    plt.figure(figsize=(3.9, 1.9))
    # plt.figure(figsize=(3.3, 1.9))
    xs = np.arange(4, 9)
    
    barWidth = 0.4
    w = 0.05
    br1 = xs - barWidth/2
    br2 = [x + barWidth for x in br1]
    
    key = "progress"
    ylabel = "Progress (%)"

    mins, maxes, means = load_time_data(cth_folder, "")
    
    plt.bar(br1, means[key], color=pp_light[4], width=barWidth, label="Baseline")
    plot_error_bars(br1, mins[key], maxes[key], pp_darkest[4], w)
    
    mins, maxes, means = load_time_data(tal_folder, "")
    plt.bar(br2, means[key], color=pp_light[5], width=barWidth, label="TAL")
    plot_error_bars(br2, mins[key], maxes[key], pp_darkest[5], w)
        
    plt.gca().get_xaxis().set_major_locator(MultipleLocator(1))
    plt.xlabel("Maximum speed (m/s)")
    plt.ylabel(ylabel)
    
    plt.legend(framealpha=0.95)
        
    name = "Data/Vehicles/" + f"compareBarPlot_{key}"
    
    std_img_saving(name)
   
      
def plot_six_barplot_series():
    keys = ["time", "success", "progress"]
    ylabels = "Time (s), Success (%), Progress (%)".split(", ")
    
    for i in range(len(keys)):
        make_barplot_cth_vs_tal_6(keys[i], ylabels[i])
        
   
   
def make_barplot_cth_vs_tal_6(key, ylabel):
    cth_folder = "Data/Vehicles/Cth_maps/"
    tal_folder = "Data/Vehicles/TAL_maps/"
    
    # plt.figure(figsize=(3.9, 1.9))
    plt.figure(figsize=(2.5, 1.9))
    # plt.figure(figsize=(3.3, 1.9))
    xs = np.arange(4)
    
    barWidth = 0.4
    w = 0.05
    br1 = xs - barWidth/2
    br2 = [x + barWidth for x in br1]
    
    # key = "progress"
    # ylabel = "Progress (%)"

    mins, maxes, means = load_time_data(cth_folder, "")
    
    plt.bar(br1, means[key], color=pp_light[4], width=barWidth, label="Baseline")
    plot_error_bars(br1, mins[key], maxes[key], pp_darkest[4], w)
    
    mins, maxes, means = load_time_data(tal_folder, "")
    plt.bar(br2, means[key], color=pp_light[5], width=barWidth, label="TAL")
    plot_error_bars(br2, mins[key], maxes[key], pp_darkest[5], w)
        
    plt.gca().get_xaxis().set_major_locator(MultipleLocator(1))
    # plt.xlabel("Maximum speed (m/s)")
    plt.xticks([0, 1, 2, 3], ["AUT", "ESP", "GBR", "MCO"])
    plt.ylabel(ylabel)
    
    
    plt.legend(framealpha=0.95, ncol=2)
        
    name = "Data/Vehicles/" + f"CthVsTal6_{key}"
    
    std_img_saving(name)
   
def make_barplot_cth_vs_tal_6_combined():
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

    for z in range(2):
        key = keys[z]
        plt.sca(axs[z])
        mins, maxes, means = load_time_data(cth_folder, "")
        
        plt.bar(br1, means[key], color=pp_light[4], width=barWidth, label="Baseline")
        plot_error_bars(br1, mins[key], maxes[key], pp_darkest[4], w)
        
        mins, maxes, means = load_time_data(tal_folder, "")
        plt.bar(br2, means[key], color=pp_light[5], width=barWidth, label="TAL")
        plot_error_bars(br2, mins[key], maxes[key], pp_darkest[5], w)
            
        plt.gca().get_xaxis().set_major_locator(MultipleLocator(1))
        plt.xticks([0, 1, 2, 3], ["AUT", "ESP", "GBR", "MCO"])
        plt.ylabel(ylabels[z])
        plt.grid(True)
    
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="center", bbox_to_anchor=(0.55, 0.01))
        
    name = "Data/Vehicles/" + f"CthVsTal6_combined"
    
    std_img_saving(name)
   
   
def make_barplot_cth_vs_tal_6_success():
    cth_folder = "Data/Vehicles/Cth_maps/"
    tal_folder = "Data/Vehicles/TAL_maps/"
    
    key = "success"
    ylabel  = "Success (%)"
    
    plt.figure(figsize=(3.9, 1.9))
    xs = np.arange(4)
    
    barWidth = 0.4
    w = 0.05
    br1 = xs - barWidth/2
    br2 = [x + barWidth for x in br1]
    
    mins, maxes, means = load_time_data(cth_folder, "")
    
    plt.bar(br1, means[key], color=pp_light[4], width=barWidth, label="Baseline")
    plot_error_bars(br1, mins[key], maxes[key], pp_darkest[4], w)
    
    mins, maxes, means = load_time_data(tal_folder, "")
    plt.bar(br2, means[key], color=pp_light[5], width=barWidth, label="TAL")
    plot_error_bars(br2, mins[key], maxes[key], pp_darkest[5], w)
        
    plt.gca().get_xaxis().set_major_locator(MultipleLocator(1))
    # plt.xlabel("Maximum speed (m/s)")
    plt.xticks([0, 1, 2, 3], ["AUT", "ESP", "GBR", "MCO"])
    plt.ylabel(ylabel)
    
    
    # plt.legend(framealpha=0.95, ncol=1, loc="center", bbox_to_anchor=(1.2, 0.5))
    plt.legend(ncol=2, loc="center", bbox_to_anchor=(0.5, 1.1))
        
    name = "Data/Vehicles/" + f"CthVsTal6_{key}_export"
    
    std_img_saving(name)
   
   
   
make_barplot_cth_vs_tal_6_combined()
# plot_six_barplot_series()
# make_barplot_cth_vs_tal_6_success()
# plot_speed_barplot_series("Data/Vehicles/Cth_speedMaps/")
# plot_barplot_series("Data/Vehicles/CthVsProgress/")
# make_reward_barplot_combined("Data/Vehicles/CthVsProgress/")