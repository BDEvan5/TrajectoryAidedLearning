import matplotlib.pyplot as plt
import numpy as np
import glob

def std_img_saving(name):
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(name + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(name + ".svg", bbox_inches='tight', pad_inches=0)
    new_name = "Data/UploadImgs/" + name.split("/")[-1]
    plt.savefig(new_name + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(new_name + ".svg", bbox_inches='tight', pad_inches=0)


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


