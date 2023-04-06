from matplotlib import pyplot as plt
# plt.rc('font', family='serif')
# plt.rc('pdf',fonttype = 42)
# plt.rc('text', usetex=True)
plt.rcParams['pdf.use14corefonts'] = True

import numpy as np
import glob
import os

import glob
from matplotlib.ticker import PercentFormatter
from matplotlib.collections import LineCollection

from TrajectoryAidedLearning.DataTools.MapData import MapData
from TrajectoryAidedLearning.Utils.StdTrack import StdTrack 
from TrajectoryAidedLearning.Utils.RacingTrack import RacingTrack
from TrajectoryAidedLearning.Utils.utils import *
from matplotlib.ticker import MultipleLocator
from TrajectoryAidedLearning.DataTools.plotting_utils import *

# SAVE_PDF = False
SAVE_PDF = True


def ensure_path_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class AnalyseTestLapData:
    def __init__(self):
        self.path = None
        self.vehicle_name = None
        self.map_name = None
        self.states = None
        self.actions = None
        self.map_data = None
        self.std_track = None
        self.summary_path = None
        self.lap_n = 0

    def explore_folder(self, path):
        vehicle_folders = glob.glob(f"{path}*/")
        print(vehicle_folders)
        print(f"{len(vehicle_folders)} folders found")

        set = 1
        for j, folder in enumerate(vehicle_folders):
            print(f"Vehicle folder being opened: {folder}")
            
            
                
            self.process_folder(folder)

    def process_folder(self, folder):
        self.path = folder

        self.vehicle_name = self.path.split("/")[-2]
        
        # if int(self.vehicle_name.split("_")[-1]) != 1: return
        
        self.map_name = self.vehicle_name.split("_")[4]
        if self.map_name == "f1":
            self.map_name += "_" + self.vehicle_name.split("_")[5]
        self.map_data = MapData(self.map_name)
        self.std_track = StdTrack(self.map_name)
        self.racing_track = RacingTrack(self.map_name)

        if not os.path.exists(self.path + "TestingVelocities/"): 
            os.mkdir(self.path + "TestingVelocities/")    
        for self.lap_n in range(5):
            if not self.load_lap_data(): break # no more laps
            self.plot_velocity_heat_map()


    def load_lap_data(self):
        try:
            data = np.load(self.path + f"Testing/Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
        except Exception as e:
            print(e)
            print(f"No data for: " + f"Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
            return 0
        self.states = data[:, :7]
        self.actions = data[:, 7:]

        return 1 # to say success

    
    def plot_velocity_heat_map(self): 
        save_path  = self.path + "TestingVelocities/"
        
        plt.figure(1)
        plt.clf()
        points = self.states[:, 0:2]
        vs = self.states[:, 3]
        
        self.map_data.plot_map_img()

        xs, ys = self.map_data.pts2rc(points)
        points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
        points = points.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(5)
        line = plt.gca().add_collection(lc)
        cbar = plt.colorbar(line,fraction=0.046, pad=0.04, shrink=0.99)
        cbar.ax.tick_params(labelsize=25)
        plt.gca().set_aspect('equal', adjustable='box')


        txt = self.vehicle_name.split("_")[3]
        if txt == "PP": txt = "Classic"
        if txt  == "Cth": txt = "Baseline"
        # if len(txt)==5: txt = "SSS"
        # elif len(txt)==3: txt = "PP"
        # plt.text(300, 400, txt, fontsize=25, ha='left', backgroundcolor='white', color="#1B4F72")
        # plt.text(1050, 130, txt, fontsize=28, ha='left', backgroundcolor='white', color="#1B4F72")

        
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        l_txt = plt.text(350, 80, txt, fontsize=28, ha='left', backgroundcolor='white', color="#1B4F72")
        name = save_path + f"{self.vehicle_name}_velocity_map_{self.lap_n}_left"
        esp_left_limits()
        std_img_saving(name)
        # del l_txt
        l_txt.set_visible(False)

        plt.text(1050, 130, txt, fontsize=28, ha='left', backgroundcolor='white', color="#1B4F72")
        name = save_path + f"{self.vehicle_name}_velocity_map_{self.lap_n}_right"
        esp_right_limits()
        std_img_saving(name)

def esp_left_limits():
    plt.xlim(20, 620)
    plt.ylim(50, 520)

def esp_right_limits():
    plt.xlim(900, 1500)
    plt.ylim(50, 520)

def analyse_folder():

    # path = "Data/Vehicles/TAL_speeds/"
    path = "Data/Vehicles/TAL_speeds_old/"
    # path = "Data/Vehicles/Cth_speeds/"
    # path = "Data/Vehicles/PP_speeds/"
    
    TestData = AnalyseTestLapData()
    TestData.explore_folder(path)


if __name__ == '__main__':
    analyse_folder()
