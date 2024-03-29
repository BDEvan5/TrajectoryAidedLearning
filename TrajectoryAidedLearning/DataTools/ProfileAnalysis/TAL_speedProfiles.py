import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection
from TrajectoryAidedLearning.DataTools.MapData import MapData
from TrajectoryAidedLearning.RewardSignals.StdTrack import StdTrack 

from TrajectoryAidedLearning.Utils.utils import *
from TrajectoryAidedLearning.DataTools.TrainingGraphs.TrainingUtils import *
from TrajectoryAidedLearning.DataTools.plotting_utils import *


class TestLapData:
    def __init__(self, path, lap_n=0):
        self.path = path
        self.vehicle_name = self.path.split("/")[-2]
        self.map_name = self.vehicle_name.split("_")[4]
        if self.map_name == "f1":
            self.map_name += "_" + self.vehicle_name.split("_")[5]
        self.map_data = MapData(self.map_name)
        self.race_track = StdTrack(self.map_name)

        self.states = None
        self.actions = None
        self.lap_n = lap_n

        self.load_lap_data()

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

    def generate_state_progress_list(self):
        pts = self.states[:, 0:2]
        progresses = [0]
        for pt in pts:
            p = self.race_track.calculate_progress_percent(pt)
            # if p < progresses[-1]: continue
            progresses.append(p)
            
        return np.array(progresses[:-1])



def make_slip_compare_graph():
    # map_name = "f1_gbr"
    map_name = "f1_esp"
    # pp_path = f"Data/Vehicles/RacingResultsWeekend/PP_Std_{map_name}_1_0/"
    # agent_path = f"Data/Vehicles/RacingResultsWeekend/Agent_Cth_{map_name}_2_1/"

    pp_path = f"Data/Vehicles/PerformanceSpeed/PP_Std5_{map_name}_1_0/"
    agent_path = f"Data/Vehicles/PerformanceSpeed/Agent_Cth_{map_name}_3_0/"


    pp_data = TestLapData(pp_path)
    agent_data = TestLapData(agent_path, 2)

    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 1.7), sharex=True)
    ax1.plot(agent_data.states[:, 6], color=pp[1], label="Agent")
    ax1.plot(pp_data.states[:, 6], color=pp[0], label="PP")


    ax1.set_ylabel("Slip angle")
    ax1.set_xlabel("Time steps")
    ax1.legend(ncol=2)

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"Data/HighSpeedEval/SlipCompare_{map_name}.pdf", bbox_inches='tight')

    plt.show()
    
    
    

def compare_tal_pp_cth_speed():
    map_name = "f1_esp"
    path  = "Data/Vehicles/"
    a1 = path + f"Cth_speeds/fast_Std_Std_Cth_{map_name}_6_1_1/"
    a2 = path + f"TAL_speeds/fast_Std_Std_TAL_{map_name}_6_1_1/"
    a3 = path + f"PP_speeds/PP_PP_Std_PP_{map_name}_6_1_0/"
    
    # colors = ["#E67E22", "#2ECC71", "#9B59B6"]
    # vehicles = [a2, a3, a1]
    # labels = ["TAL", "PP", "Baseline"]
    
    colors = ["#2ECC71", "#E67E22", "#9B59B6"]
    vehicles = [a3, a2, a1]
    labels = ["PP", "TAL", "Baseline"]
    
    fig, (ax1) = plt.subplots(1, 1, figsize=(4.2, 1.7), sharex=True)
    for i in range(len(vehicles)):
        vehicle = TestLapData(vehicles[i], 0)
        xs = vehicle.generate_state_progress_list()*100
        ax1.plot(xs, vehicle.states[:, 3], color=colors[i], label=labels[i], linewidth=2)

    ax1.set_ylabel("Speed (m/s)")
    ax1.set_xlabel("Track progress (%)")
    ax1.legend(ncol=3)
    ax1.yaxis.set_major_locator(MultipleLocator(2))
    # ax2.set_ylabel("Slip Angle")

    plt.grid(True)
    plt.xlim(-2, 40)
    plt.tight_layout()

    name = "Data/Images/compare_speed_tal_baseline_6"
    std_img_saving(name)


def compare_tal_pp_cth_slip_6():
    map_name = "f1_esp"
    path  = "Data/Vehicles/"
    a1 = path + f"Cth_speeds/fast_Std_Std_Cth_{map_name}_6_1_1/"
    a2 = path + f"TAL_speeds/fast_Std_Std_TAL_{map_name}_6_1_1/"
    a3 = path + f"PP_speeds/PP_PP_Std_PP_{map_name}_6_1_0/"

    
    # colors = ["#2ECC71", "#E67E22", "#9B59B6"]
    # vehicles = [a3, a2, a1]
    # labels = ["PP", "TAL", "Baseline"]
    colors = ["#9B59B6", "#E67E22", "#2ECC71"]
    vehicles = [a1, a2, a3]
    labels = ["Baseline", "TAL","PP"]
    
    fig, (ax1) = plt.subplots(1, 1, figsize=(4.2, 1.7), sharex=True)
    for i in range(len(vehicles)):
        vehicle = TestLapData(vehicles[i], 0)
        xs = vehicle.generate_state_progress_list()*100
        slip = np.rad2deg(vehicle.states[:, 6])
        slip = np.abs(slip)
        ax1.plot(xs, slip, color=colors[i], label=labels[i], linewidth=2)

    ax1.set_ylabel("Slip angle (deg)")
    # ax1.set_ylabel("Speed (m/s)")
    ax1.set_xlabel("Track progress (%)")
    handles, labels = ax1.get_legend_handles_labels()
    h2 = [handles[1], handles[0], handles[2]]
    l2 = [labels[1], labels[0], labels[2]]
    # ax1.legend(handles[::-1], labels[::-1], ncol=2)
    # ax1.legend(handles, labels, ncol=2)
    ax1.legend(h2, l2, ncol=2, loc="upper right")
    # ax1.legend(ncol=2, loc="upper right")
    ax1.yaxis.set_major_locator(MultipleLocator(10))
    # ax2.set_ylabel("Slip Angle")

    plt.grid(True)
    plt.xlim(-2, 40)
    plt.tight_layout()

    name = "Data/Images/compare_slip_tal_baseline_6"
    std_img_saving(name)


def TAL_baseline_PP_slip_speed_6():
    map_name = "f1_esp"
    path  = "Data/Vehicles/"
    a1 = path + f"Cth_speeds/fast_Std_Std_Cth_{map_name}_6_1_1/"
    a2 = path + f"TAL_speeds_old/fast_Std_Std_TAL_{map_name}_6_1_1/"
    a3 = path + f"PP_speeds/PP_PP_Std_PP_{map_name}_6_1_0/"

    
    # colors = ["#9B59B6", "#E67E22", "#2ECC71"]
    colors = [plot_blue, plot_red, plot_green]
    vehicles = [a1, a2, a3]
    labels = ["Baseline", "TAL","Classic"]
    
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(4.6, 2.7), sharex=True)
    lap_n = 1
    for i in range(len(vehicles)):
        vehicle = TestLapData(vehicles[i], lap_n)
        xs = vehicle.generate_state_progress_list()*100
        ax0.plot(xs, vehicle.states[:, 3], color=colors[i], label=labels[i], linewidth=2)

        
        vehicle = TestLapData(vehicles[i], lap_n)
        xs = vehicle.generate_state_progress_list()*100
        slip = np.rad2deg(vehicle.states[:, 6])
        slip = np.abs(slip)
        ax1.plot(xs, slip, color=colors[i], label=labels[i], linewidth=2)
        

    ax1.set_ylabel("Slip angle (deg)")
    ax0.set_ylabel("Speed (m/s)")
    ax1.set_xlabel("Track progress (%)")
    handles, labels = ax1.get_legend_handles_labels()
    # h2 = [handles[1], handles[0], handles[2]]
    # l2 = [labels[1], labels[0], labels[2]]
    h2, l2 = handles, labels
    fig.legend(h2, l2, ncol=3, loc="center", bbox_to_anchor=(0.5, 0))
    ax1.yaxis.set_major_locator(MultipleLocator(10))
    ax0.yaxis.set_major_locator(MultipleLocator(2))
    
    ax0.grid(True)
    plt.grid(True)
    plt.xlim(0, 60)
    # plt.xlim(-2, 40)
    plt.tight_layout()

    name = "Data/Images/TAL_baseline_PP_slip_speed_6"
    std_img_saving(name)


def TAL_PP_speed_profile_8():
    map_name = "f1_esp"
    path  = "Data/Vehicles/"
    # a1 = path + f"Cth_speeds/fast_Std_Std_Cth_{map_name}_6_1_1/"
    a2 = path + f"TAL_speeds_old/fast_Std_Std_TAL_{map_name}_8_1_0/"
    a3 = path + f"PP_speeds/PP_PP_Std_PP_{map_name}_8_1_0/"
    
    # colors = ["#2ECC71", "#E67E22"]
    colors = [plot_green, plot_red]
    vehicles = [a3, a2]
    labels = ["Classic", "Trajectory-aided learning (TAL)"]
    
    fig, (ax1) = plt.subplots(1, 1, figsize=(4.6, 1.7), sharex=True)
    for i in range(len(vehicles)):
        vehicle = TestLapData(vehicles[i], 1)
        xs = vehicle.generate_state_progress_list()*100
        ax1.plot(xs, vehicle.states[:, 3], color=colors[i], label=labels[i], linewidth=2)

    ax1.set_ylabel("Speed (m/s)")
    ax1.set_xlabel("Track progress (%)")
    fig.legend(ncol=3, loc="center", bbox_to_anchor=(0.5, 0))
    ax1.yaxis.set_major_locator(MultipleLocator(2))
    # ax2.set_ylabel("Slip Angle")

    plt.grid(True)
    plt.xlim(30, 89)
    # plt.xlim(-2, 40)
    plt.tight_layout()

    name = "Data/Images/TAL_PP_speed_profile_8"
    std_img_saving(name)



# TAL_baseline_PP_slip_speed_6()
TAL_PP_speed_profile_8()
