from matplotlib import pyplot as plt
import numpy as np
import glob
import os


class VehicleData:
    def __init__(self, vehicle_id, n=3, prefix="Data/Vehicles/Cth_speedMaps/"):
        self.vehicle_id = vehicle_id
        self.prefix = prefix = prefix
        
        self.times = []
        self.success_rates = []
        self.avg_progresses = []
        
        
        for i in range(n):
            self.process_folder(vehicle_id, i)
        
        self.save_data()
        
    def process_folder(self, name, n):
        folder = self.prefix + name + "_" + str(n) 
        
        #open summary stats
        with open(f"{folder}/SummaryStatistics.txt", 'r') as file:
            lines = file.readlines()
            line = lines[2] # first lap is heading
            line = line.split(',')
            
            time = float(line[8])
            if not np.isnan(time): self.times.append(time)
            success = float(line[12])
            if not np.isnan(success): self.success_rates.append(success)
            avg_progress = float(line[7])
            if not np.isnan(avg_progress): self.avg_progresses.append(avg_progress)
            
    def save_data(self):
        functions = [np.mean, np.std, np.amin, np.amax]
        names = ["Mean", "Std", "Min", "Max"]
        
        times = np.array(self.times)
        success_rates = np.array(self.success_rates)
        progresses = np.array(self.avg_progresses)


        with open(self.prefix + "Results_" + self.vehicle_id + ".txt", 'w') as file:
            file.write(f"Metric  , Time              , Success Rate     , Avg Progress    \n")
            for i in range(len(names)):
                file.write(f"{names[i]}".ljust(10))
                t = functions[i](times)
                file.write(f", {t:14.4f}")
                file.write(f", {functions[i](success_rates):14.4f}")
                file.write(f", {functions[i](progresses):14.4f} \n")

            
            






def aggregate_runs(path):
    vehicle_folders = glob.glob(f"{path}*/")
    vehicle_folders.sort()
    
    print(f"{len(vehicle_folders)} folders found")

    id_list = []
    for j, folder in enumerate(vehicle_folders):
        print(f"Vehicle folder being opened: {folder}")
        
        vehicle_name = folder.split("/")[-2]
        vehicle_id = vehicle_name[:-2]
        print(vehicle_id)
        
        if not vehicle_id in id_list:
            id_list.append(vehicle_id)
        
    for i in range(len(id_list)):
        
        v = VehicleData(id_list[i], n=5, prefix=path)
        



# aggregate_runs("Data/Vehicles/Cth_speedMaps/")
aggregate_runs("Data/Vehicles/CthVsProgress/")
