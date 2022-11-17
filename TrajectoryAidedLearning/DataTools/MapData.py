import numpy as np
from matplotlib import pyplot as plt
import csv, yaml
from PIL import Image
from matplotlib.collections import LineCollection

class MapData:
    def __init__(self, map_name):
        self.path = "maps/"
        # self.path = "map_data/"
        self.map_name = map_name

        self.xs, ys = None, None
        self.t_ss, self.t_xs, self.t_ys, self.t_ths, self.t_ks, self.t_vs, self.t_accs = None, None, None, None, None, None, None

        self.N = 0
        self.map_resolution = None
        self.map_origin = None
        self.map_img = None
        self.map_height = None
        self.map_width = None

        self.load_map_img()
        self.load_centerline()
        try:
            self.load_raceline()
        except: pass

    def load_map_img(self):
        with open(self.path + self.map_name + ".yaml", 'r') as file:
            map_yaml_data = yaml.safe_load(file)
            self.map_resolution = map_yaml_data["resolution"]
            self.map_origin = map_yaml_data["origin"]
            map_img_name = map_yaml_data["image"]

        self.map_img = np.array(Image.open(self.path + map_img_name).transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(np.float64)

        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 1.

        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]
        
    def load_centerline(self):
        xs, ys = [], []
        # with open(self.path + self.map_name + "_std.csv", 'r') as file:
        with open(self.path + self.map_name + "_centerline.csv", 'r') as file:
            csvFile = csv.reader(file)

            for i, lines in enumerate(csvFile):
                if i ==0:
                    continue
                xs.append(float(lines[0]))
                ys.append(float(lines[1]))

        self.xs = np.array(xs)
        self.ys = np.array(ys)

        self.N = len(xs)

    def load_raceline(self):
        ss, xs, ys, thetas, ks, vs, accs = [], [], [], [], [], [], []

        waypoints = np.loadtxt(self.path + self.map_name + '_raceline.csv', delimiter=',', skiprows=0)

        for i in range(len(waypoints)):
            if i ==0 or i ==1 or i ==2:
                continue
            lines = waypoints[i]
            ss.append(float(lines[0]))
            xs.append(float(lines[1]))
            ys.append(float(lines[2]))
            thetas.append(float(lines[3]))
            ks.append(float(lines[4]))
            vs.append(float(lines[5]))
            # accs.append(float(lines[6]))

        self.t_ss = np.array(ss)
        self.t_xs = np.array(xs)
        self.t_ys = np.array(ys)
        self.t_ths = np.array(thetas)
        self.t_ks = np.array(ks)
        self.t_vs = np.array(vs)
        # self.t_accs = np.array(accs)

    def xy2rc(self, xs, ys):
        xs = (xs - self.map_origin[0]) / self.map_resolution
        ys = (ys - self.map_origin[1]) /self.map_resolution
        return xs, ys

    def pts2rc(self, pts):
        return self.xy2rc(pts[:,0], pts[:,1])
    
    def plot_centre_line(self):
        xs, ys = self.xy2rc(self.xs, self.ys)
        plt.plot(xs, ys, '--', color='black', linewidth=1)

    def plot_race_line(self):
        xs, ys = self.xy2rc(self.t_xs, self.t_ys)

        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(self.t_vs.min(), self.t_vs.max())
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(self.t_vs)
        lc.set_linewidth(2)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line)

    def plot_map_img(self):
        self.map_img[self.map_img == 1] = 180
        self.map_img[self.map_img == 0 ] = 230
        self.map_img[0, 1] = 255
        self.map_img[0, 0] = 0
        plt.imshow(self.map_img, origin='lower', cmap='gray')

    def plot_map_data(self):
        self.plot_map_img()

        self.plot_centre_line()
        
        self.plot_race_line()

        plt.show()




def main():
    map_name = "f1_gbr"

    map_data = MapData(map_name)
    map_data.plot_map_data()

if __name__ == '__main__':

    main()