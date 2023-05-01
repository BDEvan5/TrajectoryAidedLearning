"""
Partial code source: https://github.com/f1tenth/f1tenth_gym
Example waypoint_follow.py from f1tenth_gym
Specific function used:
- nearest_point_on_trajectory_py2
- first_point_on_trajectory_intersecting_circle
- get_actuation

Adjustments have been made

"""

import numpy as np
from TrajectoryAidedLearning.Utils.utils import init_file_struct, calculate_speed
from numba import njit
import csv
import os
from matplotlib import pyplot as plt

@njit(fastmath=True, cache=True)
def add_locations(x1, x2, dx=1):
    # dx is a scaling factor
    ret = np.array([0.0, 0.0])
    for i in range(2):
        ret[i] = x1[i] + x2[i] * dx
    return ret

@njit(fastmath=True, cache=True)
def sub_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = np.array([0.0, 0.0])
    for i in range(2):
        ret[i] = x1[i] - x2[i] * dx
    return ret


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory_py2(point, trajectory):
    '''
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    '''
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    ''' starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    '''
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t


def get_distance(x1=[0, 0], x2=[0, 0]):
    d = [0.0, 0.0]
    for i in range(2):
        d[i] = x1[i] - x2[i]
    return np.linalg.norm(d)

class Trajectory:
    def __init__(self, map_name, speed=True):
        self.map_name = map_name
        self.waypoints = None
        self.vs = None
        if speed:
            self.load_csv_track()
        else:
            self.load_csv_centerline()
        self.n_wpts = len(self.waypoints)

        self.max_reacquire = 20

        self.diffs = None 
        self.l2s = None 
        self.ss = None 
        self.o_points = None
        # self.show_pts()

    def load_csv_track(self):
        track = []
        filename = 'maps/' + self.map_name + "_raceline.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        # these get expanded
        self.waypoints = track[:, 1:3]
        self.vs = track[:, 5] 

        # these don't get expanded
        self.N = len(track)
        self.o_pts = np.copy(self.waypoints)
        self.ss = track[:, 0]
        self.diffs = self.o_pts[1:,:] - self.o_pts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 
        # self.show_trajectory()

        # self._expand_wpts()

    def load_csv_centerline(self):
        track = []
        filename = 'maps/' + self.map_name + "_centerline.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        # these get expanded
        self.waypoints = track[:, 0:2]
        # self.waypoints = self.waypoints[::-1, :]
        self.vs = np.ones_like(self.waypoints[:,0]) * 2

        # these don't get expanded
        self.N = len(track)
        self.o_pts = np.copy(self.waypoints)
        self.ss = track[:, 0]
        self.diffs = self.o_pts[1:,:] - self.o_pts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 
        # self.show_trajectory()

        # self._expand_wpts()

    def _expand_wpts(self):
        n = 5 # number of pts per orig pt 
        dz = 1 / n
        o_line = self.waypoints
        o_vs = self.vs
        new_line = []
        new_vs = []
        for i in range(len(o_line)-1):
            dd = sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

                dv = o_vs[i+1] - o_vs[i]
                new_vs.append(o_vs[i] + dv * j * dz)

        self.waypoints = np.array(new_line)
        self.vs = np.array(new_vs)


    def get_current_waypoint(self, position, lookahead_distance):
        #TODO: for compuational efficiency, pass the l2s and the diffs to the functions so that they don't have to be recalculated
        wpts = np.vstack((self.waypoints[:, 0], self.waypoints[:, 1])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory_py2(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = self.vs[i]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], self.vs[i])
        else:
            raise Exception("Waypoint not found")

    def show_pts(self):
        from matplotlib import pyplot as plt
        plt.figure(3)
        plt.plot(self.waypoints[:,0], self.waypoints[:,1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle



class PurePursuit:
    def __init__(self, conf, run, init=True):
        self.name = run.run_name
        path = os.getcwd() + f"/Data/Vehicles/" + run.path  + self.name
        if init: 
            init_file_struct(path)
            self.mode = "racing"
        else:
            self.mode = "training"
            
        self.conf = conf
        self.run = run

        self.raceline = run.raceline
        self.speed_mode = run.pp_speed_mode
        self.max_speed = run.max_speed
        self.trajectory = Trajectory(run.map_name, run.raceline)

        self.lookahead = conf.lookahead 
        self.v_min_plan = conf.v_min_plan
        self.wheelbase =  conf.l_f + conf.l_r
        self.max_steer = conf.max_steer
        
    def plan(self, obs):
        state = obs['state']
        position = state[0:2]
        theta = state[2]
        if self.mode == "training":
            lookahead = 1 + 0.6 * state[3] / 8 # original....
        elif self.mode == "racing":
            lookahead = 1 + (self.max_speed/10) * state[3] /  self.max_speed
        lookahead_point = self.trajectory.get_current_waypoint(position, lookahead)

        if state[3] < self.v_min_plan:
            return np.array([0.0, 4])

        speed_raceline, steering_angle = get_actuation(theta, lookahead_point, position, self.lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
        if self.speed_mode == 'constant':
            speed = 2
        elif self.speed_mode == 'link':
            speed = calculate_speed(steering_angle, 0.8, 7)
        elif self.speed_mode == 'raceline':
            speed = speed_raceline
        else:
            raise Exception(f"Invalid speed mode: {self.speed_mode}")
            
        speed = min(speed, self.max_speed) # cap the speed

        action = np.array([steering_angle, speed])

        return action

    def lap_complete(self):
        pass
