
import numpy as np 
import csv
import math, cmath


def get_distance(x1=[0, 0], x2=[0, 0]):
    d = [0.0, 0.0]
    for i in range(2):
        d[i] = x1[i] - x2[i]
    return np.linalg.norm(d)

def find_closest_pt(pt, wpts):
    """
    Returns the two closes points in order along wpts
    """
    dists = [get_distance(pt, wpt) for wpt in wpts]
    min_i = np.argmin(dists)
    d_i = dists[min_i] 
    if min_i == len(dists) - 1:
        min_i -= 1
    if dists[max(min_i -1, 0) ] > dists[min_i+1]:
        p_i = wpts[min_i]
        p_ii = wpts[min_i+1]
        d_i = dists[min_i] 
        d_ii = dists[min_i+1] 
    else:
        p_i = wpts[min_i-1]
        p_ii = wpts[min_i]
        d_i = dists[min_i-1] 
        d_ii = dists[min_i] 

    return p_i, p_ii, d_i, d_ii

def get_tiangle_h(a, b, c):
    s = (a + b+ c) / 2
    A = np.sqrt(s*(s-a)*(s-b)*(s-c))
    h = 2 * A / c

    return h

def distance_potential(s, s_p, end, beta=0.2, scale=0.5):
    prev_dist = get_distance(s[0:2], end)
    cur_dist = get_distance(s_p[0:2], end)
    d_dis = (prev_dist - cur_dist) / scale

    return d_dis * beta

def find_reward(s_p):
    if s_p['collisions'][0] == 1:
        return -1
    elif s_p['lap_counts'][0] == 1:
        return 1
    return 0


def get_gradient(x1=[0, 0], x2=[0, 0]):
    t = (x1[1] - x2[1])
    b = (x1[0] - x2[0])
    if b != 0:
        return t / b
    return 1000000 # near infinite gradient. 


def get_bearing(x1=[0, 0], x2=[0, 0]):
    grad = get_gradient(x1, x2)
    dx = x2[0] - x1[0]
    th_start_end = np.arctan(grad)
    if dx == 0:
        if x2[1] - x1[1] > 0:
            th_start_end = 0
        else:
            th_start_end = np.pi
    elif th_start_end > 0:
        if dx > 0:
            th_start_end = np.pi / 2 - th_start_end
        else:
            th_start_end = -np.pi/2 - th_start_end
    else:
        if dx > 0:
            th_start_end = np.pi / 2 - th_start_end
        else:
            th_start_end = - np.pi/2 - th_start_end

    return th_start_end

def sub_angles_complex(a1, a2): 
    real = math.cos(a1) * math.cos(a2) + math.sin(a1) * math.sin(a2)
    im = - math.cos(a1) * math.sin(a2) + math.sin(a1) * math.cos(a2)

    cpx = complex(real, im)
    phase = cmath.phase(cpx)

    return phase
    


#TODO: njit these function
def robust_angle_difference_degree(x, y):
    """Returns the difference between two angles in DEGREES
    r = x - y"""
    x = np.deg2rad(x)
    y = np.deg2rad(y)
    r = np.arctan2(np.sin(x-y), np.cos(x-y))
    return np.rad2deg(r)

def robust_angle_difference_rad(x, y):
    """Returns the difference between two angles in RADIANS
    r = x - y"""
    return np.arctan2(np.sin(x-y), np.cos(x-y))


if __name__ == '__main__':
    test_angle_diff()
    pass
