from typing import List

import numpy as np
import math
from mfnlc.plan.common.geometry import ObjectBase, Circle, Polygon
from copy import deepcopy


def getBB(state, w = 2.0, l=3.8, ego=True, front_axis=True):
    new_state = deepcopy(state)
    x = new_state[0]
    y = new_state[1]
    angle = new_state[2]
    if ego:
        w = w / 2
        l = l / 2
        if front_axis:
            rear_to_center = 0.65
            shift = l - rear_to_center
            toCenter = True
            shift = -shift if toCenter else shift
            x = x + shift * math.cos(angle)
            y = y + shift * math.sin(angle)
    else:
        w = new_state[3]
        l = new_state[4]
    BBPoints = [(-l, -w), (l, -w), (l, w), (-l, w)]
    vertices = []
    sinAngle = math.sin(angle)
    cosAngle = math.cos(angle)
    for i in range(len(BBPoints)):
        new_x = cosAngle * (BBPoints[i][0]) - sinAngle * (BBPoints[i][1])
        new_y = sinAngle * (BBPoints[i][0]) + cosAngle * (BBPoints[i][1])
        vertices.append([new_x + x, new_y + y])
    
    return vertices

def project(a, axis):
    maxProj = -math.inf
    minProj = math.inf
    for v in a:
        proj = np.dot(axis, v)
        if proj < minProj:
            minProj = proj
        if proj > maxProj:
            maxProj = proj
    
    return minProj, maxProj

def separatingAxes(a, axes):
    for i in range(len(a)):
        current = a[i]
        next = a[(i + 1) % len(a)]
        edge = np.array(next) - np.array(current) 
        new_edge = edge / (np.sqrt(np.sum(edge ** 2)) + 1e-6)
        # print(f"new_edge : {new_edge}")
        axes.append([-new_edge[1], new_edge[0]])

def intersectPolygons(a, b, rl=True):
    axes = []
    new_a = []
    if rl:
        for pair in a:
            # print(pair)
            new_a.append((pair[0].x, pair[0].y))
        a = new_a
        new_b = []
        for pair in b:
            new_b.append((pair[0].x, pair[0].y))
        b = new_b 
    separatingAxes(a, axes)
    separatingAxes(b, axes)
    for axis in axes:
        aMinProj, aMaxProj, bMinProj, bMaxProj = 0., 0., 0., 0.
        aMinProj, aMaxProj = project(a, axis)
        bMinProj, bMaxProj = project(b, axis)
        if (aMinProj > bMaxProj) or (bMinProj > aMaxProj):
            return False 
    return True

def intersectPoint(point, polygon, epsilon=1e-4):
    """
      point = [x, y]
      polygon = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    """
    result = False
    i = 0
    j = len(polygon) - 1
    for i in range(len(polygon)):
        if ((polygon[i][1] > point[1]) != (polygon[j][1] > point[1]) and (point[0] < (polygon[j][0]\
            - polygon[i][0]) * (point[1] - polygon[i][1]) / (polygon[j][1] - polygon[i][1] + epsilon) + polygon[i][0])):
            result = not result
        j = i
        i += 1

    return result
    """
    result = False
    #i = 0
    #j = len(polygon) - 1
    #for i in range(len(polygon)):
    #    if ((polygon[i][1] > point[1]) != (polygon[j][1] > point[1]) and (point[0] < (polygon[j][0]\
    #        - polygon[i][0]) * (point[1] - polygon[i][1]) / (polygon[j][1] - polygon[i][1] + epsilon) + polygon[i][0])):
    #        result = not result
    #    j = i
    #    i += 1
    if point[0] > polygon[0][0] and point[1] > polygon[0][1] and\
       point[0] < polygon[2][0] and point[1] < polygon[2][1]:
        result = True

    return result

def intersect(a, b):
    axes = []
    separatingAxes(a, axes)
    separatingAxes(b, axes)
    for axis in axes:
        aMinProj, aMaxProj, bMinProj, bMaxProj = 0., 0., 0., 0.
        aMinProj, aMaxProj = project(a, axis)
        bMinProj, bMaxProj = project(b, axis)
        if (aMinProj > bMaxProj) or (bMinProj > aMaxProj):
            return False 
    return True

def collision_dist(traj_array, obstacles_array, safe_dist) -> np.ndarray:
    init = traj_array[0]
    end = traj_array[-1]
    lb = np.min([init, end], axis=0)
    ub = np.max([init, end], axis=0)

    dists = np.full(obstacles_array.shape[0], np.inf)
    safe_obj_indx = np.logical_or((obstacles_array < lb - safe_dist).any(axis=-1),
                                  (obstacles_array > ub + safe_dist).any(axis=-1))
    checking_obj_indx = np.logical_not(safe_obj_indx)

    dists[checking_obj_indx] = np.abs(
        np.cross(end - init, obstacles_array[checking_obj_indx] - init)) \
                               / np.linalg.norm(end - init)

    return dists

def collision(node, bbObs=None):
    """
    bbObs = getBB([bbObs.x, bbObs.y, bbObs.theta, bbObs.w, bbObs.l], ego=False)
    if node is None:
        return False
    if intersectPoint([node.x, node.y], bbObs):
        return True
    if node.theta != None:
        bb = getBB([node.x, node.y, node.theta], w=node.w, l=node.l)
        if intersectPolygons(bb, bbObs, rl=False):
            return True
    return False  # safe
    """
    bb = getBB([node.x, node.y, node.theta], w=node.w, l=node.l)
    bbObs = getBB([bbObs.x, bbObs.y, bbObs.theta, bbObs.w, bbObs.l], ego=False)
    if intersectPoint([node.x, node.y], bbObs):        
        return True

    if intersect(bb, bbObs):
        return True

    return False

class CollisionChecker:
    @staticmethod
    def overlap(obj_1: ObjectBase, obj_2: ObjectBase) -> bool:
        if isinstance(obj_1, Circle):
            if isinstance(obj_2, Circle):
                assert len(obj_2.state) == 2
                #return np.linalg.norm(obj_1.state - obj_2.state, ord=2) <= obj_1.radius + obj_2.radius
                return np.linalg.norm(obj_1.state[:2] - obj_2.state, ord=2) <= obj_1.radius + obj_2.radius
        elif isinstance(obj_1, Polygon):
            if isinstance(obj_2, Polygon):
                return collision(node=obj_1, bbObs=obj_2)
                #bb = getBB([node.x, node.y, node.theta], w=node.w, l=node.l)
                #bbObs = obj_2
                #bbObs = getBB([bbObs.x, bbObs.y, bbObs.theta, bbObs.w, bbObs.l], ego=False)
                #return (obj_1.x > bbObs[0][0] and obj_1.y > bbObs[0][1]) and \
                #       (obj_1.x < bbObs[2][0] and obj_1.y < bbObs[2][1])
            else:
                assert 1 == 0
        else:
            assert 1 == 0

    @staticmethod
    def seq_to_seq_overlap(traj: List[ObjectBase],
                           obstacles: List[ObjectBase]):
        traj_array = np.array([obj.state for obj in traj])
        obstacles_array = np.array([obj.state for obj in obstacles])

        if isinstance(traj[0], Circle):
            if isinstance(obstacles[0], Circle):
                safe_dist = traj[0].radius + obstacles[0].radius  # noqa
                dists = collision_dist(traj_array, obstacles_array, safe_dist)
                return (dists < safe_dist).any()
