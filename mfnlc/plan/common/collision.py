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
    result = False
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

def collision(node, bbObs=None, center_state=False):
    if center_state:
        bb = getBB([node.x, node.y, node.theta, node.w, node.l], ego=False)
    else:
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
            if isinstance(obj_2, Polygon):
                # Проверяем, находится ли центр круга внутри прямоугольника
                if obj_2.x - obj_2.l <= obj_1.state[0] <= obj_2.x + obj_2.l and \
                     obj_2.y - obj_2.w <= obj_1.state[1] <= obj_2.y + obj_2.w:
                    return True
                # Проверяем пересечение вертикальных границ круга и прямоугольника
                h_dist = max(0, abs(obj_1.state[0] - obj_2.x) - obj_2.l)
                v_dist = max(0, abs(obj_1.state[1] - obj_2.y) - obj_2.w)
                return (h_dist**2 + v_dist**2) <= (obj_1.radius**2)
            
            elif isinstance(obj_2, Circle):
                assert len(obj_2.state) == 2
                #return np.linalg.norm(obj_1.state - obj_2.state, ord=2) <= obj_1.radius + obj_2.radius
                return np.linalg.norm(obj_1.state[:2] - obj_2.state, ord=2) <= obj_1.radius + obj_2.radius
        elif isinstance(obj_1, Polygon):
            if isinstance(obj_2, Polygon):
                if obj_1.center_state and obj_2.center_state:
                    return collision(node=obj_1, bbObs=obj_2, center_state=True)
                elif not obj_1.center_state:
                    return collision(node=obj_1, bbObs=obj_2)
                else:
                    assert 1 == 0
            else:
                assert 1 == 0
        else:
            assert 1 == 0

    def overlap_polygon_between_states(self, obj_1: Polygon, state_1: list, 
                                       state_2: list, obj_2: Polygon) -> bool:
        assert len(state_1) == len(state_2) == 5
        obj_init = deepcopy(obj_1)
        obj_init.state = deepcopy(state_1)
        obj_init.theta = obj_1.theta
        if self.overlap(obj_init, obj_2):
            return True
        
        obj_middle = deepcopy(obj_1)
        middle_x = (state_1[0] + state_2[0]) / 2
        middle_y = (state_1[1] + state_2[1]) / 2
        obj_middle.x = middle_x
        obj_middle.y = middle_y
        obj_middle.l = np.sqrt((middle_x - state_1[0]) ** 2 + (middle_y - state_1[1]) ** 2)
        if self.overlap(obj_middle, obj_2):
            return True
        
        obj_end = deepcopy(obj_1)
        obj_end.state = deepcopy(state_2)
        obj_end.theta = obj_1.theta
        if self.overlap(obj_end, obj_2):
            return True
        
        return False

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
