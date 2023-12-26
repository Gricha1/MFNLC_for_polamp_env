from abc import ABC

import numpy as np


class ObjectBase(ABC):
    def __init__(self, state: np.ndarray):
        self.state = state


class Circle(ObjectBase):
    def __init__(self, state: np.ndarray, radius: float):
        super(Circle, self).__init__(state)
        self.radius = radius

class Polygon(ObjectBase):
    def __init__(self, state: np.ndarray, w: float, l: float):
        super(Polygon, self).__init__(state)
        assert len(self.state) == 5
        self.w = w # half for obst, full for ego
        self.l = l # half for obst, full for ego

    @property
    def x(self):
        return self.state[0]
    @property
    def y(self):
        return self.state[1]
    @property
    def theta(self):
        return self.state[2]
    
    @theta.setter 
    def theta(self, angle):
        self.state[2] = angle
