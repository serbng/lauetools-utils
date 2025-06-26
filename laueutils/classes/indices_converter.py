import numpy as np

from graintools.utils.scan import mesh_points

class Converter:
    def __init__(self, start_point: tuple, end_point: tuple, step_size: tuple):
        self.start_point = start_point
        self.end_point   = end_point
        self.step_size   = step_size
        self.scan_points = mesh_points(start_point, end_point, step_size)
        self.length      = len(self.scan_points)
        self.scan_shape  = tuple([int((end_point[i] - start_point[i])/step_size[i]) + 1 for i in range(2)])
        
    def ij_to_index(self, i: int, j: int):
        """From scan position indices (i,j) to flattened image number"""
        # index in self.scan_points
        col = (i - self.start_point[0]) / self.step_size[0]
        row = (j - self.start_point[1]) / self.step_size[1]
        return int(row * self.scan_shape[0] + col)
    
    def ij_to_xy(self):
        pass
    
    def xy_to_index(self):
        pass
    
    def xy_to_ij(self):
        pass
    
    def index_to_ij(self):
        pass
    
    def index_to_xy(self, image_number: int):
        """From flattened image number to scan position (i,j)"""
        return tuple(self.scan_points[image_number])
    