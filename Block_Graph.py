import networkx as nx
from networkx import DiGraph
import os


# class Node():
#     def __init__(self, blockID, population=None, elevation=None, lr_images=None, regionID=None, azimuth = 0.0):
#         super(Node, self).__init__()



class BlockGraph(DiGraph):
    '''
    Class that represents a single graph node
    '''

    def __init__(self, blockID=None, population=None, elevation=None, lr_images=None, regionID=None, azimuth = None, width =None, height = None, offsetx = None, offsety = None, area = None):
        super(BlockGraph, self).__init__()
        self.blockID = blockID
        self.regionID = regionID
        self.population = population
        self.elevation = elevation
        self.lr_images = lr_images

        # geometry attributes
        self.width = width # leave None for now
        self.height = height

        self.area = area
        self.azimuth = azimuth
        self.geocoord_offsetx = offsetx
        self.geocoord_offsety = offsety


    def add_obj_node(self, nodeID, posx, posy,
                 azimuth = None, includ_angle = None,
                 bldg_hor=None, bldg_ver=None, bldg_shape=None, bldg_area=None,
                 bldg_type=None, bldg_height=None,
                 road_type=None, road_length=None):

        self.add_node(nodeID, posx = posx, posy = posy,
                      azimuth=azimuth,
                      bldg_shape = bldg_shape, bldg_area = bldg_area, bldg_hor = bldg_hor, bldg_ver = bldg_ver,
                            bldg_type = bldg_type,  bldg_height = bldg_height,
                            road_type = road_type, road_length = road_length)

    def add_obj_edge(self, start_id, end_id, edge_dist, azimuth=None, edge_type=None, included_angle=None):
        self.add_edge(start_id, end_id,
                            edge_dist = edge_dist, azimuth = azimuth, edge_type = edge_type, included_angle = included_angle)






