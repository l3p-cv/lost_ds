from shapely.geometry import Point as Pt, MultiPoint
import numpy as np
import cv2

from lost_ds.geometry.api import Geometry
from lost_ds.vis.geometries import draw_points


class Point(Geometry):
    
    def __init__(self):
        super().__init__()


    def to_shapely(self, data):
        return Pt(data)
    
    
    def segmentation(self, segmentation, color, anno_data, anno_format, 
                     anno_style, radius, **kwargs):
        anno_data = self.to_abs(anno_data, anno_format, segmentation.shape)
        if radius is None:
            return segmentation
        cv2.circle(segmentation, tuple(anno_data.astype(np.int32)), radius, 
                   color, cv2.FILLED)
        return segmentation
        
    
    def crop(self, crop_pos, data, **kwargs):
        xmin, ymin, xmax, ymax = crop_pos.bounds
        point = self.to_shapely(data)
        intersection = point.intersection(crop_pos)
        if intersection.is_empty:
            return [np.nan]
        
        new_points = []
        if isinstance(intersection, MultiPoint):
            new_points = list(intersection)
        else:
            new_points = [intersection]
            
        for i, point in enumerate(new_points):
            new_point = np.array(point.coords) - [xmin, ymin]
            new_points[i] = new_point.squeeze()
        
        return new_points
    

    def validate(self, data):
        return len(data.shape)==1 and len(data)==2

    
    def _draw(self, img, data, style, text, color, line_thickness, radius):
        if line_thickness is None:
            line_thickness = -1
        return draw_points(img, data, text, color, radius, line_thickness)