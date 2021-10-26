import numpy as np
from shapely.geometry import LineString, mapping, MultiLineString
import cv2

from lost_ds.geometry.api import Geometry
from lost_ds.vis.geometries import draw_lines

class Line(Geometry):
    
    def __init__(self):
        super().__init__()
        
    
    def to_shapely(self, data):
        return LineString(data)
    
    
    def segmentation(self, segmentation, color, anno_data, anno_format, 
                     anno_style, line_thickness, **kwargs):
        anno_data = self.to_abs(anno_data, anno_format, segmentation.shape)
        if line_thickness is None:
            return segmentation
        cv2.polylines(segmentation, [anno_data.astype(np.int32)], False, color, 
                      line_thickness)
        return segmentation
    
    
    def crop(self, crop_pos, data, **kwargs):
        xmin, ymin, xmax, ymax = crop_pos.bounds
        line = self.to_shapely(data)
        intersection = line.intersection(crop_pos)
        if intersection.is_empty:
            return [np.nan]
        
        new_lines = []
        if isinstance(intersection, MultiLineString):
            new_lines = list(intersection)
        else:
            new_lines = [intersection]
            
        for i, line in enumerate(new_lines):
            new_line = np.array(
                mapping(line)['coordinates']) - [xmin, ymin]
            new_lines[i] = new_line.squeeze()
        return new_lines
    
    
    def validate(self, data):
        return len(data.shape)==2 and len(data)>1
    
    
    def _draw(self, img, data, style, text, color, line_thickness, **kwargs):
        if line_thickness is None:
            line_thickness = 2
        return draw_lines(img, data, text, color, line_thickness)
    
    