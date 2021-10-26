from shapely.geometry import mapping, MultiPolygon, Polygon as Poly
import numpy as np
import cv2

from lost_ds.geometry.api import Geometry
from lost_ds.vis.geometries import draw_polygons

class Polygon(Geometry):
    
    def __init__(self):
        super().__init__()


    def to_shapely(self, data):
        return Poly(data)
    
    
    def crop(self, crop_pos, data, **kwargs):
        xmin, ymin, xmax, ymax = crop_pos.bounds
        poly = self.to_shapely(data).buffer(0)
        intersection = poly.intersection(crop_pos)
        if intersection.is_empty:
            return [np.nan]
        
        new_polys = []
        if isinstance(intersection, MultiPolygon):
            new_polys = list(intersection)
        else:
            new_polys = [intersection]
        
        for i, polygon in enumerate(new_polys):
            new_poly = np.array(
                mapping(polygon)['coordinates']) - [xmin, ymin]
            new_polys[i] = new_poly.squeeze()
        
        return new_polys
        
                        
    def validate(self, data):
        return len(data.shape)==2 and len(data)>=4


    def segmentation(self, segmentation, color, anno_data, anno_format, 
                     anno_style, **kwargs):
        anno_data = self.to_abs(anno_data, anno_format, segmentation.shape)
        cv2.fillPoly(segmentation, [anno_data.astype(np.int32)], color)
        return segmentation


    def _draw(self, img, data, style, text, color, line_thickness, **kwargs):
        if line_thickness is None:
            line_thickness = 2
        return draw_polygons(img, data, text, color, line_thickness)