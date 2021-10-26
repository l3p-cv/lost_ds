import numpy as np
from shapely.geometry import box, Polygon, MultiPolygon
import cv2

from lost_ds.geometry.api import Geometry
from lost_ds.vis.geometries import draw_boxes

class Bbox(Geometry):
    
    styles = {
              'lost':       'xcycwh',     # LOST-style
              'opencv':     'xywh',       # COCO and openCV-style
              'shapely':    'x1y1x2y2'    # Pascal VOC and shapely style
              }    
    
    def __init__(self):
        super().__init__()
    
        
    def from_polygon(self, data, dst_style):
        '''Create a bbox from a polygon
        Args:
            row (pd.Series): row with attributes 'anno_data', 'anno_style', 
                'anno_dtype'
            dst_style (str): dst style for bbox
        Returns:
            pd.Series like row with bbox data
        '''
        # bounds gets us xmin, ymin, xmax, ymax
        bbox = np.array(Polygon(data).bounds)
        anno_style = 'x1y1x2y2'
        if dst_style is None:
            dst_style = anno_style
        data = self.transform_style(bbox, anno_style, dst_style)
        return data
    
    
    def validate(self, data):
        '''Validate bbox data by shape
        Args:
            data (list): bbox data
        Returns:
            bool True for valid False for invalid
        '''
        return len(data.shape)==1 and len(data)==4
    
    
    def to_shapely(self, data, style):
        '''Get a shapely box Instance
        Args:
            data (list, np.ndarray): bbox data
            style (str): current bbox-style
        Returns: 
            shapely box instance 
        '''
        bbox = self.transform_style(data, style, self.styles['shapely'])
        return box(*bbox)
    
    
    def crop(self, crop_pos, data, style):
        xmin, ymin, xmax, ymax = crop_pos.bounds
        bbox = self.to_shapely(data, style)
        intersection = bbox.intersection(crop_pos)
        if intersection.is_empty:
            return [np.nan]
        
        new_bboxes = []
        if isinstance(intersection, MultiPolygon):
            new_bboxes = list(intersection)
        else:
            new_bboxes = [intersection]
            
        for i, bbox in enumerate(new_bboxes):
            new_bbox = np.array(bbox.bounds).squeeze()-[xmin, ymin, xmin, ymin]
            new_bbox = self.transform_style(new_bbox, self.styles['shapely'], 
                                            style)
            new_bboxes[i] = new_bbox
        return new_bboxes
    
    
    def _draw(self, img, data, style, text, color, line_thickness, **kwargs):
        if line_thickness is None:
            line_thickness = 2
        data = self.transform_style(data, style, 'x1y1x2y2')
        return draw_boxes(img, data, text, color, line_thickness)
    
    
    def segmentation(self, segmentation, color, anno_data, anno_format, 
                     anno_style, **kwargs):
        anno_data = self.to_abs(anno_data, anno_format, segmentation.shape)
        anno_data = self.transform_style(anno_data, anno_style, 
                                        self.styles['shapely']).astype(np.int32)
        p1, p2 = anno_data[:2], anno_data[2:]
        cv2.rectangle(segmentation, tuple(p1), tuple(p2), color, cv2.FILLED)
        return segmentation
    
    
    #
    #   style transformation
    #
    
    def xcycwh_trans(self, data, style):
        '''Transform xcycwh style
        Args:
            data (np.ndarray, list): bbox data
            style (str): new bbox style, one of {xywh, x1y1x2y2}
        Returns:
            list with bbox in new style
        '''
        xc,yc,w,h = data
        x1, y1 = xc - w/2, yc - h/2
        if style == 'xywh': return [x1, y1, w, h]
        elif style == 'x1y1x2y2': return [x1, y1, x1 + w, y1 + h]
    
    
    def xywh_trans(self, data, style):
        '''Transform xywh style
        Args:
            data (np.ndarray, list): bbox data
            style (str): new bbox style, one of {xcycwh, x1y1x2y2}
        Returns:
            list with bbox in new style
        '''
        x,y,w,h = data
        if style == 'xcycwh': return [x + w/2, y + h/2, w, h]
        elif style == 'x1y1x2y2': return [x, y, x + w, y + h]
    
    
    def x1y1x2y2_trans(self, data, style):
        '''Transform x1y1x2y2 style
        Args:
            data (np.ndarray, list): bbox data
            style (str): new bbox style, one of {xcycwh, xywh}
        Returns:
            list with bbox in new style
        '''
        x1, y1, x2, y2 = data
        w, h = x2 - x1, y2 - y1
        if style == 'xcycwh': return [x1 + w/2, y1 + h/2, w, h]
        elif style == 'xywh': return [x1, y1, w, h]
            
            
    def transform_style(self, data, src_style, dst_style):
        '''transform style of bbox
        Args:
            data (numpy.ndarray): bbox
            src_style (str): current style: {'xcycwh', 'xywh', 'x1y1x2y2'}
            dst_style (str): desired style: {'xcycwh', 'xywh', 'x1y1x2y2'}
        Returns:
            numpy.ndarray: bbox in dst_style
        '''
        if src_style == dst_style:
            return data
        new_style_data = None
        if src_style == 'xywh': 
            new_style_data = self.xywh_trans(data, dst_style)
        elif src_style == 'x1y1x2y2':
            new_style_data = self.x1y1x2y2_trans(data, dst_style)
        elif src_style == 'xcycwh':
            new_style_data = self.xcycwh_trans(data, dst_style)
        if new_style_data is None:
            raise Exception('Unknown bbox-style or empty anno-data!')
        return np.array(new_style_data)
    