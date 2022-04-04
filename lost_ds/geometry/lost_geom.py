from typing import Union
import numpy as np
import pandas as pd 
from shapely.geometry import box, Polygon as Poly


from lost_ds.geometry.api import Point, Polygon, Line, Bbox

class LOSTGeometries:
    
    def __init__(self):
        self.point = Point()
        self.polygon = Polygon()
        self.bbox = Bbox()
        self.line = Line()
    
    def _get_geometry(self, dtype):
        if dtype == 'polygon':
            return self.polygon
        elif dtype == 'point':
            return self.point
        elif dtype == 'bbox':
            return self.bbox
        elif dtype == 'line':
            return self.line
    
    def serializable(self, data):
        if 'numpy' in str(type(data)):
            data = data.tolist()
        if not isinstance(data, list):
            return [[data]]
        elif not isinstance(data[0], list):
            return [data]
        elif not isinstance(data[0][0], list):
            return data           
        else:
            raise ValueError('Unsupported datatype: {}!'.format(type(data)))
    
    def validate(self, data, dtype):
        geom = self._get_geometry(dtype)
        if geom is not None:
            return geom.validate(data)
        else: 
            return True
    
    def poly_to_bbox(self, data, dst_style):
        '''Create a bbox from a polygon
        Args:
            data (pd.Series): polygon annodata
            dst_style (str): dst style for bbox
        Returns:
            pd.Series like row with bbox data
        '''
        return self.bbox.from_polygon(data, dst_style)
    
    
    def to_abs(self, data, dtype, data_format, img_shape):
        ''' Transform annotation to absolute
        
        Args:
            data (np.ndarray): data to transform
            dtype (str): anno dtype {'polygon', 'bbox', ...}
            data_format (str): format of data {'rel', 'abs'}
            img_shape (tuple or list of int): image shape like (H, W)
        
        Returns:
            np.ndarray: annotation transformed to absolute
        '''
        new_data = data
        if isinstance(data, np.ndarray):
            geom = self._get_geometry(dtype)
            new_data = geom.to_abs(data, data_format, img_shape)
        return new_data
    
    
    def to_rel(self, data, dtype, data_format, img_shape):
        ''' Transform annotation to relative
        
        Args:
            data (np.ndarray): data to transform
            dtype (str): anno dtype {'polygon', 'bbox', 'line', 'point'}
            data_format (str): format of data {'rel', 'abs'}
            img_shape (tuple or list of int): image shape like (H, W)
        
        Returns:
            np.ndarray: annotation transformed to relative
        '''
        new_data = data
        if isinstance(data, np.ndarray):
            geom = self._get_geometry(dtype)
            new_data = geom.to_rel(data, data_format, img_shape)
        return new_data
    
    
    def apply_crop(self, data, dtype, data_format, data_style, img_shape,
                   crop_position:Union[list, Poly], padding):
        ''' Crop an annotation
        
        Args:
            data (np.ndarray): anno-data to crop
            dtype (str): anno dtype {'polygon', 'bbox', 'line', 'point'}
            data_format (str): anno format {'rel', 'abs'}
            img_shape (list or tuple of int): img shape like (H, W)
            crop_position (list or shapely.Polygon): crop position in absolute 
                data format. If it is 
            padding (tuple): image padding in absolute pixel values -
                ((top_pad, bot_pad), (left_pad, right_pad))
        
        Returns:
            list of np.ndarray: cropped data in original data_format
        '''
        new_data = [data]
        if isinstance(data, np.ndarray):
            geom = self._get_geometry(dtype)
            if isinstance(crop_position, list):
                crop_position = box(*crop_position)
            tmp_format = 'abs'
            new_data = self.to_abs(data, dtype, data_format, img_shape)
            new_data = geom.offset(new_data, (padding[0][0], padding[1][0]))
            new_data = geom.apply_crop(crop_position, new_data, data_style)
            # reverse format to original
            if data_format == 'rel':
                xmin, ymin, xmax, ymax = crop_position.bounds
                crop_h, crop_w = ymax - ymin, xmax - xmin
                for i in range(len(new_data)):
                    new_data[i] = self.to_rel(new_data[i], dtype, tmp_format, 
                                              (crop_h, crop_w))
        return new_data
    
    
    def transform_bbox_style(self, data, src_style, dst_style):
        return self.bbox.transform_style(data, src_style, dst_style)
    
    
    #
    #   Drawing
    #
    def segmentation(self, segmentation, color, anno_data, anno_dtype, 
                     anno_format, anno_style, **kwargs):
        ''' draw annotation onto an image as segmentation
        
        Args:
            segmentation (np.ndarray): segmentation image to draw on
            color (int): color to use for segmentation
            anno_data (np.ndarray): annotation to draw
            anno_dtype (str): anno-type {'point', 'polygon', 'bbox', 'line'}
            anno_format (str): data format {'rel', 'abs'}
            anno_style (str): anno-style {'xy', 'xywh', 'xcycwh', 'x1y1x2y2'}

        Returns:
            image with segmentation drawn
        '''
        geom = self._get_geometry(anno_dtype)
        segmentation = geom.segmentation(segmentation, color, anno_data, 
                                         anno_format, anno_style, **kwargs)
        return segmentation
    
    
    def draw(self, img, anno_data, anno_conf, anno_label, anno_dtype, 
             anno_style, anno_format, line_thickness, color, radius):
        '''draw annotation onto an image
        Args:
            img (np.ndarray): image to draw on
            anno_data (list of np.ndarray): annotations to draw
            anno_conf (list of float, list of list of float): confidences
            anno_label (list of str, list of list of str): labels
            anno_dtype (list of string): dtypes {'bbox', 'polygon', ...}
            anno_style (list of string): style {'xy', 'xywh', ...}
            anno_format (list of string): format {'abs', 'rel'}
            line_thickness (int, dict): int or dtype mapping like {dtype: int}.
                Value of -1 does fill the annotation.
            color (tuple, dict): (B,G,R) or class mapping like {class: (B,G,R)}
            radius (int): radius of circles/points
        Returns:
            np.ndarray: image with drawn annotations
        '''
        for i, data in enumerate(anno_data):
            if data is None:
                continue
            dtype = anno_dtype[i]
            style = anno_style[i] 
            format_ = anno_format[i]
            lbl = anno_label[i]
            conf = None
            if isinstance(anno_conf, list):
                conf = anno_conf[i]
            text = self.get_text(lbl, conf)
            c = self.get_color(lbl, color)
            thickness = self.get_line_thickness(dtype, line_thickness)    
            geom = self._get_geometry(dtype)
            img = geom.draw(img, data, format_, style, text, c, thickness, 
                            radius=radius)
        return img
        
        
    def get_text(self, label, confidence):
        '''
        Args:
            label (string, list of string): single label or multi label
            confidence (float or list of float):
        '''
        if label is None:
            return label
        conf = text = None
        lbl = label
        if isinstance(label, str):
            if len(label) == 0:
                return None
            lbl = [label]
        if len(lbl) == 0:
            return None
        if isinstance(confidence, float):
            conf = [confidence]
        if conf is not None:
            text = [l+' ('+str(round(c, 2))+')' for c, l in zip(conf, lbl)]
        else:
            text = [l for l in lbl]
        text = ', '.join(text)
        return text
    
    
    def get_color(self, label, color):
        '''Decode color for class
        Args:
            label (string, list of string): single label or multi label
            color (tuple, dict): tuple with color or dict with mapping of colors
                for different classes
        Returns:
            tuple color (B,G,R)
        '''
        if isinstance(color, tuple):
            return color
        elif isinstance(color, dict):
            lbl = label
            # just pick the first one in case of multilabel
            if isinstance(label, list):
                lbl = label[0]
            lbl = lbl[0]
            if lbl in color.keys():
                return color[lbl]
            else:
                return None
        
        
    def get_line_thickness(self, dtype, thickness):
        '''Decode thickness for anno type
        Args:
            dtype (string): annotype 
            thickness (int, dict): thickness or dict with mapping of thickness
                for different annotypes
        Returns:
            int thickness
        '''
        if isinstance(thickness, int):
            return thickness
        elif isinstance(thickness, dict):
            if dtype in thickness.keys():
                return thickness[dtype]
            else: 
                return None
        