import numpy as np

from shapely.geometry import box, Polygon

class Geometry(object):
    
    styles = ['xy']
    formats = ['abs', 'rel']
    
    def __init__(self):
        pass
        
        
    def _get_wh_struct(self, data, hw):
        h, w = hw[:2]
        assert isinstance(data, np.ndarray)
        data_len = data.shape[-1]
        assert data_len in [2, 4]
        wh_struct = [w, h] * (data_len // 2)
        return wh_struct


    def offset(self, data, offset):
        ''' Add an offset to the annotation
        
        Args:
            data (np.array): absolute data to transform
            offset (tuple): (H, W) offset to apply to the data
            
        Returns:
            np.ndarray with offset added
        '''
        wh_struct = self._get_wh_struct(data, offset)
        return data + wh_struct


    def to_abs(self, data, data_format, img_shape):
        ''' Transform relative data to absolute
        
        Args:
            data (np.array): relative data to transform
            data_format (str): format of data, one of {'abs', 'rel'}
            img_shape (tuple): (H,W) of image
            
        Returns:
            np.ndarray: data transformed to absolute
        '''
        new_data = None
        if data_format == 'abs':
            new_data = data
        else:
            wh_struct = self._get_wh_struct(data, img_shape)
            new_data = np.array(data * wh_struct)
        return new_data
    
    
    def to_rel(self, data, data_format, img_shape):
        ''' Transform absolute data to relative
        
        Args:
            data (np.ndarray): absolute data to transform
            data_format (str): format of data, one of {'abs', 'rel'}
            img_shape (tuple): (H,W) of image
            
        Returns:
            np.ndarray: data transformed to relative
        '''
        new_data = None
        if data_format == 'rel':
            new_data = data
        else:
            wh_struct = self._get_wh_struct(data, img_shape)
            new_data = np.array(data / wh_struct)
        return new_data
    
    
    def segmentation(self, *args, **kwargs):
        raise NotImplementedError('Method segmentation() at {}'.
                                  format(type(self)))
    
    
    def draw(self, img, data, data_format, style, text, color, line_thickness, 
             **kwargs):
        data = self.to_abs(data, data_format, img.shape).astype(np.int32)
        img = self._draw(img, data, style, text, color, line_thickness,**kwargs)
        assert img is not None
        return img
    
    
    def apply_crop(self, crop_pos:Polygon, data, style):
        return self.crop(crop_pos, data, style=style)


    def blow_up(self):
        pass
    