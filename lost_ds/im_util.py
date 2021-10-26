import math

import numpy as np 

from lost_ds.util import get_fs
    
def get_imagesize(path, filesystem=None):
    '''Get the size of an image without reading the entire data
    
    Args:
        path (str): path to image
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized
        
    Returns:
        tuple of int: (im_h, im_w) height and width of the image
    '''
    fs = get_fs(filesystem)
    return fs.get_imagesize(path)


def pad_image(img, target_shape, fill_value):
    ''' Pad image to target shape
    
    Args:
        img (np.ndarray): image to pad
        target_shape (tuple): desired shape after padding [H, W]
        fill_value (int): Value to use for padding
        
    Returns:
        tuple(np.ndarray, tuple): 
            np.ndarray - padded image with target_shape
            tuple - padding ((top_pad, bot_pad), (left_pad, right_pad))
    '''
    d_pad_h = target_shape[0] - img.shape[0]
    d_pad_w = target_shape[1] - img.shape[1]
    top_pad = math.floor(d_pad_h / 2)
    bot_pad = math.ceil(d_pad_h / 2)
    left_pad = math.floor(d_pad_w / 2)
    right_pad = math.ceil(d_pad_w / 2)
    # do not pad in depth dimension (vertical_pad, horizontal_pad, depth_pad)
    pads = ((top_pad, bot_pad), (left_pad, right_pad), (0, 0))
    return np.pad(img, pads, constant_values=fill_value), pads[:2]
