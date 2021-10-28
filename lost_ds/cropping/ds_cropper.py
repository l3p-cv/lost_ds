import math

import numpy as np
import pandas as pd 
from shapely.geometry import box

from lost_ds.im_util import pad_image
from lost_ds.im_util import get_imagesize, get_fs
from lost_ds.geometry.lost_geom import LOSTGeometries
from lost_ds.io.file_man import FileMan


class DSCropper:
    """ Class to apply crops to images and annotations """
    def __init__(self, 
                 lost_geometry:LOSTGeometries=None, 
                 filesystem:FileMan=None) -> None:
        self.geom = lost_geometry
        if lost_geometry is None:
            self.geom = LOSTGeometries()
        self.fs = get_fs(filesystem)
        

    def crop_anno(self, img_path, df, crop_position, im_w=None, im_h=None, 
                  padding=None):
        """Calculate annos for a crop-position in an image
        
        Args:
            img_path (str): image to calculate the crop-annos
            crop_position (list): crop-position like: [xmin, ymin, xmax, ymax] 
                in absolute data-format
            im_w (int): width of image if available
            im_h (int): height of image if available
            df (pd.DataFrame): dataframe to apply 
            padding (tuple): image padding in absolute pixel values -
                ((top_pad, bot_pad), (left_pad, right_pad))
            
        Returns:
            pd.DataFrame 
        """
        if not im_w or not im_h:
            im_h, im_w = get_imagesize(img_path, self.fs)
        if padding is None:
            padding = ((0, 0), (0, 0))
        img_df = df[df['img_path']==img_path]
        assert max(crop_position) > 1, 'Crop position has to be absolute!'
        # crop_position = np.clip(crop_position, 0, [im_w, im_h, im_w, im_h])
        crop_box = box(*crop_position)
        result_annos = []
        for idx, row in img_df.iterrows():
            cropped_anno = self.geom.apply_crop(row.anno_data, row.anno_dtype, 
                                                row.anno_format, row.anno_style, 
                                                (im_h, im_w), crop_box, padding)
            for crop_anno in cropped_anno:
                new_anno = row.copy()
                new_anno['anno_data'] = crop_anno
                result_annos.append(new_anno)
                
        return pd.DataFrame(result_annos)
        

    def crop_img(self, img, crop_shape=(512, 512), overlap=(0, 0), 
                 fill_value=0):
        '''Crop a 3D image and get a 4D array with crops
        
        Args:
            img (np.array): (H, W, C) Image to crop
            crop_shape (tuple): (H, W) shape of crops
            overlap (tuple): (H, W) overlap between crops
            fill_value (float): pixel-value to fill the area where crops are 
                out of the image
        
        Returns:
            (np.ndarray, np.ndarray, tuple): 
                1. batch of cropped images as 4D-array with shape 
                    [N, crop_shape[0], crop_shape[1], channels] 
                2. positions of shape [N, 4] in format [xmin, ymin, xmax, ymax]
                3. image padding ((top_pad, bot_pad), (left_pad, right_pad))
        '''
        im_h, im_w, im_c = img.shape
        ol = overlap
        num_y_crops = math.ceil((im_h - ol[0]) / (crop_shape[0] - ol[0]))
        num_x_crops = math.ceil((im_w - ol[1]) / (crop_shape[1] - ol[1]))
        target_shape = [num_y_crops*(crop_shape[0] - ol[0]) + ol[0],
                        num_x_crops*(crop_shape[1] - ol[1]) + ol[1]]
        
        img, padding = pad_image(img, target_shape, fill_value)
        xy_steps = np.array(crop_shape) - ol
        crops = self._sliding_window_view(img,
                                    tuple(crop_shape) + (im_c, ),
                                    tuple(xy_steps) + (1, ))
        crop_pos = self._crop_positions(num_x_crops, num_y_crops, crop_shape,ol)
        
        return crops.reshape((-1, *crop_shape, im_c)), crop_pos, padding


    def _sliding_window_view(self, arr, window_shape, steps):
        ''' get sliding window view
        
        Args:
            arr (np.ndarray): array to slide over
            window_shape (tuple): shape of the window
            steps (tuple): steps of the window in each dimension
        
        Returns:
            np.ndarray: view of the defined sliding window
        '''
        # Code basically borrowed from
        # https://gist.github.com/meowklaski/4bda7c86c6168f3557657d5fb0b5395a
        in_shape = np.array(arr.shape[-len(steps):])
        window_shape = np.array(window_shape)
        steps = np.array(steps)
        nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`
        window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
        step_strides = tuple(window_strides[-len(steps):] * steps)
        strides = tuple(int(i) * nbytes for i in step_strides + window_strides)
        outshape = tuple((in_shape - window_shape) // steps + 1)
        outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)
        return np.lib.stride_tricks.as_strided(arr, shape=outshape, 
                                               strides=strides, writeable=False)


    def _crop_positions(self, num_x_crops, num_y_crops, crop_shape, overlap):
        '''calculate crop positions for an image
        
        Args:
            num_x_crops (int): number of crops in x direction
            num_y_crops (int): number of crops in y direction
            crop_shape (tuple of int): (H, W)/(y, x) size of one crop
            overlap (tuple of int): (H, W)/(y, x) overlap between crops
            
        Returns:
            np.ndarray: crop positions in shape [N, 4] in style 
                [xmin, ymin, xmax, ymax]. 
                The order of the crops is from top left to bottom right. The 
                indices are to interpret like in this 3x3 example:
                 ___________
                |_0_|_1_|_2_|
                |_3_|_4_|_5_|
                |_6_|_7_|_8_|
                
        '''
        batch_size = num_x_crops * num_y_crops
        y_crop_i = np.sort(np.tile(np.arange(num_y_crops), num_x_crops))
        x_crop_i = np.tile(np.arange(num_x_crops), num_y_crops)
        # crop ids of shape [N, 4] with [xmin, ymin, xmax, ymax]
        crop_i = np.transpose([x_crop_i, y_crop_i, 1+x_crop_i, 1+y_crop_i]) 
        crop_positions = np.full((batch_size, 4), crop_shape[::-1] * 2)
        crop_positions = crop_positions * crop_i 
        ol_offset = np.hstack([crop_i[:, :2] * overlap[::-1]] * 2)
        crop_positions -= ol_offset
        return crop_positions
        