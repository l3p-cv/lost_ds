import os

from tqdm import tqdm
import pandas as pd 
from joblib import Parallel, delayed, cpu_count
from lost_ds.functional.filter import label_selection, selection_mask, remove_empty
import numpy as np
from lost_ds.functional.transform import to_abs, polygon_to_bbox

from lost_ds.io.file_man import FileMan
from lost_ds.functional.validation import validate_empty_images
from lost_ds.cropping.ds_cropper import DSCropper
from lost_ds.util import get_fs
from lost_ds.im_util import get_imagesize


def crop_anno(img_path, crop_position, df, im_w=None, im_h=None, 
              filesystem:FileMan=None):
    """Calculate annos for a crop-position in an image
    
    Args:
        img_path (str): image to calculate the crop-annos
        crop_position (list): crop-position like: [xmin, ymin, xmax, ymax] 
            in absolute data-format
        im_w (int): width of image if available
        im_h (int): height of image if available
        df (pd.DataFrame): dataframe to apply 
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized
            
    Returns:
        pd.DataFrame 
    """
    cropper = DSCropper(filesystem=filesystem)
    return cropper.crop_anno(img_path, df, crop_position, im_w, im_h)

def crop_img(img, crop_shape=(512, 512), overlap=(0, 0), 
                fill_value=0, filesystem:FileMan=None):
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
        cropper = DSCropper(filesystem=filesystem)
        return cropper.crop_img(img, crop_shape=crop_shape, overlap=overlap, 
                                fill_value=fill_value)
    
    
def crop_dataset(df, dst_dir, crop_shape=(500, 500), overlap=(0,0),
                 write_empty=False, fill_value=0, filesystem:FileMan=None,
                 parallel=-1):
    """Crop the entire dataset with fixed crop-shape

    Args:
        df (pd.DataFrame): dataframe to apply bbox typecast
        dst_dir (str): Directory to store the new dataset
        crop_shape (tuple, list): [H, W] cropped image dimensions
        overlap (tuple, list): [H, W] overlap between crops
        write_empty (bool): Flag if crops without annos wil be written
        fill_value (float): pixel value to fill the rest of crops at borders
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized
    Returns:
        pd.DataFrame
    """
    fs = get_fs(filesystem)
    fs.makedirs(dst_dir, True)
    df = validate_empty_images(df)
    
    def crop_and_recalculate(img_path, img_df):
        cropper = DSCropper(filesystem=filesystem)
        img = cropper.fs.read_img(img_path)
        im_h, im_w, im_c = img.shape
                
        crops, positions, padding = cropper.crop_img(img, crop_shape, overlap, 
                                                     fill_value)
        img_name, img_ending = img_path.split('/')[-1].split('.')
        
        result_df = []
        for i, position in enumerate(positions):
            crop_string = f'_shape_{crop_shape[0]}_{crop_shape[1]}'
            overlap_string = f'_overlap_{overlap[0]}_{overlap[1]}'
            crop_name = img_name + '_crop_' + str(i) + crop_string + overlap_string + '.' + img_ending
            crop_path = os.path.abspath(os.path.join(dst_dir, crop_name))
            crop_df = cropper.crop_anno(img_path, img_df, position, im_w, im_h, 
                                        padding)
            data_present = crop_df['anno_data'].notnull()
            
            # padding is ((top, bot), (left, right))
            pad_y, pad_x = padding[0][0], padding[1][0]
            crop_position = position - [pad_x, pad_y, pad_x, pad_y]
            
            if data_present.any() or write_empty:
                crop_df['img_path'] = crop_path
                crop_df['crop_position'] = crop_df['img_path'].map(lambda x: crop_position)
                cropper.fs.write_img(crops[i], crop_path)
                data_df = crop_df[data_present]
                if write_empty:
                    empty_df = crop_df.loc[~data_present].drop_duplicates('img_path')
                    if not len(data_df):
                        data_df = pd.concat([data_df, empty_df])
                result_df.append(data_df)
                
        if len(result_df):
            return pd.concat(result_df)
        else:
            return None
    
    if parallel:
        crop_dfs = Parallel(n_jobs=parallel)(delayed(crop_and_recalculate)(path, img_df) 
                                        for path, img_df in tqdm(df.groupby('img_path'), 
                                                        desc='crop dataset'))
    else:
        crop_dfs = []
        for path, img_df in tqdm(df.groupby('img_path'), desc='crop dataset'):
            crop_dfs.append(crop_and_recalculate(path, img_df))
    
    ret_df = pd.concat(crop_dfs)
    
    anno_keys = [k for k in ret_df.keys() if 'anno' in k]
    # set anno-information from formerly non empty annos to empty
    ret_df.loc[ret_df['anno_data'].isnull(), anno_keys] = None
    # make anno_lbl* format compatible
    anno_keys = [k for k in ret_df.keys() if 'anno_lbl' in k]
    for k in anno_keys:
        ret_df.loc[ret_df['anno_data'].isnull(), k] = ret_df[ret_df['anno_data'].isnull()][k].map(lambda x: np.array([], 'object'))

    return ret_df


# def crop_components(df, dst_dir, base_labels=-1, lbl_col='anno_lbl', context=0, 
#                     context_alignment=None, min_size=None, 
#                     anno_dtype=['polygon'], center_lbl_key='center_lbl', 
#                     filesystem:FileMan=None, parallel=-1):
#     """Crop the entire dataset with fixed crop-shape

#     Args:
#         df (pd.DataFrame): dataframe to apply bbox typecast
#         dst_dir (str): Directory to store the new dataset
#         base_labels (list of str): labels to align the crops 
#         lbl_col (str): column holding the labels
#         context (float, tuple of floats): context to add to each component for 
#             cropping (twice -> left/right, top/bottom). If tuple of float: uses 
#             the given floats as fraction of the components (H, W) to calculate 
#             the added contexts. 
#         context_alignment (str): Define alignment of the crop-context. One of 
#             [None, 'max', 'min', 'flip']. 
#             None: Use H/W-context in H/W-dimension
#             max: use maximum of H-context, W-context for both dimensions
#             min: use minimum of H-context, W-context for both dimensions
#             flip: use H-context in W-dimension and vice versa. This makes the 
#                 crop a little bit more squarish shaped
#         min_size (int, tuple of int): minimum size of produced crops in both
#             dimensions if int or for (H, W) if tuple of int
#         anno_dtype (list of str): dtype to apply on
#         center_lbl_key (str): column containing the label the crop was aligned to
#         filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
#             if not initialized
            
#     Returns:
#         pd.DataFrame
#     """
    
#     cropper = DSCropper(filesystem=filesystem)
#     fs = get_fs(filesystem)
#     fs.makedirs(dst_dir, True)
#     df = validate_empty_images(df)
#     if len(np.setdiff1d(anno_dtype, ['polygon', 'bbox'])):
#         raise NotImplementedError(f'Component cropping does not support {anno_dtype}')
    
#     df = to_abs(df, filesystem=filesystem, verbose=False)
#     df[center_lbl_key] = False
#     context_y = context_x = context
#     if not isinstance(context, (float, int)):
#         context_y, context_x = context
    
#     min_size_y = min_size_x = 0
#     if not isinstance(min_size, int) and min_size is not None:
#         min_size_y, min_size_x = min_size
#     if isinstance(min_size, int):
#         min_size_x = min_size_y = min_size
    
#     def crop_and_recalculate(img_path, img_df):
#         if base_labels == -1:
#             base_df = img_df
#         else:
#             base_df = label_selection(base_labels, img_df, col=lbl_col)
#         if not len(base_df):
#             return None
#         crop_df = base_df[selection_mask(anno_dtype, base_df, 'anno_dtype')].copy()
#         if not len(crop_df):
#             return None
#         crop_df = polygon_to_bbox(crop_df, 'x1y1x2y2')
        
#         img = fs.read_img(img_path)
#         im_h, im_w = img.shape[:2]
#         ret_df = list()
        
#         crop_id = 0
#         for idx, row in crop_df.iterrows():
#             minx, miny, maxx, maxy = row.anno_data
#             w, h = maxx - minx, maxy - miny
#             x_marg, y_marg = context_x * w, context_y * h
#             if context_alignment == 'flip':
#                 x_marg, y_marg = y_marg, x_marg
#             elif context_alignment == 'max':
#                 x_marg = y_marg = max(x_marg, y_marg)
#             elif context_alignment == 'min':
#                 x_marg = y_marg = min(x_marg, y_marg)
                
#             range_w = int(minx - x_marg), int(maxx + x_marg)
#             range_h = int(miny - y_marg), int(maxy + y_marg)
#             w, h = range_w[1] - range_w[0], range_h[1] - range_h[0]
            
#             if w < min_size_x:
#                 diff = min_size_x - w
#                 range_w = (int(range_w[0]-diff/2), int(range_w[1] + diff/2))
#             if h < min_size_y:
#                 diff = min_size_y - h
#                 range_h = (int(range_h[0]-diff/2), int(range_h[1] + diff/2))
            
#             minx, maxx,  = np.clip(range_w, 0, im_w) 
#             miny, maxy = np.clip(range_h, 0, im_h)
            
#             crop_pos = [minx, miny, maxx, maxy]
#             crop_anno = cropper.crop_anno(img_path, img_df, crop_pos, im_w, im_h)

#             # apply crop
#             crop = img[miny: maxy, minx: maxx]
#             crop_name = f'{img_path.split("/")[-1].split(".")[0]}_crop_{crop_id}.{img_path.split(".")[-1]}'
#             crop_id += 1
#             crop_path = os.path.join(dst_dir, crop_name)
#             fs.write_img(crop, crop_path)
#             crop_anno['img_path'] = crop_path
#             crop_anno['crop_position'] = crop_anno['img_path'].map(lambda x: [minx, miny, maxx, maxy])
#             crop_anno.loc[idx, center_lbl_key] = True
#             ret_df.append(crop_anno)    
#         if len(ret_df):
#             return pd.concat(ret_df)
#         else: 
#             return None
        
#     if parallel:
#         crop_dfs = Parallel(n_jobs=parallel)(delayed(crop_and_recalculate)(path, df) 
#                                              for path, df in tqdm(df.groupby('img_path'), 
#                                                                   desc='crop dataset'))
#     else:
#         crop_dfs = []
#         for path, df in tqdm(df.groupby('img_path'), desc='crop dataset'):
#             crop_dfs.append(crop_and_recalculate(path, df))
    
#     # TODO: return empty df if noting to crop
#     return pd.concat(crop_dfs)



def crop_components(df, dst_dir=None, base_labels=-1, lbl_col='anno_lbl', context=0, 
                    context_alignment=None, min_size=None, 
                    anno_dtype=['polygon'], center_lbl_key='center_lbl', 
                    filesystem:FileMan=None, parallel=-1):
    """Crop the entire dataset with fixed crop-shape

    Args:
        df (pd.DataFrame): dataframe to apply bbox typecast
        dst_dir (str, None): Directory to store the crops. If None no cropping 
            will be done.
        base_labels (list of str): labels to align the crops 
        lbl_col (str): column holding the labels
        context (float, tuple of floats): context to add to each component for 
            cropping (twice -> left/right, top/bottom). If tuple of float: uses 
            the given floats as fraction of the components (H, W) to calculate 
            the added contexts. 
        context_alignment (str): Define alignment of the crop-context. One of 
            [None, 'max', 'min', 'flip']. 
            None: Use H/W-context in H/W-dimension
            max: use maximum of H-context, W-context for both dimensions
            min: use minimum of H-context, W-context for both dimensions
            flip: use H-context in W-dimension and vice versa. This makes the 
                crop a little bit more squarish shaped
        min_size (int, tuple of int): minimum size of produced crops in both
            dimensions if int or for (H, W) if tuple of int
        anno_dtype (list of str): dtype to apply on
        center_lbl_key (str): column containing the label the crop was aligned to
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized
            
    Returns:
        pd.DataFrame
    """
     
    cropper = DSCropper(filesystem=filesystem)
    fs = get_fs(filesystem)
    crop_image = True
    if dst_dir is None:
        crop_image = False
    if crop_image:
        fs.makedirs(dst_dir, True)
    df = validate_empty_images(df)
    if len(np.setdiff1d(anno_dtype, ['polygon', 'bbox'])):
        raise NotImplementedError(f'Component cropping does not support {anno_dtype}')
    
    df = to_abs(df, filesystem=filesystem, verbose=False)
    df[center_lbl_key] = False
    context_y = context_x = context
    if not isinstance(context, (float, int)):
        context_y, context_x = context
    
    min_size_y = min_size_x = 0
    if not isinstance(min_size, int) and min_size is not None:
        min_size_y, min_size_x = min_size
    if isinstance(min_size, int):
        min_size_x = min_size_y = min_size
    
    def crop_and_recalculate(img_path, img_df):
        # pick labels to crop
        if base_labels == -1:
            base_df = img_df
        else:
            base_df = label_selection(base_labels, img_df, col=lbl_col)
        
        if not len(base_df):
            return None
        
        # pick anno types to crop
        crop_df = base_df[selection_mask(anno_dtype, base_df, 'anno_dtype')].copy()
        
        if not len(crop_df):
            return None
        
        # crops have to be bboxes
        crop_df = polygon_to_bbox(crop_df, 'x1y1x2y2')
        
        # get image and/or information
        if crop_image:
            img = fs.read_img(img_path)
            im_h, im_w = img.shape[:2]
        else:
            im_h, im_w = get_imagesize(img_path)
        
        # iterate over images, calculate crop, crop image and anno
        ret_df = list()
        crop_id = 0
        for idx, row in crop_df.iterrows():
            
            # get crop position
            minx, miny, maxx, maxy = row.anno_data
            w, h = maxx - minx, maxy - miny
            
            # calculate context and alignment as specified in args
            x_marg, y_marg = context_x * w, context_y * h
            if context_alignment == 'flip':
                x_marg, y_marg = y_marg, x_marg
            elif context_alignment == 'max':
                x_marg = y_marg = max(x_marg, y_marg)
            elif context_alignment == 'min':
                x_marg = y_marg = min(x_marg, y_marg)
            
            # new crop shape
            range_w = int(minx - x_marg), int(maxx + x_marg)
            range_h = int(miny - y_marg), int(maxy + y_marg)
            w, h = range_w[1] - range_w[0], range_h[1] - range_h[0]
            
            # crop min size
            if w < min_size_x:
                diff = min_size_x - w
                range_w = (int(range_w[0]-diff/2), int(range_w[1] + diff/2))
            if h < min_size_y:
                diff = min_size_y - h
                range_h = (int(range_h[0]-diff/2), int(range_h[1] + diff/2))
            
            # take care the crops are inside the image
            minx, maxx,  = np.clip(range_w, 0, im_w) 
            miny, maxy = np.clip(range_h, 0, im_h)
            
            # final crop position with aligned context
            crop_pos = [minx, miny, maxx, maxy]
            
            # crop annotation
            crop_anno = cropper.crop_anno(img_path, img_df, crop_pos, im_w, im_h)
            crop_anno.loc[idx, center_lbl_key] = True
            crop_anno['crop_position'] = crop_anno['img_path'].map(lambda x: [minx, miny, maxx, maxy])
            crop_anno['crop_id'] = crop_id
            
            # crop image and save it, add path to dataframe
            if crop_image:
                crop = img[miny: maxy, minx: maxx]
                crop_name = f'{img_path.split("/")[-1].split(".")[0]}_crop_{crop_id}.{img_path.split(".")[-1]}'
                crop_path = os.path.join(dst_dir, crop_name)
                fs.write_img(crop, crop_path)
                crop_anno['img_path'] = crop_path
            
            crop_id += 1
            
            # add crop dataframe as result
            ret_df.append(crop_anno)    
            
        # return results or None if no crops occured
        if len(ret_df):
            return pd.concat(ret_df)
        else: 
            return None
        
    if parallel:
        crop_dfs = Parallel(n_jobs=parallel)(delayed(crop_and_recalculate)(path, df) 
                                             for path, df in tqdm(df.groupby('img_path'), 
                                                                  desc='crop dataset'))
    else:
        crop_dfs = []
        for path, df in tqdm(df.groupby('img_path'), desc='crop dataset'):
            crop_dfs.append(crop_and_recalculate(path, df))
    
    # TODO: return empty df if nothing to crop
    return pd.concat(crop_dfs)
