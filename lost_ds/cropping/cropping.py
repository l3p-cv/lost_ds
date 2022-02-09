import os
from typing import Iterable

from tqdm import tqdm
import pandas as pd 
from joblib import Parallel, delayed
from lost_ds.functional.filter import label_selection, selection_mask
import numpy as np
from lost_ds.functional.transform import to_abs

from lost_ds.io.file_man import FileMan
from lost_ds.functional.validation import validate_empty_images
from lost_ds.cropping.ds_cropper import DSCropper
from lost_ds.util import get_fs


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

    
def crop_dataset(df, dst_dir, crop_shape=(500, 500), overlap=(0,0),
                 write_empty=False, fill_value=0, filesystem:FileMan=None):
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
            crop_name = img_name + '_crop_' + str(i) + '.' + img_ending
            crop_path = os.path.join(dst_dir, crop_name)
            crop_df = cropper.crop_anno(img_path, img_df, position, im_w, im_h, 
                                        padding)
            data_present = crop_df.anno_data.notnull()
            
            if data_present.any() or write_empty:
                crop_df['img_path'] = crop_path
                cropper.fs.write_img(crops[i], crop_path)
                result_df.append(crop_df)
                
        return pd.concat(result_df)
    
    crop_dfs = Parallel(n_jobs=-1)(delayed(crop_and_recalculate)(path, df) 
                                    for path, df in tqdm(df.groupby('img_path'), 
                                                     desc='crop dataset'))
    # crop_dfs = []
    # for path in tqdm(df.img_path.unique(), desc='crop dataset'):
    #     crop_dfs.append(crop_and_recalculate(path))
        
    return pd.concat(crop_dfs)


def crop_components(df, dst_dir, base_labels=-1, lbl_col='anno_lbl', context=0, 
                    context_alignment=None, min_size=None, 
                    anno_dtype=['polygon'], center_lbl_key='center_lbl', 
                    filesystem:FileMan=None):
    """Crop the entire dataset with fixed crop-shape

    Args:
        df (pd.DataFrame): dataframe to apply bbox typecast
        dst_dir (str): Directory to store the new dataset
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
    fs.makedirs(dst_dir, True)
    df = validate_empty_images(df)
    df = df[selection_mask(anno_dtype, df, 'anno_dtype')]
    if not anno_dtype[0] == 'polygon': 
        raise NotImplementedError('Component cropping supports polygons only')
    
    df = to_abs(df, verbose=False)
    
    context_y = context_x = context
    if not isinstance(context, (float, int)):
        context_y, context_x = context
    
    min_size_y = min_size_x = 0
    if not isinstance(min_size, int) and min_size is not None:
        min_size_y, min_size_x = min_size
    if isinstance(min_size, int):
        min_size_x = min_size_y = min_size
    
    def crop_and_recalculate(img_path, img_df):
        if base_labels == -1:
            base_df = img_df
        else:
            base_df = label_selection(base_labels, img_df, col=lbl_col)
        img = fs.read_img(img_path)
        im_h, im_w = img.shape[:2]
        ret_df = list()
        
        for idx, row in base_df.iterrows():
            # calculate context for crop
            polygon = row.anno_data
            (minx, miny), (maxx, maxy) = polygon.min(axis=0), polygon.max(axis=0)
            w, h = maxx - minx, maxy - miny
            x_marg, y_marg = context_x * w, context_y * h
            if context_alignment == 'flip':
                x_marg, y_marg = y_marg, x_marg
            elif context_alignment == 'max':
                x_marg = y_marg = max(x_marg, y_marg)
            elif context_alignment == 'min':
                x_marg = y_marg = min(x_marg, y_marg)
                
            range_w = int(minx - x_marg), int(maxx + x_marg)
            range_h = int(miny - y_marg), int(maxy + y_marg)
            w, h = range_w[1] - range_w[0], range_h[1] - range_h[0]
            
            if w < min_size_x:
                diff = min_size_x - w
                range_w = (int(range_w[0]-diff/2), int(range_w[1] + diff/2))
            if h < min_size_y:
                diff = min_size_y - h
                range_h = (int(range_h[0]-diff/2), int(range_h[1] + diff/2))
            
            minx, maxx,  = np.clip(range_w, 0, im_w) 
            miny, maxy = np.clip(range_h, 0, im_h)
            
            crop_pos = [minx, miny, maxx, maxy]
            crop_anno = cropper.crop_anno(img_path, img_df, crop_pos, im_w, im_h)

            # apply crop
            crop = img[miny: maxy, minx: maxx]
            crop_name = f'{img_path.split("/")[-1].split(".")[0]}_crop_{idx}.{img_path.split(".")[-1]}'
            crop_path = os.path.join(dst_dir, crop_name)
            fs.write_img(crop, crop_path)
            crop_anno['img_path'] = crop_path
            crop_anno[center_lbl_key] = row[lbl_col]
            ret_df.append(crop_anno)
        if len(ret_df):
            return pd.concat(ret_df)
        else: 
            return None
    
    crop_dfs = Parallel(n_jobs=-1)(delayed(crop_and_recalculate)(path, df) 
                                    for path, df in tqdm(df.groupby('img_path'), 
                                                     desc='crop dataset'))
    
    # crop_dfs = []
    # for path, df in tqdm(df.groupby('img_path'), desc='crop dataset'):
    #     crop_dfs.append(crop_and_recalculate(path, df))
                                    
    return pd.concat(crop_dfs)