import os

from joblib.parallel import Parallel, delayed 
import numpy as np
from tqdm import tqdm

from lost_ds.functional.api import (remove_empty,
                               is_multilabel,
                               label_selection,
                               )
from lost_ds.im_util import get_imagesize, get_fs
from lost_ds.geometry.lost_geom import LOSTGeometries


def _get_and_validate_order(order, df, lbl_col, seg_lbl_col):
    
    if isinstance(order, list):
        order = {lbl: idx for idx, lbl in enumerate(order)}
    order = {k: v for k, v in sorted(order.items(), key=lambda item: item[1])}
    
    # if a multilabel column is passed we have to make sure that a maximum of 
    # one label is defined in the passed 'order'. If order definition is OK the 
    # multilabel column can be expressed by a single label column by only 
    # picking labels from order. Otherwise an Exception will be thrown
    
    if is_multilabel(df, lbl_col):
        # # remove dataset labels that do not occure in order
        df[seg_lbl_col] = df[lbl_col].apply(lambda x: 
            [lbl for lbl in x if lbl in list(order.keys())])
        if df[seg_lbl_col].apply(lambda x: len(x) > 1).sum():
            raise Exception('Found entries where multiple labels from order ' \
                'are defined! Pass an order without multilabel classes and run again')
        else:
            df[seg_lbl_col] = df[seg_lbl_col].apply(lambda x: 
                x[0] if len(x) else np.nan)
    else:
        df[seg_lbl_col] = df[lbl_col]
        
    return order, df


def semantic_segmentation(order, dst_dir, fill_value, df, 
                          anno_dtypes=['polygon'], lbl_col='anno_lbl', 
                          dst_path_col='seg_path', dst_lbl_col='seg_lbl', 
                          line_thickness=None, radius=None, filesystem=None):
    '''Create semantic segmentations from polygon-annos
    
    Args:
        df (pd.DataFrame): dataframe to generate the pixel maps from
        order (dict, list): order of the classes. Higher pixel
            values will overwrite lower ones. 
            Example: order=['Car', 'Person', 'Glasses'] or  
                order={'Car': 0, 'Person': 1, 'Glasses': 2}
                Car will get pixel value 0, Person px value 1 and so on - 
                Person overwrites Car, Glasses overwrites Person ... 
        dst_dir (str): Directory to store the pixel maps.
        fill_value (int): Pixel value for not annotated areas. Usually this
            will be something like 0 for something like 'background'.
        anno_dtypes (list of string): dtypes to use for segmentation. Possible
            values are {'polygon', 'bbox', 'line', 'point'}. Annotations with
            dtypes other than anno_dtypes will be removed from dataframe
        lbl_col (str): Column containing the training labels
        line_thickness (int): thickness of line-segmentation when using an 
            annotation of dtype line. Only takes effect when anno_dtypes 
            contains 'line'
        radius (int): radius of circle-segmentation when using an annotation
            of dtype point. Only takes effect when anno_dtypes contains 'point'
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized
        
    Returns:
        pd.DataFrame: The original dataframe with new column (dst_col) 
            containing the path to the according segmentation file. 
            Furthermore the column dst_lbl_col contains the label the
            segmentation looked up in order for creation
    '''
    fs = get_fs(filesystem)
    df = remove_empty(df=df, col='anno_data')
    order, df = _get_and_validate_order(order, df, lbl_col, dst_lbl_col)
    df = df[df.anno_dtype.isin(anno_dtypes)]
    
    def generate_seg(image_path, img_df):
        geom = LOSTGeometries()
        if not fs.exists(image_path):
            raise Exception('Image {} does not exist'.format(image_path))
        im_h, im_w = get_imagesize(image_path)
        segmentation = np.full([im_h, im_w], fill_value, dtype=np.int32)

        for label, level in order.items():
            draw = label_selection(labels=[label], df=img_df, col=dst_lbl_col)
            for _, row in draw.iterrows():
                segmentation = geom.segmentation(segmentation, level,
                                                 row.anno_data, row.anno_dtype, 
                                                 row.anno_format,row.anno_style,
                                                 line_thickness=line_thickness, 
                                                 radius=radius)
        
        filename = image_path.split('/')[-1].split('.')[0] + '.png'
        seg_path = os.path.join(dst_dir, filename)
        fs.write_img(segmentation, seg_path)
        return image_path, seg_path
    
    fs.makedirs(dst_dir, exist_ok=True)
    
    # # sequential loop
    # path_map = dict()
    # for path in tqdm(image_paths, desc='segmentation'):
    #     seg_path = generate_seg(path)
    #     path_map[path] = seg_path
    
    # parallel loop
    paths = Parallel(n_jobs=-1)(delayed(generate_seg)(path, img_df) 
                                    for path, img_df in tqdm(
                                        df.groupby('img_path'), 
                                        desc='segmentation'))
    path_map = {im_path: seg_path for im_path, seg_path in paths}
    df[dst_path_col] = df.img_path.apply(lambda x: path_map[x])
    return df
