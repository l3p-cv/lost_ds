import os 
import json
from multiprocessing import Process
from typing import Union

from tqdm import tqdm
import pandas as pd 
import numpy as np
import fsspec
from joblib import Parallel, delayed
from lost_ds.copy import copy_imgs
from lost_ds.functional.validation import validate_img_paths

from lost_ds.geometry.api import LOSTGeometries
from lost_ds.geometry.bbox import Bbox
from lost_ds.im_util import get_imagesize
from lost_ds.io.file_man import FileMan
from lost_ds.functional.filter import unique_labels

        
def to_abs(df, path_col='img_path', 
           filesystem:Union[FileMan,fsspec.AbstractFileSystem]=None, 
           verbose=True):
    ''' Transform all annos to absolute annos
    
    Args:
        df (pd.DataFrame): dataframe to transform
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized
        verbose (bool): print tqdm progress-bar
            
    Returns:
        pd.DataFrame: transformed dataframe wil absolute annotations
    '''
    df = df.copy()
    df_rel = df[df['anno_format'] != 'abs']
    cols = [path_col, 'anno_data', 'anno_dtype', 'anno_format']
    def make_abs(row):
        geom = LOSTGeometries()
        anno_data = row.anno_data
        anno_format = row.anno_format
        if isinstance(anno_data, np.ndarray) and anno_format=='rel':
            img_path = row[path_col]
            anno_dtype = row.anno_dtype
            img_shape = get_imagesize(img_path, filesystem)
            anno_data = geom.to_abs(
                anno_data, anno_dtype, anno_format, img_shape
                )
        return anno_data
        
    abs_data = Parallel(-1)(delayed(make_abs)(row)
                            for idx, row in tqdm(df_rel[cols].iterrows(), 
                                                 total=len(df_rel),
                                                 desc='to abs', 
                                                 disable=(not verbose)))
    df.loc[df.anno_format != 'abs', 'anno_data'] = pd.Series(abs_data, 
                                                             index=df_rel.index)
    df['anno_format'] = 'abs'
    return df


def to_rel(df, path_col='img_path', 
           filesystem: Union[FileMan, fsspec.AbstractFileSystem] = None,
           verbose=True):
    ''' Transform all annos to absolute annos
    
    Args:
        df (pd.DataFrame): dataframe to transform
        filesystem (FileMan): Filesystem instance. If None local filesystem 
            is used
        verbose (bool): print tqdm progress-bar
            
    Returns:
        pd.DataFrame: transformed dataframe wil absolute annotations
    '''
    df = df.copy()
    df_abs = df[df['anno_format'] != 'rel']
    cols = [path_col, 'anno_data', 'anno_dtype', 'anno_format']
    def make_rel(row):
        geom = LOSTGeometries()
        anno_data = row.anno_data
        anno_format = row.anno_format
        if isinstance(anno_data, np.ndarray) and anno_format == 'abs':
            img_path = row[path_col]
            anno_dtype = row.anno_dtype
            img_shape = get_imagesize(img_path, filesystem)
            anno_data = geom.to_rel(
                anno_data, anno_dtype, anno_format, img_shape
            )
        return anno_data
        
    rel_data = Parallel(-1)(delayed(make_rel)(row) 
                            for idx, row in tqdm(df_abs[cols].iterrows(), 
                                                 total=len(df_abs),
                                                 desc='to rel',
                                                 disable=(not verbose)))
    df.loc[df.anno_format != 'rel', 'anno_data'] = pd.Series(rel_data, 
                                                             index=df_abs.index)
    df['anno_format'] = 'rel'
    return df


def transform_bbox_style(dst_style, df):
    ''' Transform all bbox annos to a specified style
    
    Args:
        dst_style (str): desired bbox style: {'xywh', 'x1y1x2y2', 'xcycwh'}
        df (pd.DataFrame): Dataframe to transform bbox styles
            
    Returns:
        pd.DataFrame: transformed dataframe with bboxes in new format
    '''
    df = df.copy()
    geom = LOSTGeometries()
    df_bbox = df[df.anno_dtype == 'bbox']
    new_data = []
    new_style = []
    for idx, row in df_bbox.iterrows():
        if isinstance(row.anno_data, np.ndarray):
            data = geom.transform_bbox_style(row.anno_data, 
                                            row.anno_style, 
                                            dst_style)
            style = dst_style
        else:
            data = row.anno_data
            style = row.anno_style
            
        new_data.append(data)
        new_style.append(style)
    df_bbox.anno_data = new_data
    df_bbox.anno_style = new_style
    df.loc[df.anno_dtype == 'bbox'] = df_bbox
    return df


def polygon_to_bbox(df, dst_style=None):
    '''Transform all polygons to bboxes
    Args:
        df (pd.DataFrame): datafram to apply transform
        lost_geometries (LOSTGeometries, None): Use an existing instance of
            LOSTGeometries if passed. Otherwise creates a new one during RT.
        dst_style (str): desired style of bbox: {'xywh', 'x1y1x2y2', 'xcycwh'}
    Returns:
        pd.DataFrame with transformed polygons
    '''
    df = df.copy()
    geom = LOSTGeometries()
    if dst_style is None:
        dst_style = 'x1y1x2y2'
    polygons = df[df.anno_dtype == 'polygon']
    df.loc[df.anno_dtype=='polygon', 'anno_data'] = polygons.anno_data.apply(
        lambda x: geom.poly_to_bbox(x, dst_style))
    columns = ['anno_style', 'anno_dtype']
    df.loc[df.anno_dtype=='polygon', columns] = [dst_style, 'bbox']
    return df


# TODO: enable results format - idstinguish between gt and results
# TODO: allow iscrowd 
# TODO: implement segmentation to RLE for iscrowd==1 and results format
# TODO: add score for results format
# TODO: enable panoptic segmentation
def to_coco(df, remove_invalid=True, lbl_col='anno_lbl', 
            supercategory_mapping=None, copy_path=None, rename_by_index=False,
            json_path=None,
            filesystem: Union[FileMan, fsspec.AbstractFileSystem] = None):
    """Transform dataset to coco data format

    Args:
        df ([pd.DataFrame]): dataframe to apply transform
        remove_invalid (bool, optional): Remove invalid image-paths. Defaults to True.
        lbl_col (str, optional): dataframe column to look for labels. Defaults to 'anno_lbl'.
        supercategory_mapping ([dict], optional): dict like mapping for supercategories ({class: superclass}). 
            Defaults to None.
        copy_path (str, optional): copy all images to given directory
        rename_by_index (bool, optional): rename files by index to assure unique filenames
        json_path (str, optional): store coco annotations as .json file 
        filesystem (Union[FileMan, fsspec.AbstractFileSystem], optional): Filesystem instance. 
            Defaults to None.

    Returns:
        dict: containing coco data like {'categories': [...], 'images': [...], 'annotations': [...]}
    """
    df = validate_img_paths(df, remove_invalid, filesystem)
    df = transform_bbox_style('xywh', df)
    df = to_abs(df, filesystem)
    img_paths = df['img_path'].unique()
    
    categories = list()
    imgs = list()
    annos = list()
    
    # COCO-categories
    u_lbls = unique_labels(df, lbl_col)
    lbl_to_id = dict()
    for lbl_id, lbl in enumerate(u_lbls):
        supercategory = None if supercategory_mapping is None else supercategory_mapping[lbl]
        categories.append({"id": lbl_id + 1, 
                           "name": lbl,
                           "supercategory": supercategory})
        lbl_to_id[lbl] = lbl_id + 1
    
    copy_process = list()
    if copy_path is None:
        root_dirs = df['img_path'].apply(lambda x: os.path.join(x.split('/')[:-1])).unique()
        assert len(root_dirs) == 1, f'COCO-Dataset images located in multiple different dirs {root_dirs}. ' \
            'You can use copy_path argument to copy the entire dataset.'
    else:
        filesystem.makedirs(copy_path, True)
        
    for img_path in img_paths:
        
        # COCO-imgs
        im_h, im_w = get_imagesize(img_path)
        img_id = len(imgs) + 1
        filename = img_path.split('/')[-1]
        if copy_path is not None:
            file_ending = img_path.split('.')[-1]
            if rename_by_index:
                filename = f'{img_id:012d}.{file_ending}'
            dst_file = os.path.join(copy_path, filename)
            # if not filesystem.exists(dst_file):
            p = Process(target=filesystem.copy, args=(img_path, dst_file,))
            copy_process.append(p)
            p.start()
            
        imgs.append({"id": img_id,
                     "width": im_w,
                     "height": im_h,
                     "file_name": filename,
                     "org_path": img_path,
                     })
        
        # COCO-annos
        img_df = df[df['img_path']==img_path]
        for _, row in img_df.iterrows():
            anno_type = row['anno_dtype']
            segmentation = None
            bbox = None
            if anno_type == 'polygon':
                segmentation = [row['anno_data'].flatten().tolist()]
                bbox = list(LOSTGeometries().poly_to_bbox(row['anno_data'], 'xywh'))
            annos.append({"id": len(annos) + 1,
                          "image_id": img_id,
                          "category_id": lbl_to_id[row[lbl_col]],
                          "iscrowd": 0,
                          "segmentation": segmentation,
                          "bbox": bbox,
                          "area": bbox[2] * bbox[3],
                          })

    coco_anno = {"categories": categories,
                 "images": imgs,
                 "annotations": annos
                 }
    
    if copy_path:
        [p.join() for p in copy_process]
    
    if json_path:
        anno_dir = os.path.dirname(json_path)
        if not filesystem.exists(anno_dir):
            filesystem.makedirs(anno_dir, True)
        with open(json_path, 'w') as f:
            json.dump(coco_anno, f) 
    
    return coco_anno