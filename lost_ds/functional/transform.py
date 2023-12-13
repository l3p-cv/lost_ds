import os 
import json
from multiprocessing import Process
from threading import Thread
from typing import Union

from tqdm import tqdm
import pandas as pd 
import numpy as np
import fsspec
from joblib import Parallel, delayed, cpu_count
from lost_ds.copy import copy_imgs
from lost_ds.functional.validation import validate_img_paths, validate_single_labels, validate_unique_annos
from lost_ds.functional.filter import remove_empty

from lost_ds.geometry.api import LOSTGeometries
from lost_ds.geometry.bbox import Bbox
from lost_ds.im_util import get_imagesize
from lost_ds.io.file_man import FileMan
from lost_ds.functional.filter import is_multilabel, unique_labels
from lost_ds.util import get_fs

        
# def to_abs(df, path_col='img_path', 
#            filesystem:Union[FileMan,fsspec.AbstractFileSystem]=None, 
#            verbose=True, parallel=-1):
#     ''' Transform all annos to absolute annos
    
#     Args:
#         df (pd.DataFrame): dataframe to transform
#         filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
#             if not initialized
#         verbose (bool): print tqdm progress-bar
            
#     Returns:
#         pd.DataFrame: transformed dataframe wil absolute annotations
#     '''
#     df = df.copy()
#     df_rel = df[df['anno_format'] == 'rel']
#     cols = [path_col, 'anno_data', 'anno_dtype', 'anno_format']
#     def make_abs(row):
#         anno_data = row.anno_data
#         anno_format = row.anno_format
#         if isinstance(anno_data, np.ndarray) and anno_format=='rel':
#             geom = LOSTGeometries()
#             img_path = row[path_col]
#             anno_dtype = row.anno_dtype
#             img_shape = get_imagesize(img_path, filesystem)
#             anno_data = geom.to_abs(
#                 anno_data, anno_dtype, anno_format, img_shape
#                 )    
#         return anno_data
    
#     ### joblib workflow
#     if parallel:
#         abs_data = Parallel(parallel)(delayed(make_abs)(row)
#                                 for idx, row in tqdm(df_rel[cols].iterrows(), 
#                                                     total=len(df_rel),
#                                                     desc='to abs', 
#                                                     disable=(not verbose)))
#     else:
#         abs_data = list()
#         for idx, row in tqdm(df_rel[cols].iterrows(), total=len(df_rel),
#                             desc='to abs', disable=(not verbose)):
#             abs_data.append(make_abs(row))
            
#     df.loc[df['anno_format'] == 'rel', 'anno_data'] = pd.Series(abs_data, 
#                                                                 index=df_rel.index,
#                                                                 dtype=object)
#     df.loc[df['anno_format'] == 'rel', 'anno_format'] = 'abs'
#     ###

#     ### apply workflow
#     # df.loc[df['anno_format'] == 'rel', ['anno_data', 'anno_format']] = df.loc[df['anno_format'] == 'rel', cols].apply(lambda x: make_abs(x, True), axis=1, result_type='expand')
#     ###
    
#     return df

def to_abs(df, path_col='img_path', 
           filesystem:Union[FileMan,fsspec.AbstractFileSystem]=None, 
           verbose=True, parallel=-1):
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
    df_rel = df[df['anno_format'] == 'rel']
    
    def make_abs(path, df):
        df = df.copy()
        h,w = get_imagesize(path, filesystem)
        xy = np.array([w,h])
        def _abs(data):
            if isinstance(data, np.ndarray):
                sh = data.shape
                return (data.reshape((-1,2)) * xy).reshape(sh)
            else: 
                return data
        df['anno_data'] = df['anno_data'].apply(_abs)
        df['anno_format'] = 'abs'
        return df
    
    if parallel:
        abs_dfs = Parallel(parallel)(delayed(make_abs)(path, path_df) 
                                     for path, path_df in tqdm(df_rel.groupby(path_col), 
                                                               total=len(df_rel[path_col].unique()),
                                                               desc='to abs',
                                                               disable=(not verbose)))
    else:
        abs_dfs = list()
        for path, path_df in tqdm(df_rel.groupby(path_col), 
                                  total=len(df_rel[path_col].unique()),
                                  desc='to abs', 
                                  disable=(not verbose)):
            abs_dfs.append(make_abs(path, path_df))
    
    abs_dfs.append(df[df['anno_format']!='rel'])
    df_abs = pd.concat(abs_dfs)
    return df_abs


# def to_rel(df, path_col='img_path', 
#            filesystem: Union[FileMan, fsspec.AbstractFileSystem] = None,
#            verbose=True, parallel=-1):
#     ''' Transform all annos to absolute annos
    
#     Args:
#         df (pd.DataFrame): dataframe to transform
#         filesystem (FileMan): Filesystem instance. If None local filesystem 
#             is used
#         verbose (bool): print tqdm progress-bar
            
#     Returns:
#         pd.DataFrame: transformed dataframe wil absolute annotations
#     '''
#     df = df.copy()
#     df_abs = df[df['anno_format'] == 'abs']
#     cols = [path_col, 'anno_data', 'anno_dtype', 'anno_format']
#     def make_rel(row):
#         anno_data = row.anno_data
#         anno_format = row.anno_format
#         if isinstance(anno_data, np.ndarray) and anno_format == 'abs':
#             geom = LOSTGeometries()
#             img_path = row[path_col]
#             anno_dtype = row.anno_dtype
#             img_shape = get_imagesize(img_path, filesystem)
#             anno_data = geom.to_rel(
#                 anno_data, anno_dtype, anno_format, img_shape
#             )
#         return anno_data
    
#     if parallel:
#         rel_data = Parallel(parallel)(delayed(make_rel)(row) 
#                                 for idx, row in tqdm(df_abs[cols].iterrows(), 
#                                                     total=len(df_abs),
#                                                     desc='to rel',
#                                                     disable=(not verbose)))
#     else:
#         rel_data = list()
#         for idx, row in tqdm(df_abs[cols].iterrows(), total=len(df_abs),
#                              desc='to rel',disable=(not verbose)):
#             rel_data.append(make_rel(row))
            
#     df.loc[df['anno_format'] == 'abs', 'anno_data'] = pd.Series(rel_data, 
#                                                                 index=df_abs.index,
#                                                                 dtype=object)
#     df.loc[df['anno_format'] == 'abs', 'anno_format'] = 'rel'
    
#     return df


def to_rel(df, path_col='img_path', 
           filesystem: Union[FileMan, fsspec.AbstractFileSystem] = None,
           verbose=True, parallel=-1):
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
    df_abs = df[df['anno_format'] == 'abs']
    
    def make_rel(path, df):
        df = df.copy()
        h,w = get_imagesize(path, filesystem)
        xy = np.array([w,h])
        def _rel(data):
            if isinstance(data, np.ndarray):
                sh = data.shape
                return (data.reshape((-1,2)) / xy).reshape(sh)
            else: 
                return data
        df['anno_data'] = df['anno_data'].apply(_rel)
        df['anno_format'] = 'rel'
        return df
    
    if parallel:
        rel_dfs = Parallel(parallel)(delayed(make_rel)(path, path_df) 
                                     for path, path_df in tqdm(df_abs.groupby(path_col), 
                                                               total=len(df_abs[path_col].unique()),
                                                               desc='to rel',
                                                               disable=(not verbose)))
    else:
        rel_dfs = list()
        for path, path_df in tqdm(df_abs.groupby(path_col), 
                                  total=len(df_abs[path_col].unique()),
                                  desc='to rel', 
                                  disable=(not verbose)):
            rel_dfs.append(make_rel(path, path_df))
    
    rel_dfs.append(df[df['anno_format']!='abs'])
    df_rel = pd.concat(rel_dfs)
    return df_rel


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
    cols = ['anno_data', 'anno_style']
    df_bbox = df.loc[df.anno_dtype == 'bbox', cols]
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
        
    df_bbox['anno_data'] = new_data
    df_bbox['anno_style'] = new_style
    df.loc[df.anno_dtype=='bbox', cols] = df_bbox[cols]
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
    # df.loc[df.anno_dtype=='polygon', 'anno_data'] = polygons.anno_data.apply(
    #     geom.poly_to_bbox, dst_style)
    columns = ['anno_style', 'anno_dtype']
    df.loc[df.anno_dtype=='polygon', columns] = [dst_style, 'bbox']
    return df


# TODO: allow iscrowd 
# TODO: implement segmentation to RLE for iscrowd==1 and results format
# TODO: enable panoptic segmentation
def to_coco(df, remove_invalid=True, lbl_col='anno_lbl', 
            predef_img_mapping:dict=None,
            predef_lbl_mapping:dict=None,
            supercategory_mapping=None, copy_path=None, rename_by_index=False,
            json_path=None,
            filesystem: Union[FileMan, fsspec.AbstractFileSystem] = None):
    """Transform dataset to coco data format

    Args:
        df ([pd.DataFrame]): dataframe to apply transform
        remove_invalid (bool, optional): Remove invalid image-paths. Defaults to True.
        predef_img_mapping (dict, optional): predefined uid mapping for images like {img_path: uid}
        predef_lbl_mapping (dict, optional): predefined uid mapping for labels like {lbl: uid}
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
    filesystem = get_fs(filesystem)
    df = validate_single_labels(df, lbl_col, 'coco_lbl')
    df = validate_unique_annos(df)
    lbl_col = 'coco_lbl'
    assert not is_multilabel(df, lbl_col), 'Provided lbl-col {} contains multilabels'
    df = validate_img_paths(df, remove_invalid, filesystem)
    df = transform_bbox_style('xywh', df)
    df = to_abs(df, filesystem=filesystem, verbose=False)
    
    if predef_img_mapping is None:
        img_mapping = {img: idx for idx, img in enumerate(df['img_path'].unique())}
    else:
        img_mapping = predef_img_mapping
        
    categories = list()
    imgs = list()
    annos = list()
    
    # COCO-categories
    if predef_lbl_mapping is None:
        u_lbls = unique_labels(df[(df['anno_data'].notnull()) & (df[lbl_col].notnull())], lbl_col)
        lbl_to_id = {lbl: lbl_id+1 for lbl_id, lbl in enumerate(u_lbls)}
    else:
        lbl_to_id = predef_lbl_mapping
    for lbl, lbl_id in lbl_to_id.items():
        supercategory = None if supercategory_mapping is None else supercategory_mapping[lbl]
        categories.append({"id": lbl_id, 
                           "name": lbl,
                           "supercategory": supercategory})
    
    copy_process = list()
    if copy_path is None:
        root_dirs = df['img_path'].apply(lambda x: '/'.join(x.split('/')[:-1])).unique()
        # assert len(root_dirs) == 1, f'COCO-Dataset images located in multiple different dirs {root_dirs}. ' \
        #     'You can use copy_path argument to copy the entire dataset.'
    else:
        filesystem.makedirs(copy_path, True)
        
    for img_path, img_id in img_mapping.items():
                
        # COCO-imgs
        im_h, im_w = get_imagesize(img_path)
        filename = img_path.split('/')[-1]
        if copy_path is not None:
            file_ending = img_path.split('.')[-1]
            if rename_by_index:
                filename = f'{img_id:012d}.{file_ending}'
            dst_file = os.path.join(copy_path, filename)
            # if not filesystem.exists(dst_file):
            # p = Process(target=filesystem.copy, args=(img_path, dst_file,), daemon=True)
            p = Thread(target=filesystem.copy, args=(img_path, dst_file,), daemon=True)
            copy_process.append(p)
            p.start()
            
        imgs.append({"org_path": img_path,      # custom
                     "id": img_id,              # int, 
                     "width": im_w,             # int, 
                     "height": im_h,            # int, 
                     "file_name": filename,     # str, 
                     "license": None,           # int, 
                     "flickr_url": None,        # str, 
                     "coco_url": None,          # str, 
                     "date_captured": None,     # datetime,
                     })
        
        # COCO-annos
        img_df = df[df['img_path']==img_path].copy()
        if not img_df['anno_data'].notnull().any():
            continue
        img_df = remove_empty(img_df)
        for _, row in img_df.iterrows():
            anno_type = row['anno_dtype']
            anno = {"id": len(annos) + 1,
                    "image_id": img_id,
                    "category_id": lbl_to_id[row[lbl_col]],
                    "iscrowd": 0,
                    }
            area = 0
            if 'anno_confidence' in row.keys():
                if row['anno_confidence']:
                    if row['anno_confidence'] > 0:
                        score = float(row['anno_confidence'])
                        anno['score'] = score
            if anno_type == 'polygon':
                segmentation = [row['anno_data'].flatten().tolist()]
                bbox = list(LOSTGeometries().poly_to_bbox(row['anno_data'], 'xywh'))
                area = bbox[2] * bbox[3]
                anno['segmentation'] = segmentation
                anno['bbox'] = bbox
                anno['area'] = area
            elif anno_type == 'bbox':
                bbox = list(LOSTGeometries().transform_bbox_style(row['anno_data'], row['anno_style'], 'xywh'))
                area = bbox[2] * bbox[3]
                anno['bbox'] = bbox
                anno['area'] = area
            elif anno_type is None:
                pass
            else:
                raise Exception(f'Unsupported type {anno_type}')
            if area:
                annos.append(anno)

    coco_anno = {"info": {"year": None, 
                          "version": None, 
                          "description": None, 
                          "contributor": None, 
                          "url": None, 
                          "date_created": None
                          },
                 "licenses": [{"id": None, 
                               "name": None, 
                               "url": None
                               }],
                 "categories": categories,
                 "images": imgs,
                 "annotations": annos,
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