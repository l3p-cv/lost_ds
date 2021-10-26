from typing import Union

from tqdm import tqdm
import pandas as pd 
import numpy as np
import fsspec
from joblib import Parallel, delayed

from lost_ds.geometry.api import LOSTGeometries
from lost_ds.im_util import get_imagesize
from lost_ds.io.file_man import FileMan
        
def to_abs(df, filesystem:Union[FileMan,fsspec.AbstractFileSystem]=None):
    ''' Transform all annos to absolute annos
    
    Args:
        df (pd.DataFrame): dataframe to transform
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized
            
    Returns:
        pd.DataFrame: transformed dataframe wil absolute annotations
    '''
    df = df.copy()
    df_rel = df[df['anno_format'] != 'abs']
    cols = ['img_path', 'anno_data', 'anno_dtype', 'anno_format']
    def make_abs(row):
        geom = LOSTGeometries()
        anno_data = row.anno_data
        anno_format = row.anno_format
        if isinstance(anno_data, np.ndarray) and anno_format=='rel':
            img_path = row.img_path
            anno_dtype = row.anno_dtype
            img_shape = get_imagesize(img_path, filesystem)
            anno_data = geom.to_abs(
                anno_data, anno_dtype, anno_format, img_shape
                )
        return anno_data
        
    abs_data = Parallel(-1)(delayed(make_abs)(row)
                            for idx, row in tqdm(df_rel[cols].iterrows(), 
                                                 total=len(df_rel),
                                                 desc='to abs'))
    df.loc[df.anno_format != 'abs', 'anno_data'] = abs_data
    df['anno_format'] = 'abs'
    return df


def to_rel(df, filesystem: Union[FileMan, fsspec.AbstractFileSystem] = None):
    ''' Transform all annos to absolute annos
    
    Args:
        df (pd.DataFrame): dataframe to transform
        filesystem (FileMan): Filesystem instance. If None local filesystem 
            is used
            
    Returns:
        pd.DataFrame: transformed dataframe wil absolute annotations
    '''
    df = df.copy()
    df_abs = df[df['anno_format'] != 'rel']
    cols = ['img_path', 'anno_data', 'anno_dtype', 'anno_format']
    def make_rel(row):
        geom = LOSTGeometries()
        anno_data = row.anno_data
        anno_format = row.anno_format
        if isinstance(anno_data, np.ndarray) and anno_format == 'abs':
            img_path = row.img_path
            anno_dtype = row.anno_dtype
            img_shape = get_imagesize(img_path, filesystem)
            anno_data = geom.to_rel(
                anno_data, anno_dtype, anno_format, img_shape
            )
        return anno_data
        
    rel_data = Parallel(-1)(delayed(make_rel)(row) 
                            for idx, row in tqdm(df_abs[cols].iterrows(), 
                                                 total=len(df_abs),
                                                 desc='to rel'))
    df.loc[df.anno_format != 'rel', 'anno_data'] = rel_data
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

