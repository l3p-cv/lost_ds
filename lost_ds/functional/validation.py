import pandas as pd 
import numpy as np

from lost_ds.functional.filter import is_multilabel
from lost_ds.functional.split import split_by_empty
from lost_ds.util import get_fs
from lost_ds.geometry.api import LOSTGeometries

def validate_empty_images(df):
    '''Remove empty entries for images where non-empty entries do exist
    Args:
        df (pd.DataFrame): df to apply on
    Returns:
        pd.DataFrame: dataframe with validated empty images
    '''
    df = df.copy()
    not_empty_df, empty_df = split_by_empty(df)
    not_empty_imgs = not_empty_df['img_path'].unique()
    empty_df = empty_df[~empty_df['img_path'].isin(not_empty_imgs)]
    return pd.concat([not_empty_df, empty_df]).sort_index()


def validate_unique_annos(df):
    '''Remove empty entries for images where non-empty entries do exist
    Args:
        df (pd.DataFrame): df to apply on
    Returns:
        pd.DataFrame: dataframe with validated empty images
    '''
    df = df.copy()
    df['data_hash'] = df.anno_data.apply(lambda x: hash(str(x)))
    df = df.drop_duplicates(['img_path', 'data_hash'])
    return df.drop('data_hash', 'columns')


def validate_geometries(df, remove_invalid=True):
    '''Validate that annotations are unique
    Args:
        df (pd.DataFrame): df to apply on
        lost_geometries (LOSTGeometries): instance to validate the geometries
            If None a new one will be Instaciated
    Returns:
        pd.DataFrame: dataframe with validated empty images
    '''
    df = df.copy()
    geom = LOSTGeometries()
    valid_mask = df[['anno_data', 'anno_dtype']].apply(
        lambda x: geom.validate(*x), axis=1)
    if remove_invalid:
        df = df[valid_mask]
    else:
        if False in valid_mask:
            raise Exception('Found invalid geometries! {}'.
                            format(df[~valid_mask]))
    return df


def validate_img_paths(df, remove_invalid=True, filesystem=None):
    '''Validate that paths do exist
    Args:
        remove_invalid (bool): remove non existing paths from the dataframe
        df (pd.DataFrame): df to apply on
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized
    Returns:
        pd.DataFrame: dataframe with validated image paths
    '''
    df = df.copy()
    fs = get_fs(filesystem)
    images = list(df.img_path.unique())
    valid_images = []
    invalid_images = []
    for image_path in images:
        exist = fs.exists(image_path)
        if exist:
            valid_images.append(image_path)
        elif not exist:
            invalid_images.append(image_path)
    assert len(valid_images) + len(invalid_images) == len(images)
    if len(invalid_images):
        if remove_invalid:
            df = df[~df['img_path'].isin(invalid_images)]
        else:
            raise Exception('Some images do not exist! {}'.
                            format(invalid_images))
    return df


def validate_single_labels(df, lbl_col='anno_lbl', dst_col='my_lbl'):
    '''Validate single labels. If lbl_col is a multilabel column it will be 
        parsed into single labels at dst_col. 
    
    Args:
        df (pd.DataFrame): Dataframe to validate single labels
        lbl_col (str): label col to validate
        ds_col (str): new column containing single label
        
    Returns:
        pd.DataFrame with column dst_col containing single labels
    '''
    df = df.copy()
    if is_multilabel(df, lbl_col):
        df[lbl_col] = df[lbl_col].apply(lambda x: np.unique(x))
        n_lbl = df[lbl_col].apply(lambda x: len(x))
        if n_lbl.max() > 1:
            raise Exception('Found multilabels in {}. \n{}'.
                            format(lbl_col, df[n_lbl > 1][lbl_col]))
        df[dst_col] = df[lbl_col].apply(lambda x: x[0] if len(x) else np.nan)
    else:
        df[dst_col] = df[lbl_col]
    return df