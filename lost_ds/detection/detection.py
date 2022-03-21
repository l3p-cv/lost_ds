
from joblib import Parallel, cpu_count, delayed 
import pandas as pd 
import numpy as np 

from lost_ds.functional.split import split_by_empty
from lost_ds.functional.validation import (validate_empty_images, 
                                           validate_img_paths,
                                           validate_single_labels)
from lost_ds.functional.transform import (to_abs, to_rel,
                                          transform_bbox_style)
from lost_ds.util import get_fs


def bbox_nms(df: pd.DataFrame, lbl_col='anno_lbl', method='nms', iou_thr=0.5, 
             filesystem=None, **kwargs):
    
    # refer to: https://github.com/ZFTurbo/Weighted-Boxes-Fusion
    from ensemble_boxes import (nms, non_maximum_weighted, soft_nms, 
                                weighted_boxes_fusion)
    
    bbox_df = df[df['anno_dtype']=='bbox']
    bbox_df = validate_single_labels(bbox_df, lbl_col, 'nms_lbl')
    got_scores = False
    if 'anno_confidence' in bbox_df.keys():
        if len(bbox_df['anno_confidence'].notnull()):
            print('Warning! Got data with missing anno_confidence. Will remove it')
        bbox_df = bbox_df[bbox_df['anno_confidence'].notnull()]
        got_scores = True
    else: 
        bbox_df['anno_confidence'] = 1.0
    
    # make same format
    bbox_df = to_rel(bbox_df, filesystem, False)
    bbox_df = transform_bbox_style('x1y1x2y2', bbox_df)
    
    fn_dict = {'nms': nms, 'nmw': non_maximum_weighted, 'soft_nms': soft_nms, 
               'wbf': weighted_boxes_fusion}
    
    # run nms imagewise
    assert method in fn_dict.keys(), f'invalid nms method: {method} - ' \
        f'choose one of {list(fn_dict.keys())}'
    nms_fn = fn_dict[method]
    
    def apply_nms(img_path, img_df):
        boxes_list = np.vstack(img_df['anno_data']).tolist()
        scores_list = list(img_df['anno_confidence'])
        labels_list = list(img_df['nms_lbl'])
        boxes, scores, labels = nms_fn([boxes_list], [scores_list], 
                                       [labels_list], iou_thr=iou_thr, **kwargs)
        n_boxes = len(boxes)
        result = {'anno_data': list(boxes), 'anno_lbl':list(labels), 
                  'anno_style': ['x1y1x2y2']*n_boxes, 
                  'anno_dtype': ['bbox']*n_boxes, 'anno_format': ['rel']*n_boxes, 
                  'img_path': [img_path]*n_boxes}
        if got_scores:
            result['anno_confidence'] = list(scores)
        return pd.DataFrame(result)

    # results = list()
    # for img_path, img_df in bbox_df.groupby('img_path'):
    #     results.append(apply_nms(img_path, img_df))

    results = Parallel(cpu_count())(delayed(apply_nms)(img_path, img_df) 
                          for img_path, img_df in bbox_df.groupby('img_path'))
 
    return pd.concat(results)


def detection_dataset(df, lbl_col='anno_lbl', det_col='det_lbl', 
                      bbox_style='x1y1x2y2', use_empty_images=False, 
                      filesystem=None):
    '''Prepare all bboxes with commonly required operations to use them for 
       detection CNN training
    Args:
        df (pd.DataFrame): Dataframe containing bbox annotations
        lbl_col (str): column name where the anno labels are located (single 
            label or multilabel)
        det_col (str): column name where the training labels are located (single 
            label only)
        bbox_style (str): bbox anno-style. One of {'xywh', 'x1y1x2y2', 'xcycwh'}
        use_empty_images (bool, str, int): specifiy usage of empty images (image 
            without bbox annotation).
            True: keep all images, empty and non-empty
            False: only keep non-empty images and drop all empty images
            'balanced': If more empty images than non-empty ones do exist a 
                random selection will be sampled to have the same amount of 
                empty and non-empty images. If less empty than non-empty images 
                do exist all of them will be kept
            int: a specific amount of empty images will be samples randomly
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized
            
    Returns:
        pd.DataFrame: detection dataset

    Note:
        Other anno-types than 'bbox' will be ignored. You can transform 
        polygons to bboxes before by calling LOSTDataset.polygon_to_bbox()
    '''
    fs = get_fs(filesystem)
    df = validate_empty_images(df)
    # df = validate_img_paths(df, False, filesystem=fs)
    df = validate_single_labels(df, lbl_col, det_col)
    non_empty_df, empty_df = split_by_empty(df)
    bbox_df = non_empty_df[non_empty_df['anno_dtype'] == 'bbox']
    bbox_df = to_abs(bbox_df, fs, verbose=False)
    bbox_df = transform_bbox_style(dst_style=bbox_style, df=bbox_df)
    empty_df.drop_duplicates(subset=['img_path'], inplace=True)
    
    if use_empty_images:            
        n_empty = -1
        if 'int' in str(type(use_empty_images)):
            n_empty = use_empty_images
        elif use_empty_images=='balanced':
            n_empty = len(bbox_df.img_path.unique())
        if n_empty > 0:
            if len(bbox_df.img_path.unique()) > n_empty:
                empty_df = empty_df.sample(n_empty)
        bbox_df = pd.concat([bbox_df, empty_df])
        
    return bbox_df
