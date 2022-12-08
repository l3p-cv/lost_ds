import os
from typing import Union
from joblib import Parallel, cpu_count, delayed 
import pandas as pd 
import numpy as np 

from lost_ds.functional.split import split_by_empty
from lost_ds.functional.validation import (validate_empty_images, 
                                           validate_img_paths,
                                           validate_single_labels)
from lost_ds.functional.transform import (polygon_to_bbox, to_abs, to_rel,
                                          transform_bbox_style, to_coco)
from lost_ds.detection.bbox_merge import bbox_merge
from lost_ds.io.file_man import FileMan
import fsspec
from lost_ds.util import get_fs


def bbox_nms(df: pd.DataFrame, lbl_col='anno_lbl', method='nms', iou_thr=0.5, 
             filesystem=None, parallel=-1, **kwargs):
    """apply nms to dataframe to suppress overlapping bboxes

    Args:
        df (pd.DataFrame): dataframe containing bboxes to supress
        lbl_col (str, optional): columns containing labels. Defaults to 'anno_lbl'.
        method (str, optional): one of {'nms', 'nmw', 'soft_nms', 'wbf', 'merge'}. Defaults to 'nms'.
        iou_thr (float, optional): iou threshold when to supress bboxes. Defaults to 0.5.
        filesystem (fsspec, optional): filesystem. Defaults to None.

    Returns:
        pd.DataFrame: dataframe with bboxes pruned by given method
    """
    # refer to: https://github.com/ZFTurbo/Weighted-Boxes-Fusion
    from ensemble_boxes import (nms, non_maximum_weighted, soft_nms, 
                                weighted_boxes_fusion)
    df = validate_empty_images(df)
    df, empty_df = split_by_empty(df)
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
    bbox_df = to_rel(bbox_df, filesystem=filesystem, verbose=False)
    bbox_df = transform_bbox_style('x1y1x2y2', bbox_df)
    
    fn_dict = {'nms': nms, 'nmw': non_maximum_weighted, 'soft_nms': soft_nms, 
               'wbf': weighted_boxes_fusion, 'merge': bbox_merge}
    
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
        result = {'anno_data': list(boxes), lbl_col:list(labels), 
                  'anno_style': ['x1y1x2y2']*n_boxes, 
                  'anno_dtype': ['bbox']*n_boxes, 'anno_format': ['rel']*n_boxes, 
                  'img_path': [img_path]*n_boxes}
        if got_scores:
            result['anno_confidence'] = list(scores)
        return pd.DataFrame(result)

    if parallel:
        results = Parallel(parallel)(delayed(apply_nms)(img_path, img_df) 
                            for img_path, img_df in bbox_df.groupby('img_path'))
    else:
        results = list()
        for img_path, img_df in bbox_df.groupby('img_path'):
            results.append(apply_nms(img_path, img_df))
 
    return pd.concat(results + [empty_df])


def coco_eval(gt_df:pd.DataFrame, pred_df:pd.DataFrame, out_dir=None, 
              filesystem: Union[FileMan, fsspec.AbstractFileSystem] = None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    filesystem = get_fs(filesystem)
    if out_dir is None:
        out_dir = '/tmp/coco_eval'
        filesystem.makedirs(out_dir)
    gt_json = os.path.join(out_dir, 'gt.json')
    pred_json = os.path.join(out_dir, 'pred.json')
    
    # to coco
    # predefine img ids
    u_imgs = np.unique(list(gt_df['img_path']) + list(pred_df['img_path']))
    img_mapping = {p: uid for uid, p in enumerate(u_imgs)}
    # predefine lbl ids
    gt_df = validate_single_labels(gt_df, dst_col='coco_lbl')
    pred_df = validate_single_labels(pred_df, dst_col='coco_lbl')
    u_lbls = np.unique(list(gt_df['coco_lbl']) + list(pred_df['coco_lbl']))
    u_lbls = [c for c in u_lbls if pd.notna(c)]
    lbl_mapping = {p: uid+1 for uid, p in enumerate(u_lbls)}
    # dataframe to coco
    coco_gt = to_coco(gt_df, predef_img_mapping=img_mapping, 
                      predef_lbl_mapping=lbl_mapping, json_path=gt_json, 
                      filesystem=filesystem)
    coco_pred = to_coco(pred_df, predef_img_mapping=img_mapping, 
                        predef_lbl_mapping=lbl_mapping, 
                        json_path=pred_json, filesystem=filesystem)
    
    coco_gt = COCO(gt_json)
    coco_pred = COCO(pred_json)
    
    # coco eval
    coco_eval = COCOeval(coco_gt, coco_pred, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    class COCOResults:
        def __init__(self, coco_eval:COCOeval):
            self.coco_eval = coco_eval
            classes = coco_eval.params.catIds
            ious = {'class': [], 'ious': [], 'mean_iou':[]}
            for cl in classes:
                class_ious = dict((k, v) for k,v in coco_eval.ious.items() if k[1]==cl)
                max_ious = [iou.max(axis=0).reshape((-1,1)) for iou in class_ious.values() if len(iou)]
                if len(max_ious):
                    max_ious = np.concatenate(max_ious, 0)
                else:
                    max_ious = np.array([np.nan])
                ious['class'].append(cl)
                ious['ious'].append(list(max_ious.reshape((-1))))
                ious['mean_iou'].append(max_ious.mean())
            self.all_IoUs = np.concatenate(ious['ious'], -1)
            self.mean_IoU = self.all_IoUs.mean()
            self.class_IoUs = pd.DataFrame(ious)
            keys = ['APcoco', 'AP50', 'AP75', 'APcoco_small', 'APcoco_medium', 
                    'APcoco_large', 'AR1', 'AR10', 'AR100', 'AR100_small', 
                    'AR100_medium', 'AR100_large']
            self.coco_results = {k:v for k,v in zip(keys, coco_eval.stats)}
        
        def summarize(self):
            self.coco_eval.summarize()
            print('mIoU per class:')
            print(self.class_IoUs[['class', 'mean_iou']])
            print('\nmIoU all classes: ', self.mean_IoU)
            
    coco_results = COCOResults(coco_eval)
    
    return coco_results
    

def voc_eval(gt_df:pd.DataFrame, pred_df:pd.DataFrame, iou_threshold=0.5,
             APMethod='AllPointsInterpolation'):
    from podm.metrics import BoundingBox, get_pascal_voc_metrics, MethodAveragePrecision
    
    # prepare datasets
    # gt
    gt_df = to_abs(gt_df, verbose=False)
    gt_df = polygon_to_bbox(gt_df, 'x1y1x2y2')
    gt_df = transform_bbox_style('x1y1x2y2', gt_df)
    gt_df = validate_empty_images(gt_df)
    gt_df = validate_single_labels(gt_df, dst_col='anno_lbl')
    # pred
    pred_df = to_abs(pred_df, verbose=False)
    pred_df = polygon_to_bbox(pred_df, 'x1y1x2y2')
    pred_df = transform_bbox_style('x1y1x2y2', pred_df)
    pred_df = validate_empty_images(pred_df)
    pred_df = validate_single_labels(pred_df, dst_col='anno_lbl')
    
    def _cast_df(df):
        boxes = list()
        for path, path_df in df.groupby('img_path'):
            if not path_df['anno_data'].notnull().any():
                continue
            for idx, row in path_df.iterrows():
                bb = BoundingBox.of_bbox(row.img_path, row.anno_lbl,
                                        *list(row.anno_data.flatten()), # ann.xtl, ann.ytl, ann.xbr, ann.ybr, 
                                        row.anno_confidence)
                boxes.append(bb)
        return boxes    
    if not 'anno_confidence' in gt_df.keys():
        gt_df['anno_confidence'] = 1
    gt_bboxes = _cast_df(gt_df)
    pred_bboxes = _cast_df(pred_df)
    if 'allpoints' in APMethod.lower():
        method = MethodAveragePrecision.AllPointsInterpolation
    elif 'elevenpoints' in APMethod.lower():
        method = MethodAveragePrecision.ElevenPointsInterpolation
    
    if not isinstance(iou_threshold, (list, np.ndarray)):
        iou_threshold = [iou_threshold]
    
    data = {'class':[], 'iou_threshold':[], 'ap':[], 'precision':[], 'recall':[], 
            'interpolated_precision':[], 'interpolated_recall':[], 
            'tp':[], 'fp':[], 'fn':[], 'n_gt':[], 'n_det':[]}
    
    for iou_th in iou_threshold:
        results = get_pascal_voc_metrics(gt_bboxes, pred_bboxes, iou_th, method)    
        for cls, metric in results.items():
            data['class'].append(cls)
            data['iou_threshold'].append(float(iou_th))
            data['ap'].append(metric.ap)
            data['precision'].append(metric.precision)
            data['recall'].append(metric.recall)
            data['interpolated_precision'].append(metric.interpolated_precision)
            data['interpolated_recall'].append(metric.interpolated_recall)
            data['tp'].append(metric.tp)
            data['fp'].append(metric.fp)
            data['fn'].append(metric.num_groundtruth - metric.tp)
            data['n_gt'].append(metric.num_groundtruth)
            data['n_det'].append(metric.num_detection)
    
    df = pd.DataFrame(data)
    
    return df


def voc_score_iou_multiplex(gt_df, pred_df, score_key='anno_confidence', 
                            score_range=[0.25, 0.75], score_stepwidth=0.05, 
                            iou_range=[0.25, 0.9], iou_stepwidth=0.05):
    score_shift = np.arange(*score_range, score_stepwidth)
    def shift_score(score_th, gt, pred):
        iou_shift = np.arange(*iou_range, iou_stepwidth)
        pred = pred[pred[score_key] >= score_th]
        voc_df = voc_eval(gt, pred, iou_threshold=iou_shift)
        voc_df['score_threshold'] = score_th
        return voc_df
    # with ThreadPool(cpu_count()) as tp:
    #     evals = tp.map(shift_score, score_shift)
        
    evals = Parallel(-1)(delayed(shift_score)(score_th, gt_df.copy(), pred_df.copy()) 
                         for score_th in score_shift)
    
    voc_df = pd.concat(evals)
    
    return voc_df


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
    bbox_df = to_abs(bbox_df, filesystem=fs, verbose=False)
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
