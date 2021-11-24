from shapely.geometry import Polygon, MultiPolygon
import cv2
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np 
import pandas as pd 

from lost_ds.im_util import get_fs


def _segmentation_to_polygon(segmentation, pixel_mapping: dict):
        
    if len(segmentation.shape)==3:
        segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY)        
    
    im_h, im_w = segmentation.shape
    
    px_values = []
    lost_polygons = []
    
    contours, _ = cv2.findContours(segmentation, cv2.RETR_CCOMP, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = map(np.squeeze, contours)
    cont_seg = np.zeros_like(segmentation)
    valid_contours, contour_pixels = list(), list()
    
    for idx, cont in enumerate(contours):
        if len(cont) < 4:
            continue
        cont_seg = cv2.drawContours(cont_seg, [cont], 0, idx + 1, -1)# fill poly
        cont_seg = cv2.drawContours(cont_seg, [cont], 0, 0, 1) # ignore border
        valid_contours.append(cont)
        contour_pixels.append(idx + 1)
            
    for px, cont in zip(contour_pixels, valid_contours):
        org_pixels = segmentation[cont_seg==px]
        u_px, cnts = np.unique(org_pixels, return_counts=True)
        if not len(u_px):
            continue
        px_org = u_px[np.argmax(cnts)]
        cont = np.array(cont) / np.array([im_w, im_h])
        lost_polygons.append(cont)
        px_values.append(pixel_mapping[px_org])

    return lost_polygons, px_values


def segmentation_to_lost(df: pd.DataFrame, pixel_mapping, background=0, 
                         seg_key='seg_path', filesystem=None):
    
    '''Create LOST-Annotations from semantic segmentations / pixelmaps 
    
    Args:
        pixel_mapping (dict): mapping of pixel-value to anno_lbl, e.g. 
            {0:'background', 1:'thing'}
        background (int): pixel-value for background.
        seg_key (str): dataframe key containing the paths to the stored 
            pixelmaps
        df (pd.DataFrame): dataframe to apply on
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized
        
    Returns:
        pd.DataFrame with polygon-annotations in LOSTDataset format
    '''
    
    fs = get_fs(filesystem)
    
    pixel_mapping = pixel_mapping.copy()
    if background != 0:
        raise NotImplementedError('Background pixels have to be 0')
        # pm0 = pixel_mapping.get(0, None)
        # pmb = pixel_mapping[background]
        # pixel_mapping[0] = pmb
        # pixel_mapping[background] = pm0
    
    def seg_to_poly(seg_path, seg_df):
        img_paths = seg_df['img_path'].unique()
        assert len(img_paths) == 1
        lost_anno_data = dict()
        segmentation = fs.read_img(seg_path, 0)
        
        # if background != 0:
        #     bool_background = (segmentation==background)
        #     bool_zero = (segmentation==0)
        #     segmentation[bool_background] = 0
        #     segmentation[bool_zero] = background
            
        lost_polys, px_values = _segmentation_to_polygon(segmentation, 
                                                         pixel_mapping)
        assert len(lost_polys) == len(px_values)
        img_path = img_paths[0]
        lost_anno_data['anno_lbl'] = px_values
        lost_anno_data['anno_data'] = lost_polys
        lost_anno_data['img_path'] = [img_path] * len(px_values)
        lost_anno_data[seg_key] = [seg_path] * len(px_values)
        lost_anno_data['anno_format'] = ['rel'] * len(px_values)
        lost_anno_data['anno_style'] = ['xy'] * len(px_values)
        lost_anno_data['anno_dtype'] = ['polygon'] * len(px_values)
        return pd.DataFrame(lost_anno_data)
    
    n_segs = len(df[seg_key].unique())
    ret = Parallel(n_jobs=-1)(delayed(seg_to_poly)(seg_path, seg_df) 
                              for seg_path, seg_df in tqdm(df.groupby([seg_key]), 
                                                           total=n_segs, 
                                                           desc='seg. to anno'))
    
    # ret = list()
    # for seg_path, seg_df in tqdm(df.groupby([seg_key])):
    #     ret.append(seg_to_poly(seg_path, seg_df))
        
    df = pd.concat(ret)
    
    return df


