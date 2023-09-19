import cv2
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np 
import pandas as pd 

from lost_ds.im_util import get_fs


def _segmentation_to_polygon(segmentation, pixel_mapping: dict, background, 
                             cast_others):
    if len(segmentation.shape)==3:
        segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY)
    u_values = np.unique(segmentation)
    expected_values = np.unique(list(pixel_mapping.keys()) + [background])
    setdiff = np.setdiff1d(u_values, expected_values)
    if len(setdiff) and not cast_others:
        raise ValueError(f'Got pixel values {u_values} expected ' \
                         f'{expected_values} from {pixel_mapping} ' \
                         f'(unexpected values {list(setdiff)})')
    
    im_h, im_w = segmentation.shape
    
    lbl_name = []
    lost_polygons = []
    
    # iterate over all classes and get their segmentations
    for v in u_values:
        if not v in pixel_mapping.keys():
            continue
        # create color pixel map to detect contours of class v
        v_seg = segmentation.copy()
        v_seg[segmentation!=v] = 0
        v_seg[segmentation==v] = 255
        
        # get contours
        contours, _ = cv2.findContours(v_seg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours = map(np.squeeze, contours)
        cont_seg = np.zeros_like(v_seg)
        valid_contours, contour_pixels = list(), list()
        
        # validate contours
        for idx, cont in enumerate(contours):
            if len(cont) < 4:
                continue
            cont_seg = cv2.drawContours(cont_seg, [cont], 0, idx + 1, -1)# fill poly
            cont_seg = cv2.drawContours(cont_seg, [cont], 0, 0, 1) # ignore border
            valid_contours.append(cont)
            contour_pixels.append(idx + 1)
                
        for px_pos, cont in zip(contour_pixels, valid_contours):
            # pick pixels
            px_vals = segmentation[cont_seg==px_pos]
            # find unique pixels
            u_px_vals, cnts = np.unique(px_vals, return_counts=True)
            if not len(u_px_vals):
                continue
            # choose most occuring pixel as class
            px = u_px_vals[np.argmax(cnts)]
            if px != v:
                continue
            # consider unknown pixel value as background if `cast_others`
            if not px in expected_values and cast_others:
                px = 0
            # ignore background if not provided
            if px == background and not background in list(pixel_mapping.keys()):
                continue
            cont = np.array(cont) / np.array([im_w, im_h])
            lost_polygons.append(cont)
            lbl_name.append([pixel_mapping[px]])

    return lost_polygons, lbl_name

def segmentation_to_lost(df: pd.DataFrame, pixel_mapping, background=0, 
                         seg_key='seg_path', cast_others=False, filesystem=None,
                         parallel=-1):
    
    '''Create LOST-Annotations from semantic segmentations / pixelmaps 
    
    Args:
        pixel_mapping (dict): mapping of pixel-value to anno_lbl, e.g. 
            {0:'background', 1:'thing'}
        background (int): pixel-value for background.
        seg_key (str): dataframe key containing the paths to the stored 
            pixelmaps
        df (pd.DataFrame): dataframe to apply on
        cast_others (bool): Flag if pixel values not occuring in `pixel_mapping` 
            should be handeled as background. Raises ValueError if False and 
            an unknown pixel occurs
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized
        
    Returns:
        pd.DataFrame with polygon-annotations in LOSTDataset format
    '''
    
    fs = get_fs(filesystem)
    
    pixel_mapping = pixel_mapping.copy()
    if background != 0:
        raise NotImplementedError('Background pixels have to be 0')
    
    def seg_to_poly(seg_path, seg_df):
        img_paths = seg_df['img_path'].unique()
        assert len(img_paths) == 1
        lost_anno_data = dict()
        segmentation = fs.read_img(seg_path, 0)            
        lost_polys, lbl_name = _segmentation_to_polygon(segmentation, 
                                                        pixel_mapping, 
                                                        background,
                                                        cast_others)
        assert len(lost_polys) == len(lbl_name)
        img_path = img_paths[0]
        lost_anno_data['anno_lbl'] = lbl_name
        lost_anno_data['anno_data'] = lost_polys
        lost_anno_data['img_path'] = [img_path] * len(lbl_name)
        lost_anno_data[seg_key] = [seg_path] * len(lbl_name)
        lost_anno_data['anno_format'] = ['rel'] * len(lbl_name)
        lost_anno_data['anno_style'] = ['xy'] * len(lbl_name)
        lost_anno_data['anno_dtype'] = ['polygon'] * len(lbl_name)
        return pd.DataFrame(lost_anno_data)
    
    n_segs = len(df[seg_key].unique())
    if parallel:
        ret = Parallel(n_jobs=parallel)(delayed(seg_to_poly)(seg_path, seg_df) 
                                for seg_path, seg_df in tqdm(df.groupby([seg_key]), 
                                                            total=n_segs, 
                                                            desc='seg. to anno'))
    else:
        ret = list()
        for seg_path, seg_df in tqdm(df.groupby([seg_key])):
            ret.append(seg_to_poly(seg_path, seg_df))
        
    df = pd.concat(ret)
    df.reset_index(inplace=True, drop=True)
    
    return df

