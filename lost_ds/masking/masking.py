import os 
import numpy as np 
from joblib import Parallel, delayed 
from tqdm import tqdm 

from lost_ds.functional.filter import label_selection, unique_labels
from lost_ds.geometry.lost_geom import LOSTGeometries
from lost_ds.util import get_fs



def mask_dataset(df, dst_dir, masking_labels, mask_value=0, inverse=False, 
                 lbl_col='anno_lbl', dst_col='img_path', filesystem=None, 
                 parallel=-1):
    """mask images by annos of given label. Either mask the given annos itself 
    (inverse=False) or everything but the annos (inverse=True)

    Args:
        df (pd.DataFrame): dataframe to mask
        dst_dir (str): destination folder
        masking_labels (list): list of labels to use for masking. Use all labels if 'all' or -1
        mask_value (int, optional): pixel value to use for mask. Defaults to 0.
        inverse (bool, optional): fill the given annos with `mask_value` if True, and inverse if False. Defaults to False.
        lbl_col (str, optional): column name containing labels. Defaults to 'anno_lbl'.
        mask_path (str, optional): new column name for masked img paths. Defaults to 'img_path'
        filesystem (_type_, optional): _description_. Defaults to None.
    """
    fs = get_fs(filesystem)
    fs.makedirs(dst_dir, True)
    df = df.copy()
    # TODO: only polygon and bbox can be masked
    
    if masking_labels == -1 or masking_labels == 'all':
        masking_labels = unique_labels(df, lbl_col)
    
    def _mask_img(path, annos):
        mask_annos = label_selection(masking_labels, annos, lbl_col)
        dst_path = os.path.join(dst_dir, path.split('/')[-1])
        
        # mask labels occured -> apply mask to image
        if len(mask_annos):    
            img = fs.read_img(path)
            h, w = img.shape[:2]
            c = img.shape[-1] if len(img.shape) == 3 else 1
            mask = np.zeros((h, w))
            geom = LOSTGeometries()
            set_val = 1
            for _, row in mask_annos.iterrows():
                mask = geom.segmentation(mask, set_val, row.anno_data, 
                                        row.anno_dtype, row.anno_format, 
                                        row.anno_style)
            if inverse:
                set_val = 0
            px = (mask_value, )*3 if c==3 else mask_value
            img = np.where(mask[..., None]==set_val, px, img)
            fs.write_img(img, dst_path)
        
        # no mask labels occured -> copy image or write mask-only if invers
        else:
            if inverse:
                fs.write_img(np.full_like(fs.get_imagesize(path), mask_value), 
                             dst_path)
            else:
                fs.copy(path, dst_path)
    
    if parallel:
        Parallel(n_jobs=parallel)(delayed(_mask_img)(path, annos) 
                    for path, annos in tqdm(df.groupby('img_path'), 
                                            desc='mask imgs')
                    )
    else:
        for path, annos in tqdm(df.groupby('img_path'), desc='mask imgs'):
            _mask_img(path, annos)
    
    
    
    # remap new path
    df[dst_col] = df['img_path'].apply(lambda x: 
        os.path.join(dst_dir, x.split('/')[-1]))
    
    return df 


