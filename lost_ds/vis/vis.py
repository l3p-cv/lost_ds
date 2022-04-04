import os

from tqdm import tqdm
from joblib import Parallel, delayed
try:
    import seaborn as sns
except:
    pass
import numpy as np 

from lost_ds.util import get_fs
from lost_ds.geometry.lost_geom import LOSTGeometries
from lost_ds.functional.api import remove_empty


def vis_sample(img, df, line_thickness=3, color=(0, 0, 255), lbl_col='anno_lbl',
              lost_geometries:LOSTGeometries=None, blow_up=None, radius=2):
    '''Visualize annos of an image

    Args:
        img (np.ndarray): image to draw on
        df (pandas.DataFrame): The DataFrame that contains annoations to 
            visualize. If df is None a random image from df will be 
            sampled.
        color (tuple, dict of tuple): colors (B,G,R) for all annos if tuple 
            or dict for labelwise mapping like {label: color}
        line_thickness (int, dict of int): line thickness for annotations if int
            or dict for anno-type wise mapping like {dtype: thickness}
        lost_geometries (LOSTGeometries): LOSTGeometries instance to use, will
            create a new one if None
        blow_up (): TODO: implement
    Returns:
        np.array: Image painted with annotations.
    '''
    df = remove_empty(df, 'anno_data')
    if len(df) > 0:
        geom = lost_geometries
        if lost_geometries is None:
            geom = LOSTGeometries()
        
        anno_data = list(df['anno_data'])
        anno_conf = None
        if hasattr(df, 'anno_confidence'): 
            anno_conf = list(df['anno_confidence']) 
        anno_lbl = list(df[lbl_col])
        anno_dtype = list(df['anno_dtype'])
        anno_style = list(df['anno_style'])
        anno_format = list(df['anno_format'])
        
        if line_thickness == 'auto':
            thickness = 0.002*img.shape[0]
        else: 
            thickness = line_thickness
        thickness = max(1, thickness)
        img = geom.draw(img, anno_data, anno_conf, anno_lbl, anno_dtype, 
                        anno_style, anno_format, thickness, color, radius)
            
    return img


def vis_and_store(df, out_dir, lbl_col='anno_lbl', color=(0, 0, 255), 
                  line_thickness='auto', filesystem=None, radius=2):
    '''Visualize annotations and store them to a folder

    Args:
        df (pd.DataFrame): Optional dataset in lost format to visualize
        out_dir (str): Directory to store the visualized annotations
        color (tuple, dict of tuple): colors (B,G,R) for all annos if tuple 
            or dict for labelwise mapping like {label: color}
        line_thickness (int, dict of int): line thickness for annotations if int
            or dict for anno-type wise mapping like {dtype: thickness}
        lbl_col (str): column containing the labels
        radius (int): radius to draw for points/circles
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized
    '''
    fs = get_fs(filesystem)
    fs.makedirs(out_dir, exist_ok=True)
    
    def vis_img(img_path, df_vis):
        geom = LOSTGeometries()
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        if df_vis['anno_data'].notnull().any():
            img = fs.read_img(img_path)
            img = vis_sample(img=img, df=df_vis, line_thickness=line_thickness, 
                             color=color, lbl_col=lbl_col, lost_geometries=geom,
                             radius=radius)
            fs.write_img(img, out_path)
        else:
            fs.copy(img_path, out_path)
            
    Parallel(n_jobs=-1)(delayed(vis_img)(path, df_vis) 
                        for path, df_vis in tqdm(df.groupby('img_path'), 
                                                 desc='visualize'))

    # for path, df_vis in tqdm(df.groupby('img_path'), desc='visualize'):
    #     vis_img(path, df_vis) 
    

def vis_semantic_segmentation(df, out_dir, n_classes, palette='dark', 
                              seg_path_col='seg_path', filesystem=None):
    """Visualize the stored semantic segmentations by coloring it
    
    Args:
        df (pandas.DataFrame): The DataFrame that contains annoations to 
            visualize. 
        out_dir (str): path to store images
        n_classes (int): number of classes occuring in pixelmaps, number of 
            different colors needed for visualization
        palette (str): seaborn color palette i.e. 'dark', 'bright', 'pastel',...
            refer https://seaborn.pydata.org/tutorial/color_palettes.html 
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized
    """
    fs = get_fs(filesystem)
    fs.makedirs(out_dir, exist_ok=True)
    
    palette = sns.color_palette(palette, n_classes)
    palette = [(np.array(x)*255).astype(np.uint8) for x in palette]

    segmentations = df[seg_path_col].unique()

    def vis_seg(seg_path):
        seg = fs.read_img(seg_path)
        vis = np.zeros(seg.shape[:2] + (3,))
        for i in range(n_classes):
            vis = np.where(seg==i, palette[i], vis)
        fs.write_img(vis, os.path.join(out_dir, seg_path.split('/')[-1]))
        
    Parallel(n_jobs=-1)(delayed(vis_seg)(seg_path) 
            for seg_path in tqdm(segmentations, desc='vis sem. seg.'))
    