import os

from tqdm import tqdm
from joblib import Parallel, delayed

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
            
        img = geom.draw(img, anno_data, anno_conf, anno_lbl, anno_dtype, 
                        anno_style, anno_format, line_thickness, color, radius)
            
    return img


def vis_and_store(df, out_dir, lbl_col='anno_lbl', color=(0, 0, 255), 
                  line_thickness=2, filesystem=None, radius=2):
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
        if df_vis[lbl_col].notnull().sum() > 0:
            img = fs.read_img(img_path)
            img = vis_sample(img=img, df=df_vis, line_thickness=line_thickness, 
                             color=color, lbl_col=lbl_col, lost_geometries=geom,
                             radius=radius)
            fs.write_img(img, out_path)
        else:
            fs.copy(img_path, out_path)
        return None
    
    Parallel(n_jobs=-1)(delayed(vis_img)(path, df_vis) 
                        for path, df_vis in tqdm(df.groupby('img_path'), 
                                                 desc='visualize'))

    # for path, df_vis in tqdm(df.groupby('img_path'), desc='visualize'):
    #     vis_img(path, df_vis) 
    

# TODO: reimplement 
# def vis_semantic_segmentation(self, n_classes, out_dir, df=None):
#     """Visualize the semantic segmentations from dataset by coloring it
    
#     Args:
#         n_classes (int): number of classes occuring in pixelmaps
#         out_dir (str): path to store images
#         df (pandas.DataFrame): The DataFrame that contains annoations to 
#             visualize. If df is None a random image from self.df will be 
#             sampled.
    
#     Returns:
#         None
#     """
#     if df is None:
#         df = self.df

#     os.makedirs(out_dir, exist_ok=True)
#     palette = sns.color_palette('dark', n_classes)
#     palette = [(np.array(x)*256).astype(np.uint8) for x in palette]

#     segmentations = df['mask.path'].unique()

#     # def vis_seg(k, seg_path):
#     for k, seg_path in enumerate(segmentations):
#         print_progress_bar(k+1, len(segmentations), 'vis. sem. seg.:')
#         seg = cv2.imread(seg_path)
#         vis = np.zeros(seg.shape[:2]+(3,))
#         for i in range(n_classes):
#             vis = np.where(seg==i, palette[i], vis)
#         cv2.imwrite(os.path.join(out_dir, seg_path.split('/')[-1]), vis)
    
#     # Parallel(mp.cpu_count())(delayed(vis_seg)(k, path) for k, path in enumerate(segmentations))
    