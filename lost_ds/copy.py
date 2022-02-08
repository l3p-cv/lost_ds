from tqdm import tqdm
import fsspec
import os
from joblib import Parallel, delayed, cpu_count

from lost_ds.functional.mapping import remap_img_path
from lost_ds.util import get_fs


def copy_imgs(df, out_dir, col='img_path', force_overwrite=False, 
              filesystem=None):
    '''Copy all images of dataset into out_dir

    Args:
        df (pd.DataFrame): dataframe to copy
        out_dir (str): Destination folder to store images
        col (str): column containing paths to files
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized
    '''
    fs = get_fs(filesystem)
    def copy_file(src_path):
        dst_path = os.path.join(out_dir, os.path.basename(src_path))
        if fs.exists(dst_path) and not force_overwrite:
            return
        fs.copy(src_path, dst_path)
        
    fs.makedirs(out_dir, exist_ok=True)
    img_paths = list(df[col].unique())
    Parallel(n_jobs=-1)(delayed(copy_file)(path) 
                          for path in tqdm(img_paths, desc='copy imgs'))

    
def pack_ds(df, out_dir, cols=['img_path', 'mask_path', 'crop_path'],
            dirs = ['imgs', 'masks', 'crops'], filesystem=None):
    '''Copy all images from dataset to a new place and update the dataframe 
    
    Args:
        df (pd.DataFrame): Dataframe to copy
        out_dir (str): Name of the directory to store the information
        executor (Client, ThreadPoolExecutor): executor for parallelization
            if None a new ThreadPoolExecutor will be initialized
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized
        cols (list of string): column names containing file-paths
        dirs (list of string): name of new directories according to cols. The 
            dirs will contain the copied data
        
    Returns:
        pd.DataFrame with new image paths
    '''
    fs = get_fs(filesystem)
    for col, _dir in zip(cols, dirs):
        if col in df.keys():
            dout = os.path.join(out_dir, _dir)
            fs.makedirs(dout, exist_ok=True)
            copy_imgs(df=df, out_dir=dout, col=col, filesystem=fs)
            df = remap_img_path(df, dout, col)
    return df