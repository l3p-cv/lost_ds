from tqdm import tqdm
import fsspec
import os
from joblib import Parallel, delayed, cpu_count
from zipfile import ZipFile

from lost_ds.functional.mapping import remap_img_path
from lost_ds.util import get_fs


def copy_imgs(df, out_dir, col='img_path', force_overwrite=False, 
              filesystem=None, parallel=-1):
    '''Copy all images of dataset into out_dir

    Args:
        df (pd.DataFrame): dataframe to copy
        out_dir (str): Destination folder to store images
        col (str): column containing paths to files
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized
    '''
    fs = get_fs(filesystem)
    def copy_file(src_path, dst_dir):
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        if fs.exists(dst_path) and not force_overwrite:
            return
        fs.copy(src_path, dst_path)
        
    fs.makedirs(out_dir, exist_ok=True)
    img_paths = list(df[col].unique())
    if parallel:
        Parallel(n_jobs=parallel)(delayed(copy_file)(path, out_dir) 
                            for path in tqdm(img_paths, desc='copy imgs'))
    else:
        for path in tqdm(img_paths, desc='copy imgs'):
            copy_file(path, out_dir)
        

def copy_to_zip(zip_file, df, zip_dir, col='img_path', 
              filesystem=None, progress_callback=None):
    '''Copy all images of dataset into zip archive

    Args:
        df (pd.DataFrame): dataframe to copy
        zip_root (str): Root path in zip archive
        col (str): column containing paths to files
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized
        progress_callback (function): Will be called on progress
            callback definition -> progress_callback(progress), where progress 
            value is between 0...100 
    '''

    fs = get_fs(filesystem).fs
    def copy_file_to_zip(src_path, zip_file):
        dst_path = os.path.join(zip_dir, os.path.basename(src_path))
        try:
            fs.ls('')
        except:
            pass
        with fs.open(src_path, 'rb') as f:
            zip_file.writestr(dst_path, f.read())
        
    img_paths = list(df[col].unique())
    total = len(img_paths)
    next_pg = 0
    for idx, path in enumerate(img_paths):
        copy_file_to_zip(path, zip_file)
        if progress_callback is not None:
            pg = (idx+1) *100 / total
            if pg == 100:
                progress_callback(pg)
            elif pg >= next_pg:
                progress_callback(pg)
                next_pg += 5
    
def pack_ds(df, out_dir, cols=['img_path', 'mask_path', 'crop_path'],
            dirs = ['imgs', 'masks', 'crops'], filesystem=None, zip_file=None, 
            progress_callback=None):
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
        zip_file (zipfile.ZipFile or None): If not None, a ZipFile object will 
            be used to pack dataset to zip archive  
        progress_callback (function): Will be called on progress
            callback definition -> progress_callback(progress), where progress 
            value is between 0...100 
        
    Returns:
        pd.DataFrame with new image paths
    '''
    fs = get_fs(filesystem)
    for col, _dir in zip(cols, dirs):
        if col in df.keys():
            dout = os.path.join(out_dir, _dir)
            if zip_file is None:
                if progress_callback is not None:
                    raise Exception('progress_callback is only implement for packing to zip files yet!')
                fs.makedirs(dout, exist_ok=True)
                copy_imgs(df=df, out_dir=dout, col=col, filesystem=fs)
            else:
                out_base = os.path.basename(out_dir)
                out_base = os.path.splitext(out_base)[0]
                zip_dir = os.path.join(out_base, _dir)
                copy_to_zip(zip_file, df, zip_dir=zip_dir, col=col, filesystem=fs, 
                            progress_callback=progress_callback)
            df = remap_img_path(df, dout, col)
    return df