from collections import defaultdict
from tqdm import tqdm
import fsspec
import os
from joblib import Parallel, delayed, cpu_count
from zipfile import ZipFile

from lost_ds.functional.mapping import remap_img_path
from lost_ds.util import get_fs


def _make_dst_names(src_paths):
    """Map each source path to a collision-free destination filename.

    Non-colliding basenames are kept flat (e.g. ``frame001.jpg``).
    When two or more source paths share the same basename the immediate
    parent directory is prepended (e.g. ``a/frame001.jpg``), preserving
    human-readable context and guaranteeing uniqueness.

    Args:
        src_paths (list[str]): Unique source file paths.

    Returns:
        dict[str, str]: Mapping of ``src_path`` → destination relative name.
    """
    groups = defaultdict(list)
    for p in src_paths:
        groups[os.path.basename(p)].append(p)

    name_map = {}
    for basename, paths in groups.items():
        if len(paths) == 1:
            name_map[paths[0]] = basename
        else:
            for p in paths:
                parent = os.path.basename(os.path.dirname(p))
                name_map[p] = os.path.join(parent, basename)
    return name_map


def copy_imgs(df, out_dir, col='img_path', force_overwrite=False, 
              filesystem=None, parallel=-1):
    '''Copy all images of dataset into out_dir

    Args:
        df (pd.DataFrame): dataframe to copy
        out_dir (str): Destination folder to store images
        col (str): column containing paths to files
        filesystem (fsspec.filesystem, FileMan): filesystem to use. Use local
            if not initialized

    Returns:
        dict[str, str]: Mapping of source path to destination relative name
            (see :func:`_make_dst_names`).
    '''
    fs = get_fs(filesystem)
    fs.makedirs(out_dir, exist_ok=True)
    img_paths = list(df[col].unique())
    name_map = _make_dst_names(img_paths)

    def copy_file(src_path):
        dst_path = os.path.join(out_dir, name_map[src_path])
        dst_dir = os.path.dirname(dst_path)
        if dst_dir != out_dir:
            fs.makedirs(dst_dir, exist_ok=True)
        if fs.exists(dst_path) and not force_overwrite:
            return
        fs.copy(src_path, dst_path)

    if parallel:
        Parallel(n_jobs=parallel)(delayed(copy_file)(path)
                            for path in tqdm(img_paths, desc='copy imgs'))
    else:
        for path in tqdm(img_paths, desc='copy imgs'):
            copy_file(path)

    return name_map
        

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

    Returns:
        dict[str, str]: Mapping of source path to destination relative name
            (see :func:`_make_dst_names`).
    '''

    fs = get_fs(filesystem).fs
    img_paths = list(df[col].unique())
    name_map = _make_dst_names(img_paths)

    def copy_file_to_zip(src_path):
        dst_path = os.path.join(zip_dir, name_map[src_path])
        try:
            fs.ls('')
        except:
            pass
        with fs.open(src_path, 'rb') as f:
            zip_file.writestr(dst_path, f.read())

    total = len(img_paths)
    next_pg = 0
    for idx, path in enumerate(img_paths):
        copy_file_to_zip(path)
        if progress_callback is not None:
            pg = (idx+1) *100 / total
            if pg == 100:
                progress_callback(pg)
            elif pg >= next_pg:
                progress_callback(pg)
                next_pg += 5

    return name_map
    
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
                name_map = copy_imgs(df=df, out_dir=dout, col=col, filesystem=fs)
            else:
                out_base = os.path.basename(out_dir)
                out_base = os.path.splitext(out_base)[0]
                zip_dir = os.path.join(out_base, _dir)
                name_map = copy_to_zip(zip_file, df, zip_dir=zip_dir, col=col, filesystem=fs,
                            progress_callback=progress_callback)
            df = remap_img_path(df, dout, col, name_map=name_map)
    return df