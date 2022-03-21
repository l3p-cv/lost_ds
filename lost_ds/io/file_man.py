import fsspec, cv2
import pandas as pd 
import numpy as np
from PIL import Image
    
    
class FileMan(object):
    
    '''File manager class to read and write datasets and images'''
    
    def __init__(self, filesystem=None, backend='pandas'):
        self.fs = self._get_filesystem(filesystem)    
        self.backend = self._get_backend(backend)
    
    def _get_filesystem(self, filesystem):
        fs = None
        if filesystem is None:
            fs = fsspec.filesystem('file')
        elif 'fsspec' in str(filesystem):
            fs = filesystem
        elif 'FileMan' in str(filesystem):
            fs = filesystem.fs
        else:
            raise ValueError('Unsupported filesystem: {}!'.format(filesystem))
        return fs 
    
    def _get_backend(self, backend):
        lib = None
        if backend == 'pandas':
            lib = pd 
        else:
            raise ValueError('Unsupported backend: {}!'.format(backend))
        return lib
    
    
    #
    #   fs utils
    #
    
    def exists(self, path):
        return self.fs.exists(path)
    
    def isdir(self, path):
        return self.fs.isdir(path)
    
    def isfile(self, path):
        return self.fs.isfile(path)
    
    def listdir(self, path):
        return self.fs.listdir(path)
    
    def ls(self, path):
        return self.fs.ls(path)
    
    def rm(self, path, recursive=False):
        self.fs.rm(path, recursive)
    
    def makedirs(self, path, exist_ok=True):
        return self.fs.makedirs(path, exist_ok=exist_ok)

    def copy(self, src_path, dst_path, recursive=False):
        self.fs.copy(src_path, dst_path, recursive)
    
    def glob(self, path):
        return self.fs.glob(path)
    
    
    #
    #   open files
    #
    
    def open_text(self, path, mode='r'):
        return self.fs.open(path, mode)
    
    
    def open_binary(self, path, mode='r'):
        return self.fs.open(path, mode+'b')


    #
    #   R/W Images
    #
            
    def read_img(self, img_path, color_type='color'):
        if color_type == 'color':
            color = cv2.IMREAD_COLOR
        else:
            color = cv2.IMREAD_GRAYSCALE
        with self.open_binary(img_path, 'r') as f:
            arr = np.asarray((bytearray(f.read())), dtype=np.uint8)
            img = cv2.imdecode(arr, color)
        return img
    
    
    def write_img(self, img, img_path):
        success = False
        with self.open_binary(img_path, 'w') as f:
            success, arr = cv2.imencode('.' + img_path.split('.')[-1], img)
            f.write(bytearray(arr))
        return success
    
    
    def get_imagesize(self, path):
        '''Get the size of an image without reading the entire data
        
        Args:
            path (str, bytes): path or 'fs.open(path)'
            
        Returns:
            tuple of int: (im_h, im_w)
        '''
        im_h, im_w = None, None
        with self.open_binary(path) as f:
            im_w, im_h = Image.open(f).size
        return im_h, im_w

    
    #
    #   R/W Datasets
    #
    
    def _get_dstype(self, path):
        supported_types = ['csv', 'parquet']
        for t in supported_types:
            if t in path: 
                return t
        raise ValueError('Datatype of {} not supported!'.format(path))
    
    
    def load_dataset(self, ds):
        
        if isinstance(ds, str):
            return self._load_dataset(ds)
        elif isinstance(ds, dict):
            return pd.DataFrame(ds)
        elif isinstance(ds, pd.DataFrame):
            return ds
        elif 'LOSTDataset' in str(type(ds)):
            return ds.df
        elif ds is None:
            return None
        elif isinstance(ds, list):
            return pd.concat([self.load_dataset(df) for df in ds])
        else:
            raise ValueError('Unkown input-type {}'.format(type(ds)))
            
            
    def _load_dataset(self, ds_path):
        ds_type = self._get_dstype(ds_path)
        ds = None
        read_fn = getattr(self.backend, 'read_' + ds_type)
        open_fn = self.open_text if ds_type == 'csv' else self.open_binary
        with open_fn(ds_path, 'r') as f:
            ds = read_fn(f)
        return ds


    def write_dataset(self, ds, ds_path):
        ds_type = self._get_dstype(ds_path)
        open_fn = self.open_text if ds_type == 'csv' else self.open_binary
        write_fn = getattr(ds, 'to_' + ds_type)
        with open_fn(ds_path, 'w') as f:
            write_fn(f, index=False)

