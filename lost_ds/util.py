import pandas as pd 
from joblib import parallel_backend, Parallel, delayed, effective_n_jobs
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import _num_samples

from lost_ds.io.file_man import FileMan
from lost_ds.geometry.lost_geom import LOSTGeometries

   
def get_fs(filesystem, backend='pandas'):
    if isinstance(filesystem, FileMan):
        return filesystem
    else:
        return FileMan(filesystem, backend)


def to_parquet(path, df, filesystem=None):
    fs = get_fs(filesystem)
    geom = LOSTGeometries()
    store_df = df.copy()
    if 'anno_data' in df.keys():
        store_df.anno_data = store_df.anno_data.apply(lambda x: 
            geom.serializable(x))
    fs.write_dataset(store_df, path)
    
    
def parallel_apply(df:pd.DataFrame, func, n_jobs= -1, **kwargs):
    """ Pandas apply in parallel using joblib. 
    Uses sklearn.utils to partition input evenly.
    
    Args:
        df (pd.DataFrame, pd.Series): dataframe to apply the function 
        func (callable): Callable function to apply
        n_jobs (int): Desired number of workers. Default value -1 means use all 
            available cores.
        **kwargs: Any additional parameters will be supplied to the apply function
        
    Returns:
        Same as for normal Pandas DataFrame.apply()
    """
    
    if effective_n_jobs(n_jobs) == 1:
        return df.apply(func, **kwargs)
    else:
        ret = Parallel(n_jobs=n_jobs)(
            delayed(type(df).apply)(df[s], func, **kwargs)
            for s in gen_even_slices(_num_samples(df), effective_n_jobs(n_jobs)))
        return pd.concat(ret)