import os 

from lost_ds.functional.filter import (is_multilabel, 
                                      unique_labels,)


def remap_img_path(df, new_root_path, col='img_path', mode='replace', name_map=None):
    """Remap image paths e.g. for environmental changes

    Args:
        df (pd.DataFrame): dataframe to apply remap
        new_root_path (string): path of the new dataset location
        col (str, optional): Column to remap. Defaults to 'img_path'.
        mode (str, optional): One of {'replace', 'prepend'}. 
            If 'replace' the path before the filename will be replaced by new_root_path.
            If 'prepend' new_root_path will be prepended to existing path.
            Defaults to 'replace'.
        name_map (dict, optional): Mapping of original source path to destination
            relative filename as returned by :func:`lost_ds.copy._make_dst_names`.
            When provided, the destination filename is looked up from this map
            instead of using ``os.path.basename``, ensuring the dataframe paths
            stay consistent with the actual filenames written (important when two
            images from different directories share the same basename).
            Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with remapped path
    """
    df = df.copy()
    if mode == 'replace':
        def resolve(x):
            dst_name = name_map[x] if name_map and x in name_map else os.path.basename(x)
            return os.path.join(new_root_path, dst_name)
        df.loc[:, col] = df[col].map(resolve)
    elif mode == 'prepend':
        df.loc[:, col] = df[col].map(lambda x: os.path.join(new_root_path, x))
    else:
        raise Exception(NotImplementedError)
    return df
    
    
def remap_labels(df, label_mapping,  col='anno_lbl', dst_col='anno_lbl_mapped'):
    '''Map the labels in a given col
    Args:
        df (pd.DataFrame): dataframe to apply bbox transform
        label_mapping (dict): Direct mapping of labels {old: new}. Example:
            {'Cat':'Mammal', 'Dog':'Mammal', 'Car':'Vehicle', 'Tree': None}
            Note: labels with a mapping of None will be removed from dataset,
                    labels without any mapping wont be changed
        col (str): source column to read from
        dst_col (str): column to write to
        
    Returns:
        pd.DataFrame: The dataframe with the new mapped labels column lbl_col
    '''
    df = df.copy()
    label_mapping = label_mapping.copy()
    df[dst_col] = df[col]
    labels = unique_labels(df, col)
    # make sure every label occurs in the dictionary to prevent KeyError
    for l in labels:
        if not l in label_mapping.keys():
            label_mapping[l] = l
    if is_multilabel(df, dst_col):
        df[dst_col] = df[dst_col].map(lambda x: 
            [label_mapping[l] for l in x if not label_mapping[l] is None])
    else:
        df[dst_col] = df[dst_col].map(lambda x: label_mapping[x])
    return df