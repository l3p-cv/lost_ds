import os 

from lost_ds.functional.filter import (is_multilabel, 
                                      unique_labels,)


def remap_img_path(df, new_root_path, col='img_path'):
    df = df.copy()
    df.loc[:, col] = df[col].apply(lambda x: 
            os.path.join(new_root_path, os.path.basename(x)))
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
        df[dst_col] = df[dst_col].apply(lambda x: 
            [label_mapping[l] for l in x])
    else:
        df[dst_col] = df[dst_col].apply(lambda x: label_mapping[x])
    return df