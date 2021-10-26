import numpy as np


def is_multilabel(df, col):
    '''check if a column contains multilabels
    
    Args:
        df (pd.DataFrame): Dataframe to use for filtering
        col (str): column to check for multilabel
    
    Returns:
        bool: True if column contains multilabel, False if not
    '''
    if isinstance(df[col].iloc[0], (list, np.ndarray)):
        return True
    else:
        return False
    

def remove_empty(df, col='anno_data'):
    '''Remove images with empty columns in specified column
    
    Args:
        df (pd.DataFrame): Dataframe to use for filtering
        col (str): column to seatch for empty entries
    
    Returns:
        pd.DataFrame 
    '''
    return df[df[col].notnull()]


def unique_labels(df, col='anno_lbl'):
    '''Get unique dataset labels.
    
    Args:
        df (pd.DataFrame): dataframe to analyze
        col (str): Column containing list of labels

    Retrun:
        str: List of strings with unique classnames
    '''
    if is_multilabel(df=df, col=col):
        return np.unique(df[col].map(list).sum())
    return list(df[col].unique())
        
    
def selection_mask(labels, df, col='anno_lbl'):
    '''Get mask for labels in dataset
    Args:
        df (pd.DataFrame): dataset to mask
        col (str): Column containing list of labels
    
    Returns:
        pd.DataFrame: boolean mask defining which row contains one of the 
            provided labels
    '''
    if not isinstance(labels, (list, np.ndarray)):
        labels = [labels]
    if is_multilabel(df, col):
        return df[col].apply(lambda x: 
            bool(sum([l in list(x) for l in labels])))
    return df[col].apply(lambda x: x in labels)


def label_selection(labels, df, col='anno_lbl'):
    '''Get entries with a selection of labels from the dataframe
    Args:
        labels (list): list of labels to select
        df (pd.DataFrame): Frame to apply label selection
        col (str): Column containing list of labels
    Returns:
        pd.DataFrame: dataframe with label selection
    '''
    return df[selection_mask(df=df, labels=labels, col=col)]
    

def ignore_labels(labels, df, col='anno_lbl'):
    ''' Remove dataframe entries where the given labels occures

    Args:
        labels (list): list of labels to ignore
        df (pd.DataFrame): Frame to apply label ignore
        col (str): Column containing list of labels

    Returns:
        pd.DataFrame: dataframe with label selection
    '''
    return df[~selection_mask(df=df, labels=labels, col=col)]
    

def img_selection(imgs, df, invers=False):
    '''Get entries with a selection of labels from the dataframe

    Args:
        imgs (list): list of imgs to select
        invers (bool): get the selection if True, get the rest if False 
        df (pd.DataFrame): Frame to apply image selection

    Returns:
        pd.DataFrame: dataframe with image selection
    '''
    selection_mask = df.img_path.isin(imgs)
    if invers:
        selection_mask = ~selection_mask
    return df[selection_mask]