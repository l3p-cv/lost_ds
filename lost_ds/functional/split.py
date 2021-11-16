import pandas as pd
from sklearn.model_selection import train_test_split

from lost_ds.functional.filter import img_selection


def split_by_empty(df, col='anno_data'):
    '''Split dataframe into empty and non-empty entries, filtered by given col
    Args:
        df (pd.DataFrame): df to apply on
        col (str): column to check for empty entries
    Returns:
        tuple of pd.DataFrame: first dataframe contains non-emtpy entries, 
            the second dataframe contains empty entries
    '''
    empty_bool = df[col].isnull()
    empty_df = df[empty_bool]
    not_empty_df = df[~empty_bool]
    return not_empty_df, empty_df


def split_by_img_path(test_size=0.2, val_size=0.2, df=None):
    '''Split dataset based on img paths (for dataset with multiple 
        entries for one image)
    Args:
        test_size (float): fraction of images in df that will be used for 
            test dataset
        val_size (float): fraction of images in df that will be used for 
            test dataset
        df (pd.DataFrame): Dataframe to split
    Returns: 
        tuple: pd.DataFrames with dataframe split (train, test, val).
        if a size is 0.0 it will return None at the according place
    '''
    imgs = list(df.img_path.unique())
    n_images = len(imgs)
    splits = []
    for split in [test_size, val_size]:
        if split:
            size = int(split * n_images)
            set_1, set_2 = train_test_split(imgs, test_size=size)
            split_data = img_selection(list(set_2), df=df)
            splits.append(split_data)
            imgs = set_1
        else:
            splits.append(None)
    train_data = img_selection(list(imgs), df=df)
    splits.insert(0, train_data)
    return tuple(splits)


def split_multilabels(lbl_mapping, df:pd.DataFrame=None, col='anno_lbl'):
    ''' Split multilabels column into multiple single label columns
    Args:
        df (pd.DataFrame): dataframe to split
        lbl_mapping (dict): mapping from categorie to labels. 
            {categorie: [categorie lbls, ...]}
        col (str): columns containing multilabels
    Returns:
        pd.DataFrame with new columns for each categorie. Every 
            categorie has the new name col + '_' + categorie
    
    Raise:
        Exception if a multilabel contains multiple labels
            of one categorie
    '''
    df = df.copy()
    cat_to_lbl = {col+'_'+cat: lbl for cat, lbl in lbl_mapping.items()}
    lbl_to_cat = {lbl: cat for cat, lbls in cat_to_lbl.items() for lbl in lbls}
    categories = list(cat_to_lbl.keys())
    df[categories] = None
    ret_dict = {cat: None for cat in categories}
    
    def split_labels(labels):
        ret = ret_dict.copy()
        for lbl in labels:
            if lbl is not None:
                cat = lbl_to_cat[lbl]
                if ret[cat] is not None:
                    raise Exception('Got multilabel for categorie {}. ' \
                                    'Found {}'.format(cat, ret[cat], lbl))
                ret[cat] = lbl
        ret = list(ret.values())
        for lbl in labels:
            assert lbl in ret
        return ret
    
    df[categories] = pd.DataFrame(list(df[col].apply(
        lambda x: split_labels(x))), columns=categories, index=df.index)
    
    return df