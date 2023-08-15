import pandas as pd
from sklearn.model_selection import train_test_split

from lost_ds.functional.filter import img_selection, label_selection


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

def split_train_test(test_size=0.2, val_size=0.2, stratify_col=None, df=None,
                     col='img_path', random_state=42):
    df_base = df.copy()
    print(len(df_base))
    img_uid_col = 'img_uid'
    if not len(df_base):
        return df_base, df_base, df_base
    df_base['split_col'] = df_base[col].apply(lambda x: hash(str(x)))
    samples = df_base['split_col'].unique()
    n = len(samples)
    stratify = None
    ids = list(range(n))
    # validaton + filtering for stratification
    if stratify_col:
        stratify = list(df_base[stratify_col])
        assert len(samples) == len(df_base), 'Samples cannot occur multiple times in dataset when using stratify!'
        # TODO: look that each class has at least [NR_SPLIT] (== 3) unique values in stratify col
            # dict: per class look that unique img_uids are at least the number of splits
        # checking if enough (meaning 3) of each type of class exist to make splits
        # TODO: check airflow and how it calls this func (with or without startify_col???)
        # TODO: new kwarg for uid_col???
        # # img_uids = list(df_base[img_uid_col].unique()) 

        unique_classes = list(df_base[stratify_col].unique())
        print(len(df_base))
        for u_class in unique_classes:
            class_df = df_base[df_base[stratify_col] == u_class]
            # TODO: check nr of unique imgs / img_uids!!!
            unique_imgs = class_df[img_uid_col].unique()
            nr_unique_imgs = len(unique_imgs)
            print(u_class, nr_unique_imgs)
            if nr_unique_imgs < 3:
                to_drop = df_base[df_base[img_uid_col].isin(unique_imgs)]
                len_to_drop = len(to_drop)
                df_base = df_base[~df_base[img_uid_col].isin(unique_imgs)]
                print(f"""Dropped {len_to_drop} entries based off of {nr_unique_imgs} images of class {u_class}, 
                    due to it not having enough unique source-images""")
        print(len(df_base))
        # df_base = df_base.reset_index(drop=True)
        # print("Reset indexes")
        samples = df_base['split_col'].unique()
        n = len(samples)
        stratify = None
        ids = list(range(n))
    
    # doing the splitting
    splits = []
    for split in [test_size, val_size]:
        if split:
            size = int(split * n)
            set_1, set_2, ids_1, ids_2 = train_test_split(samples, ids, test_size=size, 
                                                          shuffle=True, 
                                                          random_state=random_state,
                                                          stratify=stratify)
            split_data = label_selection(list(set_2), df=df_base, col='split_col')
            splits.append(split_data)
            samples = set_1
            ids = ids_1
            if stratify_col:
                print(f"IDs1 = {ids_1}")
                print(max(ids_1), len(df_base))
                stratify = list(df_base.iloc[ids_1][stratify_col])
        else:
            splits.append(None)
    train_data = label_selection(list(samples), df=df_base, col='split_col')
    splits.insert(0, train_data)
    return tuple(splits)

def split_by_img_path(test_size=0.2, val_size=0.2, df=None, random_state=42):
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
    return split_train_test(test_size, val_size, None, df, 'img_path', 
                            random_state)


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
    
    # TODO: continue implementation
    # df[categories] = df[col].apply(split_labels, axis=1, result_type='expand')
    
    return df