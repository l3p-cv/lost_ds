import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from lost_ds.functional.filter import img_selection, label_selection
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer


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
                     col='img_path', uid_col=None, random_state=42):
    """Splits dataframe into train, test and val datasets.
    
    Args:
        test_size (float): Value between 0.0 and 1.0 describing how much data is used for the test set.
            Defaults to 0.2
        val_size (float): same as test_size, but for the validation set
        stratify_col (str): Column whose classes get distributed to the different splits if this arg is not None.
            Defaults to None
        df (pd.DataFrame): The data to split
        col (str): Column with samples (should be unique values if using stratify_col)
        img_uid_col (str): column saying which unique sources are in the dataframe.
            There have to be at least 3 unique sources (per class in stratify_col) in this column, to make the 3 splits (fairly).
            Only relevant if stratify_col is not None.
            Defaults to img_uid.
        random_state (int): Seed for random operations.
            Defaults to 42
    
    Returns:
        The created splits as a tuple
    """
    df_base = df.copy()
    if not len(df_base):
        return df_base, df_base, df_base
    df_base['split_col'] = df_base[col].apply(lambda x: hash(str(x)))
    samples = df_base['split_col'].unique()
    n = len(samples)
    stratify = None
    ids = list(range(n))

    nr_splits = 3
    if not test_size:
        nr_splits-=1
    if not val_size:
        nr_splits-=1

    # validaton + filtering for stratification
    if stratify_col:
        stratify = list(df_base[stratify_col])
        assert len(samples) == len(df_base), 'Samples cannot occur multiple times in dataset when using stratify!'

        if uid_col is not None:
            unique_classes = list(df_base[stratify_col].unique())
            # check nr of unique sources per class
            # HACK: could use validate_single_labels() here, but that causes circular imports...
            for u_class in unique_classes:
                class_df = df_base[df_base[stratify_col] == u_class]
                unique_imgs = class_df[uid_col].unique()
                nr_unique_imgs = len(unique_imgs)
                print(u_class, nr_unique_imgs)
                # has enough base-imgs per class per split
                if nr_unique_imgs < nr_splits:
                    to_drop = df_base[df_base[uid_col].isin(unique_imgs)]
                    len_to_drop = len(to_drop)
                    df_base = df_base[~df_base[uid_col].isin(unique_imgs)]
                    print(f"""Dropped {len_to_drop} entries based off of {nr_unique_imgs} images of class {u_class}, 
                        due to it not having enough unique source-images""")
            samples = df_base['split_col'].unique()
            n = len(samples)
            ids = list(range(n))
            stratify = list(df_base[stratify_col])
    
    # doing the splitting
    splits = []
    for split in [test_size, val_size]:
        if split:
            size = int(split * n)
            # something left to split? (return empty df_base if not)
            if size == 0:
                return df_base, df_base, df_base
            set_1, set_2, ids_1, ids_2 = train_test_split(samples, ids, test_size=size, 
                                                          shuffle=True, 
                                                          random_state=random_state,
                                                          stratify=stratify)
            split_data = label_selection(list(set_2), df=df_base, col='split_col')
            splits.append(split_data)
            samples = set_1
            ids = ids_1
            if stratify_col:
                stratify = list(df_base.iloc[ids_1][stratify_col])
        else:
            splits.append(None)
    train_data = label_selection(list(samples), df=df_base, col='split_col')
    splits.insert(0, train_data)
    return tuple(splits)


def create_multilabel_data(df, anno_col='anno_lbl', mult_col='mult_lbl', single_lbls='single_lbl'):
    """Creates a column with multilabel data and drops unique image-paths"""
    unique_img_df = df.drop_duplicates('img_path')
    for path, path_df in df.groupby('img_path'):
        multi_lbl_list = path_df[single_lbls].unique()
        indexes = path_df.index
        unique_img_df.at[indexes[0], mult_col] = multi_lbl_list
        
    return unique_img_df


def split_train_test_multilabel(stratify_col, test_size=0.2, val_size=0.2, df=None,
                     col='img_path', uid_col='img_uid', random_state=42):
    """Splits dataframe into train, test and val datasets via multilabel stratification.
        Slow when it comes to big datasets. Use the faster split_train_test() if possible.
    
    Args:
        stratify_col (str): Column whose classes get distributed to the different splits.
            Each entry has to be a list of labels (multilabel). The entries will be merged
            according to the uid_col, to create the real multilabels for each entry.
        test_size (float): Value between 0.0 and 1.0 describing how much data is used for the test set.
            Defaults to 0.2
        val_size (float): same as test_size, but for the validation set
        df (pd.DataFrame): The data to split
        col (str): Column with samples (should be unique values if using stratify_col)
        img_uid_col (str): column saying which unique sources are in the dataframe.
            There have to be at least enough unique sources (per class in stratify_col) in this column, to place 1 instance in each split.
            Only relevant if stratify_col is not None.
            Defaults to img_uid.
        random_state (int): Seed for random operations.
            Defaults to 42
    
    Returns:
        The created splits as a tuple
    """
    from lost_ds.functional.validation import validate_single_labels # HACK: prevent circular import
    df_base = df.copy()

    # check nr of splits
    nr_splits = 3
    if not test_size:
        nr_splits-=1
    if not val_size:
        nr_splits-=1
    # check nr of unique sources per class; drop if less than nr of splits
    df_base = validate_single_labels(df_base, lbl_col=stratify_col, dst_col='single_lbl')
    unique_img_lbl_df = df_base.drop_duplicates(subset=[uid_col,'single_lbl'])
    lbl_count = unique_img_lbl_df['single_lbl'].value_counts()
    for clss in lbl_count.keys():
        if lbl_count[clss] < nr_splits:
            to_drop_uids = df_base[df_base['single_lbl'] == clss][uid_col]
            df_base = df_base[~df_base[uid_col].isin(to_drop_uids)]
            print(f"Dropped {len(to_drop_uids)} entries of class {clss}")

    multi_lbl_df = create_multilabel_data(df_base, anno_col=stratify_col)
    multi_lbl_df = multi_lbl_df.reset_index(drop=True)
    if not len(multi_lbl_df):
        print("Return empty splits")
        return multi_lbl_df, multi_lbl_df, multi_lbl_df
    multi_lbl_df['split_col'] = multi_lbl_df[col].apply(lambda x: hash(str(x)))
    samples = multi_lbl_df['split_col'].unique()    
    n = len(samples)
    ids = np.array(list(range(n)))
    assert len(samples) == len(multi_lbl_df), 'Samples cannot occur multiple times in dataset when using stratify!'

    # doing the splitting
    split_paths = []
    multilabels = multi_lbl_df[stratify_col] # new labels after dropping
    mult_lbl_binarizer = MultiLabelBinarizer()
    hot_encoding = mult_lbl_binarizer.fit_transform(multilabels)
    for split in [test_size, val_size]:
        if split:
            size = int(split * n)
            # something left to split? (return empty base_df if not)
            if size == 0:
                return df_base, df_base, df_base
            multisplitter = MultilabelStratifiedShuffleSplit(test_size=size, random_state=random_state, n_splits=1)
            for train_index, test_index in multisplitter.split(samples, hot_encoding):
                set_1, set_2 = samples[train_index], samples[test_index]
                ids_1, ids_2 = ids[train_index], ids[test_index]
                hot_encoding_1, hot_encoding_2 = hot_encoding[train_index], hot_encoding[test_index]
            split_data = label_selection(list(set_2), df=multi_lbl_df, col='split_col')
            split_paths.append(split_data['img_path'])
            samples = set_1
            ids = ids_1
            hot_encoding = hot_encoding_1
        else:
            split_paths.append(None)
    train_data = label_selection(list(samples), df=multi_lbl_df, col='split_col')
    split_paths.insert(0, train_data['img_path'])
    split_dfs = []
    for paths in split_paths:
        if paths is not None:
            split_df = df_base[df_base['img_path'].isin(paths)]
            split_dfs.append(split_df)
        else:
            split_dfs.append(None)
    
    return tuple(split_dfs)

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