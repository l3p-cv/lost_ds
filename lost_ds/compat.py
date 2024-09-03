
from ast import literal_eval
import json
import pandas as pd
from lost_ds.core import LOSTDataset


def old_lds_to_new_lds(df):
    # load data (unstring)
    def literal_eval_parse(entry):
        if not pd.isnull(entry):
            if isinstance(entry, str):
                return literal_eval(entry.replace('nan', 'None'))
            else:
                return literal_eval(entry)
        else:
            return None

    def parse_col(col):
        try:
            return col.map(lambda entry: literal_eval_parse(entry))
        except Exception as e:
            try:
                return col.map(lambda entry: json.loads(entry))
            except Exception as e:
                return col
            
    df = df.apply(lambda x: parse_col(x), axis=0)
    # map keys
    mapping = { 'anno.idx': 'anno_uid',
                'anno.timestamp': 'anno_timestamp',
                'anno.state': 'anno_state',
                'anno.dtype':'anno_dtype',
                'anno.sim_class':'anno_sim_class',
                'anno.iteration':'anno_iteration',
                'anno.user_id':'anno_user_id',
                'anno.user':'anno_user',
                'anno.confidence':'anno_confidence',
                'anno.anno_time':'anno_anno_time',
                'anno.data': 'anno_data',
                'anno.lbl.name':'anno_lbl',
                'img.idx': 'img_uid',
                'img.timestamp': 'img_timestamp',
                'img.state': 'img_state',
                'img.sim_class': 'img_sim_class',
                'img.frame_n': 'img_frame_n',
                'img.img_path': 'img_path',
                'img.iteration': 'img_iteration',
                'img.user_id': 'img_user_id',
                'img.anno_time': 'img_anno_time',
                'img.lbl.name': 'img_lbl',
                'img.annotator': 'img_user',
                'img.is_junk': 'img_is_junk'
                }
    df_map = df.rename(columns=mapping)
    new_keys = ['anno_style', 'anno_format']
    df_map[new_keys] = None

    # # transform timestamp data
    # all_keys = list(mapping.values())
    # for k in all_keys:
    #     if 'timestamp' in k:
    #         df_map[k] = df_map[k].map(lambda x: str(x))
    
    # transform anno data
    def parse_data(row):
        dtype = row['anno_dtype']
        data = None
        style = None
        frmt = 'rel'
        if dtype == 'bbox':
            data = list(row['anno_data'].values())
            style = 'xcycwh'
        elif dtype in ['polygon', 'line']:
            data = [list(p.values()) for p in row['anno_data']]
            style = 'xy'
        elif dtype == 'point':
            data = list(row['anno_data'].values())
            style = 'xy'
        return data, style, frmt
    new_dat = df_map.apply(lambda x: parse_data(x), axis=1)
    dic = {'anno_data': [],
           'anno_style': [],
           'anno_format': []}
    
    indexes = []
    for i, dat in new_dat.iteritems():
        indexes.append(i)
        dic['anno_data'].append(dat[0])
        dic['anno_style'].append(dat[1])
        dic['anno_format'].append(dat[2])
    df_dat = pd.DataFrame(dic, index=indexes)
    df_map[['anno_data', 'anno_style', 'anno_format']] = df_dat[['anno_data', 'anno_style', 'anno_format']]
    
    drop_keys = [k for k in list(df_map.keys()) if '.' in k]
    df_map.drop(labels=drop_keys, axis='columns', inplace=True)
    
    def _lbl_parser(lbl):
        if isinstance(lbl, str): return lbl.lower()
        else: return lbl
    df_map['anno_lbl'] = df_map['anno_lbl'].map(lambda x: [_lbl_parser(lbl) for lbl in x] if isinstance(x, list) else x)
    
    return LOSTDataset(df_map)

