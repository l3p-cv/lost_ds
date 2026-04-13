import os, glob

import matplotlib.pyplot as plt
import cv2

import lost_ds as lds

# Dataset paths
coco_path = 'lost_coco_annos.parquet'
shapes_path = 'lost_shape_annos.parquet'

# directory for visualization
vis_dir = '/tmp/lds_demo'

# pass a path-string (pandas style)
ds_coco = lds.LOSTDataset(coco_path)
ds_coco['img_path'] = ds_coco['img_path'].apply(lambda x: os.path.join('/home/dkoerner/development/code/lost_ds/examples/imgs', x))
ds_coco.validate_image_paths(inplace=True)
print(ds_coco)

order = {'background': 0,
         'circle': 30,
         'square': 60,
         'line': 90,
         'others': 120,
         'food': 150,
         'animal': 180,
         'person': 210}

df_pan = lds.panoptic_segmentation(order, 'panopt', ds_coco.df, parallel=0)
print(df_pan)