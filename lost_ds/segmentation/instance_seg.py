
# def create_instance_segmentation(self, dst_dir, fill_value=0, df=None, inplace=False):
#     '''Create pixelmaps from polygon-annos
    
#     Args:
#         dst_dir (str): Directory to store the pixel maps. 
#         fill_value (int): Pixel value for not annotated areas
#         df (pd.DataFrame): Frame to generate the pixel maps from
#         inplace (bool): Overwrite the own LOST-dataframe when True
    
#     Returns:
#         tuple: (train_df, big_df) The sem seg dataframe with unique paths
#             and the big dataframe with 'mask.path' column added
#     '''

#     os.makedirs(dst_dir, exist_ok=True)

#     if df is None:
#         df = self.df
    
#     df = self.remove_empty(df=df)
#     df = self.remove_empty_labels(df=df)

#     labels = self.get_unique_labels(df=df)
                
#     df = df[df['anno.dtype']=='polygon']
#     # REPLACE pack/unpack:
#     # df['anno.data'] = df['anno.data'].apply(lambda x: json.loads(x))
#     # REPLACE pack/unpack:
#     # df['anno.lbl.name'] = df['anno.lbl.name'].apply(lambda x: json.loads(x)[0])
#     df['anno.lbl.name'] = df['anno.lbl.name'].apply(lambda x: x[0])
#     # TODO: rest of func with x instead of x[0] (label in list)
#     df['mask.path'] = None
    
#     px_values = {'anno.idx':[], 'mask.px_value':[]}

#     image_paths = self.get_unique_img_paths(df=df)
    
#     for i, image_path in enumerate(image_paths):    
#         print_progress_bar(i+1, len(image_paths), prefix='create px-maps:')
#         if not os.path.exists(image_path):
#             raise Exception('Image {} does not exist'.format(image_path))
#         im_w, im_h = get_image_size(image_path)
#         px_map = np.full([im_h, im_w], fill_value, dtype=np.int32)
#         image_annos = df[df['img.img_path']==image_path]
#         px_val = 1
#         for label in labels:
#             # get annos from current image
#             draw = image_annos[image_annos['anno.lbl.name']==label]
#             for poly, anno_id in zip(draw['anno.data'].to_list(), draw['anno.idx'].to_list()):
#                 # draw all annos
#                 poly = (np.array([list(point.values()) for point in poly]) * [im_w, im_h]).astype(np.int32)
#                 cv2.fillPoly(px_map, [poly], px_val)
#                 px_values['mask.px_value'].append(px_val)
#                 px_values['anno.idx'].append(anno_id)
#                 px_val += 1

#         img_name = image_path.split('/')[-1]
#         filename = img_name.split('.')[0]
#         filename = filename + '.png'
#         mask_path = os.path.join(dst_dir, filename)
#         cv2.imwrite(mask_path, px_map)
#         image_annos.loc[:,'mask.path'] = mask_path
#         df[df['img.img_path']==image_path] = image_annos

#     px_mapping = pd.DataFrame(px_values)
#     df = df.merge(px_mapping, on='anno.idx')
#     # REPLACE pack/unpack:
#     # df['anno.data'] = df['anno.data'].apply(lambda x: json.dumps(x))
#     # REPLACE pack/unpack:
#     # df['anno.lbl.name'] = df['anno.lbl.name'].apply(lambda x: json.dumps([x]))
    
#     if inplace:
#         self.df = df
#     return df