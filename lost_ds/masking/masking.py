# TODO: all

    # def mask_ds_from_anno(self, mask_labels, fill_value, dst_dir, invers=False, 
    #             tight_crop=False, df=None, inplace=False, write_mask_only=False,
    #              lbl_col='anno.lbl.name'):
    #     '''Mask dataset
        
    #     Args:
    #         mask_labels (list): list of labels (str) to mask in image
    #         fill_value (int): Pixel value for masked areas
    #         dst_dir (str): Directory to store the masked images
    #         invers (bool): invert the mask
    #         tight_crop (bool): crop result image tight around the mask labels
    #         df (pd.DataFrame): Frame to generate the pixel maps from
    #         inplace (bool): Overwrite the own LOST-dataframe when True
    #         write_mask_only (bool): don't write images where no mask is applied
        
    #     Returns:
    #         pd.DataFrame: The dataframe with the images pixel-map paths added
    #     '''

    #     os.makedirs(dst_dir, exist_ok=True)
    #     if df is None:
    #         df = self.df        
    #     labels = self.get_unique_labels(df=df)
    #     for label in mask_labels:
    #         if not label in labels:
    #             raise Exception('found label {} which does not occur in the dataset'
    #                             .format(label))
    #     for idx, multi_lbls in zip(df.index, df[lbl_col]):
    #         assigned_labels = sum([i in multi_lbls for i in mask_labels])
    #         if assigned_labels > 1:
    #             raise Exception('Found multiple train labels for same annotation!' \
    #                 'in row-idx {}: labels: {} row: {}'
    #                 .format(idx, multi_lbls, df.iloc[idx]))

    #     self.remove_empty(inplace=True)
    #     # self.remove_empty_labels(inplace=True)

    #     df = df[df['anno.dtype']=='polygon']
    #     image_paths = self.get_unique_img_paths(df=df)
    #     for i, image_path in enumerate(image_paths):    
    #         print_progress_bar(i+1, len(image_paths), prefix='mask and crop:')
    #         if not os.path.exists(image_path):
    #             raise Exception('Image {} does not exist'.format(image_path))

    #         img = cv2.imread(image_path)
    #         im_h, im_w, _ = img.shape
    #         label_mask = np.full([im_h, im_w], False, dtype=np.int32)
    #         image_annos = df[df['img.img_path']==image_path]
    #         min_max = []
    #         for mask_label in mask_labels:
    #             draw = self.get_label_selection([mask_label], image_annos)
    #             for poly in draw['anno.data'].to_list():
    #                 poly = (np.array([list(point.values()) for point in poly]) * [im_w, im_h]).astype(np.int32)
    #                 min_max.append(np.hstack((poly.min(axis=0), poly.max(axis=0))))
    #                 cv2.fillPoly(label_mask, [poly], True)

    #         if len(min_max):
    #             label_mask = label_mask.astype(bool)

    #             if invers:
    #                 label_mask = ~label_mask
    #             img[label_mask] = fill_value
                
    #             if tight_crop:
    #                 xy_min = np.array(min_max).min(axis=0)
    #                 xy_max = np.array(min_max).max(axis=0)
    #                 cropped_img = img[xy_min[1]:xy_max[3], xy_min[0]:xy_max[2], :]
                    
    #                 def recalculate_coords(polygon):
    #                     abs_polygon = np.array([list(point.values()) for point in polygon]) * [im_w, im_h]
    #                     abs_polygon = abs_polygon - [xy_min[0], xy_min[1]]
    #                     xclip = np.clip(abs_polygon[:,0], 0, cropped_img.shape[1])
    #                     yclip = np.clip(abs_polygon[:,1], 0, cropped_img.shape[0])
    #                     clipped_poly = np.vstack((xclip, yclip)).T
    #                     area = cv2.contourArea(clipped_poly.astype(np.int32))
    #                     new_polygon = clipped_poly / [cropped_img.shape[1], cropped_img.shape[0]]
    #                     if area < 1:
    #                         return None
    #                     return [{'x':xy[0], 'y':xy[1]} for xy in list(new_polygon)]

    #                 image_annos['anno.data'] = image_annos['anno.data'].apply(lambda x: recalculate_coords(x))
    #                 img = cropped_img
    #         else:
    #             if write_mask_only:
    #                 continue

    #         img_name = image_path.split('/')[-1]
    #         new_path = os.path.join(dst_dir, img_name)
    #         cv2.imwrite(new_path, img)
    #         image_annos['img.img_path'] = new_path
    #         df[df['img.img_path']==image_path] = image_annos

    #     df = df[df['anno.data'].notnull()]

    #     if inplace:
    #         self.df = df
    #     return df


    # def mask_ds_from_pxmap(self, mask_labels, fill_value, dst_dir, 
    #                             invers=False, df=None, inplace=False):
    #     '''Mask images of a dataset by classes
        
    #     Args:
    #         mask_labels (list): labels to mask
    #         fill_value (int): pixel value to fill masked areas
    #         dst_dir (str): directory to store the masked images
    #         invers (bool): invert masked areas
    #         df (pd.DataFrame): Frame to generate the masked set from
    #         inplace (bool): Overwrite the own LOST-dataframe when True
        
    #     Return:
    #         pd.DataFrame: Dataset with new img.img_path to the masked dataset

    #     Note: 
    #         Previous pixel map generation is required
    #     '''
        
    #     if df is None:
    #         df = self.df
        
    #     if not 'mask.path' in df.keys():
    #         raise Exception('There are no masks for available for this dataset. '\
    #                         'Create pixel maps /masks first!')
        
    #     os.makedirs(dst_dir, exist_ok=True)
        
    #     img_paths = self.get_unique_img_paths(df=df)

    #     for i, img_path in enumerate(img_paths):
    #         print_progress_bar(i+1, len(img_paths), prefix='mask images:')
    #         mask_paths = df[df['img.img_path']==img_path]['mask.path'].unique()
            
    #         if len(mask_paths) != 1:
    #             raise Exception('Expected exactly one mask for img but got {}. '\
    #                             'Got: {}'.format(len(mask_paths), mask_paths))
    #         if not os.path.exists(img_path):
    #             raise Exception('File {} does not exist!'.format(img_path))
    #         if not os.path.exists(mask_paths[0]):
    #             raise Exception('File {} does not exist!'.format(mask_paths[0]))

    #         img = cv2.imread(img_path)
    #         mask = cv2.imread(mask_paths[0], 0)
    #         img_mask = np.isin(mask, mask_labels)
    #         if invers:
    #             img_mask = ~img_mask
    #         img[img_mask] = fill_value
            
    #         filename = img_path.split('/')[-1]
    #         cv2.imwrite(os.path.join(dst_dir, filename), img)

    #     if inplace:
    #         self.df = df
    #     return df
