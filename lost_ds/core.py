from os import name
import numpy as np

from lost_ds.io.file_man import FileMan
from lost_ds.geometry.api import LOSTGeometries

from lost_ds.functional.validation import (validate_geometries,
                                           validate_empty_images,
                                           validate_unique_annos,
                                           validate_img_paths,
                                           validate_single_labels)

from lost_ds.functional.split import (split_by_empty,
                                      split_by_img_path,
                                      split_multilabels)

from lost_ds.functional.filter import (remove_empty,
                                       ignore_labels,
                                       img_selection,
                                       is_multilabel,
                                       label_selection,
                                       selection_mask,
                                       unique_labels)

from lost_ds.functional.transform import (polygon_to_bbox,
                                          to_abs, to_coco,
                                          to_rel,
                                          transform_bbox_style)

from lost_ds.functional.mapping import (remap_img_path,
                                       remap_labels,
                                       )

from lost_ds.copy import (copy_imgs,
                          pack_ds
                          )

from lost_ds.vis.vis import (vis_and_store,
                             vis_sample,
                             vis_semantic_segmentation,
                             )

from lost_ds.cropping.cropping import (crop_dataset,
                                       DSCropper,
                                       crop_components,
                                       )

from lost_ds.segmentation.semantic_seg import (semantic_segmentation,
                                               )

from lost_ds.segmentation.anno_from_seg import (segmentation_to_lost,
                                               )

from lost_ds.detection.detection import detection_dataset

from lost_ds.util import get_fs, to_parquet


class LOSTDataset(object):
    '''Represents a dataset in LOST format.'''
    
    def __init__(self, 
                 ds, 
                 filesystem=None, 
                 **kwargs):
        ''' Initialize LOSTDataset
        
        Args:
            ds (str, pd.DataFrame, LOSTDataset, list(s)): 
                Dataset(s) to use. The different types can be combined in lists.
            filesystem (fsspec.filesystem, None): Filesystem to use. If None the 
                local filesystem will be used.
            
        '''
        self.fileman = FileMan(filesystem)
        self.df = self.fileman.load_dataset(ds)
        try:
            self._parse_data()
        except TypeError as e:
            pass
        self.geom = LOSTGeometries()
        self.cropper = DSCropper(self.geom, self.fileman)
        
    #
    #   self.df management
    #
    
    def _parse_data(self):
        # parse anno_data to numpy array
        if 'anno_data' in self.df:
            self.df.anno_data = self.df.anno_data.apply(
                lambda x: np.vstack(x).squeeze())
            
            
    def to_parquet(self, path, df=None):
        ''' Store dataset as .parquet file
        
        Args:
            path (str): path to store the dataset
            df (pd.DataFrame): dataframe to store
        '''
        df = self._get_df(df)
        to_parquet(path, df, self.fileman)
        
    
    def _get_df(self, df):
        '''DataFrame selection
        
        Args:
            df (pd.DataFrame, None): dataframe to select
            
        Returns:
            pd.DataFrame: self.df if df is None else df
        '''
        if df is None:
            return self.df.copy()
        else: 
            return df
    
    
    def _update_inplace(self, df, inplace):
        '''inplace management
        
        Args:
            df (pd.DataFrame): dataframe to update
            inplace (bool): Flag if self.df should be overwritten
            
        Returns:
            None if inplace=True
            pd.DataFrame if inplace=False
        '''
        if inplace:
            self.df = df
            return None
        else:
            return df
    
    
    def get_imagesize(self, path):
        '''Get the size of an image without reading the entire data
        
        Args:
            path (str): path to image
            
        Returns:
            tuple of int: (im_h, im_w) height and width of the image
        '''
        return self.fileman.get_imagesize(path)
        
    
    def copy(self):
        return LOSTDataset(self.df.copy(), self.fileman)
    
    
    #
    #   adapt important and usefull pandas- and builtin-functions
    #
    
    def __repr__(self):
        # enables printing dataframe in interpreter by 'ds'
        return self.df.__repr__()
    
    def __str__(self):
        # enabels printing dataframe by 'print(ds)'
        return str(self.df)
    
    def __len__(self):
        # enables getting dataframe length
        return len(self.df)

    def keys(self):
        return self.df.keys()
    
    def __getitem__(self, name):
        # enables getting value by 'ds[key]'
        return self.df[name]
    
    def __setitem__(self, key, value):
        return self.df.__setitem__(key, value)
    
    # def __getattr__(self, name: str):
    #     # enable getting attributes with ds.attr
    #     # if name.startswith('__') and name.endswith('__'):
    #     #     raise ValueError
    #     try:
    #         return self.df.__getattr__(name)
    #     except:
    #         try:
    #             return getattr(self, name)
    #         except:
    #             raise AttributeError(name)
    
    #
    #   Data copy
    #
    
    def copy_imgs(self, out_dir, df=None, col='img_path', copy_path=None, 
                  force_overwrite=False):
        '''Copy all images of dataset into out_dir

        Args:
            df (pd.DataFrame): dataframe to copy
            out_dir (str): Destination folder to store images
            copy_path (str, optional): directory where to copy all images
            col (str): column containing paths to files
        '''
        df = self._get_df(df)
        copy_imgs(df=df, out_dir=out_dir, col=col, copy_path=copy_path,
                  force_overwrite=force_overwrite, filesystem=self.fileman)
            
    
    def pack_ds(self, out_dir, df=None, cols=['img_path', 'mask_path'], 
                dirs = ['imgs', 'masks', 'crops'], inplace=False):
        '''Copy all images from dataset to a new place and update the dataframe 
        
        Args:
            df (pd.DataFrame): Dataframe to copy
            out_dir (str): Name of the directory to store the information
            cols (list of string): column names containing file-paths
            dirs (list of string): name of new directories according to cols. 
                The dirs will contain the copied data
        
        Returns:
            pd.DataFrame with new image paths
        '''
        df = self._get_df(df)
        df = pack_ds(df=df, out_dir=out_dir, cols=cols, dirs=dirs, 
                     filesystem=self.fileman)
        return self._update_inplace(df, inplace=inplace)
    
    
    #
    #   Filtering
    #
    
    def remove_empty(self, df=None, col='anno_data', inplace=False):
        '''Remove images with empty columns in specified column
        
        Args:
            df (pd.DataFrame): Dataframe to use for filtering
            col (str): column to seatch for empty entries
            inplace (bool): overwrite self.df with result if True
        
        Returns:
            pd.DataFrame if inplace=False
            None if inplace=True
        '''
        df = self._get_df(df)
        df = remove_empty(df, col)
        return self._update_inplace(df, inplace)
    
    
    def ignore_labels(self, labels, df=None, col='anno_lbl', inplace=False):
        ''' Remove dataframe entries where the given labels occures

        Args:
            labels (list): list of labels to ignore
            df (pd.DataFrame): Frame to apply label ignore
            col (str): Column containing list of labels
            inplace (bool): overwrite self.df with result if True

        Returns:
            pd.DataFrame: dataframe with label selection
        '''
        df = self._get_df(df)
        df = ignore_labels(labels=labels, df=df, col=col)
        return self._update_inplace(df, inplace)
    
    
    def selection_mask(self, labels, df=None, col='anno_lbl'):
        '''Get mask for labels in dataset
        Args:
            df (pd.DataFrame): dataset to mask
            col (str): Column containing list of labels
        
        Returns:
            pd.DataFrame: row-wise boolean mask with True is a label occures in
                this row, False if not
        '''
        df = self._get_df(df)
        return selection_mask(labels=labels, df=df, col=col)
    
    
    def img_selection(self, imgs, df=None, inverse=False, inplace=False):
        '''Get entries with a selection of labels from the dataframe

        Args:
            imgs (list): list of imgs to select
            invers (bool): get the selection if True, get the rest if False 
            df (pd.DataFrame): Frame to apply image selection

        Returns:
            pd.DataFrame: dataframe with image selection
        '''
        df = self._get_df(df)
        df = img_selection(imgs=imgs, df=df, invers=inverse)
        return self._update_inplace(df, inplace)
    
    
    def is_multilabel(self, df=None, col='anno_lbl'):
        '''check if a column contains multilabels

        Args:
            df (pd.DataFrame): Dataframe to use for filtering
            col (str): column to check for multilabel

        Returns:
            pd.DataFrame: row-wise boolean mask with True if column contains 
                multilabel False if not
        '''
        df = self._get_df(df)
        return is_multilabel(df=df, col=col)
        
    
    def label_selection(self, labels, df=None, col='anno_lbl', inplace=False):
        '''Get entries with a selection of labels from the dataframe
        Args:
            labels (list): list of labels to select
            df (pd.DataFrame): Frame to apply label selection
            col (str): Column containing list of labels
        Returns:
            pd.DataFrame: dataframe with label selection
        '''
        df = self._get_df(df)
        df = label_selection(labels=labels, df=df, col=col)
        return self._update_inplace(df, inplace)
    
    
    def unique_labels(self, df=None, col='anno_lbl'):
        '''Get unique dataset labels.
        Args:
            df (pd.DataFrame): dataframe to analyze
            col (str): Column containing list of labels

        Retrun:
            str: List of strings with unique classnames
        '''
        df = self._get_df(df)
        return unique_labels(df, col)
        
    
    
    #
    #   Mapping
    #
    
    def remap_img_path(self, new_root_path, df=None, col='img_path', 
                       inplace=False):
        '''Remap img_path by prepending a root path to the filename
        
        Args:
            new_root_path (str): New root path for all files 
            col (str): The column that should be remaped
        '''
        df = self._get_df(df)
        df = remap_img_path(df, new_root_path, col)
        return self._update_inplace(df, inplace)
    
    def remap_labels(self, label_mapping, df=None, col='anno_lbl', 
                     dst_col='anno_lbl_mapped', inplace=False):
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
        df = self._get_df(df)
        df = remap_labels(df, label_mapping, col, dst_col)
        self._update_inplace(df, inplace)
    
    
    #
    #   Transformation
    #
    
    def to_abs(self, df=None, path_col='img_path', verbose=True, inplace=False):
        ''' Transform dataframe to absolute data
        
        Args:
            df (pd.DataFrame): dataframe to transform
            verbose (bool): print tqdm progress-bar
            inplace (bool): overwrite self.df
        '''
        df = self._get_df(df)
        df = to_abs(df, path_col, self.fileman, verbose)
        return self._update_inplace(df, inplace)
    
    
    def to_rel(self, df=None, path_col='img_path', verbose=True, inplace=False):
        ''' Transform dataframe to relative data
        
        Args:
            df (pd.DataFrame): dataframe to transform
            verbose (bool): print tqdm progress-bar
            inplace (bool): overwrite self.df
        '''
        df = self._get_df(df)
        df = to_rel(df, path_col, self.fileman, verbose)
        self._update_inplace(df, inplace)
    
    
    def polygon_to_bbox(self, df=None, dst_style=None, inplace=False):
        '''Transform all polygons to bboxes
        
        Args:
            df (pd.DataFrame): datafram to apply transform
            dst_style (str): desired style of bbox: 
                {'xywh', 'x1y1x2y2', 'xcycwh'}
            inplace (bool): overwrite self.df
            
        Returns:
            pd.DataFrame with transformed polygons
        '''
        df = self._get_df(df)
        df = polygon_to_bbox(df, dst_style)
        return self._update_inplace(df, inplace)
    
    
    def transform_bbox_style(self, dst_style, df=None, inplace=False):
        '''Transform anno-style of all bboxes
        
        Args:
            dst_style (str): desired bbox style {'xcycwh', 'xywh', 'x1y1x2y2'}
            df (pd.DataFrame): dataframe to transform
            inplace (bool): overwrite self.df
        
        Returns:
            pd.DataFrame: dataframe with transformed bboxes
        '''
        df = self._get_df(df)
        df = transform_bbox_style(dst_style, df)
        return self._update_inplace(df, inplace)
        
        
    def to_coco(self, df=None, remove_invalid=True, lbl_col='anno_lbl', 
                supercategory_mapping=None, copy_path=None, json_path=None,
                rename_by_index=False):
        """Transform dataset to coco data format

        Args:
            df ([pd.DataFrame], optional): dataframe to apply transform
            remove_invalid (bool, optional): Remove invalid image-paths. Defaults to True.
            lbl_col (str, optional): dataframe column to look for labels. Defaults to 'anno_lbl'.
            supercategory_mapping ([type], optional): dict like mapping for supercategories ({class: superclass}). 
                Defaults to None.
            copy_path (str, optional): copy all images to given directory
            rename_by_index (bool, optional): rename files by index to assure unique filenames
            json_path (str, optional): store coco annotations as .json file 

        Returns:
            dict: dict containing coco data like {'categories': [...], 'images': [...], 'annotations': [...]}
        """
        df = self._get_df(df)
        coco_annos = to_coco(df, remove_invalid=remove_invalid, lbl_col=lbl_col, 
                             supercategory_mapping=supercategory_mapping, 
                             copy_path=copy_path, 
                             rename_by_index=rename_by_index, 
                             json_path=json_path,
                             filesystem=self.fileman)
        return coco_annos
    
    
    #
    #   Splitting
    #
    
    def split_by_empty(self, df=None, col='anno_data'):
        '''Split dataframe into empty and non-empty entries by given collumn
        
        Args:
            df (pd.DataFrame): df to apply on
            col (str): column to check for empty entries
            
        Returns:
            tuple of pd.DataFrame: first dataframe contains non-emtpy entries, 
                the second dataframe contains empty entries
        '''
        df = self._get_df(df)
        return split_by_empty(df, col)
    
    
    def split_by_img_path(self, test_size=0.2, val_size=0.2, df=None):
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
        df = self._get_df(df)
        return split_by_img_path(test_size=test_size, val_size=val_size, df=df)
    
    
    def split_multilabels(self, lbl_mapping, df=None, col='anno_lbl', inplace=False):
        ''' Split multilabels column into multiple single label columns
        Args:
            df (pd.DataFrame): dataframe to split
            lbl_mapping (dict): mapping from categorie to labels. 
                {categorie: [categorie lbls, ...]}
            col (str): columns containing multilabels
            inplace (bool): overwrite LOSTDataset.df with result dataframe
        Returns:
            pd.DataFrame with new columns for each categorie. Every 
                categorie has the new name col + '_' + categorie
        
        Raise:
            Exception if a multilabel contains multiple labels
                of one categorie
        '''
        df = self._get_df(df)
        df = split_multilabels(df=df, lbl_mapping=lbl_mapping, col=col)
        return self._update_inplace(df, inplace=inplace)
        
    
    
    #
    #   Validation
    #
    
    def validate_geometries(self, df=None, remove_invalid=True, inplace=False):
        '''Validate geometries 
        
        Args:
            remove_invalid (bool): remove invalid geometries from the dataframe
            df (pd.DataFrame): DataFrame to check
            inplace (bool): overwrite LOSTDataset.df with result dataframe
            
        Returns:
            None or pd.DataFrame
        '''
        df = self._get_df(df)
        df = validate_geometries(df, remove_invalid)
        return self._update_inplace(df, inplace)


    def validate_empty_images(self, df=None, inplace=False):
        '''Remove empty entries for images where non-empty entries do exist
        
        Args:
            df (pd.DataFrame): df to apply on
            inplace (bool): overwrite own dataframe with result dataframe
            
        Returns:
            pd.DataFrame: dataframe with validated empty images
        '''
        df = self._get_df(df)
        df = validate_empty_images(df)
        return self._update_inplace(df, inplace)
    
    
    def validate_unique_annos(self, df=None, inplace=False):
        '''Validate that annotations are unique
        
        Args:
            df (pd.DataFrame): df to apply on
            inplace (bool): overwrite own dataframe with result dataframe
            
        Returns:
            pd.DataFrame: dataframe with validated empty images
        '''
        df = self._get_df(df)
        df = validate_unique_annos(df)
        return self._update_inplace(df, inplace)


    def validate_image_paths(self, df=None, remove_invalid=True, inplace=False):
        '''Validate that paths do exist
        
        Args:
            remove_invalid (bool): remove non existing paths from the dataframe
            df (pd.DataFrame): df to apply on
            inplace (bool): overwrite own dataframe with result dataframe
            
        Returns:
            pd.DataFrame: dataframe with validated image paths
        '''
        df = self._get_df(df)
        df = validate_img_paths(df, remove_invalid, self.fileman)
        return self._update_inplace(df, inplace)
    
    
    def validate_single_labels(self, df=None, lbl_col='anno_lbl', 
                               dst_col='my_lbl', inplace=False):
        '''Validate if labels in a multilabel column are single labels

        Args:
            df (pd.DataFrame): Dataframe to validate single labels
            lbl_col (str): label col to validate
            ds_col (str): new column containing single label
            inplace (bool): overwrite own dataframe with result dataframe
            
        Returns:
            pd.DataFrame with additioinal column dst_col containing single labels
        '''
        df = self._get_df(df)
        df = validate_single_labels(df, lbl_col, dst_col)
        return self._update_inplace(df, inplace)
    
    
    
    #
    #   Visualization
    #
    
    def vis_and_store(self, out_dir, df=None, lbl_col='anno_lbl', 
                      color=(0, 0, 255), line_thickness='auto', radius=2):
        '''Visualize annotations and store them to a folder

        Args:
            df (pd.DataFrame): Optional dataset in lost format to visualize
            out_dir (str): Directory to store the visualized annotations
            color (tuple, dict of tuple): colors (B,G,R) for all annos if tuple 
                or dict for labelwise mapping like {label: color}
            line_thickness (int, dict of int): line thickness for annotations if int
                or dict for anno-type wise mapping like {dtype: thickness}
            lbl_col (str): column containing the labels
            radius (int): radius to draw for points/circles
        '''
        df = self._get_df(df)
        vis_and_store(df, out_dir, lbl_col, color, line_thickness, self.fileman,
                      radius)
    
    
    def vis_semantic_segmentation(self, out_dir, n_classes, palette='dark', 
                                  seg_path_col='seg_path', df=None):
        """Visualize the stored semantic segmentations by coloring it
    
        Args:
            out_dir (str): path to store images
            n_classes (int): number of classes occuring in pixelmaps, number of 
                different colors needed for visualization
            palette (str): seaborn color palette i.e. 'dark', 'bright', 'pastel',...
                refer https://seaborn.pydata.org/tutorial/color_palettes.html 
            df (pandas.DataFrame): The DataFrame that contains annoations to 
                visualize. 
        """
        df = self._get_df(df)
        vis_semantic_segmentation(df, out_dir, n_classes, palette, seg_path_col, 
                                  self.fileman)


    #
    #   Cropping
    #
    
    def crop_dataset(self, dst_dir, crop_shape=(500, 500), overlap=(0, 0), 
                     df=None, write_emtpy=False, fill_value=0, inplace=False):
        """Crop the entire dataset with fixed crop-shape

        Args:
            df (pd.DataFrame): dataframe to apply bbox typecast
            dst_dir (str): Directory to store the new dataset
            crop_shape (tuple, list): [H, W] cropped image dimensions
            overlap (tuple, list): [H, W] overlap between crops
            write_empty (bool): Flag if crops without annos wil be written
            fill_value (float): pixel value to fill the rest of crops at borders
            inplace (bool): overwrite own dataframe with result dataframe
            
        Returns:
            pd.DataFrame
        """
        df = self._get_df(df)
        df = crop_dataset(df=df, dst_dir=dst_dir, crop_shape=crop_shape, 
                          overlap=overlap, write_empty=write_emtpy, 
                          fill_value=fill_value, filesystem=self.fileman)
        return self._update_inplace(df, inplace)
    
    
    def crop_anno(self, img_path, crop_position, im_w=None, im_h=None, df=None):
        """Calculate annos for a crop-position in an image
        
        Args:
            img_path (str): image to calculate the crop-annos
            crop_position (list): crop-position like: [xmin, ymin, xmax, ymax] 
                in absolute data-format
            im_w (int): width of image if available
            im_h (int): height of image if available
            df (pd.DataFrame): dataframe to apply 
            
        Returns:
            pd.DataFrame 
        """
        df = self._get_df(df)
        return self.cropper.crop_anno(img_path, df, crop_position, im_w, im_h)
    
    
    def crop_components(self, dst_dir, base_labels=-1, lbl_col='anno_lbl', 
                        context=0, df=None, context_alignment=None, 
                        min_size=None, anno_dtype=['polygon'], 
                        center_lbl_key='center_lbl', inplace=False):
        """Crop the entire dataset with fixed crop-shape

        Args:
            df (pd.DataFrame): dataframe to apply bbox typecast
            dst_dir (str): Directory to store the new dataset
            base_labels (list of str): labels to align the crops 
            lbl_col (str): column holding the labels
            context (float, tuple of floats): context to add to each component for 
                cropping (twice -> left/right, top/bottom). If tuple of float: uses 
                the given floats as fraction of the components (H, W) to calculate 
                the added contexts. 
            context_alignment (str): Define alignment of the crop-context. One of 
                [None, 'max', 'min', 'flip']. 
                None: Use H/W-context in H/W-dimension
                max: use maximum of H-context, W-context for both dimensions
                min: use minimum of H-context, W-context for both dimensions
                flip: use H-context in W-dimension and vice versa. This makes the 
                    crop a little bit more squarish shaped
            min_size (int, tuple of int): minimum size of produced crops in both
                dimensions if int or for (H, W) if tuple of int
            anno_dtype (list of str): dtype to apply on
            center_lbl_key (str): column containing the label the crop was aligned to
                
        Returns:
            pd.DataFrame
        """
        df = self._get_df(df)
        df = crop_components(df, dst_dir, base_labels, lbl_col, context, 
                             context_alignment, min_size, anno_dtype, 
                             center_lbl_key, self.fileman)
        return self._update_inplace(df, inplace)
        
    #
    #   Segmentation
    #
    
    def semantic_segmentation(self, order, dst_dir, fill_value, df=None, 
                              anno_dtypes=['polygon'], lbl_col='anno_lbl', 
                              dst_path_col='seg_path', dst_lbl_col='seg_lbl', 
                              line_thickness=None, radius=None, inplace=False):
        '''Create semantic segmentations from polygon-annos
    
        Args:
            df (pd.DataFrame): dataframe to generate the pixel maps from
            order (dict, list): order of the classes. Higher pixel
                values will overwrite lower ones. 
                Example: order=['Car', 'Person', 'Glasses'] or  
                    order={'Car': 0, 'Person': 1, 'Glasses': 2}
                    Car will get pixel value 0, Person px value 1 and so on - 
                    Person overwrites Car, Glasses overwrites Person ... 
            dst_dir (str): Directory to store the pixel maps.
            fill_value (int): Pixel value for not annotated areas. Usually this
                will be something like 0 for something like 'background'
            anno_dtypes (list of string): dtypes to use for segmentation. Possible
                values are {'polygon', 'bbox', 'line', 'point'}. Annotations with
                dtypes other than anno_dtypes will be removed from dataframe
            lbl_col (str): Column containing the training labels
            line_thickness (int): thickness of line-segmentation when using an 
                annotation of dtype line. Only takes effect when anno_dtypes 
                contains 'line'
            radius (int): radius of circle-segmentation when using an annotation
                of dtype point. Only takes effect when anno_dtypes contains 
                'point'
            inplace (bool): overwrite self.df with result df
                
        Returns:
            pd.DataFrame: The original dataframe with new column (dst_col) 
                containing the path to the according segmentation file. 
                Furthermore the column dst_lbl_col contains the label the
                segmentation looked up in order for creation
        '''
        df = self._get_df(df)
        df = semantic_segmentation(order, dst_dir, fill_value, df, anno_dtypes, 
                                   lbl_col, dst_path_col, dst_lbl_col, 
                                   line_thickness, radius, self.fileman)
        return self._update_inplace(df, inplace)
    
    
    def segmentation_to_lost(self, pixel_mapping, background=0, 
                             seg_key='seg_path', df=None, inplace=False):
        '''Create LOST-Annotations from semantic segmentations / pixelmaps 
        
        Args:
            pixel_mapping (dict): mapping of pixel-value to anno_lbl, e.g. 
                {0:'background', 1:'thing'}
            background (int): pixel-value for background.
            seg_key (str): dataframe key containing the paths to the stored 
                pixelmaps
            df (pd.DataFrame): dataframe to apply on
            
        Returns:
            pd.DataFrame with polygon-annotations in LOSTDataset format
        '''
        df = self._get_df(df)
        df = segmentation_to_lost(df, pixel_mapping=pixel_mapping, 
                                      background=background, seg_key=seg_key, 
                                      filesystem=self.fileman)
        return self._update_inplace(df, inplace)
    
    
    #
    #   Detection
    #
    
    def detection_dataset(self, df=None, lbl_col='anno_lbl', det_col='det_lbl', 
                          bbox_style='x1y1x2y2', use_empty_images=False, 
                          inplace=False):
        '''Prepare all bboxes to use them for detection CNN training
        Args:
            df (pd.DataFrame): Dataframe containing bbox annotations
            lbl_col (str): column name where the anno labels are located (single 
                label or multilabel)
            det_col (str): column name where the training labels are located (single 
                label only)
            bbox_style (str): bbox anno-style {'xywh', 'x1y1x2y2', 'xcycwh'}
            use_empty_images (bool, str, int): specifiy usage of empty images 
                (empty image means image without bbox annotation).
                True: keep all images, empty and non-empty
                False: only keep non-empty images and drop all empty images
                'balanced': If more empty images than non-empty ones do exist a 
                    random selection will be sampled to have the same amount of 
                    empty and non-empty images. If less empty than non-empty images 
                    do exist all of them will be kept
                int: a specific amount of empty images will be samples randomly
            inplace (bool): overwrite self.df with result df
                
        Returns:
            pd.DataFrame: detection dataset

        Note:
            Other anno-types than 'bbox' will be ignored. You can transform 
            polygons to bboxes before by calling LOSTDataset.polygon_to_bbox()
        '''
        df = self._get_df(df)
        df = detection_dataset(df, lbl_col, det_col, bbox_style, 
                               use_empty_images, self.fileman)
        return self._update_inplace(df, inplace)