# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased]
### Changed
- switched package build system to uv
- fix bug in unique labels
- add crop shape and overlap shape to filename when cropping ds

## [1.1.2] - 2024-01-30
### Changed
- Moved package building to pyproject.toml
### Fixed
- Fixed Pipy build

## [1.1.1] - 2024-01-22
### Fixed
- Try to fix pypy build

## [1.1.0] - 2024-01-22
### Changed
- `split_train_test` now filters samples with too few unique sources to prevent errors in sklearn
- refactored to_abs and to_rel method for easier readability and better performance
### Added
- retry reading a parquet file without opening as buffer if it fails
- semantic_segmentation can store files with numeric names
- new function: `split_train_test_multilabel` for fairly splitting multilabel datasets
- added requirements
### Fixed
- vis_and_store can handle multilabel, singlelabel and None correctly now
- polygon validation fixed. Set required amount of points from 4 to 3.
- fixed in `validate_unique_annos` pandas drop() future warning
- `crop_dataset` does not produce empty duplicates of crop positions anymore
- `segmentation_to_lost` fix bug where some contours of different classes were merged accidentially
- `mask_dataset` won't overwrite input dataframe anymore
- added a serialization case where a columns has empty lists

## [1.0.0] - 2023-01-10
### Added
- fixed voc_eval bug where gt_df bboxes weren't handled correctly
- fixed coco_eval bug - different dataframes having same uids for images and labels now
- to_coco meethod has arguments predef_img_mapping and predef_lbl_mapping now to hand over predefined label and image uids
- export method crop_img to lost_ds (lost_ds.crop_img and ds.crop_img formerly lost_ds.DSCropper.crop_img)
- resolve FutureWarning at transform_bbox_style
- make crop_dataset compatible with pandas > 1.5.0 
- crop_component indexing is imagewise instead of global
- improve some filter and transform functions
### Breaking changes
- Add parallel option for multiple functions
- fix typo in some methods arguments

## [0.5.0] - 2022-12-02
### Added
- method `split_train_test` which allows the stratify option for i.e. classification datasets
- `coco_eval` method for detection (mAP, Average Recall)
- `voc_eval` method for detection (tp, fp, fn, precision, revall, ap)
- `to_coco` improvements and bugfixes
- `voc_score_iou_multiplex` method: shifting bbox score and iou_thresholds to find optimal thresolds
- cropping method return additional column 'crop_position' now
- added arg 'random_state' for dataset-splitting
- added arg for optional parallelistaion
- added color selection for vis and store - can take column now
- improved detetion metrics

## [0.4.0] - 2022-05-01 
### Changed
- file_man: Allow other Fsspec filesystems
### Added
- Progress callback for pack_ds
- Argument `fontscale` at `vis_and_store` to enable manually control for textsize
- Argument `cast_others` at `segmentation_to_lost` to allow ignoring unspecified pixel values

## [0.3.0] - 2022-04-05 
### Added 
- Added zip support for pack_ds method

## [0.2.0] - 2022-04-04 
### Added 
- `bbox_nms` method to directly apply non-max-suppresseion for bboxes
- Argument `path_col` in methods `to_abs` and `to_rel` to enable explicit column specification to look for image paths
- Argument `mode` for method `remap_img_path` to enable not only replacement but also prepending of root paths
- Set argument default of `inplace` of method `pack_ds` to False
- Enable visualization of annos without `anno_lbl` specified

## [0.1.0] - 2022-02-09
### Changes
- `validate_unique_annos` uses data-hash for dropping duplicates now
- `vis_and_store` text-boxes won't be outside the image anymore
- `vis_and_store` can determine optimal text-size for labeling when passing 
    arg. line_thickness 'auto'.
### Added
- dependencies for lost_ds in requirements.txt + setup.py
- examples dataset and code-snippets
- improved function to load datasets at LOSTDataset
- new function `segmentation_to_lost` to convert pixelmaps to lost-annotations
- new function `crop_components` to crop dataset based on annotations
- new function `vis_semantic_segmentation` to color sem-seg. dataset
- new function `to_coco` to generate and store coco datasets from LOSTDataset

## [0.0.0] - 2021-10-26
### Added
- First version