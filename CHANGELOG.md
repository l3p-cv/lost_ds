# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased] 
### Added 
- `bbox_nms` method to directly apply non-max-suppresseion for bboxes
- Argument `path_col` in methods `to_abs` and `to_rel` to enable explicit column specification to look for image paths
- Argument `mode` for method `remap_img_path` to enable not only replacement but also prepending of root paths
- Set argument default of `inplace` of method `pack_ds` to False
- Enable visualization of annos without `anno_lbl` specified
- Argument `fontscale` at `vis_and_store` to enable manually control for textsize

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