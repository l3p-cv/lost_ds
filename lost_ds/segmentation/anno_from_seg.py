
def _segmentation_to_polygon(self, segmentation, background=0, format='lost'):
    if len(segmentation.shape)==3:
        segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY)

    im_h, im_w = segmentation.shape
    
    px_values = []
    lost_polygons = []
    
    colors = list(np.unique(segmentation))
    if background in colors:
        colors.remove(background)
        
    for color in colors:
        
        class_seg = np.zeros_like(segmentation)
        class_seg[segmentation==color] = 255
        contours, _ = cv2.findContours(class_seg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour = contour.squeeze()
            contour = np.array(contour) / np.array([im_w, im_h])
            if len(contour) < 3:
                continue
            lost_polygons.append([{'x': p[0], 'y': p[1]} for p in list(contour)])
            px_values.append(int(color))
    
    return lost_polygons, px_values


def segmentation_to_lost(self, background=0, df=None):
    """
    Returns: pd.DataFrame with lost-annos
    """
    if df is None:
        df = self.df
        
    lost_anno_data = {'anno.lbl.name': [],
                    'anno.data': [],
                    'img.img_path': [],
                    'mask.path': [],
                    'anno.dtype': []}
    
    segmentations = df['mask.path'].unique()
    def seg_to_poly(i, seg_path):
    # for i, seg_path in enumerate(segmentations):
        print_progress_bar(i+1, len(segmentations), prefix='seg-to-lost: ')
        lost_anno = lost_anno_data.copy()
        segmentation = cv2.imread(seg_path, 0)
        lost_polys, px_values = self._segmentation_to_polygon(segmentation, format='lost')
        assert len(lost_polys) == len(px_values)
        lost_anno['anno.lbl.name'] += px_values
        lost_anno['anno.data'] += lost_polys
        lost_anno['img.img_path'] += [df[df['mask.path']==seg_path]['img.img_path'].iloc[0]] * len(px_values)
        lost_anno['mask.path'] += [seg_path] * len(px_values)
        lost_anno['anno.dtype'] += ['polygon'] * len(px_values)
        return json.dumps(lost_anno)
    
    ret = Parallel(n_jobs=-1)(delayed(seg_to_poly)(i, seg_path)for i, seg_path in enumerate(segmentations))
    
    lost_anno = lost_anno_data.copy()
    for r in ret:
        entry = json.loads(r)
        for k in lost_anno.keys():
            lost_anno[k] += entry[k]
    
    lost_df = pd.DataFrame(lost_anno)
    
    return lost_df


