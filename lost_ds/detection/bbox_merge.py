import numpy as np

def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / (union_area + 1e-7), np.finfo(np.float32).eps)

    return ious


def bboxes_intersection(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    return (inter_area > 0)


def _bbox_merge(bboxes):
    """Method to merge bboxes instead of nms them
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)
    """
    bboxes = np.array(bboxes)
    
    if bboxes.size == 0:
        return np.array([])
    
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        bbox_mask = np.ones((len(cls_bboxes)))
        
        for box_idx in range(len(cls_bboxes)):
            if not bbox_mask[box_idx]:
                continue
            current_bbox = cls_bboxes[box_idx]
            other_ids = list(range(len(cls_bboxes)))
            other_ids.remove(box_idx)
            # bboxes[drop current box]
            other_bboxes = cls_bboxes[other_ids]
            # other_bboxes[drop alaready pruned boxes]
            # iou = bboxes_iou(current_bbox[None, :4], other_bboxes[list(bbox_mask[other_ids] != 0)][:, :4])
            intersection = bboxes_intersection(current_bbox[None, :4], other_bboxes[list(bbox_mask[other_ids] != 0)][:, :4])
            
            while any(intersection):
                valid_bboxes = bbox_mask[other_ids] != 0
                
                # get overlapping bboxes to combine
                gather_bboxes = other_bboxes[valid_bboxes][intersection]
                combine_bbox = np.concatenate((current_bbox[None], gather_bboxes), 0)
                xmin, xmax, ymin, ymax = combine_bbox[:, [0, 2]].min(), combine_bbox[:, [0, 2]].max(), combine_bbox[:, [1, 3]].min(), combine_bbox[:, [1, 3]].max()
                score = combine_bbox[:, 4].max()
                current_bbox = np.array([xmin, ymin, xmax, ymax, score, cls])
                cls_bboxes[box_idx] = current_bbox
                
                # remember pruned bboxes
                combined_ids = np.array(other_ids)[valid_bboxes][intersection]
                bbox_mask[combined_ids] = 0
                
                # find iou for next iteration
                remaining_bboxes = other_bboxes[valid_bboxes][~intersection] # iou < iou_threshold]
                # iou = bboxes_iou(current_bbox[None, :4], remaining_bboxes[:, :4])
                intersection = bboxes_intersection(current_bbox[None, :4], remaining_bboxes[:, :4])
        
        best_bboxes.append(cls_bboxes[bbox_mask != 0])

    return np.array(best_bboxes).reshape((-1, 6))


def bbox_merge(boxes_list, scores_list, labels_list, iou_thr, **kwargs):
    for bboxes, scores, labels in zip(boxes_list, scores_list, labels_list):
        bboxes = np.array(bboxes).reshape((-1, 4))
        scores = np.array(scores_list).reshape((-1, 1))
        labels = np.array(labels_list).reshape((-1, 1))
        merged = _bbox_merge(np.hstack((bboxes, scores, labels)))
        return merged[:, :4], merged[:, 4], merged[:, 5]