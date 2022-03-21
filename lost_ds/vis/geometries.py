import cv2
import numpy as np
import cv2

    
#
#   draw text
#

H_MAX_FRAC = 0.05
W_MAX_FRAC = 0.3

def calculate_text(text, thickness, img_h):
    # calculate fontscale when text is 5% of img-height
    fontscale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, 
                                           max(int(H_MAX_FRAC * img_h), 10), 
                                           thickness)
    
    (text_w, text_h), baseline = cv2.getTextSize(text,
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                fontscale, 
                                                thickness)
    return text_w, text_h, baseline, fontscale



def get_text_size(text, max_width, max_height, thickness):
    text_w, text_h, baseline, fontscale = calculate_text(text, thickness, 
                                                         max_height/H_MAX_FRAC)
    if text_w > max_width:
       dst_w_frac = max_width / text_w
       dst_h = text_h * dst_w_frac
       text_w, text_h, baseline, fontscale = calculate_text(text, thickness, 
                                                            dst_h/H_MAX_FRAC)
    
    return text_w, text_h, baseline, fontscale



def draw_text(img, text, x, y, color, thickness=2):
    '''Draw text onto an image
    Args:
        img (numpy.ndarray): Image to draw on
        text (str): Text to draw
        x (int): x-coordinate of geometries point where to draw the text
        y (int): y-coordinate of geometries point where to draw the text
        color (tuple): BGR color to use
        thickness (int): Text and line thickness
    Returns:
        np.ndarray Image with text drawn and framed
    Note:
        The Text will be white (255,255,255), on black background, framed by
        a rectangle having the specified color.
    '''
    if text is None:
        return img
    
    text_w, text_h, baseline, fontscale = get_text_size(text,
                                                        img.shape[1]*W_MAX_FRAC, 
                                                        img.shape[0]*H_MAX_FRAC, 
                                                        thickness)
    
    ymin = y - text_h - baseline - thickness
    if ymin < 0:
        y += abs(ymin)
        ymin = thickness
    x -= thickness
    xmax = x + text_w
    if xmax > img.shape[1]:
        x -= xmax - img.shape[1] - thickness
        xmax = img.shape[1] - thickness
        
    # black filled rectangle
    cv2.rectangle(img, (x, ymin), (xmax, y), (0,0,0),
                  thickness=cv2.FILLED)
    # colored rectangle frame to get a colored frame to the black rect
    cv2.rectangle(img, (x-thickness, ymin-thickness),
                  (xmax + thickness, y + thickness),
                  color, thickness=thickness)
    # white text on black rectangle
    cv2.putText(img, text, (x, y-baseline//2), cv2.FONT_HERSHEY_SIMPLEX, 
                fontscale, (255, 255, 255), thickness)
    
    return img



#
#   Draw geometries
#


def get_data(data, text, color):
    if color is None:
        color = (255, 255, 255)    
    if not isinstance(data, list):
        data = [data.astype(np.int32)]
    if not isinstance(text, list):
        text = [text]
    if not isinstance(color, list):
        color = [color] * len(data)
    
    assert len(data) == len(text) == len(color)
    
    return data, text, color


def draw_points(img, data, text, color=(0, 0, 255), radius=2, 
                line_thickness=-1):
    '''Draw points onto the given image

    Args:
        img (str or array): Path to an image file or an image as numpy rgb array.
        data (np.ndarray, list of np.ndarray): points in format [x, y] 
            Values have to be absolute.
        text (str, list of str): text to draw onto the annotato
        color (tuple, list of tuple): BGR color to use
        line_thickness (int): Thickness of Point

    Returns:
        np.array: Image in opencv style
    '''
    data, text, color = get_data(data, text, color)
    
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    for point, txt, c in zip(data, text, color):
        cv2.circle(img, tuple(point), radius, c, line_thickness)
        img = draw_text(img, txt, *point-[0, radius], c, max(line_thickness, 1))
    return img


def draw_polygons(img, data, text, color=(0, 0, 255), line_thickness=2, offset=None):
    '''Draw polygons onto the given image

    Args:
        img (np.ndarray): image as numpy array.
        data (np.ndarray, list of np.ndarray): polygons format 
            [[x,y],[x,y],[x,y],...]. Values have to be absolute.
        text (string, list of strings): text to draw onto the annotation
        conf_list (list): List of confidences for each annotation.
        color (tuple, list of tuple): BGR colors to use
        line_thickness (int): Thickness of the polygon lines.

    Returns:
        np.array: Image in opencv style
    '''
    data, text, color = get_data(data, text, color)
    
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    for poly, txt, c in zip(data, text, color):
        if offset is not None and isinstance(offset, (list, tuple, np.ndarray)):
            poly = poly + offset
        y_values = poly[:, 1]
        txt_y_i = np.argmin(y_values)
        txt_x = poly[txt_y_i][0]
        txt_y = y_values.min()
        cv2.polylines(img, [poly], True, c, thickness=line_thickness)
        img = draw_text(img, txt, txt_x, txt_y, c, line_thickness)
    return img


def draw_boxes(img, data, text, color=(0, 0, 255), line_thickness=1):
    '''Draw bboxes onto the given image

    Args:
        img (np.ndarray): image as numpy array
        data (np.ndarray, list of np.ndarray): bboxes in format, 
            [xmin, ymin, xmax, ymax]. Values have to be absolute.
        text (str, list of str): text to draw onto the annotation, e.g. label 
            of confidence 
        color (tuple, list of tuples): BGR color that is used for all boxes.
        line_thickness (int): Thickness of the bounding box lines.
        
    Returns:
        np.array: Image with bboxes drawn
    '''
    
    data, text, color = get_data(data, text, color)
    
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for bb, txt, c in zip(data, text, color):
        cv2.rectangle(img, (bb[0],bb[1]), (bb[2],bb[3]), c, line_thickness)
        img = draw_text(img, txt, bb[0], bb[1], c, line_thickness)
    return img


def draw_lines(img, data, text, color=(0, 0, 255), line_thickness=2):
    '''Draw poly lines onto the given image

    Args:
        img (np.ndarray): image as numpy array.
        data (np.ndarray, list of np.ndarray): lines format 
            [[x,y],[x,y],[x,y],...]. Values have to be absolute.
        text (string, list of strings): text to draw onto the annotation
        conf_list (list): List of confidences for each annotation.
        color (tuple, list of tuple): BGR colors to use
        line_thickness (int): Thickness of the lines.

    Returns:
        np.array: Image in opencv style
    '''
    data, text, color = get_data(data, text, color)
    
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    for poly, txt, c in zip(data, text, color):
        y_values = poly[:, 1]
        txt_y_i = np.argmin(y_values)
        txt_x = poly[txt_y_i][0]
        txt_y = y_values.min()
        cv2.polylines(img, [poly], False, c, line_thickness)
        img = draw_text(img, txt, txt_x, txt_y, c, line_thickness)
    return img
    