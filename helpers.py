from imageio import imread
from skimage.transform import resize
import numpy as np


def file_name_to_img(filename):
    image = imread("../images_512/" + filename)
    if len(image.shape) == 2:
        image = image.reshape((*image.shape, 1))
    if image.shape[0] == 1:
        image = np.tile(image, (1, 1, 3))

    return image


def get_query_file_names():
    with open("query_set.csv") as f:
        all_query_file_names = f.readlines()
    return [fn for fn in [fn.strip() for fn in all_query_file_names] if fn != ""]


def get_query_file_paths():
    return ["../images_512/" + fn for fn in get_query_file_names()]


def get_match_file_names():
    with open("match_set.csv") as f:
        all_match_file_names = f.readlines()
    return [fn for fn in [fn.strip() for fn in all_match_file_names] if fn != ""]


def get_query_file_paths():
    return ["../images_512/" + fn for fn in get_match_file_names()]


def resize_height(im, new_height):
    height, width, _ = im.shape
    scale = new_height / height

    return resize(im, (round(height * scale), round(width * scale)), anti_aliasing=True)


def resize_width(im, new_width):
    height, width, _ = im.shape
    scale = new_width / width

    return resize(im, (round(height * scale), round(width * scale)), anti_aliasing=True)


def bb_intersection_over_union(boxA, boxB):
    if boxA[0] < 0:
        if boxB[0] < 0:
            return 1
        else:
            return 0

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
