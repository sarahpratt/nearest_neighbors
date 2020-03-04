import json
from collections import defaultdict
import random
from typing import List, Dict, Callable
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rescale, resize
import json

from helpers import get_query_file_names, resize_height, resize_width, file_name_to_img

with open("resnet_query_to_best_dist_match_pairs.json", "r") as f:
    resnet_query_to_best_dist_match_pairs = json.load(f)

with open("swig_query_to_best_dist_match_pairs.json", "r") as f:
    swig_query_to_best_dist_match_pairs = json.load(f)

with open("det_query_to_best_dist_match_pairs.json", "r") as f:
    det_query_to_best_dist_match_pairs = json.load(f)

with open("imsitu_query_to_best_dist_match_pairs.json", "r") as f:
    imsitu_query_to_best_dist_match_pairs = json.load(f)

list_of_query_to_best_dist_match_pairs = [
    swig_query_to_best_dist_match_pairs,
    imsitu_query_to_best_dist_match_pairs,
    det_query_to_best_dist_match_pairs,
    resnet_query_to_best_dist_match_pairs,
]


def interleave(l0, l1):
    return [v for p in zip(l0, l1) for v in p]


query_file_names = get_query_file_names()

res = 1024
for i, query_file_name in enumerate(
    random.sample(query_file_names, len(query_file_names))
):
    print(query_file_name)
    query_image = resize_height(file_name_to_img(query_file_name), res // 5)

    to_cat = []
    for query_to_dist_match_pairs in list_of_query_to_best_dist_match_pairs:
        match_file_names = [x[1] for x in query_to_dist_match_pairs[query_file_name]]

        match_images = [
            resize_height(file_name_to_img(mfn), res // 5) for mfn in match_file_names
        ]

        print(i)
        to_cat.append(
            resize_width(np.concatenate([query_image] + match_images, axis=1), res)
        )

    plt.imshow(
        np.concatenate(
            interleave(to_cat, [np.zeros((10, res, 3), dtype=np.int8)] * (len(to_cat)))[
                :-1
            ],
            axis=0,
        )
    )
    plt.title(query_file_name)
    plt.show()
