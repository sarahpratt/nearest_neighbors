import json
from collections import defaultdict
from typing import List, Dict, Callable

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from helpers import (
    resize_height,
    resize_width,
    bb_intersection_over_union,
    get_query_file_names,
    get_match_file_names,
)

if __name__ == "__main__":
    print("Reading only_dets.csv csv into dict.")
    with open("dev.json") as f:
        dev = json.load(f)

    file_name_to_detected_objects_info = defaultdict(
        lambda: {"contains_set": set(), "objects": []}
    )

    seen_img_verb_pairs = set()
    with open("only_dets.csv") as f:
        current_fn = None
        objects = []
        for line in f:
            line = line.strip()
            line_split = line.split(",")
            file_name, object, a, b, c, d = line_split

            if current_fn != file_name:
                if objects is not None:
                    if current_fn in file_name_to_detected_objects_info:
                        print(
                            (
                                "WARNING: {} has already been added to"
                                " file_name_to_detected_objects. Skipping..."
                            ).format(current_fn)
                        )
                    else:
                        file_name_to_detected_objects_info[current_fn] = {
                            "contains_set": {o["object"] for o in objects},
                            "objects": objects,
                        }
                current_fn = file_name
                objects = []

            if a == "":
                continue

            h = dev[file_name]["height"]
            w = dev[file_name]["width"]

            box = [
                float(a) / float(w),
                float(b) / float(h),
                float(c) / float(w),
                float(d) / float(h),
            ]

            center_x = (int(c) + int(a)) / (2 * float(w))
            center_y = (int(d) + int(b)) / (2 * float(w))
            objects.append(
                {
                    "object": object,
                    "center_x": center_x,
                    "center_y": center_y,
                    "box": box,
                }
            )


def detection_scorer(query_objects_info: Dict, match_objects_info: Dict):
    query_contains_set = query_objects_info["contains_set"]
    match_contains_set = match_objects_info["contains_set"]

    if len(query_contains_set & match_contains_set) == 0:
        if len(query_contains_set) == len(match_contains_set):
            return 1
        else:
            return 0

    query_objects = query_objects_info["objects"]
    match_objects = match_objects_info["objects"]

    score = 0
    for query_object_ind, query_object in enumerate(query_objects):
        subscore = 0
        for match_object_ind, match_object in enumerate(match_objects):
            if query_object["object"] != match_object["object"]:
                continue

            subscore = max(
                subscore,
                1
                + bb_intersection_over_union(query_object["box"], match_object["box"]),
            )

        score += subscore
    return score / len(query_objects)


all_match_file_names = get_match_file_names()
all_query_file_names = get_query_file_names()

scoring_func = detection_scorer

query_fn_to_top_five_matches = {}
max_to_find = 2000
for i, query_file_name in enumerate(all_query_file_names):
    print(i, query_file_name)
    top_list_for_query = []
    query_info = file_name_to_detected_objects_info[query_file_name]

    if query_file_name in query_fn_to_top_five_matches:
        continue

    for match_file_name in all_match_file_names:
        match_info = file_name_to_detected_objects_info[match_file_name]

        top_list_for_query.append(
            (scoring_func(query_info, match_info), match_file_name)
        )

    top_list_for_query.sort(reverse=True)
    query_fn_to_top_five_matches[query_file_name] = top_list_for_query[:5]

    if len(query_fn_to_top_five_matches) == max_to_find:
        break

with open("det_query_to_best_dist_match_pairs.json", "w") as f:
    json.dump(query_fn_to_top_five_matches, f)

best_score_query_fn_list = [
    (matches[0][0], qfn) for qfn, matches in query_fn_to_top_five_matches.items()
]
best_score_query_fn_list.sort(reverse=True)
sorted_query_file_names = [x[1] for x in best_score_query_fn_list]

to_cat = []
res = 1024
# for i, (query_file_name, matches) in enumerate(
#     (qfn, query_fn_to_top_five_matches[qfn]) for qfn in sorted_query_file_names
# ):
for i, (query_file_name, matches) in enumerate(
    (qfn, query_fn_to_top_five_matches[qfn]) for qfn in sorted(sorted_query_file_names)
):
    print(i)
    query_image = resize_height(io.imread("../images_512/" + query_file_name), res // 5)
    match_images = [
        resize_height(io.imread("../images_512/" + mfn), res // 5) for _, mfn in matches
    ]

    print(i)
    to_cat.append(
        resize_width(np.concatenate([query_image] + match_images, axis=1), res)
    )

    if (i + 1) % 10 == 0:
        plt.imshow(np.concatenate(to_cat, axis=0))
        plt.show()
        to_cat = []
