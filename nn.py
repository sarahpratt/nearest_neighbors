import json
from collections import defaultdict
from typing import List, Dict, Callable
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rescale, resize


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


if __name__ == "__main__":
    print("Reading predictions csv into dict.")
    with open("dev.json") as f:
        dev = json.load(f)

    all_ims = defaultdict(list)

    seen_img_verb_pairs = set()
    with open("prediction.csv") as f:
        i = 1
        words = []
        for line in f:
            line = line.split("\n")[0]
            line = line.split(",")
            # if int(line[-1]) != -1 and line[2] != '':
            h = dev[line[0]]["height"]
            w = dev[line[0]]["width"]

            box = [
                float(line[3]) / float(w),
                float(line[4]) / float(h),
                float(line[5]) / float(w),
                float(line[6]) / float(h),
            ]
            # box = [float(line[3]), float(line[4]), float(line[5]), float(line[6])]

            center_x = (int(line[5]) + int(line[3])) / (2 * float(w))
            center_y = (int(line[6]) + int(line[4])) / (2 * float(w))
            # print(line[1])
            words.append(
                {
                    "verb": line[1],
                    "word": line[2],
                    "center_x": center_x,
                    "center_y": center_y,
                    "box": box,
                }
            )

            if i % 6 == 0:
                assert words[0]["verb"] == line[0]
                already_in = False

                img_verb_pair = (line[0], words[0]["verb"])
                if img_verb_pair not in seen_img_verb_pairs:
                    all_ims[line[0]].append(words)
                    seen_img_verb_pairs.add(img_verb_pair)
                else:
                    print("Repeat {}".format(img_verb_pair))

                words = []

            i += 1


def noun_conditional_squared_iou(query_words, match_words, ious):
    return sum(
        (qw == mw) * iou ** 2 for qw, mw, iou in zip(query_words, match_words, ious)
    ) / len(query_words)


def noun_conditional_marginal_iou(query_words, match_words, ious):
    return sum(
        (qw == mw) * (1 + iou) for qw, mw, iou in zip(query_words, match_words, ious)
    ) / len(query_words)


def noun_conditional_ignore_iou(query_words, match_words, ious):
    return sum((qw == mw) for qw, mw in zip(query_words, match_words)) / len(
        query_words
    )


def create_verb_conditional_scorer(frame_scorer: Callable):
    def verb_conditional_scorer(query_info: List[Dict], match_info: List[Dict]):
        query_len = len(query_info)
        match_len = len(match_info)
        max_score = 0
        for query_frame_ind, query_frame in enumerate(query_info):
            query_multiplier = 1 - query_frame_ind / query_len
            # if max_score >= query_multiplier:
            #     break

            for match_frame_ind, match_frame in enumerate(match_info):
                full_multiplier = query_multiplier * (1 - match_frame_ind / match_len)

                # if max_score >= full_multiplier:
                #     break

                if query_frame[0]["verb"] != match_frame[0]["verb"]:
                    continue

                query_words = [
                    frame["word"] for frame in query_frame if frame["word"] != "Pad"
                ]
                match_words = [
                    frame["word"] for frame in match_frame if frame["word"] != "Pad"
                ]
                query_boxes = (
                    frame["box"] for frame in query_frame if frame["word"] != "Pad"
                )
                match_boxes = (
                    frame["box"] for frame in match_frame if frame["word"] != "Pad"
                )

                ious = (
                    bb_intersection_over_union(qb, mb)
                    for qb, mb in zip(query_boxes, match_boxes)
                )

                # scores[-1] += multiplier * sum(
                #     (qw == mw) * (1 + iou) + iou for qw, mw, iou in zip(query_words, match_words, ious)
                # ) / len(query_words)

                # scores[-1] += multiplier * sum(
                #     (1 + (qw == mw)) * iou for qw, mw, iou in zip(query_words, match_words, ious)
                # ) / len(query_words)

                max_score = max(
                    max_score,
                    full_multiplier * frame_scorer(query_words, match_words, ious),
                )
        return max_score

    return verb_conditional_scorer


with open("match_set.csv") as f:
    all_match_file_names = f.readlines()
all_match_file_names = [
    fn for fn in [fn.strip() for fn in all_match_file_names] if fn != ""
]

with open("query_set.csv") as f:
    all_query_file_names = f.readlines()
all_query_file_names = [
    fn for fn in [fn.strip() for fn in all_query_file_names] if fn != ""
]

scoring_func = create_verb_conditional_scorer(frame_scorer=noun_conditional_marginal_iou)
# scoring_func = create_verb_conditional_scorer(frame_scorer=noun_conditional_ignore_iou)
query_fn_to_top_five_matches = {}
max_to_find = 20
for i, query_file_name in enumerate(all_query_file_names):
    print(i, query_file_name)
    top_list_for_query = []
    query_info = all_ims[query_file_name]

    if query_file_name in query_fn_to_top_five_matches:
        continue

    for match_file_name in all_match_file_names:
        match_info = all_ims[match_file_name]

        top_list_for_query.append(
            (scoring_func(query_info, match_info), match_file_name)
        )

    top_list_for_query.sort(reverse=True)
    query_fn_to_top_five_matches[query_file_name] = top_list_for_query[:5]

    if len(query_fn_to_top_five_matches) == max_to_find:
        break

best_score_query_fn_list = [
    (matches[0][0], qfn) for qfn, matches in query_fn_to_top_five_matches.items()
]
best_score_query_fn_list.sort(reverse=True)
sorted_query_file_names = [x[1] for x in best_score_query_fn_list]


def resize_height(im, new_height):
    height, width, _ = im.shape
    scale = new_height / height

    return resize(im, (round(height * scale), round(width * scale)), anti_aliasing=True)


def resize_width(im, new_width):
    height, width, _ = im.shape
    scale = new_width / width

    return resize(im, (round(height * scale), round(width * scale)), anti_aliasing=True)


# for query_file_name, matches in query_fn_to_top_five_matches.items():
#     query_image = resize_height_to_512(io.imread("../images_512/" + query_file_name))
#     match_images = [resize_height_to_512(io.imread("../images_512/" + mfn)) for _, mfn in matches]
#
#     plt.imshow(np.concatenate([query_image] + match_images, axis=1))
#     plt.show()

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
