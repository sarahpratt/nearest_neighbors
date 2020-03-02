import json
from collections import defaultdict
import pdb


def bb_intersection_over_union(boxA, boxB):
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

with open('dev.json') as f:
    dev = json.load(f)



all_ims = defaultdict(list)

with open('prediction.csv') as f:
    i =  0
    words = []
    for line in f:
        line = line.split('\n')[0]
        line = line.split(',')
        #if int(line[-1]) != -1 and line[2] != '':
        h = dev[line[0]]['height']
        w = dev[line[0]]['width']

        box = [float(line[3])/float(w), float(line[4])/float(h), float(line[5])/float(w), float(line[6])/float(h)]
        #box = [float(line[3]), float(line[4]), float(line[5]), float(line[6])]

        center_x = (int(line[5]) + int(line[3]))/(2*float(w))
        center_y = (int(line[6]) + int(line[4]))/(2*float(w))
        #print(line[1])
        words.append({'verb': line[1], 'word': line[2], 'center_x': center_x, 'center_y': center_y, 'box': box})

        if i%6 == 0:
            all_ims[line[0]].append(words)
            words = []

        i += 1



print('querying')

query_image = 'milking_157.jpg'
print(query_image)
print()
with open('match_set.csv') as f:
    best_value = 0
    best_image = ''
    for line in f:
        image = line.split('\n')[0]

        annot = all_ims[query_image][0]
        match_annot = all_ims[image][0]
        #
        # if all_ims[query_image][0][0]['verb'] == all_ims[image][0][0]['verb']:
        #     print(image)

        total = 0
        for item in annot:
            word = item['word']
            best = 0
            for match_item in match_annot:
                if word == match_item['word']:
                    curr = 1 + bb_intersection_over_union(match_item['box'], item['box'])
                    #curr = (1- abs((match_item['center_x'] - match_item['center_x']))) * (1- abs((match_item['center_y'] - match_item['center_y'])))
                    #curr = 1
                    if curr > best:
                        best = curr
            total += best


            if total > best_value:
                best_value = total
                best_image = image
                #print(best_image)

print(best_image)



