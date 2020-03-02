import json
import math
import random

with open('dev.json') as f:
    dev = json.load(f)

with open('query_set.csv', 'w') as f:
    with open('match_set.csv', 'w') as ff:
        for image in dev:
            if random.random() > 0.2:
                ff.write(image + '\n')
            else:
                f.write(image + '\n')

