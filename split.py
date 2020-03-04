import json
import random
from collections import Counter, defaultdict
from typing import Dict, Any

with open("dev.json") as f:
    dev: Dict[str, Any] = json.load(f)

verb_counts = Counter([key.split("_")[0].strip().lower() for key in dev])
assert all(count == 50 for count in verb_counts.values())

file_names = list(dev.keys())
random.seed(1)
random.shuffle(file_names)

total_queries_per_verb = 2
query_verb_to_fns = defaultdict(lambda: [])
match_verb_to_fns = defaultdict(lambda: [])

for fn in dev.keys():
    verb = fn.split("_")[0].strip().lower()
    if len(query_verb_to_fns[verb]) == total_queries_per_verb:
        match_verb_to_fns[verb].append(fn)
    else:
        query_verb_to_fns[verb].append(fn)

query_fn_set = set(sum(query_verb_to_fns.values(), []))
match_fn_set = set(sum(match_verb_to_fns.values(), []))

with open("query_set.csv", "w") as f:
    with open("match_set.csv", "w") as ff:
        for fn in sorted(list(dev.keys())):
            if fn in query_fn_set:
                f.write(fn + "\n")
            else:
                ff.write(fn + "\n")


# with open('query_set.csv', 'w') as f:
#     with open('match_set.csv', 'w') as ff:
#         for image in dev:
#             if random.random() > 0.2:
#                 ff.write(image + '\n')
#             else:
#                 f.write(image + '\n')
#
