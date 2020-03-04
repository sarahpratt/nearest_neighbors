import copy
import json
import os

import torch
from PIL import Image
from imageio import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
import numpy as np


class ScaleBothSides(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of both edges, and this can change aspect ratio.
    size: output size of both edges
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)


def resnet_input_transform(input_image, im_size):
    """Takes in numpy ndarray of size (H, W, 3) and transforms into tensor for
       resnet input.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    all_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            ScaleBothSides(im_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transformed_image = all_transforms(input_image)
    return transformed_image


class DatasetFromCSV(Dataset):
    def __init__(self, csv_path):
        with open(csv_path, "r") as f:
            self.file_paths = ["../images_512/" + fn.strip() for fn in f.readlines()]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        image = imread(file_path)
        if len(image.shape) == 2:
            image = image.reshape((*image.shape, 1))
        if image.shape[-1] == 1:
            image = np.tile(image, (1, 1, 3))
        return resnet_input_transform(image, 224)


match_csv_path = "match_set.csv"
query_csv_path = "query_set.csv"

match_dataset = DatasetFromCSV(match_csv_path)
query_dataset = DatasetFromCSV(query_csv_path)

match_dataloader = DataLoader(
    dataset=match_dataset, batch_size=16, num_workers=4, shuffle=False, drop_last=False
)

model = resnet50(pretrained=True)

if torch.cuda.is_available():

    def to_device(input):
        return input.to(0)


else:

    def to_device(input):
        return input.cpu()


model = to_device(model)
model.eval()

results = []
for i, batch in enumerate(match_dataloader):
    print(i)
    with torch.no_grad():
        results.append(model(to_device(batch)).cpu())

all_match_results = torch.cat(results, dim=0)

query_dataloader = DataLoader(
    dataset=query_dataset, batch_size=16, num_workers=4, shuffle=False
)
results = []
for i, batch in enumerate(query_dataloader):
    print(i)
    with torch.no_grad():
        results.append(model(to_device(batch)).cpu())

all_query_results = torch.cat(results, dim=0)

a_sq = (all_query_results ** 2).sum(-1)
b_sq = (all_match_results ** 2).sum(-1)
ab = torch.matmul(all_query_results, all_match_results.permute(1, 0))

query_match_distances = torch.sqrt(a_sq.view(-1, 1) + b_sq.view(1, -1) - 2 * ab)

query_file_paths = copy.copy(query_dataset.file_paths)
match_file_paths = copy.copy(match_dataset.file_paths)


def to_file_name(path):
    return os.path.split(path)[-1]


query_to_best_dist_match_pairs = {}
for i, qfp in enumerate(query_file_paths):
    print(i)
    query_file_name = to_file_name(qfp)
    dists_to_matches = query_match_distances[i, :]

    match_inds_sorted = torch.argsort(dists_to_matches)

    query_to_best_dist_match_pairs[query_file_name] = [
        (float(dists_to_matches[ind].item()), to_file_name(match_file_paths[ind]))
        for ind in match_inds_sorted[:5]
    ]

with open("resnet_query_to_best_dist_match_pairs.json", "w") as f:
    json.dump(query_to_best_dist_match_pairs, f)
