import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformation on input or loss calculation
def get_coords(batch):
    graph = batch[3]
    source_coord = graph.pos[batch[0]]
    target_coord = graph.pos[batch[1]]
    return source_coord, target_coord

def normalize_coord(coord):
    # norway terrain spans a 20km box, use average hieght as 1700m
    coord = (coord - torch.FloatTensor([10, 10, 1.7])) / torch.FloatTensor([10, 10, 1])
    return coord

