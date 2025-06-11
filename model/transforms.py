# %%
import math
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

def get_positional_embedding_1d(length, dim):
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)  # (length, 1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
    pe = torch.zeros((length, dim))
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe  # (length, dim)

def get_positional_embedding_2d(embedding_res, dimension):
    emb_h = get_positional_embedding_1d(embedding_res, dimension // 2)
    emb_w = get_positional_embedding_1d(embedding_res, dimension // 2)
    # Combine them to form 2D embeddings
    emb_h = emb_h[:, None, :].expand(-1, embedding_res, -1)
    emb_w = emb_w[None, :, :].expand(embedding_res, -1, -1)
    pos_embed = torch.cat([emb_h, emb_w], dim=-1)  # (res, res, C)
    return pos_embed
    #pos_embed = pos_embed.permute(1, 2, 0).unsqueeze(0)  # to fit the input of F.interpolate
    #pos_embed = F.interpolate(pos_embed, (output_res, output_res), mode="bilinear")

class PositionEmbeddingQuery():
    def __init__(self, box_span, embedding_res, dimension):
        self.box_span = box_span
        self.embedding_res = embedding_res
        self.step = self.box_span / (self.embedding_res - 1)
        self.embedding = get_positional_embedding_2d(embedding_res, dimension).permute(1,2,0)
    
    def query(self, coord):
        # expect unnormalized coord
        coord = coord[:, :2] # only lon and lat
        bins = (coord // self.step).long()
        offset = coord % self.step
        queried_emb = (self.embedding[:, bins[0], bins[1]] * (1 - offset[0]) * (1 - offset[1]) + 
                       self.embedding[:, bins[0] + 1, bins[1]] * offset[0] * (1 - offset[1]) + 
                       self.embedding[:, bins[0], bins[1] + 1] * (1 - offset[0]) * offset[1] + 
                       self.embedding[:, bins[0] + 1, bins[1] + 1] * offset[0] * offset[1]
        ) / self.embedding_res ** 2
        return queried_emb


# %%
