import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.utils import to_undirected, to_networkx

from .point_sampler import sample_queries


def load_terrain_grid(res=200, antialias="nearest"):
    with open("/data/sam/terrain/norway-smallest.txt", 'r') as f:
        n, m = map(int, f.readline().split())
        terrain = np.loadtxt(f, dtype=np.float32)

    terrain = torch.from_numpy(terrain)
    terrain = F.interpolate(terrain[None, None, :, :], (res,res), mode=antialias).squeeze()

    downsample_rate = 2000 / res
    coord = torch.arange(res) * 10 * downsample_rate
    grid = torch.meshgrid(coord, coord)
    return torch.stack(grid + (terrain, )).permute(1,2,0)



def grid_to_mesh(terrain, add_diag=True):
    N = terrain.shape[0]
    vertices = torch.arange(N * N).reshape((N, N))
    vertices_x = terrain.reshape((-1, 3))

    vertical_edge = torch.stack((vertices[:, :-1], vertices[:, 1:]),dim=-1).reshape((-1, 2))
    horizontal_edge = torch.stack((vertices[:-1], vertices[1:]),dim=-1).reshape((-1, 2))
    edge_index = torch.cat((vertical_edge, horizontal_edge))

    vertical_edge_attr = (terrain[:, :-1] - terrain[:, 1:]).norm(dim=-1).flatten()
    horizontal_edge_attr = (terrain[:-1] - terrain[1:]).norm(dim=-1).flatten()
    edge_attr = torch.cat((vertical_edge_attr, horizontal_edge_attr))
    
    if add_diag:
        diag_edge_1 = torch.stack((vertices[:-1, :-1], vertices[1:, 1:]),dim=-1).reshape((-1, 2))
        diag_edge_2 = torch.stack((vertices[:-1, 1:], vertices[1:, :-1]),dim=-1).reshape((-1, 2))
        diag_edge_attr_1 = (terrain[:-1, :-1] - terrain[1:, 1:]).norm(dim=-1).flatten()
        diag_edge_attr_2 = (terrain[:-1, 1:] - terrain[1:, :-1]).norm(dim=-1).flatten()
        edge_index = torch.cat((edge_index, diag_edge_1, diag_edge_2))
        edge_attr = torch.cat((edge_attr, diag_edge_attr_1, diag_edge_attr_2))

    mesh = Data(pos=vertices_x, edge_index=edge_index.T, weight=edge_attr)
    mesh.edge_index, mesh.weight = to_undirected(mesh.edge_index, mesh.weight, N * N)
    return mesh


class ShortestPathDataset(Dataset):
    def __init__(self, path, terrain, samples, n_sources, add_diag=True, method = "random"):
        if not os.path.exists(path):
            self.construct_dataset(path, terrain, samples, n_sources, add_diag=True, method = method)
        else:
            # load dataset
            dataset_dict = torch.load(path, weights_only=False)
            self.sources = dataset_dict["source"]
            self.targets = dataset_dict["target"]
            self.distances = dataset_dict["distance"]
            self.mesh = dataset_dict["mesh"]
            self.nx_mesh = to_networkx(self.mesh, edge_attrs=['weight'], to_undirected=True)
            self.n_samples = len(self.sources)      
    
    def construct_dataset(self, path, terrain, samples, n_sources, add_diag=True, method = "random"):
        H, W = terrain.shape[:2]
        self.n_samples = samples
        self.mesh = grid_to_mesh(terrain, add_diag=add_diag)
        self.nx_mesh = to_networkx(self.mesh, edge_attrs=['weight'], to_undirected=True)

        # precompute samples and shortest path
        samples_per_source = samples // n_sources
        self.sources = []
        self.targets = []
        self.distances = []
        for i in tqdm(range(n_sources)):
            source, target, distance = sample_queries(method, self.nx_mesh, samples_per_source)
            self.sources += source
            self.targets += target
            self.distances += distance
        self.distances = torch.FloatTensor(self.distances)
        # save dataset
        dataset_dict = {
            "source": self.sources,
            "target": self.targets,
            "distance": self.distances,
            "mesh": self.mesh
        }
        torch.save(dataset_dict, path, )

    def __len__(self,):
        return self.n_samples
    
    def __getitem__(self, index):
        src, tar, dist = self.sources[index], self.targets[index], self.distances[index]
        return self.mesh.pos[src], self.mesh.pos[tar], dist
