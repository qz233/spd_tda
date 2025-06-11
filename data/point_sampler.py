# The core part of the experiment: to find a better sampling method
# for each function defined here: take a networkx graph and number of samples
# return single source shortest distance to sampled points.
import numpy as np
import torch
import networkx as nx
from ripser import ripser, lower_star_img
from persim import plot_diagrams

def random_sampling(graph: nx.Graph, samples_per_source, src=None):
    if src is None:
        src = np.random.randint(0, graph.number_of_nodes())
    distance = nx.single_source_dijkstra_path_length(graph, src)
    # random
    target = np.random.choice(graph.number_of_nodes(), (samples_per_source, ), replace=False)
    distance = [distance[t] for t in target]
    return [src] * samples_per_source, target.tolist(), distance


def distance_based(graph: nx.Graph, samples_per_source, src=None):
    if src is None:
        src = np.random.randint(0, graph.number_of_nodes())
    distance = nx.single_source_dijkstra_path_length(graph, src)
    # random
    vertices = np.arange(graph.number_of_nodes())
    row_num = int(np.around(graph.number_of_nodes() ** 0.5))
    hops = abs(src // row_num - vertices // row_num) + abs(src % row_num - vertices % row_num) + 1
    probs = 1 / hops
    probs = probs / probs.sum()
    target = np.random.choice(vertices, (samples_per_source, ), p=probs, replace=False)
    distance = [distance[t] for t in target]
    return [src] * samples_per_source, target.tolist(), distance


def find_critical_points(terrain, threshold):
    # the original terrain has same-height points we must break the tie
    N = terrain.shape[0]
    terrain[:, :, 2] += torch.rand((N, N)) * 1e-5
    lower_dgm = lower_star_img(terrain[:, :, 2])
    upper_dgm = - lower_star_img(- terrain[:, :, 2])
    long_pers_lower_dgm = lower_dgm[lower_dgm[:, 1]- lower_dgm[:, 0] > threshold]
    long_pers_upper_dgm = upper_dgm[upper_dgm[:, 0]- upper_dgm[:, 1] > threshold]
    long_pers_dgm = np.concatenate([long_pers_lower_dgm, long_pers_upper_dgm])
    print(f"{long_pers_dgm.shape[0]} significant critical point pairs")

    flatten_terrain = terrain.flatten(0, 1)
    critical_idx_0 = [np.argmin(abs(flatten_terrain[:, 2] - long_pers_lower_dgm[i, 0])) for i in range(long_pers_lower_dgm.shape[0])]
    critical_idx_2 = [np.argmin(abs(flatten_terrain[:, 2] - long_pers_upper_dgm[i, 0])) for i in range(long_pers_upper_dgm.shape[0])]
    critical_idx_1 = [np.argmin(abs(flatten_terrain[:, 2] - long_pers_lower_dgm[i, 1])) for i in range(long_pers_lower_dgm.shape[0])] + \
                    [np.argmin(abs(flatten_terrain[:, 2] - long_pers_upper_dgm[i, 1])) for i in range(long_pers_upper_dgm.shape[0])]
    critical_idx_1 = list(set(critical_idx_1))

    critical_idx = torch.stack(critical_idx_0 + critical_idx_1 + critical_idx_2)
    # shuffle it
    critical_idx = critical_idx[torch.randperm(critical_idx.shape[0])]
    return critical_idx



class QuerySampler():
    def __init__(self, method, terrain=None):
        self.method = method
        self.counter = 0
        if "critical" in self.method:
            self.critical_points = find_critical_points(terrain, 0.08)

    def sample(self, graph, samples_per_source):
        if self.method == "random":
            return random_sampling(graph, samples_per_source)
        elif self.method == "distance":
            return distance_based(graph, samples_per_source)
        elif self.method == "critical":
            src = self.critical_points[self.counter].item()
            quries = random_sampling(graph, samples_per_source, src)
            self.counter += 1
            return quries
        elif self.method == "critical_distance":
            src = self.critical_points[self.counter].item()
            quries = distance_based(graph, samples_per_source, src)
            self.counter += 1
            return quries
        elif self.method == "critical_distance_mix":
            if np.random.rand() > 0.5:
                src = self.critical_points[self.counter].item()
                quries = distance_based(graph, samples_per_source, src)
                self.counter += 1
                return quries
            else:
                return distance_based(graph, samples_per_source)
        else:
            return NotImplementedError(f"{self.method} is not supported.")
        