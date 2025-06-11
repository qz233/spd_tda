import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv
from .transforms import get_coords, normalize_coord

class LinearLayer(nn.Module):
    def __init__(self, input_dim, out_dim, add_norm=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, out_dim)
        if add_norm:
            self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        x = self.linear(x)
        if hasattr(self, "norm"):
            x = self.norm(x)
        return F.silu(x)

class MLP(nn.Module):
    def __init__(self, input_dim, out_dim, hid_dim, n_layers, add_norm=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [LinearLayer(input_dim, hid_dim, add_norm)] +
            [LinearLayer(hid_dim, hid_dim, add_norm) for _ in range(n_layers - 2)]
        )
        self.out_proj = nn.Linear(hid_dim, out_dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out_proj(x)
    
class MLPShortestPath(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = MLP(config.input_dim, config.output_dim, config.hidden_dim, config.num_layers)

    def forward(self, batch):
        # return loss
        pred = self.predict(batch)
        loss = F.mse_loss(pred, batch[2])
        return loss

    def predict(self, batch):   
        queries = torch.cat(get_coords(batch))
        src_emb, tar_emb = self.backbone(queries).chunk(2, dim=0)
        pred = torch.norm(src_emb - tar_emb, p=1, dim=-1)
        return pred 


class GNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kind="gcn", dropout_rate=0.5):
        super().__init__()
        if kind == "gcn":
            self.conv = GCNConv(input_dim, output_dim)
        elif kind == "gin":
            self.conv = GINConv(nn.Sequential(nn.Linear(input_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim)))
        elif kind == "gat":
            self.conv = GATConv(input_dim, output_dim)
        self.norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.silu(x)
        x = self.dropout(x)
        return x
    

class GNN(nn.Module):
    def __init__(self, kind, input_dim, output_dim, hidden_dim, num_layers, dropout_rate=0.5):
        super().__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList([GNNLayer(input_dim, hidden_dim, kind=kind, dropout_rate=dropout_rate)] + 
                                    [GNNLayer(hidden_dim, hidden_dim, kind=kind, dropout_rate=dropout_rate) for _ in range(num_layers - 1)])
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.layers[0](x, edge_index)
        for layer in self.layers[1:]:
            x = x + layer(x, edge_index)
        return self.out_proj(x)


class GNNShortestPath(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = GNN(kind=config.gnn_kind,
                            input_dim=config.input_dim, 
                            output_dim=config.output_dim, 
                            hidden_dim=config.hidden_dim, 
                            num_layers=config.num_layers,
                            dropout_rate=getattr(config, "dropout_rate", 0.2))
        
    def forward(self, batch):
        # return loss
        pred = self.predict(batch)
        loss = F.mse_loss(pred, batch[2])
        return loss

    def predict(self, batch):   
        src, tar, graph = batch[0], batch[1], batch[3]
        graph_emb = self.backbone(graph.pos, graph.edge_index)
        src_emb = graph_emb[src]
        tar_emb = graph_emb[tar]
        pred = torch.norm(src_emb - tar_emb, p=1, dim=-1)
        return pred 


def build_model(config):
    if config.model_type == "mlp":
        return MLPShortestPath(config)
    elif config.model_type == "gnn":
        return GNNShortestPath(config)
    else:
        raise NotImplementedError(f"{config.model_type} not defined.")