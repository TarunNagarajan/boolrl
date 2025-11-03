from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data

class GNNQNetwork(nn.Module):
    def __init__(self, gnn_input_size: int, global_feature_size: int, hidden_size: int, action_size: int):
        super(GNNQNetwork, self).__init__()
        self.conv1 = GATConv(gnn_input_size, hidden_size, heads=4, dropout=0.2)
        self.bn1 = nn.BatchNorm1d(hidden_size * 4)
        self.conv2 = GATConv(hidden_size * 4, hidden_size, heads=4, dropout=0.2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 4)
        self.conv3 = GATConv(hidden_size * 4, hidden_size, heads=1, concat=False, dropout=0.2)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        self.fc1 = nn.Linear(hidden_size + global_feature_size, hidden_size * 2)
        self.bn_fc1 = nn.BatchNorm1d(hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn_fc2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, data: Union[Data, torch.Tensor]) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        global_features = data.global_features

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))

        x = global_mean_pool(x, batch)

        if global_features.dim() == 3:
            global_features = global_features.squeeze(1)
        x = torch.cat([x, global_features], dim=1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        return self.fc3(x)
