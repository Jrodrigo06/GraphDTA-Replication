from torch_geometric.nn import GCNConv, global_max_pool
import torch
from torch_geometric.data import Data
import torch.nn.functional as F

# Convolutional layer example
conv1 = GCNConv(in_channels=13, out_channels=32)

x = torch.randn((5, 13))  # 5 nodes, each with 13 features
edge_index = torch.tensor([[0, 1, 2, 3, 4, 0],
                           [1, 0, 3, 2, 0, 4]],
                            dtype=torch.long)  # 6 edges

data = Data(x=x, edge_index=edge_index)

out = conv1(data.x, data.edge_index)
print(out.shape)  # Should be [5, 32]

h1 = F.relu(out)
print(h1.shape)  # Should be [5, 32]