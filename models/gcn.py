import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool

# Class for Graph Convolutional Network (GCN) for Drug-Target Binding Affinity Prediction
class GCNGraphDTA(torch.nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 13,  # numer of atom features
        hidden_dim:    int = 32,
        num_layers:   int = 3
    ):
        '''
        Initializes the GCNGraphDTA model with multiple GCN layers.
        Args:
            node_feat_dim (int): Dimension of the node features.
            hidden_dim (int): Dimension of the hidden layers.
            num_layers (int): Number of GCN layers.
        '''

        super().__init__()  
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(node_feat_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
    def forward(self, data, prot_vec):
        """
        data.x          [total_nodes, node_feat_dim]
        data.edge_index [2, total_edges]
        data.batch      [total_nodes] mapping nodes→graph
        prot_vec        [batch, prot_emb_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for conv in self.convs:
            x = conv(x, edge_index)   
            x = F.relu(x)              
        
  
        drug_graph_vec = global_max_pool(x, batch)  

        return drug_graph_vec