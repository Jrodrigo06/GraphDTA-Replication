import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool

# Class for Graph Convolutional Network (GCN) for Drug-Target Binding Affinity Prediction
class GCNGraphDTA(torch.nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 13,  # numer of atom features
        hidden_dim:    int = 128,
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
        
        self.prot_emb_dim = 128  # Dimension of protein embedding 
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + self.prot_emb_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        
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
        combined_vec = torch.cat([drug_graph_vec, prot_vec], dim=1) # This combines drug and protein features
        out = self.fc(combined_vec)  # Final prediction layer 

        return out