import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool

# Class for Graph Attention Network (GAT) for Drug-Target Binding Affinity Prediction
class GATGraphDTA(torch.nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 13, 
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8, 
        dropout: float = 0.1
    ):
        '''
        Initializes the GATGraphDTA model with multiple GAT layers.
        Args:
            node_feat_dim (int): Dimension of the node features.
            hidden_dim (int): Dimension of the hidden layers.
            num_layers (int): Number of GAT layers.
            num_heads (int): Number of attention heads in each GAT layer.
            dropout (float): Dropout probability.
        '''

        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(node_feat_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, concat=True))
    
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, concat=False))
        
        self.prot_emb_dim = 128 
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
        combined_vec = torch.cat([drug_graph_vec, prot_vec], dim=1) 
        out = self.fc(combined_vec)  

        return out

