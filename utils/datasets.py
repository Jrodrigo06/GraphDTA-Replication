import pickle, torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader


class GraphDTAPklDataset(Dataset):
    """
    Dataset class for loading drug-target binding affinity data from a pickle file.
    Each entry in the dataset contains node features, edge indices, protein sequences, and affinity values.
    """
    def __init__(self, pkl_file: str):
        """
        Initializes the dataset by loading data from a pickle file.
        Args:
            pkl_file (str): Path to the pickle file containing the dataset.
        """
        self.db = self._load_pkl(pkl_file)
    
    def _load_pkl(self, pkl_file: str):
        """
        Loads the dataset from a pickle file.
        Args:
            pkl_file (str): Path to the pickle file.
        Returns:
            List[Dict]: A list of dictionaries, each containing node features, edge indices, protein sequences, and affinity values.
        """
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        k = data[0]
        for key in ['node_features', 'edge_index', 'protein_sequence', 'affinity']:
            if key not in k:
                raise KeyError(f"Key '{key}' not found in the dataset.")
        return data
    
    def __len__(self):
        """
        Returns the number of entries in the dataset.
        Returns:
            int: Number of entries in the dataset.
        """
        return len(self.db)
    
    def __getitem__(self, idx: int):
        e = self.db[idx]
        # dictionary with keys: 'node_features', 'edge_index', 'protein_sequence', 'affinity'
        x = torch.tensor(e['node_features'], dtype=torch.float32) # Node features
        ei = torch.tensor(e['edge_index'], dtype=torch.long) # Edge indices
        y = torch.tensor([e['affinity']], dtype=torch.float32) # Affinity value
        g = Data(x=x, edge_index=ei, y=y)
        return g, e['protein_sequence']  
    
def collate_graphdta(batch):
    """
    Collate function to combine a batch of graph data and protein sequences.
    Args:
        batch (List[Tuple[Data, str]]): A list of tuples where each tuple contains a graph data object and a protein sequence.
        Returns:
            Tuple[Batch, List[str], torch.Tensor]: A tuple containing a Batch object of graph data, a list of protein sequences, and a tensor of affinity values.
    """
    graphs, prot_seqs = zip(*batch)                
    graph_batch = Batch.from_data_list(list(graphs))
    y=torch.stack([g.y for g in graphs])
    return graph_batch, list(prot_seqs), y

if __name__ == "__main__":
    ds = GraphDTAPklDataset("data/processed_davis.pkl")
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_graphdta)

    graph_batch, prot_seqs, y = next(iter(loader))
    print("x:", graph_batch.x.shape)                 # [sum_nodes, 13]
    print("edge_index:", graph_batch.edge_index.shape)
    print("batch idx:", graph_batch.batch.shape)     # [sum_nodes]
    print("prot_seqs:", type(prot_seqs), len(prot_seqs))  # list, B
    print("y:", y.shape)                             # [B, 1]