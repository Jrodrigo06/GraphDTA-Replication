from utils.smiles_to_graph_data import smiles_to_graph_data

class TestSmilesToGraph:

    def test_smiles_to_graph_data(self):
        smiles = "CCO"  
        node_features, edge_index = smiles_to_graph_data(smiles)
        
        # Check the shape of node features
        assert node_features.shape[0] == 3  # Ethanol has 3 atoms
        assert node_features.shape[1] == 13 # Each atom feature vector has length 13
        
        # Check the shape of edge index
        assert edge_index.shape[0] == 2  # Edge index should have two rows
        assert edge_index.shape[1] == 4 # Ethanol has 2 heavy bonds, each represented twice (undirected)