# utils/preprocess_dataset.py

import pickle
from utils.smiles_to_graph_data import smiles_to_graph_data

FIELD_LENGTH = 5  # Number of fields in each line of the dataset


def load_dataset(file_path: str, save_path: str) -> None:
    '''
    Loads a dataset from a file, processes each SMILES string into graph data, and saves the processed data to a pickle file.
    
    Args:
        file_path (str): Path to the input dataset file.
        save_path (str): Path to save the processed data. Defaults to "processed_data.pkl".
        Returns:
            None
    '''
    data = []

    with open(file_path, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) != FIELD_LENGTH:
                continue 

            _, _, smiles, protein_seq, affinity = fields
            try:
                node_features, edge_index = smiles_to_graph_data(smiles)
                affinity = float(affinity)
            except:
                continue  

            data.append({
                "node_features": node_features,
                "edge_index": edge_index,
                "protein_sequence": protein_seq,
                "affinity": affinity
            })

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    input_file = "data/davis.txt"  
    output_file = "data/processed_davis.pkl"  
    load_dataset(input_file, output_file)
    input_file = "data/davis-filter.txt"  
    output_file = "data/processed_davis_filter.pkl"
    load_dataset(input_file, output_file)
    input_file = "data/kiba.txt"
    output_file = "data/processed_kiba.pkl"
    load_dataset(input_file, output_file)