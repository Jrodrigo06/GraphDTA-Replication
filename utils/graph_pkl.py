# utils/preprocess_dataset.py

import pickle
from utils.smiles_to_graph_data import smiles_to_graph_data
import os

FIELD_LENGTH = 5  # Number of fields in each line of the dataset



def load_dataset(file_path: str, save_path: str) -> None:
    data = []

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            raw_fields = line.strip().split()
            if len(raw_fields) < FIELD_LENGTH:
                continue

            _, _, smiles, protein_seq, affinity = raw_fields[:5]  

            try:
                node_features, edge_index = smiles_to_graph_data(smiles)
                affinity = float(affinity)
                data.append({
                    "node_features": node_features,
                    "edge_index": edge_index,
                    "protein_sequence": protein_seq,
                    "affinity": affinity
                })
            except Exception as e:
                print(f"Line {i} failed to process SMILES: {smiles} | Error: {e}")
                continue

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} entries to {save_path}")

if __name__ == "__main__":
    import os

    def test_pkl(file_path):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"[Loaded {len(data)} entries from {file_path}")

        if not data:
            print("No data found in the file.")
            return

        print(f"    First entry keys: {list(data[0].keys())}")
        print(f"    Affinity: {data[0]['affinity']}")
        print(f"    Node features shape: {data[0]['node_features'].shape}")
        print(f"    Edge index shape: {data[0]['edge_index'].shape}")

    input_file = "data/davis.txt"
    output_file = "data/processed_davis.pkl"
    load_dataset(input_file, output_file)
    test_pkl(output_file)

    input_file = "data/davis-filter.txt"
    output_file = "data/processed_davis_filter.pkl"
    load_dataset(input_file, output_file)
    test_pkl(output_file)

    input_file = "data/kiba.txt"
    output_file = "data/processed_kiba.pkl"
    load_dataset(input_file, output_file)
    test_pkl(output_file)