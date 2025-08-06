from rdkit import Chem
import numpy as np
from typing import List, Tuple

def smiles_to_graph_data(smiles: str) -> Tuple[np.ndarray, np.ndarray]: # type: ignore
    ''' 
      Converts a SMILES string to graph data.
        Args:
            smiles (str): SMILES representation of a molecule.
        Returns:
            node_features: [num_atoms, num_features]
            edge_index: [2, num_edges] for PyTorch-style graphs
    '''

    mol = Chem.MolFromSmiles(smiles) # type: ignore

    atoms = mol.GetAtoms()

    for atom in atoms:
        symbol = atom.GetSymbol()

def get_atom_features(atom) -> List[int]:
    '''
    Returns a feature vector for an atom.
    '''

    symbol = atom.GetSymbol()
    degree = atom.GetDegree()
    number_of_hydrogens = atom.GetTotalNumHs()
    implicet_valence = atom.GetImplicitValence()
    is_aromatic = int(atom.GetIsAromatic())