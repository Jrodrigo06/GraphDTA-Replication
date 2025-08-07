from rdkit import Chem
from rdkit.Chem.rdchem import ValenceType
import numpy as np
from typing import List, Tuple
import torch

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

    node_features = []
    for atom in atoms:
        atom_features = get_atom_features(atom)
        node_features.append(atom_features)
    node_features = np.array(node_features, dtype=np.float32)

    edges = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edges.append((start, end))
        edges.append((end, start))
    edge_index = np.array(edges).T

    return node_features, edge_index


def get_atom_features(atom) -> List[int]:
    '''
    Returns a feature vector for an atom. 
    Arguments:
        atom: RDKit atom object.
    Returns:
        List[int]: Feature vector for the atom.
    '''

    symbol_encoded = one_hot_encode_symbol(atom.GetSymbol())
    degree = atom.GetDegree()
    number_of_hydrogens = atom.GetTotalNumHs()
    implicet_valence = atom.GetValence(ValenceType.IMPLICIT)
    is_aromatic = int(atom.GetIsAromatic())

    return symbol_encoded + [degree, number_of_hydrogens, implicet_valence, is_aromatic]


def one_hot_encode_symbol(symbol: str) -> List[int]:
    '''
    One-hot encodes the atom symbol.
    Args:
        symbol (str): Atom symbol.
    Returns:
        List[int]: One-hot encoded vector for the atom symbol.
    '''
    ATOM_LIST = ['Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S'] 
    one_hot_vector = [0] * len(ATOM_LIST)
    if symbol in ATOM_LIST:
        index = ATOM_LIST.index(symbol)
        one_hot_vector[index] = 1
    return one_hot_vector
    