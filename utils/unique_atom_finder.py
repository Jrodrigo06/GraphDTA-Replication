from rdkit import Chem

def extract_unique_atom_symbols(file_path: str):
    '''
    Extracts unique atom symbols from a file containing SMILES strings.
    
    Args:
        file_path (str): Path to the file containing SMILES strings.
        Returns:
            List[str]: Sorted list of unique atom symbols.
            '''
    
    atom_symbols = set()

    with open(file_path, 'r') as f:
        for i, raw_line in enumerate(f):
            if i < 5:
                print("Raw:", raw_line.strip())
            line = raw_line.strip().split()
            if len(line) < 3:
                continue
            smiles = line[2]
            mol = Chem.MolFromSmiles(smiles)
            for atom in mol.GetAtoms():
                atom_symbols.add(atom.GetSymbol())

    return sorted(atom_symbols)

if __name__ == "__main__":
    path = "data/davis.txt" 
    atoms = extract_unique_atom_symbols(path)
    print("Unique Atom Symbols:")
    print(atoms)
