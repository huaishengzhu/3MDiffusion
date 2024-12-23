import sys
import argparse 
# print("Finish hello")
from poly_hgraph import *
from multiprocessing import Pool
from collections import Counter
from rdkit import Chem
COMMON_ATOMS = [('B', 0), ('B', -1), ('Br', 0), ('Br', -1), ('Br', 2), ('C', 0), ('C', 1), ('C', -1), ('Cl', 0), ('Cl', 1), ('Cl', -1), ('Cl', 2), ('Cl', 3), ('F', 0), ('F', 1), ('F', -1), ('I', -1), ('I', 0), ('I', 1), ('I', 2), ('I', 3), ('N', 0), ('N', 1), ('N', -1), ('O', 0), ('O', 1), ('O', -1), ('P', 0), ('P', 1), ('P', -1), ('S', 0), ('S', 1), ('S', -1), ('Se', 0), ('Se', 1), ('Se', -1), ('Si', 0), ('Si', -1)]
COMMON_ATOMS = {x[0]:x[1] for i,x in enumerate(COMMON_ATOMS)}
def smiles2molecule(smiles: str, kekulize=True):
    '''turn smiles to molecule'''
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    if kekulize:
        Chem.Kekulize(mol)
    return mol
# print("Finish Importing")
def process(data):
    # vocab = set()
    smiles = []
    index = []
    num_nodes = 0
    for i in range(0,len(data)):
        line = data[i]
        s = line.strip("\r\n ")
        mol = smiles2molecule(s, kekulize=True)
        flag = True
        for j in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(j).GetSymbol() not in COMMON_ATOMS:

                flag = False
                break
        
        if not flag:
            continue
        num_nodes = max(num_nodes, mol.GetNumAtoms())
        if mol.GetNumAtoms()>=30:
            continue
        index.append(i)
        smiles.append(s)

    return smiles, index

def fragment_process(data):
    counter = Counter()
    for smiles in data:
        mol = get_mol(smiles)
        fragments = find_fragments(mol)
        for fsmiles, _ in fragments:
            counter[fsmiles] += 1
    return counter

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', type=int, default=4)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--min_frequency', type=int, default=100)
    args = parser.parse_args()
    data = []
    data_all = []
    with open(args.input_file, 'r') as fin:
        for line in fin.readlines()[1:]:
            data.append(line.strip().split("\t")[1])
            data_all.append(line)
                # data.append(mol)
                # print(mol)
    print(COMMON_ATOMS)
    print("Finish Loading")
    batch_size = len(data) // args.ncpu + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    pool = Pool(args.ncpu)
    counter_list = pool.map(fragment_process, batches)
    counter = Counter()
    for cc in counter_list:
        counter += cc
    fragments = [fragment for fragment,cnt in counter.most_common() if cnt >= args.min_frequency]
    MolGraph.load_fragments(fragments)

    smiles, index = process(data)
    write_lines = []
    for i in range(0, len(smiles)):
        write_lines.append(data_all[index[i]])
        # smiles[i] = smiles[i]+"\n"
    smiles_new = ['CID\tSMILES\tdescription\n']
    smiles_new = smiles_new + smiles
    print("Flitering the number of samples:", len(data) - len(smiles))
    # new_data_path = './valid_new.txt'
    with open(args.output_file, 'w') as fout:
            fout.writelines(write_lines)


    