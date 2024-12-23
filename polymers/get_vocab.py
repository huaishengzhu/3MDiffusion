import sys
import argparse 
from collections import Counter
from poly_hgraph import *
from rdkit import Chem
from multiprocessing import Pool

def process(data):
    vocab = set()
    for i in range(0, len(data)):
        line = data[i]
        s = line.strip("\r\n ")
        try:
            hmol = MolGraph(s)
            for node,attr in hmol.mol_tree.nodes(data=True):
                smiles = attr['smiles']
                vocab.add( attr['label'] )
                if attr['label'][1] == 'C1=[CH:1][CH2:2]N=[CH:2][NH:2]1':
                    print("----")
                for i,s in attr['inter_label']:
                    vocab.add( (smiles, s) )
        except:
            continue
    return vocab

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
    parser.add_argument('--min_frequency', type=int, default=100)
    parser.add_argument('--ncpu', type=int, default=1)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--input_fragment_file', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()
    data = []
    with open(args.input_file, 'r') as fin:
        for line in fin.readlines()[1:]:
            data.append(line.strip().split("\t")[1])
    # data = list(set(data))

    batch_size = len(data) // args.ncpu + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    pool = Pool(args.ncpu)
    counter_list = pool.map(fragment_process, batches)
    counter = Counter()
    for cc in counter_list:
        counter += cc
    
    fragments = [fragment for fragment,cnt in counter.most_common() if cnt >= args.min_frequency]
    with open(args.output_file + "./fragment.txt", 'w') as fin:
        for fra in fragments:
            fin.write(fra+"\n")
    MolGraph.load_fragments(fragments)
    
    pool = Pool(args.ncpu)
    vocab_list = pool.map(process, batches)
    vocab = [(x,y) for vocab in vocab_list for x,y in vocab]
    vocab = list(set(vocab))

    fragments = set(fragments)
    with open(args.output_file + "./vocab.txt", 'w') as fin:
        for x,y in sorted(vocab):
        # print(x, y)
            fin.write(x+" "+y+"\n")