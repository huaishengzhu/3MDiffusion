import torch
from torch.utils.data import Dataset
from rdkit import Chem
import os, random, gc
import pickle

from poly_hgraph.chemutils import get_leaves
from poly_hgraph.mol_graph import MolGraph

class MoleculeDataset(Dataset):

    def __init__(self, data, vocab, avocab, batch_size, data_text, model_blip, text_only=False):
        self.batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.batches_text = [data_text[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.vocab = vocab
        self.avocab = avocab
        self.model_blip = model_blip
        self.text_only = text_only

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        if self.text_only:
            return MolGraph.tensorize_text(self.batches_text[idx], self.model_blip)
        return MolGraph.tensorize(self.batches[idx], self.vocab, self.avocab, self.batches_text[idx], self.model_blip)

class MolEnumRootDataset(Dataset):

    def __init__(self, data, vocab, avocab):
        self.batches = data
        self.vocab = vocab
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.batches[idx])
        leaves = get_leaves(mol)
        smiles_list = set( [Chem.MolToSmiles(mol, rootedAtAtom=i, isomericSmiles=False) for i in leaves] )
        smiles_list = sorted(list(smiles_list)) #To ensure reproducibility

        safe_list = []
        for s in smiles_list:
            hmol = MolGraph(s)
            ok = True
            for node,attr in hmol.mol_tree.nodes(data=True):
                if attr['label'] not in self.vocab.vmap:
                    ok = False
            if ok: safe_list.append(s)
        
        if len(safe_list) > 0:
            return MolGraph.tensorize(safe_list, self.vocab, self.avocab)
        else:
            return None

class MolPairDataset(Dataset):

    def __init__(self, data, vocab, avocab, batch_size):
        self.batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.vocab = vocab
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        x, y = zip(*self.batches[idx])
        x = MolGraph.tensorize(x, self.vocab, self.avocab)[:-1] #no need of order for x
        y = MolGraph.tensorize(y, self.vocab, self.avocab)
        return x + y
    
class TextDataset(Dataset):

    def __init__(self, data, data_text, batch_size, model_blip):
        self.batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.batches_text = [data_text[i : i + batch_size] for i in range(0, len(data_text), batch_size)]
        self.model_blip = model_blip


    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return MolGraph.tensorize_text(self.batches_text[idx], self.model_blip), self.batches[idx]
    
class DataFolder(object):

    def __init__(self, data_folder, batch_size, shuffle=True):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.length = self.get_len()

    def get_len(self):
        count =  0
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                batches = pickle.load(f)
                for batch in batches:
                    count+=1
        return count  

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                batches = pickle.load(f)

            if self.shuffle: random.shuffle(batches) #shuffle data before batch
            for batch in batches:
                yield batch

            del batches
            gc.collect()
    
    def __len__(self):
        return self.length

