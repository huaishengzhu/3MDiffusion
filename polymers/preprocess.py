from multiprocessing import Pool
import math, random, sys
import pickle
import argparse
from functools import partial
import torch
import numpy

from poly_hgraph import MolGraph, common_atom_vocab, PairVocab
import rdkit
sys.path.append("..")
from model.blip2_stage1 import Blip2Stage1
def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c,d,e = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    c = [convert(x) for x in c[0]], [convert(x) for x in c[1]], [convert(x) for x in c[2]], [convert(x) for x in c[3]]
    # print(c[0])
    return a, b, c, d, e


def tensorize(mol_batch, vocab, batch_text, model_blip):
    # print(len(mol_batch))
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab, batch_text, model_blip)
    return to_numpy(x)

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--ncpu', type=int, default=4)
    parser.add_argument('--saved_blipmodel', type=str, default='../all_checkpoints/stage1/epoch=49.ckpt', help='path to loade the blip model')

    args = parser.parse_args()

    with open(args.vocab + "vocab.txt") as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    
    with open(args.vocab + "fragment.txt") as f:
        fragments = [x.strip("\r\n ") for x in f]
    model_blip = Blip2Stage1.load_from_checkpoint(args.saved_blipmodel).cuda()
    MolGraph.load_fragments([fra for fra in fragments])
    # MolGraph.load_fragments([x[0] for x in vocab if eval(x[-1])])
    args.vocab = PairVocab(vocab, cuda=False)

    
    data = []
    # data_text = []
    count = 0
    stop = 0
    with open(args.train) as fin:
        for line in fin.readlines()[1:]:
            count+=1
            # if stop == 60:
            #     break
            stop+=1
            try:
                MolGraph(line.strip().split("\t")[1])
            except:
                continue
            data.append((line.strip().split("\t")[1], line.strip().split("\t")[2]))
            # data_text.append()
        # data = [line.strip("\r\n ").split()[0] for line in f]
    print("Flitering Training Samples:", count-len(data))
    print("Total Number of Samples:", len(data))
    # pool = Pool(args.ncpu) 
    random.seed(1)
    random.shuffle(data)
    batches = []
    batches_text = []
    for i in range(0, len(data), args.batch_size):
        temp = []
        temp_text = []
        for da in data[i : i + args.batch_size]:
            temp.append(da[0])
            temp_text.append(da[1])
        batches.append(temp)
        batches_text.append(temp_text)

    # batches = [batches[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]
    # batches_text = [batches_text[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]
    all_data = []
    
    # func = partial(tensorize, vocab = args.vocab)
    all_data = [tensorize(batch, args.vocab, batch_text, model_blip) for batch, batch_text in zip(batches, batches_text)]
    # all_data = pool.map(func, batches)
    print(len(all_data))
    num_splits = len(all_data) // 1000
    if num_splits ==0:
        num_splits = 1
    le = (len(all_data) + num_splits - 1) // num_splits

    for split_id in range(num_splits):
        st = split_id * le
        sub_data = all_data[st : st + le]

        with open('tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)