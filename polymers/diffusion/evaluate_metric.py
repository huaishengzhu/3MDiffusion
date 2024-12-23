from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def to_fingerprints(mols):
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in mols]
    return fps

def eval_novelty(pred_actives, true_actives):

    true_fps = to_fingerprints(true_actives)
    pred_fps = to_fingerprints(pred_actives)
    novel_actives = []
    fraction_similar = 0
    for i in range(len(pred_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], true_fps)
        if max(sims) < 0.4:
            novel_actives.append(pred_actives[i])
        else:
            fraction_similar += 1

    novelty = 1 - fraction_similar / len(pred_actives)

    return novelty

def eval_diversity(pred_actives):
    if len(pred_actives) == 0 or len(pred_actives) == 1:
        return []

    pred_fps = to_fingerprints(pred_actives)

    diversity = []
    for i in range(1, len(pred_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], pred_fps[:i])
        diversity+=[1-s for s in sims]

    return diversity