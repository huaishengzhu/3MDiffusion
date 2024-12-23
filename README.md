# 3M-Diffusion: Latent Multi-Modal Diffusion for Language-Guided Molecular Structure Generation


[![arXiv](https://img.shields.io/badge/arXiv-2205.12454-b31b1b.svg)](https://arxiv.org/abs/2403.07179)

**This repository is an official PyTorch implementation of**  ["3M-Diffusion: Latent Multi-Modal Diffusion for Language-Guided Molecular Structure Generation"](https://openreview.net/pdf?id=DomBynQsqt) (COLM 2024)



<img width="987" alt="image" src="https://github.com/huaishengzhu/3MDiffusion/assets/53974866/746a4bd0-36d7-4a1a-bf65-1bb79f211331">

<img width="987" alt="image" src="https://github.com/huaishengzhu/3MDiffusion/assets/53974866/d26672ea-d7e4-401d-92d3-013cc53134b8">

## Environment Preparation
Run the following command to create a new anaconda environment 3MDiffusion:
```
conda env create -f environment.yml
```

or download the `molca.tar.gz` conda environment in [the link](https://drive.google.com/drive/folders/1TcK0oC3-jr9Lvg14W9Fw3ASBeNCyx-Zw?usp=sharing) and run the following command:

```
tar -xzf molca.tar.gz -C [path of your conda's environment]
```

## Download the pretrained  Text encoder for 3M Diffusion:

Put ``epoch=49.pt`` into the folder all_checkpoints/stage1 and ``graphcl_80.pth`` into polymers/gin_pretrained through [the link](https://drive.google.com/drive/folders/1TcK0oC3-jr9Lvg14W9Fw3ASBeNCyx-Zw?usp=sharing).

## Training for VAE

### Filter small molecule

This folder contains the molecule generation script. The polymer generation experiment in the paper can be reproduced through the following steps:

```
python preprocess_filter.py --input_file ../data/ChEBI-20_data/train.txt --output_file ../data/ChEBI-20_data/train_filter.txt 
python preprocess_filter.py --input_file ../data/ChEBI-20_data/test.txt --output_file ../data/ChEBI-20_data/test_filter.txt 
```

### Motif Extraction
Extract substructure vocabulary from a given set of molecules:
```
mkdir vocab_chebi
python get_vocab.py --min_frequency 100 --ncpu 8 --input_file ../data/ChEBI-20_data/train_filter.txt --output_file ./vocab_chebi/
```
The `--min_frequency` means to discard any large motifs with lower than 100 occurances in the dataset. The discarded motifs will be decomposed into simple rings and bonds. Change `--ncpu` to specify the number of jobs for multiprocessing.

### Data Preprocessing
Preprocess the dataset using the vocabulary extracted from the first step: 
```
python preprocess.py --train ../data/ChEBI-20_data/train_filter.txt --vocab ./vocab_chebi/ --ncpu 8 
mkdir train_processed
mv tensor* train_processed/
```

### Training
Train the generative model with KL regularization weight beta=0.1 and VAE latent dimension 24. You can change it by `--beta` and `--latent_size` argument.
```
mkdir -p ckpt/tmp
python vae_train.py --train train_processed/ --vocab ./vocab_chebi/ --save_dir ckpt/tmp
```

## Training for Diffusion Model


```
cd polymers 

python main.py --adam_weight_decay 0.00001 --num_train_steps 100000 --batch_size 64 --tx_dim 256 --tx_depth 8 --objective pred_x0 --num_samples 1000 --scale_shift --beta_schedule linear --loss_type l2   --wandb_name train_100_smi_d8_decoder --timesteps 100 --sampling_timesteps 50 --text_hidden_dim 256 --train ./train_processed_chebi/ --vocab ./vocab_chebi_30/ --model ./ckpt/tmp-chebi-clip/model.49 --lr 0.001 --epochs 500 --test ../data/ChEBI-20_data/test_filter.txt --output_dir ./results_chebi/
```

## Evaluation

We provide the example for inference of ChEBI-20 dataset.

To repreoduce the results, you firstly need to download five files from [the link](https://drive.google.com/drive/folders/1TcK0oC3-jr9Lvg14W9Fw3ASBeNCyx-Zw?usp=sharing). Put `epoch=49.pt` into the folder `all_checkpoints/stage1`, `graphcl_80.pth` into `polymers/gin_pretrained`, `model-winoise100_train_decoder.pt` into `polymers/results_chebi`, `model.49` into `ckpt/tmp-chebi-clip/` and `tensors-0.pkl` into `train_processed_chebi`.

Then you can run the following code for inference on ChEBI-20 dataset:
```
cd polymers

python evaluate_diffusion.py --adam_weight_decay 0.00001 --num_train_steps 100000 --batch_size 64 --tx_dim 256 --tx_depth 8 --objective pred_x0 --num_samples 1000 --scale_shift --beta_schedule linear --loss_type l2   --wandb_name train_100_smi_d8_decoder --timesteps 100 --sampling_timesteps 50 --text_hidden_dim 256 --train ./train_processed_chebi/ --vocab ./vocab_chebi_30/ --model ./ckpt/tmp-chebi-clip/model.49 --lr 0.001 --epochs 500 --test ../data/ChEBI-20_data/test_filter.txt --output_dir ./results_new/ --resume_dir ./results_chebi/
```

## Citation

If you find this work useful, please cite our paper:
```bibtex
@inproceedings{zhu20243m,
  title={3M-Diffusion: Latent Multi-Modal Diffusion for Language-Guided Molecular Structure Generation},
  author={Zhu, Huaisheng and Xiao, Teng and Honavar, Vasant G},
  booktitle={First Conference on Language Modeling},
  year={2024}
}
```


