# DSAA5013 Group 4 Project

Reproduction of the paper 

> **[Elucidating the Design Space of Dataset Condensation](https://arxiv.org/abs/2404.13733) ** <br>

on CIFAR-10/100 datasets. 

Paper authors: <br>

><em>Shitong Shao, Zikai Zhou, Huanran Chen, </em> and <em>Zhiqiang Shen*</em> <br>

Original open source code: <br>

> https://github.com/shaoshitong/EDC
>
> ***Note that we identified some bugs in the original codebase, which prevented us from successfully  running it. The detailed debugging procedures and resolutions are documented in the report.***

## Get Started

### Prerequisites
To run the code, you need to install the following packages and dependency:
```bash
# create the conda environment
conda create <name env> -file env.yaml

# install the dependency and packages
pip install -r requirements.txt
```

### File Directory
The file directory consists of four parts:
```dotenv
├─Branch_CIFAR_100
│  ├─squeeze  # Get the statistics
│  │  └─models
│  ├─recover   # Get the synthetic images
│  │  └─models
│  ├─relabel   # Relabel the sythetic images
│  │  └─models
│  └─train     # Evaluate the synthetic dataset
```
### Example
First run `squeeze.sh` in `squeeze` folder to obtain the statistics:
```bash
bash squeeze.sh
```
Then run `recover.sh` in `recover` folder to obtain the synthetic datasets. Change `train-data-path` to your CIFAR-10/100 dataset path. Note that you can also change the hyperparameter `ipc-number` to determine the number of synthetic datasets per class. 
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python recover.py \
    --arch-name "resnet18" \
    --exp-name "EDC_CIFAR_100_Recover_IPC_10" \
    --batch-size 100 --category-aware "global" \
    --lr 0.05 --drop-rate 0.0 \
    --ipc-number 10 --training-momentum 0.8 \
    --iteration 2000 \
    --train-data-path  /path/to/dataset/cifar100 \
    --r-loss 0.01 --initial-img-dir "None" \
    --verifier --store-best-images --gpu-id 0,1,2,3
```
Then run the `relabel.sh` in `relabel`  folder to generate soft labels for synthetic datasets. Note that the `-b` (batch size) should be the same as that (`batch-size`) in `recover.sh`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_soft_label_with_db.py \
    -b 100 \
    -j 8 \
    --epochs 300 \
    --fkd-seed 42 \
    --input-size 224 \
    --min-scale-crops 0.5 \
    --max-scale-crops 1 \
    --use-fp16 --candidate-number 4 \
    --fkd-path /path/to/store/soft/label \
    --mode 'fkd_save' \
    --mix-type 'cutmix' \
    --data /path/to/synthetic/dataset
```
Evaluating the synthetic dataset by running `train.sh`  in `train` folder. Change `val-dir` to your local CIFAR-10/100 validation dataset and `train-dir` to your synthetic dataset used in `recover.sh`:
```bash
wandb enabled
wandb offline

CUDA_VISIBLE_DEVICES=0 python direct_train.py \
    --wandb-project 'final_RN18_fkd' \
    --batch-size 50 --epochs 1000 \
    --model "ResNet18" \
    --ls-type cos2 --loss-type "mse_gt" --ce-weight 0.025 \
    -T 20 --sgd --sgd-lr 0.1 --adamw-lr 0.001 --gpu-id 0 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    --mix-type 'cutmix' --weight-decay 0.0005 \
    --output-dir ./save/final_RN18_fkd/ \
    --train-dir /path/to/synthetic/dataset \
    --val-dir '/path/to/dataset/cifar100/val/'
```

## Group 4 members

Zhang Kunming, Wang Xinlong, Tan ZhongXi, Lei Yongxin