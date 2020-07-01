# Auto-DrAC: Automatic Data-Regularized Actor-Critic
This is a PyTorch implementation of the methods proposed in

[**Automatic Data Augmentation for Generalization in Deep Reinforcement Learning**](https://arxiv.org/pdf/2006.12862.pdf) by 

Roberta Raileanu, Max Goldstein, Denis Yarats, Ilya Kostrikov, and Rob Fergus.

# Citation
If you use this code in your own work, please cite our paper:
```
@article{raileanu2020automatic,
  title={Automatic Data Augmentation for Generalization in Deep Reinforcement Learning},
  author={Raileanu, Roberta and Goldstein, Max and Yarats, Denis and Kostrikov, Ilya and Fergus, Rob},
  journal={arXiv preprint arXiv:2006.12862},
  year={2020}
}
```

# Requirements
The code was run on a GPU with CUDA 10.2.
To install all the required dependencies: 

```
conda create -n auto-drac python=3.7
conda activate auto-drac

git clone git@github.com:rraileanu/auto-drac.git
cd auto-drac
pip install -r requirements.txt

git clone https://github.com/openai/baselines.git
cd baselines 
python setup.py install 

pip install procgen
```


# Instructions
```
cd auto-drac
```

## Train DrAC with *crop* augmentation on BigFish
```
python train.py --env_name bigfish --aug_type crop
```

## Train UCB-DrAC on BigFish
```
python train.py --env_name bigfish --use_ucb
```

## Train RL2-DrAC on BigFish
```
python train.py --env_name bigfish --use_rl2
```

## Train Meta-DrAC on BigFish
```
python train.py --env_name bigfish --use_meta
```

# Procgen Results 
**UCB-DrAC** achieves state-of-the-art performance on the [Procgen benchmark](https://openai.com/blog/procgen-benchmark/) (easy mode), significantly improving the agent's generalization ability over standard RL methods such as PPO.  

Test Results on Procgen

![Procgen Test](/figures/test.png)

Train Results on Procgen

![Procgen Train](/figures/train.png)

# Agent Videos
You can find some videos of the agent's behavior while training on our [website](https://sites.google.com/view/ucb-drac).


# Acknowledgements
This code was based on an open sourced [PyTorch implementation of PPO](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).

We also used [kornia](https://github.com/kornia/kornia) for some of the augmentations.
