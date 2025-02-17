# QAvatar: Robust Cross-Domain RL Under Distinct State-Action Representations

This repository is the official implementation of [QAvatar: Robust Cross-Domain RL Under Distinct State-Action Representations]

## Requirements

To install requirements:

```setup
pip3 install -r requirements.txt
```

## Training
**Before training and evaluating, you have to replace all {Absolute Path put QAvatar} in the following command with your own absolute path which you put QAvatar in.**

At first, run this command to train the source model and collect the source-domain data:

```train
cd {Absolute Path put QAvatar}/QAvatar/source/stable-baselines3-1.7.0/
python3 test.py --seed 1 --device 'cuda:0' --folder '{Absolute Path put QAvatar}/QAvatar/' --env 'Hopper-v3'
python3 data_collect.py --seed 1 --folder '{Absolute Path put QAvatar}/QAvatar/' --env 'Hopper-v3'
python3 data_collect_action.py --seed 1 --folder '{Absolute Path put QAvatar}/QAvatar/' --env 'Hopper-v3'
```

Secondly, run this command to train the normalizing flow model in the paper:
```train
cd {Absolute Path put QAvatar}/QAvatar/source/flowpg
python -m experiments.train_flow_forward --seed 1 --device 'cuda:0' --folder '{Absolute Path put QAvatar}/QAvatar/'
python -m experiments.train_flow_forward_action --seed 1 --device 'cuda:0' --folder '{Absolute Path put QAvatar}/QAvatar/'
```

Thirdly, run this command to train the target-domain model in the paper:
```train
cd {Absolute Path put QAvatar}/target_domain/
python3 train_target_model.py --seed 1 --device 'cuda:0' --folder '{Absolute Path put QAvatar}/QAvatar/'
```

## Evaluation

Run this command to evaluate the model after finish training:

```eval
cd {Absolute Path put QAvatar}/target_domain/
python3 eval_model.py --folder '{Absolute Path put QAvatar}/QAvatar/'
```
