# CROSS-DOMAIN REINFORCEMENT LEARNING VIA PREFERENCE CONSISTENCY

## CDPC implementation
```
cd ~/CDPC
```

1. Train source and target policy
```
python sac_v2.py --seed=2 --device="cuda" --domain="source" --env="cheetah" --ep=2000
python sac_v2.py --seed=2 --device="cuda" --domain="target" --env="cheetah" --ep=500
```

2. BC
```
python BC_training.py --seed=2 --n_traj=10 --expert_ratio=1.0
python BC_training_source.py --seed=2 --n_traj=10 --expert_ratio=1.0
```

3. CDPC
```
python main_v2.py --seed=2 --n_traj=1000 --expert_ratio=1.0 --num_ep=200 --device="cuda" --env="cheetah" --wandb --decoder_batch=256
```

4. BC_multiple
```
see BC_training_multiple.py and evaluation_multiple.py
```

## Baselines implementation
1. SAC-Off-TR
```
python baselines/sac_off_tr.py --seed=7 --n_traj=10000 --expert_ratio=0.2 --ep=500 --device="cuda" --env="reacher"
```

2. BC: imitation
```
python baselines/bc.py --seed=7 --n_traj=10000 --expert_ratio=0.2 --ep=500 --device="cuda" --env="reacher" --bc_ratio=0.1
```

3. BC
```
python baselines/bc2.py --seed=7 --n_traj=10000 --expert_ratio=0.2 --ep=500 --device="cuda" --env="reacher" --bc_ratio=0.1
```