# CROSS-DOMAIN REINFORCEMENT LEARNING VIA PREFERENCE CONSISTENCY

## CDPC implementation
```
cd ~/CDPC
```

1. Train source and target policy
```
python sac_v2.py --seed=7 --device="cuda" --domain="source" --env="reacher"
python sac_v2.py --seed=7 --device="cuda" --domain="target" --env="reacher"
```

2. (OPTIONAL) train flow model
```
cd ~/CDPC/flowpg
python -m experiments.train_flow_forward --seed=2 --device='cuda' --env="reacher"
```

3. CDPC
```
python main.py --seed=7 --n_traj=10000 --expert_ratio=0.2 --decoder_ep=500 --device="cuda" --env="reacher"
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