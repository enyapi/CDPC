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

2. Collect offline data in target domain
```
python collect_offline_data.py --seed=7 --n_traj=10000 --expert_ratio=0.2 --device="cuda" --env="reacher"
```

3. Train MPC policy and Dynamic model
```
python MPC_DM_model.py --seed=7 --device="cuda" --expert_ratio=0.2 --env="reacher"
```

4. (OPTIONAL) train flow model
```
cd ~/CDPC/flowpg
python -m experiments.train_flow_forward --seed=2 --device='cuda' --env="reacher"
```

5. CDPC
```
python main.py --seed=7 --n_traj=10000 --expert_ratio=0.2 --decoder_ep=500 --device="cuda" --env="reacher"
python main.py --seed=7 --n_traj=1000 --expert_ratio=0.2 --decoder_ep=200 --device="cuda" --env="cheetah"
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