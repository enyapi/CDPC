# CROSS-DOMAIN REINFORCEMENT LEARNING VIA PREFERENCE CONSISTENCY

## CDPC implementation
1. Train source and target policy
```
python sac_v2.py --seed=7 --device="cuda" --domain="source" --env="reacher"
python sac_v2.py --seed=7 --device="cuda" --domain="target" --env="reacher"
```

2. CDPC
```
python main.py --seed=7 --targetData_ep=10000 --expert_ratio=0.2 --decoder_ep=500 --device="cuda" --env="reacher"
```

## Baselines implementation
1. SAC-Off-TR
```
python baselines/sac_off_tr.py --seed=7 --targetData_ep=10000 --expert_ratio=0.2 --ep=500 --device="cuda" --env="reacher"
```