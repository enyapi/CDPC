# CROSS-DOMAIN REINFORCEMENT LEARNING VIA PREFERENCE CONSISTENCY

1. Train source and target policy
```
python sac_v2.py --seed=7 --device="cuda" --domain="source" --env="reacher"
python sac_v2.py --seed=7 --device="cuda" --domain="target" --env="reacher"
```

2. CDPC
```
python main.py --seed=7 --targetData_ep=10000 --expert_ratio=0.8 --decoder_ep=500 --device="cuda" --env="reacher"
```
