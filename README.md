# 1. Setup

```python
conda env create --file PCR.yaml
mim install mmcv-full==1.7.2
```

# 2. Generate Meta-Dataset
```python
# Vehicle 
bash scripts/metaset_generate_inc/all.sh
# Pedestrian
bash scripts_pedestrian/metaset_generate_inc_person/all.sh
```

# 3. Compute PCR score 
```python
# Vehicle detection - r50+retinanet
bash scripts/autoeval_PCR/all.sh
# You can change configs and work_dir_ours/MODEL/epoch_36.pth for a specific model(MODEL)

# Pedestrian detection - r50+retinanet
bash scripts_pedestrian/autoeval_PCR/all.sh
# You can change configs and work_dir_ours/MODEL/epoch_36.pth for a specific model(MODEL)
```

# 4. Compute RMSE
```python
python tools/rmse.py
```
