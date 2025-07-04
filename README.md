# 1. Setup

```python
conda env create --file PCR.yaml
mim install mmcv-full==1.7.2
```

# 2. Generate Meta-Dataset
```python
bash scripts/metaset_generate_inc/all.sh
```

# 3. Compute PCR score 
```python
# 
bash scripts/autoeval_PCR/all.sh
```
