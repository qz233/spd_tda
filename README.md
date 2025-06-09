# spd_tda
Use topological data analysis tools to improve shortest path distance learning.
Currently we use 500x500 dataset for efficiency

### prepare dataset
(Will skip already preloaded dataset, so it fine to add new config to the end and run this command again)
```sh
python run.py --prepare --config ./config/prepare_random_500_ds.yaml

```

### Run MLP
```sh
export CUDA_VISIBLE_DEVICES=7
python run.py --config ./config/random_sample_mlp.yaml
```
result (might need to run mutliple times and do averaging):
relative error: 0.81%, acc: 92.05