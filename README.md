# spd_tda
Use topological data analysis tools to improve shortest path distance learning.
Currently we use 500x500 dataset for efficiency

### prepare dataset
(Will skip already preloaded dataset, so it fine to add new config to the end and run this command again)
```sh
python run.py --prepare --config ./config/prepare_random_500_ds.yaml
python run.py --prepare --config ./config/prepare_modified_500_ds.yaml

```

### Run MLP
```sh
export CUDA_VISIBLE_DEVICES=7
python run.py --config ./config/random_sample_mlp.yaml

export CUDA_VISIBLE_DEVICES=6
python run.py --config ./config/distance_sample_mlp.yaml

export CUDA_VISIBLE_DEVICES=7
python run.py --config ./config/critical_distance_mlp.yaml

export CUDA_VISIBLE_DEVICES=7
python run.py --config ./config/critical_mix_mlp.yaml
```
1.5 min for training 100 epoch.

result averaged over 5 runs.

| | relative error | accuracy|
|- | - | - |
| mlp, random sample| $0.828 \pm 0.018 \%$ | $91.55 \pm 0.31$|
| mlp, distance-based sample| $0.826 \pm 0.027 \%$ | $92.29 \pm 0.26$|
| mlp, critical points-based sample| $0.873 \pm 0.024 \%$ | $90.72 \pm 0.33$|
| mlp, critical + distance sample| $0.847 \pm 0.038\%$ | $91.72 \pm 0.42$|
| mlp, critical mixed sample| $0.794 \pm 0.025\%$ | $92.41 \pm 0.26$|


### Run GAT
```sh
export CUDA_VISIBLE_DEVICES=7
python run.py --config ./config/random_sample_gat.yaml
```
The last epoch result is not satifying, so test best
```sh
export CUDA_VISIBLE_DEVICES=7
python run.py --test --config ./config/random_sample_gat.yaml
```
result relative error: 1.42%, acc: 77.06, after 100 epoch(~1h)

Remark: Maybe we are justified to use mlp on this task. It is way much fast (10x~50x) than gnn, raise comparible result (as in the paper) within much less epoches. 
