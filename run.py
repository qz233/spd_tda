import os
import argparse
import torch
import time
from omegaconf import OmegaConf

from trainer import Trainer
from model import build_model
from data import build_dataloader
from utils import average_multiple_run

parser = argparse.ArgumentParser(description="run experiements (note that all configs are stored in ./config, argparse only take the config file path)")
parser.add_argument("--config", help="config .yaml path")
parser.add_argument("--prepare", action='store_true', help="prepare datasets")
parser.add_argument("--test", action='store_true', help="test model results")


def train(config):
    print("======= loading dataset and model =======")
    train_dataloader = build_dataloader(config, "train")
    valid_dataloader = build_dataloader(config, "valid")
    test_dataloader = build_dataloader(config, "test")
    
    print(f"======= train =======")
    all_results = []
    for i in range(getattr(config, "num_train_runs", 1)):
        model = build_model(config)
        trainer = Trainer(model, train_dataloader, valid_dataloader, config)
        t = time.time()
        trainer.train()
        print(f"[Train time]: {(time.time() - t) / 60:.2f} min")
        metric = trainer.eval(test_dataloader)
        print(f"Result on test: {metric}")
        all_results.append(metric)
    print(f"final result: {average_multiple_run(all_results)}")

def test(config):
    test_dataloader = build_dataloader(config, "test")
    model = build_model(config)
    model.load_state_dict(torch.load(config.pretrain_path, map_location="cpu"))
    trainer = Trainer(model, None, None, config)
    print(f"Result on test: {trainer.eval(test_dataloader)}")

def prepare_dataset(config):
    for key in config.keys():
        ds_config = getattr(config, key)
        print(f"for {key}")
        build_dataloader(ds_config, ds_config.type)

def main():
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    print(f"Running with following config: \n {OmegaConf.to_yaml(config)}")
    if args.prepare:
        prepare_dataset(config)
    elif args.test:
        test(config)
    else:
        train(config)

if __name__ == "__main__":
    main()










