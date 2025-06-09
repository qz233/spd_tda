import os
import argparse
import torch
from omegaconf import OmegaConf

from trainer import Trainer
from model import build_model
from data import build_dataloader

parser = argparse.ArgumentParser(description="run experiements (note that all configs are stored in ./config, argparse only take the config file path)")
parser.add_argument("--config", help="config .yaml path")
parser.add_argument("--prepare", action='store_true', help="prepare datasets")


def train(config):
    print("======= loading dataset and model =======")
    train_dataloader = build_dataloader(config, "train")
    valid_dataloader = build_dataloader(config, "valid")
    test_dataloader = build_dataloader(config, "test")
    model = build_model(config)
    
    print(f"======= train =======")
    trainer = Trainer(model, train_dataloader, valid_dataloader, config)
    trainer.train()
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
    else:
        train(config)

if __name__ == "__main__":
    main()










