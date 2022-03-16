import torch

from optuna import trial

from transformers import get_scheduler, TransfoXLConfig, TransfoXLLMHeadModel, TransfoXLTokenizer
from longcontext.utils.dataset import get_dataloader
from longcontext.utils.training import train

import argparse
import yaml
import os


def main(main_config):
    train_config = main_config["transformer_xl"].pop("train")
    loader_config = main_config["transformer_xl"].pop("data_loader")


    # Get tokenizer
    tokenizer = TransfoXLTokenizer.from_pretrained(**main_config["transformer_xl_tokenizer"])
    tokenizer.model_max_length = loader_config["max_length"]

    # Get Dataloaders processed by TransfoXLTokenizer
    train_loader, valid_loader, _ = get_dataloader(tokenizer, **loader_config)

    # Create Model
    config = TransfoXLConfig(
        vocab_size=tokenizer.vocab_size,
        eos_token_id=tokenizer.eos_token_id,
        **main_config["transformer_xl"]
    )
    model = TransfoXLLMHeadModel(config)

    # Get gpu if possible and put model on it
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Set optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=main_config["optimizer"]["learning_rate"])

    # Setup learning_rate scheduler
    steps_per_epoch = len(train_loader) // train_config["aggregate"]
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=steps_per_epoch,
        num_training_steps=train_config["epochs"] * steps_per_epoch
    )

    # train
    train(model, train_loader, optimizer, valid_loader=valid_loader, device=device, lr_scheduler=lr_scheduler, **train_config)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Transformer-XL training routine")
    parser.add_argument("-c", "--config", required=True, type=str, help="Path to the YAML-config for training. Should be found in the \"config/\"-folder")
    args = parser.parse_args()

    config = {}
    if os.path.isfile(args.config):
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
    else:
        raise ValueError("The path to the config file is invalid.")

    main(config)