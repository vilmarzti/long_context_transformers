import os
import torch

from longcontext.transformers.compressive_transformer import (
    CompressiveTransformerWithLMHead,
    CompressiveTransformerConfig
)

from longcontext.utils.dataset import get_dataloader
from longcontext import train

from transformers import (
    TransfoXLTokenizer,
    get_scheduler
)

import argparse
import yaml


def main(main_config):
    train_config = main_config["compressive_transformer"].pop("train")
    loader_config = main_config["compressive_transformer"].pop("data_loader")

    # Get tokenizer
    tokenizer = TransfoXLTokenizer.from_pretrained(main_config["transformer_xl_tokenizer"]["path"])

    # Get dataloaders for training
    train_loader, valid_loader, _ = get_dataloader(tokenizer, **loader_config)

    # Create Model
    model_config = CompressiveTransformerConfig(vocab_size=tokenizer.vocab_size, **main_config["compressive_transformer"])
    model = CompressiveTransformerWithLMHead(model_config)
    
    # Put model onto GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Setup optimizer
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
    parser = argparse.ArgumentParser("Compressive-Transformer training routine")
    parser.add_argument("-c", "--config", required=True, type=str, help="Path to the YAML-config for training. Should be found in the \"config/\"-folder")
    args = parser.parse_args()

    config = {}
    if os.path.isfile(args.config):
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
    else:
        raise ValueError("The path to the config file is invalid.")

    main(config)