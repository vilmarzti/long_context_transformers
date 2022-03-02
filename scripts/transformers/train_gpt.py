import argparse
import torch
import os
import yaml

from longcontext.utils.dataset import get_dataloader
from longcontext.utils.training import train

from transformers import (
    OpenAIGPTConfig,
    OpenAIGPTTokenizer,
    OpenAIGPTLMHeadModel,
    get_scheduler
)


def main(main_config):
    # Get tokenizer
    tokenizer = OpenAIGPTTokenizer(**main_config["gpt_tokenizer"])
    tokenizer.model_max_length = main_config["data_loader"]["max_length"]

    # Get Dataloaders processed by TransfoXLTokenizer
    train_loader, valid_loader, _ = get_dataloader(tokenizer, **main_config["data_loader"])

    config = OpenAIGPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=main_config["data_loader"]["max_length"],
        **main_config["gpt"]
    )

    model = OpenAIGPTLMHeadModel(config)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Set optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=main_config["optimizer"]["learning_rate"])

    # Setup learning_rate scheduler
    steps_per_epoch = len(train_loader) // main_config["train"]["aggregate"]
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=steps_per_epoch,
        num_training_steps=main_config["train"]["epochs"] * steps_per_epoch
    )

    # train
    train(model, train_loader, optimizer, valid_loader=valid_loader, device=device, lr_scheduler=lr_scheduler, **main_config["train"])


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
