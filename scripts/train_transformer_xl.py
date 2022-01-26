import torch
from transformers import TransfoXLConfig, TransfoXLLMHeadModel


def main():
    # Create Model
    config = TransfoXLConfig(
        n_layer=6
    )
    model = TransfoXLLMHeadModel(config)

    # Get gpu if possible and put model on it
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    main()