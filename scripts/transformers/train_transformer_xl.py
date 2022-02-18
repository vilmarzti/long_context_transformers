import torch

from transformers import AdamW, TransfoXLConfig, TransfoXLLMHeadModel, TransfoXLTokenizer
from longcontext.utils.dataset import get_dataloader
from longcontext.utils.training import train


def main():
    # Create Model
    config = TransfoXLConfig(
        n_layer=2
    )
    model = TransfoXLLMHeadModel(config)

    # Get gpu if possible and put model on it
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Get tokenizer
    tokenizer = TransfoXLTokenizer.from_pretrained("data/tokenizer-xl-wiki2.json")

    # Get Dataloaders processed by TransfoXLTokenizer
    train_loader, valid_loader, _ = get_dataloader(tokenizer, 8, 24)

    # Set optimizer
    optimizer = AdamW(model.parameters())

    # train
    train(model, train_loader, optimizer, 10, valid_loader, device=device)
    

if __name__ == "__main__":
    main()