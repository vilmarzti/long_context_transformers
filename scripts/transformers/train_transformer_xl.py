import torch

from transformers import AdamW, TransfoXLConfig, TransfoXLLMHeadModel, TransfoXLTokenizer
from longcontext.utils.dataset import get_dataloader
from longcontext.utils.training import train


def main():
    # Get tokenizer
    tokenizer = TransfoXLTokenizer.from_pretrained("data/tokenizer-xl-wiki2.json")

    # Get Dataloaders processed by TransfoXLTokenizer
    train_loader, valid_loader, _ = get_dataloader(tokenizer, 2, 128)

    # Create Model
    config = TransfoXLConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=2,
        cutoffs=[1000, 5000, 15000],
        return_dict=True
    )
    model = TransfoXLLMHeadModel(config)

    # Get gpu if possible and put model on it
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    # Set optimizer
    optimizer = AdamW(model.parameters())

    # train
    train(model, train_loader, optimizer, 10, valid_loader, device=device)
    

if __name__ == "__main__":
    main()