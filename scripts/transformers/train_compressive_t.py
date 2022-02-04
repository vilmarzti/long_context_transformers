import torch

from longcontext.transformers.compressive_transformer import (
    CompressiveTransformerWithLMHead,
    CompressiveTransformerConfig
)

from longcontext.utils.dataset import get_dataloader
from longcontext.utils.training import train

from transformers import (
    AdamW,
    TransfoXLTokenizer
)


def main():
    config = CompressiveTransformerConfig(4, 10, n_layer=6)
    model = CompressiveTransformerWithLMHead(config)
    
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