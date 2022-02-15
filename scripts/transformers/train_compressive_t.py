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
    # Get tokenizer
    tokenizer = TransfoXLTokenizer.from_pretrained("data/tokenizer-xl-wiki2.json")

    # Create Model
    config = CompressiveTransformerConfig(4, 10, vocab_size=tokenizer.vocab_size, n_layer=1, return_dict=True, cutoffs=[1000, 5000, 15000])
    model = CompressiveTransformerWithLMHead(config)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Get Dataloaders processed by TransfoXLTokenizer
    train_loader, valid_loader, _ = get_dataloader(tokenizer, 8, 24)

    # Set optimizer
    optimizer = AdamW(model.parameters())

    # train
    train(model, train_loader, optimizer, 100, valid_loader, device=device)

if __name__ == "__main__":
    main()