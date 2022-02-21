import torch

from longcontext.transformers.compressive_transformer import (
    CompressiveTransformerWithLMHead,
    CompressiveTransformerConfig
)

from longcontext.utils.dataset import get_dataloader
from longcontext import train

from transformers import (
    AdamW,
    TransfoXLTokenizer
)


def main():
    # Get tokenizer
    tokenizer = TransfoXLTokenizer.from_pretrained("data/tokenizer-xl-wiki2.json")

    # Create Model
    config = CompressiveTransformerConfig(4, 10, vocab_size=tokenizer.vocab_size, n_layer=4, return_dict=True, output_hidden_states=True, cutoffs=[1000, 5000, 15000])
    model = CompressiveTransformerWithLMHead(config)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Get Dataloaders processed by TransfoXLTokenizer
    train_loader, valid_loader, _ = get_dataloader(tokenizer, 4, 100, 100)

    # Set optimizer
    optimizer = AdamW(model.parameters())

    # train
    train(model, train_loader, optimizer, 100, valid_loader, device=device, subsequence_len=50)

if __name__ == "__main__":
    main()