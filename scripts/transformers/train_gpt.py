import torch

from longcontext.utils.dataset import get_dataloader
from longcontext.utils.training import train

from transformers import (
    OpenAIGPTConfig,
    OpenAIGPTTokenizer,
    OpenAIGPTLMHeadModel,
    get_scheduler
)

def main():
    epochs = 30
    max_length = 32
    batch_size = 10
    samples = 1028
    valid_samples = 128

    # Get tokenizer
    tokenizer = OpenAIGPTTokenizer(
        vocab_file="data/tokenizer-gpt-wiki2/vocab.json",
        merges_file="data/tokenizer-gpt-wiki2/merges.txt",
        unk_token="<unk>"
    )

    tokenizer.model_max_length = max_length

    # Get Dataloaders processed by TransfoXLTokenizer
    train_loader, valid_loader, _ = get_dataloader(tokenizer, samples=samples, batch_size=batch_size, max_length=max_length, valid_samples=valid_samples)

    config = OpenAIGPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=max_length,
        n_layer=6
    )

    model = OpenAIGPTLMHeadModel(config)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Set optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=len(train_loader), num_training_steps=epochs*len(train_loader))

    # train
    train(model, train_loader, optimizer, epochs, valid_loader, device=device, lr_scheduler=lr_scheduler)





if __name__ == "__main__":
    main()