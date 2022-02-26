import torch

from transformers import get_scheduler, TransfoXLConfig, TransfoXLLMHeadModel, TransfoXLTokenizer
from longcontext.utils.dataset import get_dataloader
from longcontext.utils.training import train


def main():
    epochs = 30
    max_length = 32
    batch_size = 10
    samples = 1024

    # Get tokenizer
    tokenizer = TransfoXLTokenizer.from_pretrained("data/tokenizer-xl-wiki2")
    tokenizer.model_max_length = max_length

    # Get Dataloaders processed by TransfoXLTokenizer
    train_loader, valid_loader, _ = get_dataloader(tokenizer, samples=samples, batch_size=10, max_length=max_length, valid_samples=128)

    # Create Model
    config = TransfoXLConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=6,
        cutoffs=[2222, 4444, 22222],
        return_dict=True
    )
    model = TransfoXLLMHeadModel(config)

    # Get gpu if possible and put model on it
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Set optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=5, num_training_steps=epochs*samples/batch_size)

    # train
    train(model, train_loader, optimizer, epochs, valid_loader, device=device, subsequence_len=max_length, lr_scheduler=lr_scheduler)
    

if __name__ == "__main__":
    main()