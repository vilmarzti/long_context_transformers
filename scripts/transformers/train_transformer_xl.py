import torch

from transformers import get_scheduler, AdamW, TransfoXLConfig, TransfoXLLMHeadModel, TransfoXLTokenizer
from longcontext.utils.dataset import get_dataloader
from longcontext.utils.training import train



def main():
    epochs = 100
    # Get tokenizer
    tokenizer = TransfoXLTokenizer.from_pretrained("data/tokenizer-xl-wiki2.json")

    # Get Dataloaders processed by TransfoXLTokenizer
    train_loader, valid_loader, _ = get_dataloader(tokenizer, batch_size=4,  samples=16, max_length=32, valid_samples=4)

    # Create Model
    config = TransfoXLConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=4,
        cutoffs=[2222, 4444, 22222],
        return_dict=True
    )
    model = TransfoXLLMHeadModel(config)

    # Get gpu if possible and put model on it
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Set optimizer
    optimizer = AdamW(model.parameters())

    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=epochs)

    # train
    train(model, train_loader, optimizer, epochs, valid_loader, device=device, subsequence_len=8, lr_scheduler=lr_scheduler)
    

if __name__ == "__main__":
    main()