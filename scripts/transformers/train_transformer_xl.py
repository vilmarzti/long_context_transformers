import torch

from transformers import get_scheduler, TransfoXLConfig, TransfoXLLMHeadModel, TransfoXLTokenizer
from longcontext.utils.dataset import get_dataloader
from longcontext.utils.training import train


def main():
    epochs = 50
    max_length = 32
    batch_size = 8
    samples = 400 * batch_size
    valid_samples =  40 * batch_size
    

    # Get tokenizer
    tokenizer = TransfoXLTokenizer.from_pretrained("data/tokenizer-xl-wiki2")
    tokenizer.model_max_length = max_length

    # Get Dataloaders processed by TransfoXLTokenizer
    train_loader, valid_loader, _ = get_dataloader(tokenizer, samples=samples, batch_size=batch_size, max_length=max_length, valid_samples=valid_samples)

    # Create Model
    config = TransfoXLConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=12,
        n_heads=8,
        cutoffs=[2222, 4444, 22222],
        return_dict=True,
        mem_len=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = TransfoXLLMHeadModel(config)

    # Get gpu if possible and put model on it
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Set optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=len(train_loader)//8, num_training_steps=epochs*len(train_loader)//8)

    # train
    train(model, train_loader, optimizer, epochs, valid_loader, device=device, subsequence_len=16, lr_scheduler=lr_scheduler)
    

if __name__ == "__main__":
    main()