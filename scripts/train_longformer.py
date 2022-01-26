import torch

from transformers import LongformerForMaskedLM, LongformerConfig, PreTrainedTokenizerFast, AdamW, get_scheduler

from longcontext.utils import train
from longcontext.utils.dataset import get_dataloader

epochs = 10

def tokenize_dataset(tokenizer, samples):
    return tokenizer(samples, padding="max_length", truncate=True)

def main():
    # Create model
    config = LongformerConfig(
        vocab_size=30000,
        attention_window=64,
        max_position_embeddings=219,
        num_hidden_layers=6
    )
    model = LongformerForMaskedLM(config)

    # Put model on GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Load pre-trained Tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/tokenizer-wiki2.json")

    # Hack I need to fix
    tokenizer.pad_token = 3

    # Get DataLoaders from wikitext2
    train_loader, valid_loader, _ = get_dataloader(tokenizer, 8, 3000)

    # Set optimizer
    optimizer = AdamW(model.parameters())
    
    # Schedule learning rate
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epochs*len(train_loader)
    )

    # train the model
    train(model, train_loader, optimizer, epochs, valid_loader, lr_scheduler, "cuda") 


if __name__ == "__main__":
    main()
