import torch

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import LongformerForMaskedLM, LongformerConfig, PreTrainedTokenizerFast, AdamW, get_scheduler
from longcontext.utils import train

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

    # Load Dataset
    dataset = load_dataset("wikitext", name="wikitext-2-v1")

    # Tokenize Dataset
    tokenized_dataset = dataset.map(
        lambda samples: tokenizer(samples["text"], padding="max_length", truncation=True, max_length=218),
        batched=True
    )
    tokenized_dataset.set_format("torch")

    # Remove text and type_ids
    tokenized_dataset = tokenized_dataset.remove_columns(["text", "token_type_ids"])
    
    # Get Datasets
    train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
    valid_dataset = tokenized_dataset["validation"].shuffle(seed=42).select(range(200))

    # Get Dataloader
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    valid_loader = DataLoader(valid_dataset, batch_size=8)

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