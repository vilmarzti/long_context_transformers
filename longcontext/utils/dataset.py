"""
    This module provides all the files that deal with the dataset,
    such as loading or preparing.
"""
from torch.utils.data import DataLoader
from datasets import load_dataset

def get_dataloader(tokenizer, batch_size=8, samples=1000):
    """Creates DataLoaders from wikitext2 encoded with a tokenizer

    Args:
        tokenizer (object): A Tokenizer from huggingface. Should 
            be trained on wikitext2
        batch_size (int, optional): The batch size of the DataLoader. 
            Defaults to 8.
        samples (int, optional): How many samples from the original 
            dataset to take. Defaults to 1000.

    Returns:
        tuple[DataLoader]: Dataloaders containing train, validation
            and test set.
    """
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
    if samples: 
        train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(samples))
        valid_dataset = tokenized_dataset["validation"].shuffle(seed=42).select(range(samples))
        test_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(samples))
    else:
        train_dataset = tokenized_dataset["train"].shuffle(seed=42)
        valid_dataset = tokenized_dataset["validation"].shuffle(seed=42)
        test_dataset = tokenized_dataset["test"].shuffle(seed=42)

    # Get Dataloader
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    valid_loader = DataLoader(valid_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    return (train_loader, valid_loader, test_loader)