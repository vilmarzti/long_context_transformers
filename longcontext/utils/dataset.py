"""
    This module provides all the files that deal with the dataset,
    such as loading or preparing.
"""
from dataclasses import replace
from torch.utils.data import DataLoader
from datasets import load_dataset
import datasets


def get_dataloader(tokenizer, batch_size=8, samples=None, max_length=256, valid_samples=None, padding="max_length", replace_pad=False):
    """Creates DataLoaders from wikitext2 encoded with a tokenizer

    Args:
        tokenizer (object): A Tokenizer from huggingface. Should 
            be trained on wikitext2
        batch_size (int, optional): The batch size of the DataLoader. 
            Defaults to 8.
        samples (int, optional): How many samples from the original 
            dataset to take. If None are provided then it takes all samples.
            Defaults to None.
        max_length (int, optional): The length the input_ids should have.
            Defaults to 256
        valid_samples (int, optional): How many validation samples we have.
            If None all the validation samples are considered. Defaults
            to None
        padding (bool, optional): Whether we pad all the sequences below
            max_length to max_length or discred them. Defaults to True
        replace_bad (bool, optional): Whether we replace the padding token ids
            with -100. This is useful when computing the Cross-Entropy-loss as
            these tokens won't be considered

    Returns:
        tuple[DataLoader]: Dataloaders containing train, validation
            and test set.
    """

    # Remove Verbose
    datasets.logging.set_verbosity_error()

    # Load Dataset
    dataset = load_dataset("wikitext", name="wikitext-103-v1")

    # Exclude empty text samples
    dataset = dataset.filter(lambda sample: len(sample["text"]) > 0)

    # Tokenize Dataset
    tokenized_dataset = dataset.map(
        lambda samples: tokenizer(samples["text"], max_length=max_length, truncation=True, padding=padding, return_attention_mask=True, verbose=False),
        batched=True
    )

    # Filter for equal length if no padding is provided
    tokenized_dataset = tokenized_dataset.filter(lambda sample: len(sample["input_ids"]) == max_length)

    if replace_pad:
        # Set pad_token id to -100 such that it get's ignored when computing the cross-entropy
        tokenized_dataset = tokenized_dataset.map(
            lambda sample: {
                "input_ids": [input_id if input_id != tokenizer.pad_token_id else -100 for input_id in sample["input_ids"]],
                **{i:sample[i] for i in sample if i!="input_ids"}
            }
    )

    # Change to tensors
    tokenized_dataset.set_format("torch")

    # Remove text and type_ids
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    if "token_type_ids" in tokenized_dataset.column_names["train"]:
        tokenized_dataset.remove_columns(["token_type_ids"])
    
    # Get Datasets
    if samples: 
        train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(samples))
    else:
        train_dataset = tokenized_dataset["train"].shuffle(seed=42)
   
    if valid_samples:
        valid_dataset = tokenized_dataset["validation"].shuffle(seed=42).select(range(valid_samples))
        test_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(valid_samples))
    else:
        valid_dataset = tokenized_dataset["validation"].shuffle(seed=42)
        test_dataset = tokenized_dataset["test"].shuffle(seed=42)
 
    # Get Dataloader
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return (train_loader, valid_loader, test_loader)