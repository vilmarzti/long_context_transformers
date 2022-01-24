import torch

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import LongformerModel, LongformerConfig, PreTrainedTokenizerFast

def tokenize_dataset(tokenizer, samples):
    return tokenizer(samples, padding="max_length", truncate=True)

def main():
    # Create model
    config = LongformerConfig(vocab_size=30000)
    model = LongformerModel(config)

    # Load pre-trained Tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/tokenizer-wiki2.json")

    # Load Dataset
    dataset = load_dataset("wikitext", name="wikitext-2-v1", split="train")

    # Tokenize Dataset
    tokenized_dataset = dataset.map(
        lambda samples: tokenizer(samples["text"], padding="max_length", truncation=True, max_length=512),
        batched=True
    )


    for txt in dataset:
        tokens = tokenizer.encode(txt["text"], return_tensors="pt")
        input_ids = torch.tensor(tokens)

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long)


        outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)



if __name__ == "__main__":
    main()
