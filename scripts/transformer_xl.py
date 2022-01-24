import torch

from transformers import LongformerModel, LongformerConfig
from tokenizers import Tokenizer
from datasets import load_dataset

def main():

    dataset = load_dataset("wikitext", name="wikitext-2-v1", split="train")

    config = LongformerConfig()
    model = LongformerModel(config)
    tokenizer = Tokenizer.from_file("data/tokenizer-wiki2.json")

    for txt in dataset:
        tokens = tokenizer.encode(txt["text"], return_tensors="pt")
        input_ids = torch.tensor(tokens)

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long)


        outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)



if __name__ == "__main__":
    main()
