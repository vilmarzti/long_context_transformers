import torch

from transformers import LongformerModel, LongformerTokenizer, LongformerConfig
from datasets import load_dataset

def main():

    dataset = load_dataset("wikitext", name="wikitext-2-v1", split="train")

    config = LongformerConfig()
    model = LongformerModel(config)
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

    for txt in dataset:
        tokens = tokenizer.encode(txt["text"], return_tensors="pt")
        input_ids = torch.tensor(tokens)

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long)


        outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)



if __name__ == "__main__":
    main()
