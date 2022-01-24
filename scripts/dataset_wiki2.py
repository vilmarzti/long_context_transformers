from datasets import load_dataset

dataset = load_dataset("wikitext", name="wikitext-2-v1")
print(dataset)