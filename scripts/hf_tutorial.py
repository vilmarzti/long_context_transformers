from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilber-base-uncase-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "Hello World!"

model_inputs = tokenizer(sequence, return_tensors="pt")
model(**model_inputs)

