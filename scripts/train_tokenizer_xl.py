from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import TransfoXLTokenizer

def main():

    tokenizer = TransfoXLTokenizer(
        special=[
            "<unk>",
            "<eos>",
            "<bos>",
            "<cls>",
            "<msk>",
            "<pad>"
        ]
    )
    
    tokenizer.unk_token = "<unk>"
    tokenizer.eos_token = "<eos>"
    tokenizer.bos_token = "<bos>"
    tokenizer.cls_token = "<cls>"
    tokenizer.pad_token = "<pad>"
    tokenizer.mask_token = "<msk>"

    for split in ["test", "train", "valid"]:
        tokenizer.vocab_file = f"./data/wikitext-2/wiki.{split}.tokens"
        tokenizer.build_vocab()

    # Split by whitespace
    tokenizer.pre_tokenizer = Whitespace()

    # Train on wikitext-2
    files = [f"./data/wikitext-2/wiki.{split}.tokens" for split in ["test", "train", "valid"]]

    # Add start and end sequence tag at beginning/end
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>", # As far as I know we don't use sentence pairs
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>"))
        ]
    )

    # Add padding token
    tokenizer.pad_token = tokenizer.token_to_id("<pad>")

    tokenizer.save("data/tokenizer-wiki2.json")


if __name__ == "__main__":
    main()