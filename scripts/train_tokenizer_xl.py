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
        tokenizer.count_file(f"./data/wikitext-2/wiki.{split}.tokens", True)
    tokenizer.build_vocab()

    tokenizer.save_pretrained("data/tokenizer-xl-wiki2.json")


if __name__ == "__main__":
    main()