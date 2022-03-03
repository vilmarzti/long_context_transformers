from transformers import TransfoXLTokenizer

def main():


    tokenizer = TransfoXLTokenizer(
        delimiter=" ",
        pad_token="<pad>"
    )

    for split in ["test", "train", "valid"]:
        tokenizer.count_file(f"./data/wikitext-2/wiki.{split}.tokens", add_eos=True)
    tokenizer.build_vocab()

    tokenizer.save_pretrained("data/tokenizer-xl-wiki2")


if __name__ == "__main__":
    main()