from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

def main():
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(special_tokens=[
        "<unk>", # Unknown
        "<s>", # Beginning of Sequence
        "</s>", # End of Sequence
        "<pad>", # Padding 
        "<msk>", # Mask
        ],
        vocab_size=30000
    )

    # Split by whitespace
    tokenizer.pre_tokenizer = Whitespace()

    # Train on wikitext-2
    files = [f"./data/wikitext-2/wiki.{split}.tokens" for split in ["test", "train", "valid"]]
    tokenizer.train(files, trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>", # As far as I know we don't use sentence pairs
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>"))
        ]
    )

    tokenizer.save("data/tokenizer-wiki2.json")



if __name__ == "__main__":
    main()