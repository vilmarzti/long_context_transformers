from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import os
import json
import sys


def main():
    pass


if __name__ == "__main__":
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(special_tokens=["<unk>", "<pad>"])

    tokenizer.pre_tokenizer = Whitespace()

    files = [f"data/wikitext-2/wiki.{split}.tokens" for split in ["test", "train", "valid"]]
    tokenizer.train(files, trainer)

    if not os.path.isdir("data/tokenizer-gpt-wiki2"):
        os.mkdir("data/tokenizer-gpt-wiki2")
    
    tokenizer.model.save("data/tokenizer-gpt-wiki2")