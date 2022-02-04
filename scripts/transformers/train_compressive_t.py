import torch

from longcontext.transformers.compressive_transformer import (
    CompressiveTransformerWithLMHead,
    CompressiveTransformerConfig
)

from longcontext.utils.dataset import get_dataloader
from longcontext.utils.training import train


def main():
    config = CompressiveTransformerConfig(4, 10)
    model = CompressiveTransformerWithLMHead(config)
    

if __name__ == "__main__":
    main()