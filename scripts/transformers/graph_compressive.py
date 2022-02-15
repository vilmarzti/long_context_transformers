import torch
from torch.utils.tensorboard import SummaryWriter

from longcontext.transformers.compressive_transformer import (
    CompressiveTransformerWithLMHead,
    CompressiveTransformerConfig
)

from longcontext.utils.dataset import get_dataloader

from transformers import (
    TransfoXLTokenizer
)


def main():
    config = CompressiveTransformerConfig(4, 10, n_layer=6)
    model = CompressiveTransformerWithLMHead(config)
    
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = "cpu"
    model.to(device)

    # Get tokenizer
    tokenizer = TransfoXLTokenizer.from_pretrained("data/tokenizer-xl-wiki2.json")

    # Get Dataloaders processed by TransfoXLTokenizer
    train_loader, _, _ = get_dataloader(tokenizer, 1, 24)

    writer = SummaryWriter("data/tensorboard/")
    for batch in train_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        input_ids, attention_mask = [x.to(device) for x in [input_ids, attention_mask]]
        writer.add_graph(model, (input_ids, torch.BoolTensor([False]), torch.BoolTensor([False]), torch.BoolTensor([False]), attention_mask, input_ids))
    writer.close()
        


if __name__ == "__main__":
    main()