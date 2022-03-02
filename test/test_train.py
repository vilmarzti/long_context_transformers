import torch
import unittest
import yaml

from transformers import (
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLConfig,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
)

from longcontext.transformers.compressive_transformer import (
    CompressiveTransformerWithLMHead,
    CompressiveTransformerConfig
)

from longcontext.utils.dataset import get_dataloader
from longcontext import train


class TrainingTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Get Config
        with open("config/transformer_test.yaml", "r") as file:
            cls.main_config = yaml.safe_load(file)
        cls.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
   
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        del cls.main_config
        del cls.device

    def test_compressive_transformer(self):
        tokenizer = TransfoXLTokenizer.from_pretrained(self.main_config["transformer_xl_tokenizer"]["path"])

        # Get dataloaders for training
        train_loader, valid_loader, _ = get_dataloader(tokenizer, **self.main_config["data_loader"])

        config = CompressiveTransformerConfig(vocab_size=tokenizer.vocab_size, **self.main_config["compressive_transformer"])
        model = CompressiveTransformerWithLMHead(config)
        model = model.to(self.device)

        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.main_config["optimizer"]["learning_rate"])

        try:
            # train
            train(model, train_loader, optimizer, valid_loader=valid_loader, device=self.device, **self.main_config["train"])
        except:
            self.fail("Could not train Compressive Transformer for 1 epoch")
    
    def test_transformer_xl(self):
        tokenizer = TransfoXLTokenizer.from_pretrained(self.main_config["transformer_xl_tokenizer"]["path"])

        # Get dataloaders for training
        train_loader, valid_loader, _ = get_dataloader(tokenizer, **self.main_config["data_loader"])

        # Create Model
        config = TransfoXLConfig(
            vocab_size=tokenizer.vocab_size,
            eos_token_id=tokenizer.eos_token_id,
            **self.main_config["transformer_xl"]
        )
        model = TransfoXLLMHeadModel(config)
        model = model.to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.main_config["optimizer"]["learning_rate"])

        try:
            train(model, train_loader, optimizer, valid_loader=valid_loader, device=self.device, **self.main_config["train"])
        except:
            self.fail("Could not train Transformer-XL for 1 epoch")
    
    def test_gpt(self):
        # Get tokenizer
        tokenizer = OpenAIGPTTokenizer(**self.main_config["gpt_tokenizer"])
        tokenizer.model_max_length = self.main_config["data_loader"]["max_length"]

        # Get Dataloaders processed by TransfoXLTokenizer
        train_loader, valid_loader, _ = get_dataloader(tokenizer, **self.main_config["data_loader"])

        config = OpenAIGPTConfig(
            vocab_size=tokenizer.vocab_size,
            n_positions=self.main_config["data_loader"]["max_length"],
            **self.main_config["gpt"]
        )

        model = OpenAIGPTLMHeadModel(config)
        model = model.to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.main_config["optimizer"]["learning_rate"])
        try:
            train(model, train_loader, optimizer, valid_loader=valid_loader, device=self.device, **self.main_config["train"])
        except:
            self.fail("Could not train OpenAIGPTLMHead model")
    

if __name__ == "__main__":
    unittest.main()


