import unittest

from transformers import (
    AdamW,
    LongformerConfig,
    TransfoXLConfig,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer
)

from longcontext.transformers.compressive_transformer import (
    CompressiveTransformerWithLMHead,
    CompressiveTransformerConfig
)

from longcontext.transformers.longformer import LongFormerLMHeadModel

from longcontext.utils.dataset import get_dataloader
from longcontext.utils.training import train


class TrainingTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Get tokenizer
        cls.tokenizer = TransfoXLTokenizer.from_pretrained("data/tokenizer-xl-wiki2.json")

        # Get Dataloaders processed by TransfoXLTokenizer
        cls.train_loader, cls.valid_loader, _ = get_dataloader(cls.tokenizer, 4, 16)
    
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        del cls.tokenizer
        del cls.train_loader
        del cls.valid_loader


    def test_compressive_transformer(self):
        # Create Model
        config = CompressiveTransformerConfig(4, 10, vocab_size=self.tokenizer.vocab_size, n_layer=2, return_dict=True, output_hidden_states=True, cutoffs=[1000, 5000, 15000])
        model = CompressiveTransformerWithLMHead(config)
        
        # Set optimizer
        optimizer = AdamW(model.parameters())

        try:
            # train
            train(model, self.train_loader, optimizer, 1, self.valid_loader)
        except:
            self.fail("Could not train Compressive Transformer for 1 epoch")
    
    def test_transformer_xl(self):
        config = TransfoXLConfig(
            n_layer=6
        )
        model = TransfoXLLMHeadModel(config)

        # Set optimizer
        optimizer = AdamW(model.parameters())

        try:
            # train
            train(model, self.train_loader, optimizer, 1, self.valid_loader)
        except:
            self.fail("Could not train Transformer-XL for 1 epoch")
    
    @unittest.SkipTest
    def test_longformer(self):
        # Create model
        config = LongformerConfig(
            vocab_size=self.tokenizer.vocab_size,
            attention_window=64,
            max_position_embeddings=219,
            num_hidden_layers=6
        )
        model = LongFormerLMHeadModel(config)

        optimizer = AdamW(model.parameters())
        
        try:
            train(model, self.train_loader, optimizer, 1, self.valid_loader) 
        except:
            self.fail("Could not train long-former for 1 epoch")

if __name__ == "__main__":
    unittest.main()


