import unittest

from transformers import (
    LongformerConfig,
    TransfoXLConfig,
    TransfoXLLMHeadModel
)

from longcontext.transformers.compressive_transformer import (
    CompressiveTransformerWithLMHead,
    CompressiveTransformerConfig
)
from longcontext.transformers.longformer import LongFormerLMHeadModel


class CreationTestCase(unittest.TestCase):

    def test_create_xl(self):
        try:
            config = TransfoXLConfig()
            model = TransfoXLLMHeadModel(config)
        except:
            self.fail("Transformer-XL can not be instantiated")
    
    def test_create_compressive(self):
        try:
            config = CompressiveTransformerConfig(4, 512)
            model = CompressiveTransformerWithLMHead(config)
        except:
            self.fail("Compressive Transformer could not be instantiated")
        
    def test_create_long(self):
        try:
            config = LongformerConfig()
            model = LongFormerLMHeadModel(config)
        except:
            self.fail("LongFormer could not be instantiated")


if __name__ == "__main__":
    unittest.main()
