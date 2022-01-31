""" This module ports the Compressive Transformer from labml-ai
to huggingface for a more suitable training experience
"""

from transformers import PreTrainedModel, PretrainedConfig


class CompressiveTransformerConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super.__init__(**kwargs)

        self.model_type = "compressive_transformer"
        self.is_composition = False
        self.keys_to_ignore_at_inference = []


class CompressiveTransformerPretrainedModel(PreTrainedModel):
    def __init__(self, config):
        pass

    def _init_weights():
        pass

class CompressiveTransfomerModel(CompressiveTransformerPretrainedModel):
    def __init__(self, config):
        pass
    
    def forward(self, input_ids, attention_masks):
        pass

class CompressiveTransformerWithLMHead(CompressiveTransformerPretrainedModel):
    def __init__(self, config):
        pass

    def forward(self, input_ids, attention_masks, labels):
        pass