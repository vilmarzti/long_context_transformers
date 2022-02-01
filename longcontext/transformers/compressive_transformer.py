""" Compressive Transformer Module
This module ports the Compressive Transformer from labml-ai
to huggingface for a more suitable training experience.
It contains all the needed classes for creating and training
the compressive Transformer

Compare to the docs:
    https://huggingface.co/docs/transformers/add_new_model

Code adapted from:
    GPT2: https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/gpt2/modeling_gpt2.py
    labml-ai : https://github.com/labmlai/annotated_deep_learning_paper_implementations
"""

from transformers import PreTrainedModel, PretrainedConfig
from torch import nn
from labml_nn.transformers.compressive import RelativeMultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, ):
        pass

    def forward(self):
        pass


class CompressiveBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dropout_rate = config.dropout_rate
        self.num_attention_heads = config.num_attention_heads

        self.self_attn = RelativeMultiHeadAttention(self.num_attention_heads, self.hidden_size, self.dropout_rate)
        self.feed_forward = nn.Fe
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, input_ids, past_keys_value=None, attention_mask=None, labels=None):
        pass


class CompressiveTransformerConfig(PretrainedConfig):
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)

        self.model_type = "compressive_transformer"
        self.is_composition = False
        self.keys_to_ignore_at_inference = []
        self.dropout = dropout


class CompressiveTransformerPretrainedModel(PreTrainedModel):
    config_class = CompressiveTransformerConfig

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

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