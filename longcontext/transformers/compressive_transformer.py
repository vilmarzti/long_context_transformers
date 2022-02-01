""" Compressive Transformer Module
This module ports the Compressive Transformer from labml-ai
to huggingface for a more suitable training experience.
It contains all the needed classes for creating and training
the compressive Transformer

Compare to the docs:
    https://huggingface.co/docs/transformers/add_new_model

Code adapted from:
    GPT2: https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/gpt2/
    labml-ai : https://github.com/labmlai/annotated_deep_learning_paper_implementations
"""

from transformers import PreTrainedModel, PretrainedConfig
from torch import nn
from labml_nn.transformers.compressive import RelativeMultiHeadAttention


class CompressiveTransformerConfig(PretrainedConfig):
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)

        self.model_type = "compressive_transformer"
        self.is_composition = False
        self.keys_to_ignore_at_inference = []
        self.dropout = dropout


class CompressiveFF(nn.Module):
    """This class implements the Feedforward block of the Compressive
    Transformer block. It has two linear layers and an activation, dropout
    in between.

    Compare to: 
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/feed_forward.py
        https://github.com/huggingface/transformers/blob/db7d6a80e82d66127b2a44b6e3382969fdc8b207/src/transformers/models/gpt2/modeling_gpt2.py#L349
    
    Attributes:
        hidden_size (int): The input dimension to the Feedforward block
        n_inner (int): The dimension the input expands to
        dropout_rate (int): The dropout rate in between the linear layers
        ff_1 (nn.Linear): The first linear layer
        ff_2 (nn.Linear): The second linear layer
        dropout (nn.Dropout): The dropout layer applied after the first linear layer
        activation (nn.ReLU): Relu applied after first linear layer
    """

    def __init__(self, config):
        """ Read config for creating the appropriate layers

        Args:
            config (CompressiveTransformerConfig): Config with the values for the FF-block.
                It should contain the values hidden_size, n_inner and config_rate
        """
        super().__init__()

        # Save appropriate Values
        self.hidden_size = config.hidden_size
        self.n_inner = config.n_inner if config.n_inner is not None else config.hidden_size * 4
        self.dropout_rate = config.dropout_rate

        # Create Layers for FeedForward-block
        self.ff_1 = nn.Linear(self.hidden_size, self.n_inner)
        self.ff_2 = nn.Linear(self.n_inner, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        """ Apply the FeedForward block

        Args:
            hidden_states (torch.tensor): The intermediate values in a compressive Transforer block

        Returns:
            torch.tensor: The values after the feed-forward block is applied
        """
        # Apply pass through first linear layer
        ff1 = self.activation(self.ff_1(hidden_states))

        # add dropout
        dropout = self.dropout(ff1)

        # Pass through second linear layer 
        ff2 = self.ff2(dropout)
        return ff2


class CompressiveBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dropout_rate = config.dropout_rate
        self.num_attention_heads = config.num_attention_heads

        self.self_attn = RelativeMultiHeadAttention(self.num_attention_heads, self.hidden_size, self.dropout_rate)
        self.feed_forward = CompressiveFF(config)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, input_ids, past_keys_value=None, attention_mask=None, labels=None):
        pass



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