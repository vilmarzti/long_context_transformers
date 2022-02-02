"""
    This module contains the class for the LongFormerWithLMHead
    as the current LongFormer model does not support
    autoregressive language modeling

    Resource for autoregressive:
        https://github.com/neqkir/attention-mechanism
        `Left-ward flow`in Attention is all you need
"""
from transformers import LongformerPreTrainedModel
from torch import nn


class LongFormerLMHeadModel(LongformerPreTrainedModel):
    """
    This class uses the LongFormer and places a linear layer at the end
    such that we can predict the next word in an autoregressive way.

    Note:
        Compare to implementation of GPT2LMHeadModel at
        https://github.com/huggingface/transformers/blob/05fa1a7ac17bb7aa07b9e0c1e138ecb31a28bbfe/src/transformers/models/gpt2/modeling_gpt2.py
    
    Attributes:
        lm_head    (torch.nn.Linear)        : The linear layer for predicting the next word
    """

    def __init__(self, config):
        """
        Initliazes a basic LongFormer Model and the linear layer on top

        Args:
            config (LongFormerConfig): The config for creation of the LongFormer. 
                The config should include the hidden size and the vocab_size
        """
        # Set up the Longformer
        super.__init__(config)

        # Set up the linear layer for LM
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Post cleanup
        self.post_init()
    
    def forward():
        pass