""" Compressive Transformer Module
This module ports the Compressive Transformer from labml-ai
to huggingface for a more suitable training experience.
It contains all the needed classes for creating and training
the compressive Transformer

Compare to the docs:
    https://huggingface.co/docs/transformers/add_new_model

Code adapted from:
    GPT2: https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/gpt2/
    labml-ai: https://github.com/labmlai/annotated_deep_learning_paper_implementations
    Transfomer-XL: https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/transfo_xl/modeling_transfo_xl.py
"""
import torch
from torch import nn

from transformers import PreTrainedModel, PretrainedConfig
from transformers.models.transfo_xl.modeling_transfo_xl import RelPartialLearnableMultiHeadAttn, AdaptiveEmbedding


class CompressiveTransformerConfig(PretrainedConfig):
    def __init__(self, dropout_rate, head_size, init="uniform", init_range=0.01, init_std=0.02, **kwargs):
        super().__init__(**kwargs)

        self.model_type = "compressive_transformer"
        self.is_composition = False
        self.keys_to_ignore_at_inference = []
        self.dropout_rate = dropout_rate
        self.head_size = head_size
        self.init = init
        self.init_range = init_range
        self.std_init = init_std 


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


class CompressiveLayer(nn.Module):
    """A layer of the Compressive transfomer. It uses the RelPartialLearnableMultiheadAttn
    from the Transformer-XL and the above defined CompressiveFF.

    Compare to:
        https://github.com/huggingface/transformers/blob/db7d6a80e82d66127b2a44b6e3382969fdc8b207/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L376
        https://nn.labml.ai/transformers/compressive/index.html

    Attributes:
        hidden_size (int): The dimensionality of the encoded sequence between layer (input, output)
        dropout_rate (int): The rate with which to perform dropout
        num_attention_heads (int): The number of heads in this layer
        head_size (int): The size of the current head
        self_attn (RelPartialLearnableMultiHeadAttn): The Class which applies the relative attention
        feed_forward: The feed-forward block in this layer
        norm_self_attn: The norm that will be applied before going through the attention
    """
    def __init__(self, config):
        """Initialize the compressive Layer which performs relative self attention with a corresponding
        feed-forward block.

        Args:
            config (CompressiveTransformerConfig): Config with the appropriate values for instantiating
                the compressiv layer
        """
        super().__init__()

        # Read appropriate values from config
        self.hidden_size = config.hidden_size
        self.dropout_rate = config.dropout_rate
        self.num_attention_heads = config.num_attention_heads
        self.head_size = config.head_size

        # Create Layers for the CompressiveTransformer layer
        self.self_attn = RelPartialLearnableMultiHeadAttn(
            self.num_attention_heads,
            self.hidden_size,
            self.head_size,
            self.dropout_rate,
        )
        self.feed_forward = CompressiveFF(config)
        self.norm_self_attn = nn.LayerNorm([self.hidden_size])
    
    def _concat_memories(self, input_ids, mem=None, c_mem=None):
        """ Concatenate the tensors of input_ids, memory and compressed memory

        Args:
            input_ids (torch.tensor): The encoded sequence
            mem (torch.tensor, optional): The memories from previous runs. Defaults to None.
            c_mem (torch.tensor, optional): The compressed memories of previous runs. Defaults to None.

        Returns:
            torch.tensor: Concatenated version of input_ids, memory and compressed memory
        """
        # Concatenate memory and compressed memory
        if mem is None:
            return input_ids
        elif c_mem is not None:
            # TODO: Look at dimension
            mem = torch.cat((mem, c_mem), dim=0)
        
        # TODO how to order norm
        norm = self.norm_self_attn(mem)

        # Combine input_ids with memory and compressed memory
        combined = torch.cat((input_ids, norm), dim=0)

        return combined

    def forward(self, input_ids, positional_embedding, mem=None, c_mem=None, attention_mask=None, output_attention=False):
        """Forward pass thorught this Compressive Transformer layer

        Args:
            input_ids (torch.tensor): The encoded sequence.
            positional_embedding (torch.tensor): The embeddings used for the relative positional encoding
            mem (torch.tensor, optional): The memories from previous runs. Defaults to None.
            c_mem (torch.tensor, optional): The compressed memories from previous runs. Defaults to None.
            attention_mask (torch.tensor, optional): Attention mask on which tokens to attend. The values
                should be either 0 or 1. Defaults to None.
            output_attention (bool, optional): Whether to output attentions. Defaults to False.

        Returns:
            torch.tensor: The encoded sequence after a forward pass through this layer
        """
        # Norm before attention
        norm_attn = self.norm_self_attn(input_ids)
        combined = self._concat_memories(norm_attn, mem, c_mem)

        # Apply relative self attention
        attention_scores = self.self_attn(
            combined,
            positional_embedding,
            attn_mask=attention_mask,
            output_attention=output_attention
        )

        # Final feed forward block
        feed_forward = self.feed_forward(attention_scores[0])
        outputs = [feed_forward] + attention_scores[1:]

        return outputs


class CompressiveTransformerPretrainedModel(PreTrainedModel):
    """The base class of the Commpressive Transformer Model

    Compare to:
        https://github.com/huggingface/transformers/blob/db7d6a80e82d66127b2a44b6e3382969fdc8b207/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L464

    Class Attributes:
        config_class (PreTrainedconfig): The type of config this Model accepts
        is_parallelizable (bool): Whether this Model support parallelization.
        base_model_prefix (str): The identifier of the base-model
        main_input_name (str): How to identify the main input
    """
    config_class = CompressiveTransformerConfig
    is_parallelizable = False
    base_model_prefix = "transformer"
    main_input_name = "input_ids"

    def _init_weights(self, weight):
        """How to initialize the weights in this model. We can either do it 
        through a normal-function or initialize uniformly. This function is
        typically called through `self.apply(self._init_weights)`

        Args:
            weight (torch layer): The layer we want to initialize

        Raises:
            ValueError: If the config doesn't have the appropriate values set
        """
        if self.config.init == "uniform":
            nn.init.uniform_(weight, -self.config.init_range, self.config.init_range)
        elif self.config.init == "normal":
            nn.init.normal_(weight, 0.0, self.config.init_std)
        else:
            raise ValueError("The `init` in config has been set to an unkown initilization")

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