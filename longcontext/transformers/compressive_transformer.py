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

from transformers import (
    PreTrainedModel,
    TransfoXLConfig
)

from transformers.models.transfo_xl.modeling_transfo_xl import (
    RelPartialLearnableMultiHeadAttn,
    AdaptiveEmbedding,
    PositionalEmbedding
)


class CompressiveTransformerConfig(TransfoXLConfig):
    """The configuration for the Compressive transformer. As most of the 
    Code for the C-transformer is taken from the Transfomer-XL implementation
    of Huggingface, the config is very similar.

    Compare to:
        https://huggingface.co/docs/transformers/model_doc/transfo-xl#transformers.TransfoXLConfig

    Attributes:
        c_mem_length (int): How many instances of compressed memory there
            are.
        compression_rate (int): How strongly the memories get compressed
    """
    def __init__(self, compression_rate=4, c_mem_length=None, **kwargs):
        """See Class doc-string for further information

        Args:
            compression_rate (int, optional): By which factors the memories get
                compressed. Defaults to 4.
            c_mem_length (int, optional): How many instances of compressed
                memory there are. Defaults to None.
            kwargs (dict): Additional settings for the TransfoXLConfig or 
                pretrained config.
        """
        # Init TransfoXLConfig
        super().__init__(**kwargs)

        # Set necessary properties
        self.model_type = "compressive_transformer"
        self.is_composition = False
        self.keys_to_ignore_at_inference = []

        # Set properties specific to Compressive transformer
        self.c_mem_length = c_mem_length
        self.compression_rate = compression_rate

        # Rename some of the properties for readability
        self.embedding_size = kwargs.get("d_embed")
        self.hidden_size = kwargs.get("d_model")
        self.head_size = kwargs.get("n_heads")
        self.num_heads = kwargs.get("n_heads")
        self.dropout_rate = kwargs.get("dropout")
        self.num_layers = kwargs.get("n_layers")
        self.clamp_length = kwargs.get("clamp_len")


class Conv1dCompression(nn.Module):
    """One of the compression functions used in the Compressive Layer
    of a Transformer. It applies a 1d convultion

    Compare to:
        https://nn.labml.ai/transformers/compressive/index.html
    
    Attributes:
        conv (nn.Conv1d): A 1d-convolution used for compression of old
            memories
    """
    def __init__(self, compression_rate, hidden_size):
        """[summary]

        Args:
            compression_rate (int): By which factor the memories need to be compressed
            hidden_size (int): The size of the tokens in the transformer
        """
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=compression_rate,
            stride=compression_rate
        )

    def forward(self, memories):
        """ Compress Memories with 1d-convolution

        Args:
            memories (torch.Tensor): The memories we want to compress

        Returns:
            torch.Tensor: The compressed memories
        """
        # TODO: Check shape
        memories = memories.permute(1, 2, 0)
        memories = self.conv(memories)
        memories = memories.permute(2, 0, 1)
        return memories


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

    # TODO: Input dimensions
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
        feed_forward (nn.Module): The feed-forward block in this layer
        norm_self_attn (nn.Module): The norm that will be applied before going through the attention
        compression (nn.Module): The compression function that compresses new memories. Currently only
            1d-convolution is supported
    """
    def __init__(self, config):
        """Initialize the compressive Layer which performs relative self attention with a corresponding
        feed-forward block.

        Args:
            config (CompressiveTransformerConfig): Config with the appropriate values for instantiating
                the compressiv layer. Check The class definition for further explanations
        """
        super().__init__()

        # Read appropriate values from config
        self.hidden_size = config.hidden_size
        self.dropout_rate = config.dropout_rate
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.compression_rate = config.compression_rate

        # Create Compression function
        self.compression = Conv1dCompression(self.compression_rate, self.hidden_size)

        # Create Layers for the CompressiveTransformer layer
        self.self_attn = RelPartialLearnableMultiHeadAttn(
            self.num_heads,
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

    # TODO: Input dimensions
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
        # Initialize uniformly
        if self.config.init == "uniform":
            nn.init.uniform_(weight, -self.config.init_range, self.config.init_range)
        # Initialize with normal around 0 and specified std
        elif self.config.init == "normal":
            nn.init.normal_(weight, 0.0, self.config.init_std)
        # Raise error if initialization method is not supported
        else:
            raise ValueError("The `init` in config has been set to an unkown initilization")

        
class CompressiveTransfomerModel(CompressiveTransformerPretrainedModel):
    """The base model for the Compressive Transformer. This will be used by 
    other models further down the line

    Args:
        config_args (any): Config settings that are saved by the model for 
            later usage. See the CompressiveTransformerConfig and the 
            TransfoXLConfig for further details.
        word_emb (nn.Module): Embedding layer for the tokens that get fed into
            the transformer
        dropout (nn.Module): The dropout layer that is applied to the input-/
            positional embeddings.
        layers (nn.ModuleList): List of the compressive layers in this transformer
        pos_embedding (nn.Module): A module that creates the tensors for the 
            positional embedding.
        
    """
    def __init__(self, config):
        """ Initialize properties and layers from a config

        Args:
            config (CompressiveTransformerConfig):See the 
                CompressiveTransformerConfig and the TransfoXLConfig for further 
                details.
        """
        super().__init__(config)

        # Get Properties form config for later use.
        # CompressiveTransformerConfig for details
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.cutoffs = config.cutoffs
        self.dropout_rate = config.dropout_rate
        self.num_layers = config.num_layers
        self.mem_length = config.mem_length
        self.c_mem_length = config.c_mem_length
        self.same_length = config.self_length
        self.clamp_length = config.clamp_len
        self.compression_rate = config.compression_rate

        # Initiate word embedding
        self.word_emb = AdaptiveEmbedding(
            self.vocab_size,
            self.embedding_size,
            self.hidden_size,
            self.cutoffs,
            div_val=config.div_val
        )

        # Dropout layer for later use
        self.dropout = nn.Dropout(self.dropout_rate)

        # Add layers to the model
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(CompressiveLayer(config))

        # Add positional embedding (as described in original Transformer-XL paper)
        self.pos_embedding = PositionalEmbedding(self.hidden_size)

        self.post_init()
    
    def init_memories(self, memory_size, batch_size):
        """Initialize memory tensors with appropriate size

        Compare to:
            https://github.com/huggingface/transformers/blob/db7d6a80e82d66127b2a44b6e3382969fdc8b207/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L839

        Args:
            memory_size (int): How many instances of this memory there are.
            batch_size (int): The batch_size of the current run.

        Returns:
            List[torch.Tensor]: A list of tensors which are filled by zeros.
            TODO: shape of return value
        """
        if memory_size > 0:
            # Initialize memory and read parameters for checking
            # Device and dtype
            memory = []
            param = next(self.parameters)

            # Fill Memory with zeros
            for _ in range(self.num_layers):
                memory.append(
                    torch.zeros(
                        memory_size,
                        batch_size,
                        self.hidden_size,
                        dtype=param.dtype,
                        device=param.device
                    )
                )
            return memory
        else:
            return None
        
    @torch.no_grad()
    def merge_and_compress(self, memory, compressed_memory, new_memory):
        """ Merge the current memory and compressed memory with the new memory.
        Apply compression if necessary.

        Compare to:
            https://nn.labml.ai/transformers/compressive/experiment.html

        Args:
            memory (List[torch.Tensor]): Old memory from the previous step
            compressed_memory (List[torch.Tensor]): Old compressed memory from
                the previous step.
            new_memory ([type]): The new hidden states that should be added to 
                the memory.

        Returns:
            (List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]): A 
            3-tuple that consists of the new (uncompressed) memory, the new
            compressed memory and the memories that have been compressed. 
            The last part of this tuple is needed for the attention-reconstruction
            loss
        """
        if memory is None and new_memory is None:
            return None, None
        
        # Concatenate old memories with new memories
        if memory:
            memory = [torch.cat((m, x), dim=0) for m, x in zip(memory, new_memory)]
        else:
            memory = new_memory
        
        # Compress the oldest memory
        if len(memory[0]) > self.mem_length:
            # Calculate the number of compressed memories to create
            num_c_mem = (len(memory[0]) - self.mem_length + self.compression_rate - 1) // self.compression_rate

            # Number of memories to compress
            num_mem_to_compress = num_c_mem * self.compression_rate

            # Memories that need to be compressed
            mem_to_compress = []

            # Memories that don't need to be compressed
            uncompressed_mem = []

            # Split memory into needs to be compressed and doesn't need to be compressed
            for m in memory:
                cm, m = torch.split(m, [num_mem_to_compress, len(m) - num_mem_to_compress])

                mem_to_compress.append(cm)
                uncompressed_mem.append(m)
            
            # Assign new memory
            new_memory = uncompressed_mem

            # Compress appropriate memories
            new_c_memory = []
            for i, layer in enumerate(self.layers):
                new_c_memory = layer.compress(mem_to_compress[i])
            
            # After compressing, concat with old compressed
            if compressed_memory:
                compressed_memory = [torch.cat([m, nm], dim=0) for m, nm in zip(compressed_memory, new_c_memory)]
            else:
                compressed_memory = new_c_memory 

            # Truncate compressed memory to given length
            if len(compressed_memory[0]) > self.c_mem_length:
                compressed_memory = [m[-self.c_mem_length:] for m in compressed_memory]#
        else:
            mem_to_compress = []

        # Also return the mem_to_compress for Reconstruction loss
        return new_memory, compressed_memory, mem_to_compress


    def forward(self, input_ids, mems=None, c_mems=None, head_mask=None, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False):
        """ The forward pass in the base Compressive Transformer layer

        Compare to:
            https://github.com/huggingface/transformers/blob/db7d6a80e82d66127b2a44b6e3382969fdc8b207/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L878
            https://nn.labml.ai/transformers/compressive/index.html

        TODO: mention Input-dimension
        Args:
            input_ids (torch.Tensor): Tensor with input ids from tokenization
            mems (List[torch.Tensor], optional): The previous hidden states as
                a list, where each entry corresponds to a layer. Defaults to None.
            c_mems (List[torch.Tensor], optional): The previous compressed hidden
                states. A list with torch.Tensor's where each entry corresponds 
                to a layer Defaults to None.
            head_mask (torch.Tensor, optional): The mask that says which heads
                exclude. Defaults to None.
            attention_mask (torch.Tensor, optional): Tensor that says which 
                tokens are padding and should be ignored. 1 for keep and 0
                for ignore. Defaults to None.
            output_attentions (bool, optional): Whether to output the attention
                scores. Defaults to False.
            output_hidden_states (bool, optional): Whether to output the hidden
                states . Defaults to False.
            return_dict (bool, optional): Whether the output should be a dict
                or a tuple. If true a dict is returned. Defaults to False.

        Raises:
            ValueError: If no input_ids have been supplied

        TODO: Prepare dict class and modify next lines.
        Returns:
            dict or tuple: The calculated values.
        """
        # Set output_attentions if not passed into function
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # Set output_hidden_states if not passed into function
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Set return_dict value if not passed into function
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Transpose for unified library interface. See comment in TransfoXLModel
        if input_ids is not None:
            input_ids = input_ids.transpose(0, 1).contigous()
            query_length, batch_size = input_ids.shape
        else:
            raise ValueError("input_ids has to be specified for forward pass")
        
        # Initialize memories if not given
        if mems is None:
            mems = self.init_memories(self.mem_length, batch_size)
        
        # Intialize compressed memories if not given
        if c_mems is None:
            c_mems = self.init_memories(self.c_mem_length, batch_size)

        # Get head_mask into appropriate size. Currently only one dimension is supported
        # That means that we disable the heads for all layers in the same way
        if head_mask is not None and head_mask.dim == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
        else:
            head_mask = [None] * self.num_layers

        # Put input_ids into embeddings
        input_embeddings = self.word_emb(input_ids)

        # TODO: Check following if-clause of the code and how attention_masks work
        # Resources:
        #   https://medium.com/analytics-vidhya/masking-in-transformers-self-attention-mechanism-bad3c9ec235c
        # 

        # Get appropriate memory lengths. This is necessary if the parameters mems/c_mems
        # Are not given (i.e. they are passed as None)
        mem_length = mems[0].size(0) if mems is not None else 0
        c_mem_length = c_mems[0].size(0) if c_mems is not None else 0
        key_length = mem_length + c_mem_length + query_length

        # If we use the same attention length for all tokens
        # Taken from the source code of the TransfoXLConfig
        if self.same_length and attention_mask is None:
            # Get tensor of shape [query_length, key_length] with the device set the same as the
            # input embeddings
            all_ones = input_embeddings.new_ones(
                (query_length, key_length),
                dtype=torch.uint8
            )

            # TODO: Is it really the mask length
            # Get the length of the mask
            mask_length = key_length - self.mem_length - self.c_mem_length

            # TODO: Why shift?
            if mask_length > 0:
                mask_shift_len = query_length - mask_length
            else:
                mask_shift_len = query_length

            # TODO: Check attention mask (sum of upper and lower triangular matrix?)
            # Get the attention mask
            attention_mask = (
                torch.triu(all_ones, 1 + mask_length) + 
                torch.tril(all_ones, -mask_shift_len)
            )[:, :, None]

        # If attention_mask is already provided, do nothing
        elif attention_mask is not None:
            pass

        # If same_length for all tokens is not set
        else:
            attention_mask = torch.triu(
                input_embeddings.new_ones(
                    (query_length, key_length),
                    dtype=torch.uint8
                ),
                diagonal=1 + mem_length + c_mem_length
            )[:, :, None]
        
        # PREPARE FORWARD-PASS

        # Create a tensor with the given positions
        position_sequence = torch.arange(
            key_length, # length
            -1,         # start
            -1,         # step
            -1.0,       # TODO: what parameter is this? Check what happens if removed
            device=input_embeddings.device,
            dtype=input_embeddings.dtype
        )

        # Clamp to clamp_length
        if self.clamp_length > 0:
            position_sequence.clamp_(max=self.clamp_length)

        # Get positional embedding
        position_embeddings = self.pos_embedding(position_sequence)

        # Apply dropout to positional-/word-embeddings
        input_embeddings = self.dropout(input_embeddings)
        position_embeddings = self.dropout(position_embeddings)

        # FORWARD_PASS

        hidden_state = input_embeddings
        hidden_states = [] if output_hidden_states else None
        attentions = [] if output_attentions else None
        for i, layer in enumerate(self.layers):
            # Save hidden state
            if output_hidden_states:
                hidden_states.append(hidden_state)

            # Get appropriate (compressed) memory for the given layer
            current_memory = None if mems is None else mems[i]
            current_c_memory = None if c_mems is None else c_mems[i]

            # Pass through layer
            layer_output = layer(
                input_embeddings,
                position_embeddings,
                mem=current_memory,
                c_mem=current_c_memory,
                attention_mask=attention_mask,
                output_attentions=output_attentions
            )

            # Get the hidden state for the next layer
            hidden_state = layer_output[0]

            # Add attention if it is needed in output
            if output_attentions:
                attentions.append(layer_output[1])
        
        # Apply dropout before outputting
        final_output = self.dropout(hidden_state)

        # Process hidden states if part of output
        if output_hidden_states:
            hidden_states.append(hidden_state)
            # Set up for library standard shape [bsz, len, hidden_dim]
            hidden_states = tuple(v.transpose(1, 0).contiguous() for v in hidden_states)

        # Process attentions if part of output
        if output_attentions:
            # Transpose to library standard shape [bsz, n_heads, query_seq_len, key_seq_len]
            attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)
        
        # Process output to library standard shape
        final_output = final_output.transpose(1, 0).contiguous()

        # Create new memories from the computed hidden states
        new_mems, new_c_mems, mem_to_compress = self.merge_and_compress(mems, c_mems, hidden_states)

        # One version of the output
        if not return_dict:
            return tuple(v for v in [final_output, new_mems, new_c_mems, mem_to_compress, hidden_states, attentions] if v is not None)
        
        # TODO: CompressiveTransformer specific output-dict


class CompressiveTransformerWithLMHead(CompressiveTransformerPretrainedModel):
    def __init__(self, config):
        pass

    def forward(self, input_ids, attention_masks, labels):
        pass