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
import torch.nn.functional as F
from torch import nn

from transformers import (
    TransfoXLConfig,
    TransfoXLPreTrainedModel
)

from transformers.models.transfo_xl.modeling_transfo_xl import (
    PositionwiseFF,
    RelPartialLearnableMultiHeadAttn,
    AdaptiveEmbedding,
    PositionalEmbedding,
)

from transformers.models.transfo_xl.modeling_transfo_xl_utilities import ProjectedAdaptiveLogSoftmax

from .outputs import (
    CompressiveTransformerLMHeadModelOutput,
    CompressiveTransformerModelOutput
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
        """Initialize the 1dConvolution that compresses the memory

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


class RelativeMultiheadAttention(RelPartialLearnableMultiHeadAttn):
    """Relative positional Attention based on the Transformer-XL
    self attention with additional method for attention-reconstruction
    loss
    """

    def content_based_attention(self, sequence, mem):
        """This performs the attention operation without any relative
        positional mechanism

        Args:
            sequence (torch.FloatTensor): The sequence we embedded
            mem (torch.FloatTensor): The memory from previous sequences

        Returns:
            torch.FloatTensor: The values we get after applying the
                attention mechanism.
        """
        # Transpose back to Transformer-XL shape
        sequence = sequence.transpose(1, 0).contiguous()

        # Read values for later processing
        query_length = sequence.size(0)
        batch_size = sequence.size(1)

        # Concatenate memory and sequence
        cat = torch.cat([mem, sequence], dim=0)

        # Disable gradient adjustments through qkv
        self.qkv_net.requires_grad_(False)

        # Apply lnorm if specified
        if self.pre_lnorm:
            w_heads = self.qkv_net(self.layer_norm(cat))
        else:
            w_heads = self.qkv_net(cat)

        # Enable gradients for qkv again
        self.qkv_net.requires_grad_(True)

        # Get Query, Key and Values
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        w_head_q = w_head_q[-query_length:]

        key_length = w_head_k.size(0)

        # TODO: check shapes
        # qlen x bsz x n_head x d_head
        w_head_q = w_head_q.view(
            query_length, batch_size, self.n_head, self.d_head)
        # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(
            key_length, batch_size, self.n_head, self.d_head)
        # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(
            key_length, batch_size, self.n_head, self.d_head)

        attn_score = torch.einsum("ibnd,jbnd->ijbn", (w_head_q, w_head_k))

        attn_prob = F.softmax(attn_score, dim=1)

        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))

        # TODO: check shape
        attn_vec = attn_vec.contiguous().view(attn_vec.size(
            0), attn_vec.size(1), self.n_head * self.d_head)

        return attn_vec


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

    def __init__(self, config, **kwargs):
        """Initialize the compressive Layer which performs relative self attention with a corresponding
        feed-forward block.

        Args:
            config (CompressiveTransformerConfig): Config with the appropriate values for instantiating
                the compressiv layer. Check The class definition for further explanations
        """
        super().__init__()

        # Read appropriate values from config
        self.dropout_rate = config.dropout
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.lnorm_epsilon = config.layer_norm_epsilon
        self.compression_rate = config.compression_rate
        self.pre_lnorm = config.pre_lnorm

        # Create Compression function
        self.compression = Conv1dCompression(
            self.compression_rate,
            self.d_model
        )

        # Create Layers for the CompressiveTransformer layer
        self.self_attn = RelativeMultiheadAttention(
            self.n_head,
            self.d_model,
            self.d_head,
            self.dropout_rate,
            **kwargs
        )

        # Feed Forward Layer after Multi-Head attention
        self.feed_forward = PositionwiseFF(
            self.d_model,
            self.d_inner,
            self.dropout_rate,
            self.pre_lnorm,
            self.lnorm_epsilon
        )

        self.norm_self_attn = nn.LayerNorm([self.d_model])

    def _concat_memories(self, mem, c_mem=None):
        """ Concatenate the tensors of input_ids, memory and compressed memory

        Args:
            mem (torch.tensor, optional): The memories from previous runs.
            c_mem (torch.tensor, optional): The compressed memories of previous runs. Defaults to None.

        Returns:
            torch.tensor: Concatenated version of memory and compressed memory
        """
        # Concatenate memory and compressed memory
        if c_mem is not None:
            mem = torch.cat((mem, c_mem), dim=0)

        return mem

    # TODO: Input dimensions
    def forward(self, input_ids, positional_embedding, mem=None, c_mem=None, attention_mask=None, output_attentions=False):
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
        combined = self._concat_memories(mem, c_mem)

        # Apply relative self attention
        attention_scores = self.self_attn(
            input_ids,
            positional_embedding,
            attn_mask=attention_mask,
            output_attentions=output_attentions,
            mems=combined
        )

        # Final feed forward block
        feed_forward = self.feed_forward(attention_scores[0])

        # TODO: Check `[1:]`
        outputs = [feed_forward] + attention_scores[1:]

        return outputs


class CompressiveTransformerPretrainedModel(TransfoXLPreTrainedModel):
    """The base class of the Commpressive Transformer Model it inherits from
    the TransfoXLPretrainedModel

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
    load_tf_weights = None


class CompressiveTransfomerModel(CompressiveTransformerPretrainedModel):
    """The base model for the Compressive Transformer. This will be used by
    other models further down the line

    Attrs:
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
        self.embedding_size = config.d_embed
        self.hidden_size = config.d_model
        self.num_heads = config.n_head
        self.cutoffs = config.cutoffs
        self.dropout_rate = config.dropout
        self.num_layer = config.n_layer
        self.mem_length = config.mem_len
        self.c_mem_length = config.c_mem_length
        self.same_length = config.same_length
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
        for _ in range(self.num_layer):
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
            param = next(self.parameters())

            # Fill Memory with zeros
            for _ in range(self.num_layer):
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

    def calculate_reconstruction_loss(self, layer, hidden_state, memory):
        """Calculates the reconstruction loss for a given layer

        Args:
            layer (CompressiveTransfomerLayer): The layer that includes the
                compression function and the attention mechanism
            hidden_state (torch.FloatTensor): The hidden states associated
                with the given layer
            memory (torch.FloatTensor): The memories to compress associated
                with the given layer.

        Returns:
            torch.FloatTensor:
        """
        # Detach embeddings and memories
        hidden_state = hidden_state.detach()
        memory = memory.detach()

        # Compress the current memory with the given layer
        c_memory = layer.compression(memory)

        # Perform detached normalization of h_state memory and c_memory
        hidden_state, memory, c_memory = (
            F.layer_norm(
                embed,
                layer.norm_self_attn.normalized_shape,
                weight=layer.norm_self_attn.weight.detach(),
                bias=layer.norm_self_attn.bias.detach(),
                eps=layer.norm_self_attn.eps
            ) for embed in [hidden_state, memory, c_memory]
        )

        # Apply self_attention without gradients
        attn_memory = layer.self_attn.content_based_attention(hidden_state, memory)
        attn_c_memory = layer.self_attn.content_based_attention(hidden_state, c_memory)

        layer_compression_loss = F.mse_loss(attn_memory, attn_c_memory)
        return layer_compression_loss

    def attention_reconstruction_loss(self, hidden_states, memories):
        """Generates the Attention-Reconstruction loss for each layer as
        described in the paper.

        Compare to:
            https://nn.labml.ai/transformers/compressive/index.html

        Args:
            hidden_states (List[torch.FloatTensor]): The computed hidden states
                for each layer.
            memories (List[torch.FloatTensor]): The memories that got compressed
                in the current step

        Returns:
            torch.FloatTensor: The summed up attention-reconstruction losses from
                each layer
        """
        losses = []
        for i, layer in enumerate(self.layers):
            losses.append(self.calculate_reconstruction_loss(
                layer,
                hidden_states[i],
                memories[i]
            ))

        return torch.sum(torch.stack(losses))

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
            memory = [torch.cat((m, x), dim=0)
                      for m, x in zip(memory, new_memory)]
        else:
            memory = new_memory

        # Compress the oldest memory
        if memory[0].shape[0] > self.mem_length:
            # Calculate the number of compressed memories to create
            num_c_mem = (memory[0].shape[0] - self.mem_length +
                        self.compression_rate - 1) // self.compression_rate

            # Number of memories to compress
            num_mems_to_compress = num_c_mem * self.compression_rate

            # Memories that need to be compressed
            mems_to_compress = []

            # Memories that don't need to be compressed
            uncompressed_mem = []

            # Split memory into needs to be compressed and doesn't need to be compressed
            for m in memory:
                cm, m = torch.split(
                    m,
                    [num_mems_to_compress, m.shape[0] - num_mems_to_compress]
                )

                mems_to_compress.append(cm)
                uncompressed_mem.append(m)

            # Assign new memory
            new_memory = uncompressed_mem

            # Compress appropriate memories
            new_c_memory = []
            for i, layer in enumerate(self.layers):
                new_c_memory.append(layer.compression(mems_to_compress[i]))

            # After compressing, concat with old compressed
            if compressed_memory:
                compressed_memory = [
                    torch.cat([m, nm], dim=0) for m, nm in zip(
                        compressed_memory, new_c_memory)]
            else:
                compressed_memory = new_c_memory

            # Truncate compressed memory to given length
            if compressed_memory[0].shape[0] > self.c_mem_length:
                compressed_memory = [m[-self.c_mem_length:]
                                     for m in compressed_memory]
        else:
            mems_to_compress = []

        # Also return the mems_to_compress for Reconstruction loss
        return new_memory, compressed_memory, mems_to_compress

    def forward(self, input_ids, mems=None, c_mems=None, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=None):
        """ The forward pass in the base Compressive Transformer layer

        Compare to:
            https://github.com/huggingface/transformers/blob/db7d6a80e82d66127b2a44b6e3382969fdc8b207/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L878
            https://nn.labml.ai/transformers/compressive/index.html

        Args:
            input_ids (torch.Tensor): Tensor with input ids from tokenization
            mems (List[torch.Tensor], optional): The previous hidden states as
                a list, where each entry corresponds to a layer. Defaults to None.
            c_mems (List[torch.Tensor], optional): The previous compressed hidden
                states. A list with torch.Tensor's where each entry corresponds 
                to a layer Defaults to None.
            head_mask (torch.Tensor, optional): The mask that says which heads
                exclude. Defaults to None.
            output_attentions (bool, optional): Whether to output the attention
                scores. Defaults to False.
            output_hidden_states (bool, optional): Whether to output the hidden
                states . Defaults to False.
            return_dict (bool, optional): Whether the output should be a dict
                or a tuple. If true a dict is returned. Defaults to False.

        Raises:
            ValueError: If no input_ids have been supplied

        Returns:
            (CompressiveTransformerModelOutput or tuple): The calculated values. 
            See CompressiveTransformerModelOutput for values that get returned
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
        if input_ids is not None and input_ids.dtype == torch.int64:
            input_ids = input_ids.transpose(0, 1).contiguous()
            query_length, batch_size = input_ids.shape
        else:
            raise ValueError("input_ids has to be specified for forward pass")
        
        # Initialize memories if not given
        if mems is None or len(mems) == 0:
            mems = self.init_memories(self.mem_length, batch_size)

        # Intialize compressed memories if not given
        if c_mems is None or len(c_mems) == 0:
            c_mems = self.init_memories(self.c_mem_length, batch_size)

        # Get head_mask into appropriate size. Currently only one dimension is supported
        # That means that we disable the heads for all layers in the same way
        if head_mask is not None:
            head_mask = head_mask.unsqueeze(
                0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            head_mask = head_mask.expand(self.num_layer, -1, -1, -1, -1)
        else:
            head_mask = [None] * self.num_layer

        # Put input_ids into embeddings
        input_embeddings = self.word_emb(input_ids)

        # Get appropriate memory lengths. This is necessary if the parameters mems/c_mems
        # Are not given (i.e. they are passed as None)
        mem_length = mems[0].size(0) if mems is not None else 0
        c_mem_length = c_mems[0].size(0) if c_mems is not None else 0
        key_length = mem_length + c_mem_length + query_length

        # Resources:
        #   https://medium.com/analytics-vidhya/masking-in-transformers-self-attention-mechanism-bad3c9ec235c
        #   https://atcold.github.io/pytorch-Deep-Learning/en/week12/12-1/
        auto_reg_attention_mask = torch.triu(
            input_embeddings.new_ones((query_length, key_length),
            dtype=torch.uint8),
            diagonal=1 + mem_length + c_mem_length
        )[:,:,None]

        att_mask = auto_reg_attention_mask
        
        # PREPARE FORWARD-PASS

        # Create a tensor with the given positions
        position_sequence = torch.arange(
            key_length - 1,  # length
            -1,         # start
            -1,         # step
            # -1.0,       # TODO: what parameter is this? Check what happens if removed
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
        hidden_states = []
        attentions = [] if output_attentions else None
        for i, layer in enumerate(self.layers):
            hidden_states.append(hidden_state)

            # Get appropriate (compressed) memory for the given layer
            current_memory = None if mems is None else mems[i]
            current_c_memory = None if c_mems is None else c_mems[i]

            # Pass through layer
            layer_output = layer(
                hidden_state,
                position_embeddings,
                mem=current_memory,
                c_mem=current_c_memory,
                attention_mask=att_mask,
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
        hidden_states.append(hidden_state)

        # Create new memories from the computed hidden states
        new_mems, new_c_mems, mems_to_compress = self.merge_and_compress(mems, c_mems, hidden_states)

        # Set up for library standard shape [bsz, len, hidden_dim]
        hidden_states = tuple(v.transpose(1, 0).contiguous()
                              for v in hidden_states)

        # Process attentions if part of output
        if output_attentions:
            # Transpose to library standard shape [bsz, n_heads, query_seq_len, key_seq_len]
            attentions = tuple(t.permute(2, 3, 0, 1).contiguous()
                               for t in attentions)

        # Process output to library standard shape
        final_output = final_output.transpose(1, 0).contiguous()

        # One version of the output
        if not return_dict:
            return tuple(v for v in [final_output, new_mems, new_c_mems, mems_to_compress, hidden_states, attentions] if v is not None)

        return CompressiveTransformerModelOutput(
            last_hidden_state=final_output,
            mems=new_mems,
            c_mems=new_c_mems,
            mems_to_compress=mems_to_compress,
            hidden_states=hidden_states,
            attentions=attentions
        )


class CompressiveTransformerWithLMHead(CompressiveTransformerPretrainedModel):
    """ The decoder with a linear-layer and projected adaptive logsoftmax on top
    Note that the linear projection is included in this version of the softmax.

    Atrr:
        transfomer (CompressiveTransformerModel): The base C-Transfomer model.
        crit (ProjectedAdaptiveLogSoftmax) : The softmax function for after the.
    """
    def __init__(self, config):
        super().__init__(config)

        self.transformer = CompressiveTransfomerModel(config)

        # Create an addaptive Softmax layer
        self.crit = ProjectedAdaptiveLogSoftmax(
            config.vocab_size,
            config.d_embed,
            config.d_model,
            config.cutoffs,
            config.div_val
        )

        self.post_init()

    def forward(self, input_ids, mems=None, c_mems=None, head_mask=None, labels=None, output_attentions=False, output_hidden_states=False, return_dict=None):
        """Makes one forwards pass for a given input sequence

        Args:
            input_ids (torch.LongTensor): The input token ids. Has size (B, S). Where
                B is the batch size and S is the sequence-length.
            mems (List[torch.FloatTensor], optional): A list with the previous hidden states for each layer.
                Defaults to None.
            c_mems (_type_, optional): A list with the compressed memories for each layer.
                Defaults to None.
            head_mask (torch.FloatTensor, optional): See Transformer-XL for documentation. Defaults to 
                None
            labels (torch.LongTensor, optional): The token ids of the desired output. Defaults to None.
            output_attentions (bool, optional): Whether to output the attentions. Defaults to False.
            output_hidden_states (bool, optional): Whether we should output the hidden-states. Defaults to False.
            return_dict (bool, optional): If we should return a dict. Defaults to None.

        Returns:
            (tuple or CompressiveTransformerLMHeadModelOutpu): A tuple or dict with the outputs we desired.
        """
        # Decide whether to return a dict or a tuple
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if input_ids is not None and input_ids.dtype == torch.int64:
            batch_size = input_ids.size(0)
            sequence_length = input_ids.size(1)

        # Forward pass through base Compressive Transformer
        transformer_output = self.transformer(
            input_ids,
            mems=mems,
            c_mems=c_mems,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # Read values from tuple/dict
        if return_dict:
            last_hidden_state = transformer_output.last_hidden_state
            mems = transformer_output.mems
            mems_to_compress = transformer_output.mems_to_compress
            hidden_states = transformer_output.hidden_states
        else:
            last_hidden_state = transformer_output[0]
            mems = transformer_output[1]
            mems_to_compress = transformer_output[3]
            hidden_states = transformer_output[4]


        # Compute losses
        ce_losses = self.crit(last_hidden_state, labels)

        prediction_scores = ce_losses.view(
            batch_size, sequence_length, -1
        ) if labels is None else ()

        ce_losses = ce_losses.view(
            batch_size, sequence_length - 1
        ) if labels is not None else None
        
        attention_reconstruction_loss = None
        if len(mems_to_compress) > 0:
            attention_reconstruction_loss = self.transformer.attention_reconstruction_loss(
                hidden_states=hidden_states,
                memories=mems_to_compress
            )

        # Accumulate prediction and reconstruction loss
        if labels is not None:
            loss = ce_losses.mean() + attention_reconstruction_loss if attention_reconstruction_loss else ce_losses.mean()
        elif attention_reconstruction_loss:
            loss = attention_reconstruction_loss
        else:
            loss = None
        
        if loss is not None:
            loss = loss.unsqueeze(0).unsqueeze(0)
        
        if not return_dict:
            output = (prediction_scores,) + transformer_output[1:]
            return ((loss,) + output) if loss is not None else output

        return CompressiveTransformerLMHeadModelOutput(
            losses=loss,
            prediction_loss=ce_losses,
            attention_reconstruction_loss=attention_reconstruction_loss,
            prediction_scores=prediction_scores,
            mems=transformer_output.mems,
            hidden_states=transformer_output.hidden_states,
            attentions=transformer_output.attentions,
            c_mems=transformer_output.c_mems
        )
