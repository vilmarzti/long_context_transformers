from typing import List
import torch
from dataclasses import dataclass

from transformers.models.transfo_xl.modeling_transfo_xl import (
    TransfoXLModelOutput,
    TransfoXLLMHeadModelOutput,
)

from transformers.file_utils import ModelOutput

@dataclass
class GeneralOutput(ModelOutput):
    """General output format that is used when accumulating over outputs. Is used
    in Perplexity.

    Attrs:
        loss (torch.FloatTensor):
        prediction_loss (torch.FloatTensor): the scores before softmax used to predict the next words
    """
    loss: torch.FloatStorage = None
    prediction_loss : torch.FloatTensor = None


@dataclass
class CompressiveTransformerModelOutput(TransfoXLModelOutput):
    """Base Class for CompressiveTransformerModel output. It inherits from the
    TransfoXLModelOutput and adds a field for the compressed Memory

    Compare to:
        https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L606

    Additional Args:
        c_mem (List[torch.FloatTensor]): A list with the compressed memories for
            each layer in the C-Transformer

    """
    c_mems: List[torch.FloatTensor] = None
    mems_to_compress: List[torch.FloatTensor] = None


@dataclass
class CompressiveTransformerLMHeadModelOutput(TransfoXLLMHeadModelOutput):
    """Base class for CompressiveTransformerLMHeadModel output. It inherits
    from the TransfoXLLMHeadModelOutput and adds a field for the compressed
    memory.

    Compare to:
        https://github.com/huggingface/transformers/blob/4167519c3efa0f0fe867e82abe32c141147e0675/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L671

    Additional Args:
        c_mem (List[torch.FloatTensor]): A list with the compressed memories for
        each layer in the Compressive Transformer
    """
    prediction_loss: torch.FloatTensor = None
    prediction_scores: torch.FloatTensor = None
    attention_reconstruction_loss: torch.FloatTensor = None
    c_mems: List[torch.FloatTensor] = None

