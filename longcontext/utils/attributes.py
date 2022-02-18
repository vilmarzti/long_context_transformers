from transformers.models.transfo_xl.modeling_transfo_xl import TransfoXLLMHeadModelOutput 

from .config import transformer_xl, compressive_transformer, longformer

from longcontext.transformers.compressive_transformer import CompressiveTransformerLMHeadModelOutput
from longcontext.transformers.longformer import LongFormerLMHeadModelOutput 

def get_attribute(output, key):
    out = None
    if isinstance(output, TransfoXLLMHeadModelOutput):
        if key in  transformer_xl.keys():
            out = output[transformer_xl[key]]

    elif isinstance(output, CompressiveTransformerLMHeadModelOutput):
        if key in compressive_transformer.keys():
            out = output[compressive_transformer[key]]

    elif isinstance(output, LongFormerLMHeadModelOutput):
        if key in longformer.keys() :
            out = output[longformer[key]]
    
    else:
        raise ValueError(f"Output of type {type(output)} not found.")
    
    if out is None:
        out = output[key]

    return out