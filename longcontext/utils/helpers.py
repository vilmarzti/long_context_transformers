import numpy as np
import torch

from transformers import TransfoXLLMHeadModel
from transformers.models.transfo_xl.modeling_transfo_xl import TransfoXLLMHeadModelOutput 

from longcontext.transformers.compressive_transformer import CompressiveTransformerLMHeadModelOutput, CompressiveTransformerWithLMHead
from longcontext.transformers.longformer import LongFormerLMHeadModelOutput 

from .config import transformer_xl, compressive_transformer, longformer


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
        try:
            out = output[key]
        except KeyError:
            raise ValueError(f"Output of type {type(output)} not found.")
    
    if out is None:
        out = output[key]

    return out

def construct_args(model, input_ids, attention_mask, memories=None, output_labels=True):
    if isinstance(model, TransfoXLLMHeadModel):
        args = [input_ids]
        kwargs = {
            "mems": memories["mems"]
        }
    elif isinstance(model, CompressiveTransformerWithLMHead):
        args = [input_ids]
        kwargs = {
            "attention_mask": attention_mask,
            "mems": memories["mems"],
            "c_mems": memories["c_mems"]
        }
    else:
        ValueError(f"Function for model of type {type(model)} has not been implemented")
    
    if output_labels:
        kwargs["labels"] = input_ids
    
    return args, kwargs

def forward_pass(model, input_ids, attention_mask, subsequence_len=-1, use_labels=True):
    # Transformer-XL and Compressive Transformer learn with memories over the sequence
    if isinstance(model, (TransfoXLLMHeadModel, CompressiveTransformerWithLMHead)):
        # Split inot subsequences
        if subsequence_len > 1:
            input_ids = torch.split(input_ids, subsequence_len, dim=1)
            attention_mask = torch.split(attention_mask, subsequence_len, dim=1)
        else:
            ValueError("Please provide a subsequence-length for forwardpass with Compressive or Transformer-XL")

        # Memories that get saved over the run over the subsequences
        memories = {
            "mems": None,
            "c_mems": None
        } 

        # Loss that get saved over the run over the subsequences
        final_outputs = {
            "loss": None,
            "prediction_scores": [] 
        }

        for ids, mask in zip(input_ids, attention_mask):
            # Construct the arguments for the model
            args, kwargs = construct_args(model, ids, mask, memories, output_labels=use_labels)

            # Run through
            outputs = model(*args, **kwargs)

            # Save memories and compressed memories
            memories["mems"] = get_attribute(outputs, "mems")
            if isinstance(model, CompressiveTransformerWithLMHead):
                memories["c_mems"] = get_attribute(outputs, "c_mems")
            
            # Accumulate loss
            if final_outputs["loss"] == None:
                final_outputs["loss"] = get_attribute(outputs, "loss").unsqueeze(0)
            else:
                final_outputs["loss"] = torch.cat(
                    (final_outputs["loss"], get_attribute(outputs, "loss").unsqueeze(0)), 
                    dim=0
                )
            
            if not use_labels:
                final_outputs["prediction_scores"] = np.cat(
                    (final_outputs["prediction_scores"], get_attribute(outputs, "prediction_loss").to_numpy()),
                    dim=1
                )

        # Mean over the accumulated losses
        final_outputs["loss"] = final_outputs["loss"].mean()
        final_outputs["outputs"] = final_outputs["loss"].mean()

        # Reassign
        outputs = final_outputs
    # Normal Forward Pass
    else:
        args, kwargs = construct_args(model, input_ids, attention_mask)
        outputs = model(*args, **kwargs)
    return outputs
