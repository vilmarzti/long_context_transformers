import numpy as np
import torch

from transformers import OpenAIGPTLMHeadModel, TransfoXLLMHeadModel
from transformers.models.transfo_xl.modeling_transfo_xl import TransfoXLLMHeadModelOutput


from ..transformers.outputs import (
    CompressiveTransformerLMHeadModelOutput,
    GeneralOutput
)

from ..transformers.compressive_transformer import CompressiveTransformerWithLMHead

from transformers.modeling_outputs import CausalLMOutput

from .config import (
    transformer_xl,
    compressive_transformer,
    gpt,
    general
)


def get_attribute(output, key):
    out = None
    if isinstance(output, TransfoXLLMHeadModelOutput):
        if key in  transformer_xl.keys():
            out = getattr(output, transformer_xl[key])

    elif isinstance(output, CompressiveTransformerLMHeadModelOutput):
        if key in compressive_transformer.keys():
            out = getattr(output, compressive_transformer[key])
    
    elif isinstance(output, CausalLMOutput):
        if key in gpt.keys():
            out = getattr(output, gpt[key])

    elif isinstance(output, GeneralOutput):
        if key in general.keys():
            out = getattr(output, general[key])

    else:
        try:
            out = getattr(output, key)
        except KeyError:
            raise ValueError(f"Output for type {type(output)} and {key} not found.")
    
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
            "mems": memories["mems"],
            "c_mems": memories["c_mems"]
        }
    elif isinstance(model, OpenAIGPTLMHeadModel):
        args = [input_ids]
        kwargs = {
            "attention_mask": attention_mask
        }
    else:
        ValueError(f"Function for model of type {type(model)} has not been implemented")
    
    if output_labels:
        kwargs["labels"] = input_ids
    
    return args, kwargs

def forward_pass(model, input_ids, attention_mask, subsequence_len=-1, use_labels=True):
    general_outputs = GeneralOutput()

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

        for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
            # Construct the arguments for the model
            args, kwargs = construct_args(model, ids, mask, memories, output_labels=use_labels)

            # Run through
            outputs = model(*args, **kwargs)

            # Save memories and compressed memories
            memories["mems"] = get_attribute(outputs, "mems")
            if isinstance(model, CompressiveTransformerWithLMHead):
                memories["c_mems"] = get_attribute(outputs, "c_mems")
            
            # Accumulate loss and prediction scores
            if i == 0:
                # Init loss and prediction scores
                general_outputs.loss = get_attribute(outputs, "loss")

                if not use_labels:
                    general_outputs.prediction_scores = get_attribute(outputs, "prediction_scores").detach().cpu().numpy()
            else:
                if not (general_outputs.loss is None):
                    # Accumulate
                    general_outputs.loss = torch.cat(
                        (
                            general_outputs.loss,
                            get_attribute(outputs, "loss")
                        ), 
                        dim=1
                    )

                if not use_labels:
                    general_outputs.prediction_scores = np.concatenate(
                        (
                            general_outputs.prediction_scores, 
                            get_attribute(outputs, "prediction_scores").detach().cpu().numpy()
                        ),
                        axis=1
                    )

        # Reassign
        outputs = general_outputs

    # Forwards for GPT
    elif isinstance(model, OpenAIGPTLMHeadModel):
        # Construct inputs
        args, kwargs = construct_args(model, input_ids, attention_mask)

        # Pass through model
        outputs = model(*args, **kwargs)

        # Read values
        general_outputs.loss = get_attribute(outputs, "loss")
        general_outputs.prediction_scores = get_attribute(outputs, "prediction_scores").detach().cpu().numpy()

        outputs = general_outputs

    # Normal Forward Pass
    else:
        args, kwargs = construct_args(model, input_ids, attention_mask)
        outputs = model(*args, **kwargs)
    return outputs
