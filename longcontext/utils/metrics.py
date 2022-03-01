import torch

import numpy as np
from scipy.special import softmax
from transformers import TransfoXLLMHeadModel

from longcontext.utils.helpers import get_attribute

from .helpers import forward_pass

@torch.no_grad()
def perplexity(model, input_ids, attention_mask, subsequence_len=-1):
    """Computes the perplexity of a model for a given input_sequence by
    producing the next-word probabilities of all subsequences. 
    For numerical stability we compute the exp(log(PPL))

    Args:
        model (PreTrainedModel): The Model for which we want to compute 
            the perplexity
        input_ids (torch.FloatTensor): Batch of input tokens. Is of size (B, S) where
        B is the batch size and S is the sequence length.
        attention_mask (torch.LongTensor): Mask which indicate padding in the sequences.
            Has the same size as input_ids
        subsequence_len (int, optional): Used for Transformer-XL and compressive
            Transformer to determine the length of the processed subsequences.

    Returns:
        List[torch.FloatTensor]: A list with the associated perplexities. Note that some
            sequences are just padding. These sequences will return no perplexity
    """

    # Make list of sequences
    sentence_token = torch.unbind(input_ids)
    sentence_mask = torch.unbind(attention_mask)
    
    # Compute Perplexity for a sentence
    perplexities = []
    for tokens, attn in zip(sentence_token, sentence_mask):
        sub_sequence_log_probs = []

        # Generate probabilities for subsequence of sentence
        for i in range(2, tokens.size(0)):
            # don't generate padding
            if attn[i] == 0.0:
                break

            token_head = tokens[:i].unsqueeze(dim=0)
            mask_head = attn[:i].unsqueeze(dim=0)

            outputs = forward_pass(model, token_head, mask_head, subsequence_len, use_labels=False)

            # Get probability for the chosen last word
            prediction_scores = get_attribute(outputs, "prediction_scores")

            # Get log probabilities. Tranformer-XL outputs them directly
            if isinstance(model, TransfoXLLMHeadModel):
                log_probabilities = prediction_scores[0, -1]
            else:
                log_probabilities = np.log(softmax(prediction_scores[0], axis=-1))[-1]

            token_prob = log_probabilities[tokens[i]]

            sub_sequence_log_probs.append(token_prob.item())

        # Compute perplexity. If the whole sentence is padding skip
        if len(sub_sequence_log_probs) > 0:
            # Perplexity direct
            # perplexity = torch.pow(torch.prod(1.0 / sub_sequence_probs.double()), 1/sub_sequence_probs.size(0))

            # exp(log(PPL))
            perplexity = np.exp((-1.0 / len(sub_sequence_log_probs)) * np.sum(sub_sequence_log_probs, dtype=np.longdouble), dtype=np.longdouble)

            perplexities.append(perplexity.item())

    return perplexities

