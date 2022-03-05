import torch

import numpy as np
import numpy.ma as ma

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

    # Compute Perplexity for a sentence
    sequence_log_probs = []
    input_ids_numpy = input_ids.cpu().numpy()

    # Generate probabilities for subsequence of sentence
    for i in range(1, input_ids.size(1)):
        token_head = input_ids_numpy[:, :i]
        mask_head = attention_mask[:, :i]

        # Forwards pass
        outputs = forward_pass(
            model,
            input_ids[:, :i],
            attention_mask[:, :i],
            subsequence_len,
            use_labels=False
        )

        # Get probability for the chosen last word
        prediction_scores = get_attribute(outputs, "prediction_scores")

        # Get log probabilities. Tranformer-XL outputs them directly
        if isinstance(model, TransfoXLLMHeadModel):
            log_probabilities = prediction_scores[:, -1]
        else:
            log_probabilities = np.log(softmax(prediction_scores, axis=-1))[:, -1]

        # Get prdicted token log probabilities
        token_log_prob = np.array([log_probabilities[i, id] for i, id in enumerate(input_ids_numpy[:, i])])

        # Append to sequence
        if i == 1:
            sequence_log_probs = token_log_prob[:, None]
        else:
            sequence_log_probs = np.append(sequence_log_probs, token_log_prob[:, None], axis=1)

    # Mask log_probabilities
    sequence_log_probs = ma.array(sequence_log_probs, mask=(attention_mask[:, 1:] == 0).cpu().numpy())

    # Compute lengths of each batch
    sequence_lengths = np.count_nonzero(sequence_log_probs.mask == 0, axis=1)

    sum_log_probs = ma.sum(sequence_log_probs, axis=1).compressed()

    # Compute perplexity.
    perplexity = np.exp((-1.0 / sequence_lengths) * sum_log_probs)

    return perplexity

