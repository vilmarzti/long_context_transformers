import torch
import torch.nn.functional as F
import numpy as np

from transformers import TransfoXLLMHeadModel

from longcontext.utils.attributes import get_attribute

@torch.no_grad()
def perplexity(model, input_ids, attention_mask):
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
        sub_sequence_probs = []

        # Generate probabilities for subsequence of sentence
        for i in range(2, tokens.size(0)):
            # don't generate padding
            if attn[i] == 0.0:
                break

            token_head = tokens[:i].unsqueeze(dim=0)
            mask_head = attn[:i].unsqueeze(dim=0)

            if isinstance(model, TransfoXLLMHeadModel):
                outputs = model(token_head, output_hidden_states=True)
            else:
                outputs = model(token_head, attention_mask=mask_head)

            # Get probability for the chosen last word
            prediction_scores = get_attribute(outputs, "prediction_scores")
            probababilities = F.softmax(prediction_scores, dim=-1)[0,-1]
            token_prob = probababilities[token_head[0, -1]]

            sub_sequence_probs.append(token_prob.item())

        # Compute perplexity. If the whole sentence is padding skip
        if len(sub_sequence_probs) > 0:
            # Perplexity direct
            # perplexity = torch.pow(torch.prod(1.0 / sub_sequence_probs.double()), 1/sub_sequence_probs.size(0))

            # exp(log(PPL))
            perplexity = np.exp((-1.0 / len(sub_sequence_probs)) * np.log(sub_sequence_probs).sum())

            perplexities.append(perplexity.item())

    return perplexities

