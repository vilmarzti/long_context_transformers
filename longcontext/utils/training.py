"""
    This module contains the training routine for our transformer models.
"""
import torch
import torch.nn.functional as F

from tqdm import tqdm

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

            sub_sequence_probs.append(token_prob)

        # Compute perplexity. If the whole sentence is padding skip
        if len(sub_sequence_probs) > 0:
            sub_sequence_probs = torch.stack(sub_sequence_probs)

            # Perplexity direct
            # perplexity = torch.pow(torch.prod(1.0 / sub_sequence_probs.double()), 1/sub_sequence_probs.size(0))

            # exp(log(PPL))
            perplexity = torch.exp((-1.0 / sub_sequence_probs.size(0)) * torch.log(sub_sequence_probs).sum())

            perplexities.append(perplexity.item())

    return perplexities


def train(model, train_loader, optimizer, epochs, valid_loader=None, lr_scheduler=None, device="cpu"):
    """ The training loop with validation.

    Args:
        model (torch.nn.Module): A transformer where the call function should 
            return an object with and element 'loss'.
        train_loader (torch.utils.data.Dataloader): A generator for the data
        optimizer (object): The optimizer for the transformer
        epochs (integer): How many epochs to train
        valid_loader (torch.utils.data.Dataloader, optional): Optional generator
            for the validation data. If None is provided then no validation is 
            performed. Defaults to None.
        lr_scheduler (object, optional): The scheduler for the learning rate. 
            Defaults to None.
        device (string, optional): Either "cpu" or "cuda" for training on cpu/gpu.
            Defaults to "cpu"
    """
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            # Get the appropriate columns
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Move to GPU if possible
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Let it run through the Model
            # The TransformerXL model doesn't have an attention_mask input
            if not isinstance(model, TransfoXLLMHeadModel):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            else:
                outputs = model(input_ids=input_ids, labels=input_ids)

            # Accumulate losses if necessary
            loss = get_attribute(outputs, "loss")

            # Reduce loss if necessary
            if loss.dim() > 0:
                loss = loss.mean()

            # Backprop
            loss.backward()
            optimizer.step()

            if lr_scheduler:
                lr_scheduler.step()

            # Reset optimizer
            optimizer.zero_grad()
        
        # set model for training
        model.eval()

        if valid_loader:
            with torch.no_grad():
                # Go through 
                losses = []
                perplexities = []
                for batch in tqdm(valid_loader, desc=f"Validation epoch {epoch}"):
                    # Get the appropriate columns
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)

                    # Move to GPU if possible
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)

                    # Let it run through the Model
                    # The TransformerXL model doesn't have an attention_mask input
                    if not isinstance(model, TransfoXLLMHeadModel):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    else:
                        outputs = model(input_ids=input_ids, labels=input_ids)

                   
                    loss = get_attribute(outputs, "outputs")

                    if loss.dim() > 0:
                        loss = loss.mean()

                    losses.append(loss.item())

                    # compute perplexity
                    perplexities.extend(perplexity(model, input_ids, attention_mask))

                average_ppl = sum(perplexities) / len(perplexities)
                average_loss = sum(losses) / len(losses)
                
                print(f"{epoch} in {epochs} done ")
                print(f"avg_loss: {average_loss} avg_ppl: {average_ppl}")

    return model