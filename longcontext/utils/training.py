"""
    This module contains the training routine for our transformer models.
"""
import torch

from transformers import TransfoXLLMHeadModel


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
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            # Get the appropriate columns
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Move to GPU if possible
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Let it run through the Model
            # The TransformerXL model doesn't have an attention_mask input
            if model is not TransfoXLLMHeadModel:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            else:
                outputs = model(input_ids=input_ids, labels=input_ids)

            # Accumulate losses if necessary
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            elif hasattr(outputs, "losses"):
                loss = torch.sum(outputs.losses)
            else:
                raise AttributeError("outputs neither contain attribute `loss` or `losses`")

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
                average_loss = 0
                average_nll = 0
                for batch in valid_loader:
                    # Get the appropriate columns
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)

                    # Move to GPU if possible
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)

                    # Let it run through the Model
                    # The TransformerXL model doesn't have an attention_mask input
                    if model is not TransfoXLLMHeadModel:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    else:
                        outputs = model(input_ids=input_ids, labels=input_ids)

                    # Accumulate losses if necessary
                    if hasattr(outputs, "loss"):
                        loss = outputs.loss
                    elif hasattr(outputs, "losses"):
                        loss = torch.mean(outputs.losses)
                    else:
                        raise AttributeError("outputs neither contain attribute `loss` or `losses`")
                    
                    average_loss += loss.item()

                    # compute perplexity
                    # https://huggingface.co/docs/transformers/perplexity
                    max_length = 32
                    stride = 16

                    neg_log_likelihoods =[]
                    for i in range(0, input_ids.size(1), stride):
                        begin_loc = max(i + stride - max_length, 0)
                        end_loc = min(i + stride, input_ids.size(1))
                        target_len = end_loc - i

                        nll_input_ids = input_ids[:, begin_loc: end_loc]
                        nll_attention_mask = attention_mask[:, begin_loc: end_loc]
                        nll_target_ids = nll_input_ids.clone()
                        nll_target_ids[:, :-target_len] = -100

                        outputs = model(nll_input_ids, attention_mask=nll_attention_mask, labels=nll_target_ids)
                        nll = outputs[0] * target_len
                        neg_log_likelihoods.append(nll)

                    perplexity = torch.exp(torch.stack(neg_log_likelihoods).sum() /end_loc)
                    average_nll += perplexity.item()

                
                print(f"{epoch} in {epochs} done ")
                print(f"avg_loss: {average_loss/len(valid_loader)} avg_ppl: {average_nll/len(valid_loader)}")

    return model