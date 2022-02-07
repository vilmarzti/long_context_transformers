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
    for epoch in range(epochs):
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
                average_loss=0
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
                
                print(f"{epoch} in {epochs} done")
                print(average_loss/len(valid_loader))

    return model