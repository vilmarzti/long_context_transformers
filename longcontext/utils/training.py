"""
    This module contains the training routine for our transformer models.
"""
import torch
import numpy as np
from os import path
from datetime import datetime

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from longcontext.utils.helpers import forward_pass, get_attribute
from longcontext.utils.metrics import perplexity


def train(model, train_loader, optimizer, epochs=30, valid_loader=None, lr_scheduler=None, device="cpu", subsequence_len=-1, save_path=None, aggregate=1):
    """ The training loop with validation.

    Args:
        model (torch.nn.Module): A transformer where the call function should 
            return an object with and element 'loss'.
        train_loader (torch.utils.data.Dataloader): A generator for the data
        optimizer (object): The optimizer for the transformer
        epochs (integer): How many epochs to train. Defaults to 30.
        valid_loader (torch.utils.data.Dataloader, optional): Optional generator
            for the validation data. If None is provided then no validation is 
            performed. Defaults to None.
        lr_scheduler (object, optional): The scheduler for the learning rate. 
            Defaults to None.
        device (string, optional): Either "cpu" or "cuda" for training on cpu/gpu.
            Defaults to "cpu"
        subsequence_len (int, optional): The length of the subsequences for 
            Transformer-XL and Compressive Transformer.
        aggregate: (int, optional): The number of batches we aggregate over. Defaults to 1
    """
   
    print(f"\nStarting training of model {type(model).__name__} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters\n")

    if save_path is None:
        save_path = path.join("data", "transformers", "_".join([datetime.now().strftime("%Y_%m_%d_%H"), type(model).__name__]))
 
    writer = SummaryWriter()
    last_ppl = float("-inf")
    for epoch in range(1, epochs + 1):
        model.train()
        average_loss_train =[]
        aggregate_loss = torch.FloatTensor(0).to(device)
        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)):
            # Get the appropriate columns
            input_ids = batch["input_ids"]

            # Get attention_mask if supported
            try:
                attention_mask = batch["attention_mask"]
            except KeyError:
                attention_mask = torch.ones_like(input_ids)

            # Move to GPU if possible
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward Pass
            outputs = forward_pass(model, input_ids, attention_mask, subsequence_len=subsequence_len)

            # Get loss
            loss = get_attribute(outputs, "loss")

            # Reduce loss if necessary
            if loss.dim() > 0:
                loss = loss.mean()
            
            if i % aggregate == 0:
                aggregate_loss = loss
            else:
                aggregate_loss += loss

            if i % aggregate == aggregate - 1:
                # Backprop
                aggregate_loss.backward()
                optimizer.step()

                if lr_scheduler:
                    lr_scheduler.step()
            
                # Reset optimizer
                optimizer.zero_grad()
                del aggregate_loss

            average_loss_train.append(loss.cpu().detach().item())
        
        average_loss_train = np.mean(average_loss_train)

        # set model for evaluation
        model.eval()
        if valid_loader:
            with torch.no_grad():
                # Go through 
                losses = []
                perplexities = []
                for batch in tqdm(valid_loader, desc=f"Validation epoch {epoch}", leave=False):
                    # Get the appropriate columns
                    input_ids = batch["input_ids"]

                    # Get attention_mask if supported
                    try:
                        attention_mask = batch["attention_mask"]
                    except KeyError:
                        attention_mask = torch.ones_like(input_ids)
                    
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)


                    outputs = forward_pass(model, input_ids, attention_mask, subsequence_len=subsequence_len)
                  
                    loss = get_attribute(outputs, "loss")

                    if loss.dim() > 0:
                        loss = loss.mean()

                    losses.append(loss.cpu().detach().item())

                    # compute perplexity
                    perplexities.extend(perplexity(model, input_ids, attention_mask, subsequence_len))

                average_ppl = np.mean(perplexities, dtype=np.longdouble)
                average_loss = np.mean(losses)

                writer.add_scalars("Loss", {"valid": average_loss}, epoch)
                writer.add_scalars("Loss", {"train": average_loss_train}, epoch)
                writer.add_scalar("Perplexity/test", average_ppl, epoch)

               
                print(f"Epoch {epoch:<4n} in {epochs:<4n}: \navg_loss_train: {average_loss_train:<8.4n} avg_loss_test: {average_loss:<8.4n} avg_ppl: {average_ppl:<8.4n}")

                if average_ppl > last_ppl:
                    model.save_pretrained(save_path)
                    last_ppl = average_ppl

    return model