"""
    This module contains the training routine for our transformer models.
"""
import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from longcontext.utils.helpers import forward_pass, get_attribute
from longcontext.utils.metrics import perplexity


def train(model, train_loader, optimizer, epochs, valid_loader=None, lr_scheduler=None, device="cpu", subsequence_len=-1):
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
        subsequence_len (int, optional): The length of the subsequences for 
            Transformer-XL and Compressive Transformer.
    """
    writer = SummaryWriter()


    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            # Get the appropriate columns
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

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

            # Backprop
            loss.backward()
            optimizer.step()

            if lr_scheduler:
                lr_scheduler.step()

            # Reset optimizer
            optimizer.zero_grad()
        
        # set model for evaluation
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

                    outputs = forward_pass(model, input_ids, attention_mask, subsequence_len=subsequence_len)
                  
                    loss = get_attribute(outputs, "outputs")

                    if loss.dim() > 0:
                        loss = loss.mean()

                    losses.append(loss.item())

                    # compute perplexity
                    perplexities.extend(perplexity(model, input_ids, attention_mask, subsequence_len))

                average_ppl = sum(perplexities) / len(perplexities)
                average_loss = sum(losses) / len(losses)

                writer.add_scalar("Loss/test", average_loss, epoch)
                writer.add_scalar("Perplexity/test", average_ppl, epoch)

               
                print(f"{epoch} in {epochs} done ")
                print(f"avg_loss: {average_loss} avg_ppl: {average_ppl}")

    return model