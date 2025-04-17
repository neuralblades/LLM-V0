import torch
import numpy as np
import math
from tqdm import tqdm

def calculate_perplexity(model, dataloader, device):
    """ Calculate perplixity on a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Calculating perplixity"):
            x, y = x.to(device), y.to(device)
            output = model(x)

            # Reshape for loss calculation
            output = output.view(-1, output.size(-1))
            y = y.view(-1)

            loss = criterion(output, y)
            total_loss += loss.item()
            total_tokens += y.numel()

    # Perplexity = exp(average negative log-likelihood)
    perplexity = math.exp(total_loss / total_tokens)
    return perplexity