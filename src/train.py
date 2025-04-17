import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm

def train(model, train_dataloader, val_dataloader=None, epochs=10, lr=0.001, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # For tracking metrics
    train_losses = []
    val_perplexities = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for x, y in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)

            #Forward pass
            optimizer.zero_grad()
            output = model(x)

            #Reshape for loss calculation
            output = output.view(-1, output.size(-1))
            y = y.view(-1)

            #Calculate loss
            loss = criterion(output, y)
            total_loss += loss.item()

            #Backward pass and optimization
            loss.backward()
            optimizer.step() 

        avg_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_loss)

        #Validation if provided
        if val_dataloader is not None:
            from src.evaluate import calculate_perplexity
            val_perplexity = calculate_perplexity(model, val_dataloader, device)
            val_perplexities.append(val_perplexity)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Perplexity: {val_perplexity:.4f}, Time: {time.time() - start_time:.2f}s")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")

    return model, {"train_losses": train_losses, "val_perplexities": val_perplexities}
