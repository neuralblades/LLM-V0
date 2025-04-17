import torch 
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length=64):
        self.tokenizer = tokenizer
        self.seq_length = int(seq_length)  # Ensure seq_length is an integer
        self.tokens = tokenizer.encode(text)  

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_length)  # Add max to avoid negative values
    
    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.seq_length]
        y = self.tokens[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x), torch.tensor(y)
    

def get_dataloader(text, tokenizer, seq_length=64, batch_size=32):
    dataset = TextDataset(text, tokenizer, seq_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)