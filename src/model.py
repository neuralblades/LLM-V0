import torch.nn as nn
import torch

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        hidden, _ = self.rnn(embeds)
        output = self.fc(hidden)
        return output
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, vocab_size, embedding_dim=64, hidden_dim=128):
        model = cls(vocab_size, embedding_dim, hidden_dim)
        model.load_state_dict(torch.load(path))
        return model