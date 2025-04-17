class Tokenizer:
    def __init__(self, texts):
        # Build vocabulary (character-level for simplicity)
        self.chars = sorted(list(set(''.join(texts))))
        self.vocab_size = len(self.chars)

        # Create mapping dictionaries
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[idx] for idx in indices])
    
    def save (self, path):
        import json
        with open(path, 'w') as f:
            json.dump({'chars': self.chars}, f)
        
    @classmethod
    def load (cls, path):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        tokenizer = cls(['']) # create empty tokenizer
        tokenizer.chars = data['chars']
        tokenizer.vocab_size = len(tokenizer.chars)
        tokenizer.char_to_idx = {ch: i for i, ch in enumerate(tokenizer.chars)}
        tokenizer.idx_to_char = {i: ch for i, ch in enumerate(tokenizer.chars)}
        return tokenizer