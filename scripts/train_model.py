import os
import json
import torch
from src.tokenizer import Tokenizer
from src.model import LanguageModel
from src.dataset import TextDataset, get_dataloader
from src.train import train

def main():
    # Load configuration
    with open('configs/default_config.json', 'r') as f:
        config = json.load(f)

    #Load data
    with open(os.path.join('data/raw', config['data_file']), 'r', encoding='utf-8') as f:
        text = f.read()

    #create tokenizer
    tokenizer = Tokenizer([text])

    # Split data for validation (80% train, 20% validation)
    tokens = tokenizer.encode(text)
    split_idx = int(len(tokens) * 0.8)
    train_text = tokenizer.decode(tokens[:split_idx])
    val_text = tokenizer.decode(tokens[split_idx:])

    # Create dataloader
    train_dataloader = get_dataloader(
        train_text,
        tokenizer,
        seq_length=config['seq_length'],
        batch_size=config['batch_size']
    )

    val_dataloader = get_dataloader(
        val_text,
        tokenizer,
        seq_length=config['seq_length'],
        batch_size=config['batch_size']
    )

    # Create model
    model = LanguageModel(
        tokenizer.vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim']
    )

    #set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    #train model
    model, metrics = train(
        model,
        train_dataloader,
        val_dataloader,
        epochs=config['epochs'],
        lr=config['learning_rate'],
        device=device
    )

    # Save metrics
    os.makedirs('saved_models', exist_ok=True)
    with open('saved_models/training_metrics.json', 'w') as f:
        json.dump({
            'train_losses': [float(x) for x in metrics['train_losses']],
            'val_perplexities': [float(x) for x in metrics['val_perplexities']]
        }, f)

    #save model and tokenizer
    model.save('saved_models/model.pt')
    tokenizer.save('saved_models/tokenizer.json')

    print("Training completed. Model and tokenizer saved.")

if __name__ == "__main__":
    main()
