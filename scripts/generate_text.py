import os
import json
import torch
import argparse 
from src.tokenizer import Tokenizer
from src.model import LanguageModel
from src.generate import generate_text


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate text using trained langusge mmodel')
    parser.add_argument('--seed', type=str, default='The ', help='Seed text to start generation')
    parser.add_argument('--length', type=int, default=500, help='Number of  characters to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (higher = more random)')
    parser.add_argument('--model_path', type=str, default='saved_models/model.pt', help='Path to the saved model')
    parser.add_argument('--tokenizer', type=str, default='saved_models/tokenizer.json', help='Path to the saved tokenizer')

    args = parser.parse_args()
    #load the configuration
    with open('configs/default_config.json', 'r') as f:
        config = json.load(f)

    #load tokenizer
    tokenizer = Tokenizer.load(args.tokenizer)

    #create model with the same parameters as during training
    model = LanguageModel.load(
        args.model_path,
        tokenizer.vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim']
    )

    #set device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #generate text
    generated_text = generate_text(
        model,
        tokenizer,
        args.seed,
        length=args.length,
        temperature=args.temperature,
        device=device
    )

    #print the generated text
    print("\nGenerated Text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    main()
