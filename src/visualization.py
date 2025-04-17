import matplotlib.pyplot as plt
import json

def plot_training_metrics(metrics_file='saved_models/training_metrics.json'):
    """Plot training and validation metrics"""
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 5))

    # Plot training loss
    epochs =  list(range(1, len(metrics['train_losses']) + 1))
    ax1.plot(epochs, metrics['train_losses'])
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    # Plot validation Perplexity
    if metrics['val_perplexities']:
        ax2.plot(epochs, metrics['val_perplexities'])
        ax2.set_title('Validation Perplexity')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.grid(True)

    plt.tight_layout()
    plt.savefig('saved_models/training_metrics.png')
    plt.show()
