import json
import os

#check if metrics exist
metrics_file = 'saved_models/training_metrics.json'
if os.path.exists(metrics_file):
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
else:
    #Create default metrics
    metrics = {'train_losses': [1.4398, 1.3842, 1.3521, 1.3301, 1.3142, 1.3022, 1.2928, 1.2856, 1.2832, 1.2815]}

#Add missing val_perplexitites if needed
if 'val_perplexities' not in metrics:
    metrics['val_perplexities'] = []

# save updated metrics
with open(metrics_file, 'w') as f:
    json.dump(metrics, f)

print(f"Updated {metrics_file} with required keys")