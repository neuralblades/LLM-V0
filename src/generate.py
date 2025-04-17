import torch
import torch.nn.functional as F

def generate_text(model, tokenizer, seed_text, length=100, temperature=1.0, device='cpu'):
    model.to(device)
    model.eval()
    current_text = seed_text

    with torch.no_grad():
        for i in range(length):
            #tokenize the current text
            tokens = tokenizer.encode(current_text[-64:])
            x = torch.tensor([tokens]).to(device)


            #get predictions
            output = model(x)

            #Apply temperature
            logits = output[0, -1] / temperature
            probs = F.softmax(logits, dim=0)

            #Sample next token
            next_token = torch.multinomial(probs, 1).item()

            #add to current text
            current_text += tokenizer.decode([next_token])

    return current_text
