import json
import torch
import torch.nn as nn
import numpy as np

import string

allowed_chars = string.ascii_lowercase + " "

# Load and preprocess the text data
with open("pokemon.txt", "r") as f:
    lines = [line.strip().lower() for line in f.readlines()]

# Filter out unwanted characters
lines = [''.join(c for c in line if c in allowed_chars) for line in lines]
text = "".join(lines)
chars = sorted(list(set(text)))
char_to_int = {char: i for i, char in enumerate(chars)}

int_text = [char_to_int[char] for char in text]

def one_hot_encode(sequence, num_categories):
    result = torch.zeros((len(sequence), num_categories))
    for i, category in enumerate(sequence):
        result[i, category] = 1
    return result

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden.expand(input.size(0), -1)), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

input_size = output_size = len(chars)
hidden_size = 1024
model = RNN(input_size, hidden_size, output_size)
loss_fn = nn.NLLLoss()
learning_rate = 0.005
num_epochs = 1000
batch_size = 128
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    hidden = model.init_hidden()
    losses = []
    for i in range(0, len(int_text) - batch_size, batch_size):
        inputs = one_hot_encode(int_text[i:i+batch_size], input_size)
        targets = torch.tensor(int_text[i+1:i+batch_size+1])
        outputs, hidden = model(inputs, hidden.detach())
        loss = loss_fn(outputs, targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss={np.mean(losses)}")

def generate_name(start, name_length):
    generated_name = start
    hidden = model.init_hidden()
    for _ in range(name_length):
        input = one_hot_encode([char_to_int[generated_name[-1]]], input_size)
        output, hidden = model(input, hidden)
        probabilities = torch.exp(output).detach().numpy().flatten()
        next_char = np.random.choice(chars, p=probabilities)
        generated_name += next_char
    return generated_name.capitalize()

num_names = 10
name_length = 10
generated_names = [generate_name('u', name_length) for _ in range(num_names)]
print(generated_names)

# Save the trained model and configuration
def save_model(model, model_file, config_file):
    torch.save(model.state_dict(), model_file)
    config = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
    }
    with open(config_file, "w") as f:
        json.dump(config, f)
        
save_model(model, "RNN_pokemon.pth", "RNN_pokemon_config.json")