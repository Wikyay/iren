import json
from pathlib import Path
import string
import numpy as np
import streamlit as st
from torch import nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = output[:, -1, :]
        output = self.fc(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
        
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

        


# Save the trained model
def save_model(model, model_file):
    torch.save(model.state_dict(), model_file)

def one_hot_encode(sequence, num_categories):
    result = torch.zeros((len(sequence), num_categories))
    for i, category in enumerate(sequence):
        result[i, category] = 1
    return result

def generate_name(model, start, name_length):
    generated_name = start
    if model_type == "LSTM":
        hidden = model.init_hidden(1)
        for _ in range(name_length):
            print("name =", generated_name)
            input = one_hot_encode([char_to_int[generated_name[-1]]], config['input_size']).view(1, 1, -1)
            output, hidden = model(input, hidden)
            probabilities = torch.exp(output).detach().numpy().flatten()
            next_char = np.random.choice(chars, p=probabilities)
            generated_name += next_char
    else: 
        hidden = model.init_hidden()
        for _ in range(name_length):
            input = one_hot_encode([char_to_int[generated_name[-1]]],config['input_size'])
            output, hidden = model(input, hidden)
            probabilities = torch.exp(output).detach().numpy().flatten()
            next_char = np.random.choice(chars, p=probabilities)
            generated_name += next_char
        
    return generated_name.capitalize()


# Load a saved model and configuration
def load_model(model_file, config, model_type):
    
    if model_type == "LSTM":
        model = LSTM(config["input_size"], config["hidden_size"], config["output_size"], config["num_layers"])
    else:
        model = RNN(config["input_size"], config["hidden_size"], config["output_size"])

    model.load_state_dict(torch.load(model_file))
    model.eval()
    return model

#Init some required values

allowed_chars = string.ascii_lowercase + " "

# Load and preprocess the text data
with open("pokemon.txt", "r") as f:
    lines = [line.strip().lower() for line in f.readlines()]

# Filter out unwanted characters
lines = [''.join(c for c in line if c in allowed_chars) for line in lines]
text = "".join(lines)
chars = sorted(list(set(text)))
char_to_int = {char: i for i, char in enumerate(chars)}
print(char_to_int)



# Streamlit app
st.title("Pokémon Name Generator")

model_type = st.selectbox("Model", ['RNN', 'LSTM'])

# Load the saved model
model_path = f"{model_type}_pokemon.pth"
with open(f"{model_type}_pokemon_config.json", "r") as f:
    config = json.load(f)

print(config)

loaded_model = load_model(model_path, config, model_type)

# Input textbox and generate button
input_letter = st.text_input("Enter the first letters of your desired Pokemon name:")
name_length = st.slider("Name length:", 3, 25)

generate_button = st.button("Generate")

# Generate and display names in a table
if generate_button and input_letter:
    print(config)
    num_names = 10
    st.write(name_length - )
    generated_names = [generate_name(loaded_model, input_letter, name_length) for _ in range(num_names)]
    st.write("Generated names:")
    st.table(generated_names, )