import json
from pathlib import Path
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
    hidden = model.init_hidden(1)
    for _ in range(name_length):
        input = one_hot_encode([char_to_int[generated_name[-1]]], input_size).view(1, 1, -1)
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

with open("pokemon.txt", "r") as f:
    lines = [line.strip() for line in f.readlines()]

text = "".join(lines).lower()
chars = sorted(list(set(text)))
char_to_int = {char: i for i, char in enumerate(chars)}



# Streamlit app
st.title("Pokémon Name Generator")

model_type = st.selectbox("Model", ['RNN', 'LSTM'])

config = RNN.config()

# Load the saved model
loaded_model = load_model(model_type, )

with open(f"{model_type}_pokemon_config.json", "r") as f:
        config = json.load(f)

# Input textbox and generate button
input_letter = st.text_input("Enter the first letter of a Pokémon name:")
input_letter = st.slider("Name length:", 3, 10)

generate_button = st.button("Generate")

# Generate and display names in a table
if generate_button and input_letter:
    num_names = 10
    name_length = 10
    generated_names = [generate_name(input_letter, name_length) for _ in range(num_names)]
    st.write("Generated names:")
    st.write("\n".join(generated_names))