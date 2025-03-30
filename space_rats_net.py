import os
import copy
import random
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import space_rats as sr

device = torch.device("mps") # Use 'cuda' for GPU, 'cpu' for CPU, 'mps' for multi-processing
print(f"Using device: {device}")

class SpaceRatsDataset(Dataset):
    def __init__(self, data):
        """
        data: List of tuples [(bot_locations, knowledge_bases, num_turns_left), ...]
              bot_locations: 2D array of size D x D, knowledge_bases: 2D array of size D x D
              num_turns_left: Number of turns left in the game
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bot_locations, knowledge_bases, num_turns_left = self.data[idx]
        return (torch.tensor(bot_locations, dtype=torch.float32),
                torch.tensor(knowledge_bases, dtype=torch.float32),
                torch.tensor(num_turns_left, dtype=torch.float32))

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=2, conv_sizes=[32, 64], fc_sizes=[64, 32], output_size=1, D=30):
        super(NeuralNetwork, self).__init__()
        # Convolutional layers
        self.input_bn = nn.BatchNorm2d(input_size)
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=conv_sizes[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_sizes[0])
        self.conv2 = nn.Conv2d(in_channels=conv_sizes[0], out_channels=conv_sizes[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_sizes[1])

        # Pool, Global Pool, and Self-Attention
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_sizes[1] * D * D, fc_sizes[0])
        self.fc2 = nn.Linear(fc_sizes[0], fc_sizes[1])
        self.output = nn.Linear(fc_sizes[1], output_size)

        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, bot_locations, knowledge_bases):
        # bot_location: 2D array of size D x D, knowledge_base: 2D array of size D x D
        x = torch.stack([bot_locations, knowledge_bases], dim=1)  # Shape: (batch_size, 2, 30, 30)

        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Fully connected layers
        x = torch.flatten(x, 1) # Shape: (batch_size, 64 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

def evaluate_and_plot(model, test_loader, criterion):
    """Evaluate the model on the test set and plot the predicted vs actual values"""
    # Load the best model
    model.load_state_dict(torch.load("best_space_rats_model.pth"))

    # Evaluate on the test set
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for (bot_locations, knowledge_bases, num_turns_left) in tqdm(test_loader, desc="Testing"):
            bot_locations, knowledge_bases, num_turns_left = bot_locations.to(device), knowledge_bases.to(device), num_turns_left.to(device)
            outputs = model(bot_locations, knowledge_bases).squeeze()
            loss = criterion(outputs, num_turns_left)
            test_loss += loss.item() * bot_locations.size(0)

            # Store predictions and actuals for plotting
            all_predictions.extend(outputs.to('cpu'))  # Convert to numpy for plotting
            all_actuals.extend(num_turns_left.to('cpu'))  # Convert to numpy for plotting

    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(all_actuals, all_predictions, alpha=0.6)
    plt.plot([min(all_actuals), max(all_actuals)], [min(all_actuals), max(all_actuals)], color='red', linestyle='--')  # Line of perfect prediction
    plt.xlabel('Actual Values (Num Turns Left)')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual (Test Set)')
    plt.show()

def train_net(data, learning_rate=0.001, num_epochs=50, batch_size=64):
    """Train the neural network using the data"""
    random.seed(520)
    # Split data into training (80%), validation (10%), and testing (10%)
    train_data, test_data = train_test_split(data, test_size=0.2)
    val_data, test_data = train_test_split(test_data, test_size=0.5)

    # Create DataLoaders
    train_loader = DataLoader(SpaceRatsDataset(train_data), batch_size=batch_size, shuffle=True) # Shuffle the training data
    val_loader = DataLoader(SpaceRatsDataset(val_data), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SpaceRatsDataset(test_data), batch_size=batch_size, shuffle=False)

    # Loss function and optimizer
    model = NeuralNetwork().to(device)
    criterion = nn.SmoothL1Loss() # Huber loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4) # AdamW optimizer
    
    train_losses = []
    test_losses = []
    best_val_loss = float('inf')  # Initialize to infinity to track the best model

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for (bot_locations, knowledge_bases, num_turns_left) in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            optimizer.zero_grad()
            bot_locations, knowledge_bases, num_turns_left = bot_locations.to(device), knowledge_bases.to(device), num_turns_left.to(device)
            outputs = model(bot_locations, knowledge_bases).squeeze()
            loss = criterion(outputs, num_turns_left)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bot_locations.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Evaluate on the validation set
        model.eval()  # Set the model to evaluation mode (no gradient computation)
        val_loss = 0.0
        with torch.no_grad():
            for (bot_locations, knowledge_bases, num_turns_left) in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                bot_locations, knowledge_bases, num_turns_left = bot_locations.to(device), knowledge_bases.to(device), num_turns_left.to(device)
                outputs = model(bot_locations, knowledge_bases).squeeze()
                loss = criterion(outputs, num_turns_left)
                val_loss += loss.item() * bot_locations.size(0)
        val_loss /= len(val_loader.dataset)
        test_losses.append(val_loss)

        # Save the model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_space_rats_model.pth')  # Save the model with the lowest validation loss

        # Print the losses for the current epoch
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    # Plot the training and testing loss
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Over Epochs")
    plt.legend()
    plt.show()

    # Evaluate the model on the test set and plot the predicted vs actual values
    evaluate_and_plot(model, test_loader, criterion)

def test_net(D=30, alpha=0.15, iterations=1000, space_rat_strategy='stationary', seed=520):
    """Test the neural network using the space rats environment"""
    # Generate games with the same grid configuration
    random.seed(seed)
    game = sr.SpaceRats(D, alpha, space_rat_strategy)
    game.initialize_grid(placement=False)
    games_list = [copy.deepcopy(game) for _ in range(iterations)]
    # Place the bot space rat in each game
    for game in games_list:
        game.place_bot_space_rat()
        game.collect_data = True

    # Parallelize game processing using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        data = list(executor.map(sr.apply_bot2_strategy, games_list))
    return data

def transform_game_data(data, iterations):
    """Process the data to be used for training the neural network"""
    transformed_data = []
    for i in range(iterations):
        num_turns = data[i][-1][2]
        # Subtract the number of turns taken from the total number of turns to get the number of turns left
        game_turns = [(data[i][j][0], data[i][j][1], num_turns - data[i][j][2]) for j in range(len(data[i]))]
        transformed_data.extend(game_turns)
    return transformed_data

def save_data(data, filename="space_rats_simulation_data.npz"):
    """Save the data to a compressed NumPy archive."""
    print("Saving data to", filename)
    arrays1, arrays2, integers = zip(*data)
    np.savez_compressed(filename, bot_locations=arrays1, knowledge_bases=arrays2, num_turns=integers)

def load_data(filename="space_rats_simulation_data.npz", iterations=1000, strat='stationary'):
    """
    Load the data from a compressed NumPy archive if it exists.
    If the file does not exist, generate the data using space_rats.py and save it.
    """
    if not os.path.exists(filename):
        print("Data file not found. Generating data...")
        # Use space_rat_strategy='mobile'/'stationary' to test the bots with mobile or stationary space rat
        data = test_net(D=30, alpha=0.15, iterations=iterations, space_rat_strategy=strat)
        data = transform_game_data(data, iterations)
        save_data(data, filename)
        return data
    with np.load(filename) as f:
        print("Data loaded from", filename)
        return list(zip(f['bot_locations'], f['knowledge_bases'], f['num_turns']))

def plot_visual(data):
    """Plot a histogram of the number of turns left in the game"""
    num_turns_left = [d[2] for d in data]
    print("Mean number of turns left:", np.mean(num_turns_left))
    plt.hist(num_turns_left, bins=100, edgecolor='black')
    plt.axvline(np.mean(num_turns_left), color='r', linestyle='dashed', linewidth=1)
    plt.xlabel("Number of Turns Left")
    plt.ylabel("Frequency")
    plt.title("Histogram of Number of Turns Left")
    plt.show()

if __name__ == "__main__":
    dim, iter = 30, 7500 # Change the grid size and number of iterations as needed (100 ~ 48 sec, 1000 ~ 8 min, etc.)
    alpha = 0.15
    print("Alpha value:", alpha)

    # Ensure the data file is in the same directory as the script
    data = load_data(filename="space_rats_simulation_data.npz", iterations=iter, strat='stationary')
    plot_visual(data)
    train_net(data, learning_rate=0.001, num_epochs=20, batch_size=64) # Modify the hyperparameters as needed
