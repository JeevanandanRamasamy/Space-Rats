# Space Rats: Bot Strategy Comparison and Neural Network Training

This repository contains a Python implementation for testing bot strategies in the Space Rats game and training a deep learning model to predict the number of turns left based on the bot's environment. The project consists of two main parts:

1. **Bot Strategy Testing**: Compare different bot strategies and analyze how they perform across multiple iterations for different alpha values.
2. **Neural Network Training**: Train a neural network model using the collected game data to predict the number of turns left based on bot locations and knowledge bases.

## Installation

### Prerequisites
This project requires Python 3.x and the following libraries:

- `numpy`
- `matplotlib`
- `tqdm`
- `sklearn`
- `torch`
- `space_rats` (custom game environment module)

You can install the necessary dependencies by running:

```bash
pip install numpy matplotlib tqdm sklearn torch
```

## Cloning the Repository

To clone this repository, run:
```bash
git clone https://github.com/JeevanandanRamasamy/space-rats.git
cd space-rats
```

## Bot Strategy Testing

The `test_bots` function tests two bot strategies and compares their performance based on the number of turns taken in the Space Rats game. The strategies can be specified as `stationary` or `mobile`.

### Overview of test_bots

1. **Input**:
   - Grid size (`D`)
   - Learning rate (`alpha`)
   - Number of iterations (`iterations`)
   - Strategy (`space_rat_strategy`), either `stationary` or `mobile`.
   - Random seed for reproducibility.
2. **Functionality**:
   - Initializes the game with the specified parameters.
   - Runs both bots with parallelized processing using `ProcessPoolExecutor`.
   - Computes and returns the number of turns taken by each bot for each iteration.
3. **Output**:
   - Returns two lists containing the number of turns taken by each bot.

### plot_turn_counts

This function plots the average number of turns taken by both bots for different alpha values. The plot is generated using `matplotlib` and provides a visual comparison between the bot strategies.

**Example Usage**
```python
dim, iter = 30, 100
bot1_average_list, bot2_average_list = [], []
alpha_values = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2,
                0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4]
for alpha in alpha_values:
    print("Alpha value:", alpha)
    bot1_turns, bot2_turns = test_bots(D=dim, alpha=alpha, iterations=iter, space_rat_strategy='mobile')
    bot1_average_list.append(sum(bot1_turns) / iter)
    bot2_average_list.append(sum(bot2_turns) / iter)
plot_turn_counts(alpha_values, bot1_average_list, bot2_average_list)
```

## Neural Network Training

This section trains a neural network model using the data generated from the Space Rats game. The model is designed to predict the number of turns left in the game based on the bot’s environment.

### Data Preprocessing

Data is collected and transformed into a format suitable for training the neural network. The dataset includes:
- `bot_locations`: 2D array representing the positions of the bots.
- `knowledge_bases`: 2D array representing the knowledge of each bot at a specific grid location.
- `num_turns_left`: The number of turns left in the game for each state.

### Model Architecture

The neural network is a CNN-based model that processes the bot locations and knowledge bases. The architecture consists of:
1. **Convolutional Layers**:
   - Two convolutional layers with Batch Normalization for better training stability.
   - Activation function: ReLU.
2. **Global Pooling**:
   - Adaptive Average Pooling is used to reduce the feature map to a single value.
3. **Fully Connected Layers**:
   - Two fully connected layers for feature transformation.
   - Dropout layer to prevent overfitting.
4. **Output Layer**:
   - A single output representing the predicted number of turns left.

**Example**
```python
train_net(data, learning_rate=0.001, num_epochs=20, batch_size=64)
```

### Training

- The dataset is split into training (80%), validation (10%), and testing (10%).
- The model is trained using the AdamW optimizer with a learning rate of 0.001 and the Smooth L1 loss function.
- The best model is saved based on the lowest validation loss.

### Evaluation

The model is evaluated on the test set, and a scatter plot is generated comparing the predicted number of turns left with the actual values.
```python
evaluate_and_plot(model, test_loader, criterion)
```

## Results

The primary output of this project is the comparison of bot strategies and the performance of the trained neural network.
1. Bot Strategy Comparison:
   - A plot comparing the number of turns taken by both bots for various alpha values is generated.
   - The bot strategies are tested over multiple iterations to assess their performance.
2. Model Evaluation:
   - The model is evaluated on the test set, and a scatter plot is generated to compare predicted vs actual values for the number of turns left.

## Usage

To run the entire pipeline (bot testing and neural network training), simply execute the train_net function after loading the data:
```python
# Ensure the data file is in the same directory as the script
data = load_data(filename="space_rats_simulation_data.npz", iterations=7500, strat='stationary')
plot_visual(data)
train_net(data, learning_rate=0.001, num_epochs=20, batch_size=64)
```
