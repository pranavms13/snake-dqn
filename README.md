# Snake Game AI

This project implements a Snake game with an AI agent that learns to play the game using reinforcement learning. The AI is trained using a Deep Q-Network (DQN) approach.

## Project Structure

```
├── game.py
├── model.pth
├── requirements.txt
├── training_stats.txt
└── README.md
```

- `game.py`: Contains the main code for the Snake game and the AI agent.
- `model.pth`: The saved model weights for the AI agent.
- `training_stats.txt`: Contains training statistics such as the number of games played and the highest score achieved.
- `requirements.txt`: Requirements file
- `README.md`: This file.

## Requirements

- Python 3.x
- Pygame
- NumPy
- PyTorch
- Matplotlib

## Installation

1. Clone the repository:
```sh
    git clone https://github.com/pranavms13/snake-nn
    cd snake-nn
```

2. Install the required packages:
```sh
    pip install -r requirements.txt
```

## Usage

To train the AI agent, run:
```sh
python game.py
```

The training process will start, and the AI agent will learn to play the Snake game. The training statistics and model weights will be saved periodically.

## Classes and Functions

### `game.py`

- `Direction`: Enum class for the direction of the snake.
- `Point`: Class representing a point on the game board.
- `SnakeGame`: Class representing the Snake game.
  - `__init__(self, w=640, h=480)`: Initializes the game.
  - `reset(self)`: Resets the game.
  - `play_step(self, action)`: Executes a game step based on the action.
  - `is_collision(self, pt=None)`: Checks for collisions.
  - `_update_ui(self)`: Updates the game UI.
  - `_move(self, action)`: Moves the snake based on the action.
  - `_place_fruit(self)`: Places a fruit on the game board.

- `Agent`: Class representing the AI agent.
  - `__init__(self)`: Initializes the agent.
  - `get_state(self, game)`: Gets the current state of the game.
  - `remember(self, state, action, reward, next_state, game_over)`: Stores the experience in memory.
  - `train_long_memory(self)`: Trains the agent using long-term memory.
  - `train_short_memory(self, state, action, reward, next_state, game_over)`: Trains the agent using short-term memory.
  - `get_action(self, state)`: Gets the action to be taken by the agent.

- `Linear_QNet`: Class representing the neural network for the Q-learning agent.
  - `__init__(self, input_size, hidden_size, output_size)`: Initializes the neural network.
  - `forward(self, x)`: Forward pass of the neural network.
  - `save(self, file_name='model.pth')`: Saves the model weights.
  - `load(self, file_name='model.pth')`: Loads the model weights.

- `QTrainer`: Class for training the Q-learning agent.
  - `__init__(self, model, lr, gamma)`: Initializes the trainer.
  - `train_step(self, state, action, reward, next_state, game_over)`: Performs a training step.

- `train()`: Main function to train the AI agent.

## License

This project is licensed under the MIT License.
```
Feel free to modify the README as needed to better fit your project.
```