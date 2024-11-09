import sys
import pygame
import random
import numpy as np
from enum import Enum
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Initialize Pygame
pygame.init()
font = pygame.font.SysFont('arial', 25)

# Constants
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001  # Learning rate

# Direction enumeration
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Point class
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

# Game class
class SnakeGame:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Reset the game
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x - 20, self.head.y),
                      Point(self.head.x - 40, self.head.y)]
        self.score = 0
        self.fruit = None
        self._place_fruit()
        self.frame_iteration = 0

    def _place_fruit(self):
        x = random.randint(0, (self.w - 20) // 20) * 20
        y = random.randint(0, (self.h - 20) // 20) * 20
        self.fruit = Point(x, y)
        if self.fruit in self.snake:
            self._place_fruit()

    def play_step(self, action):
        self.frame_iteration += 1
        # Collect user input (not needed for AI)
        # Move
        self._move(action)  # Update the head
        self.snake.insert(0, self.head)

        # Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Check if fruit eaten
        if self.head.x == self.fruit.x and self.head.y == self.fruit.y:
            self.score += 1
            reward = 10
            self._place_fruit()
        else:
            self.snake.pop()

        # Update UI and clock
        self._update_ui()
        pygame.time.Clock().tick(120)

        # Return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > self.w - 20 or pt.x < 0 or pt.y > self.h - 20 or pt.y < 0:            
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI')
        self.display.fill((0, 0, 0))

        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(pt.x, pt.y, 20, 20))
            pygame.draw.rect(self.display, (0, 200, 0), pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(self.fruit.x, self.fruit.y, 20, 20))

        text = font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            # No change
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            # Left turn
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += 20
        elif self.direction == Direction.LEFT:
            x -= 20
        elif self.direction == Direction.DOWN:
            y += 20
        elif self.direction == Direction.UP:
            y -= 20

        self.head = Point(x, y)

# Agent class
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # Popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Fruit location
            game.fruit.x < game.head.x,  # Fruit left
            game.fruit.x > game.head.x,  # Fruit right
            game.fruit.y < game.head.y,  # Fruit up
            game.fruit.y > game.head.y  # Fruit down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))  # Popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # List of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # Random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games  # Adjust epsilon
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # Prediction by the model
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

# Neural Network
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        self.load_state_dict(torch.load(file_name))

# Trainer class
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            # Only one parameter to train
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        # 1: Predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not game_over
        # pred.clone()
        # preds[argmax(action)] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

# Main training loop
import random
import matplotlib.pyplot as plt
from IPython import display

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()

    # Load model and training stats if they exist
    try:
        agent.model.load()
        with open('training_stats.txt', 'r') as f:
            lines = f.readlines()
            agent.n_games = int(lines[0].split(':')[1].strip())
            record = int(lines[1].split(':')[1].strip())
        print(f'Loaded model and training stats. Number of games: {agent.n_games}, Highest score: {record}')
    except FileNotFoundError:
        print('No saved model or training stats found, starting training from scratch.')

    try:
        while True:
            # Get old state
            state_old = agent.get_state(game)

            # Get move
            final_move = agent.get_action(state_old)

            # Perform move and get new state
            reward, game_over, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # Train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

            # Remember
            agent.remember(state_old, final_move, reward, state_new, game_over)

            if game_over:
                # Train long memory (experience replay), plot result
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    # Save the model
                    agent.model.save()
                    with open('training_stats.txt', 'w') as f:
                        f.write(f'Number of games: {agent.n_games}\n')
                        f.write(f'Highest score: {record}\n')

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                # plot(plot_scores, plot_mean_scores)
    except KeyboardInterrupt:
        # Handle Control+C to gracefully save the model, highscore, game number and exit
        print("\nTraining interrupted. Saving model and exiting...")
        # Check if the current score is the highest
        if score > record:
            record = score
            # Save the model
            agent.model.save()
            with open('training_stats.txt', 'w') as f:
                f.write(f'Number of games: {agent.n_games}\n')
                f.write(f'Highest score: {record}\n')
            print('New record saved.')
        sys.exit()

# Run the training
if __name__ == '__main__':
    train()
