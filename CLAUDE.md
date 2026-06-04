# CLAUDE.md

Guidance for working in this repository.

## Project

Snake game with a Deep Q-Network (DQN) agent that learns to play via reinforcement
learning. PyTorch for the model, pygame for the game/rendering, numpy for state.

Everything lives in a single file: `game.py`.

## Run

```sh
pip install -r requirements.txt
python game.py          # starts/resumes training; opens a pygame window
```

Training runs in an infinite loop. Press **Ctrl+C** to stop — the handler saves the
model and stats before exiting. There are no tests and no build step.

## Layout of `game.py`

- `Direction` / `Point` — small value types for the grid.
- `SnakeGame` — game logic, rendering (`_update_ui`), and `play_step(action)` which
  advances one frame and returns `(reward, game_over, score)`.
- `Agent` — owns the model + replay memory; `get_state` encodes the 11-feature state,
  `get_action` does epsilon-greedy action selection.
- `Linear_QNet` — the network (11 → 256 → 3) plus `save`/`load`.
- `QTrainer.train_step` — the Bellman/Q-learning update.
- `train()` — the main loop (short-memory train each step, long-memory replay on
  game over).

## Conventions / gotchas

- **Grid is 20px.** The board is 640×480; all coordinates are multiples of 20.
- **Action is a one-hot `[straight, right, left]`** relative to current heading, not
  absolute direction.
- **State is 11 ints**: 3 danger flags, 4 direction flags, 4 fruit-location flags.
  If you change the state vector, update the model's `input_size` (currently 11).
- **Device.** A global `DEVICE` (CUDA → MPS → CPU) is selected at import. Any new
  tensor must be created with `device=DEVICE` (or `.to(DEVICE)`), and `torch.load`
  must pass `map_location=DEVICE`, or you'll get device-mismatch errors.
- **Persistence.** `model.pth` holds weights; `training_stats.txt` holds games-played
  and the high score. Both are loaded on startup and saved only when a new record
  is hit (and on Ctrl+C). `model.pth` is committed to the repo.
- **Rendering throttles training.** `play_step` calls `pygame.time.Clock().tick(120)`,
  capping the loop at 120 FPS. For fast/headless training this cap and `_update_ui`
  would need to be bypassed.
