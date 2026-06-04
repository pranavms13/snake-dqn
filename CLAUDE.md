# CLAUDE.md

Guidance for working in this repository.

## Project

Snake game with a Deep Q-Network (DQN) agent that learns to play via reinforcement
learning. PyTorch for the model, pygame for the game/rendering, numpy for state.

Everything lives in a single file: `game.py`.

## Run

```sh
uv sync                            # creates .venv and installs deps from pyproject.toml/uv.lock
uv run python game.py              # starts/resumes training; opens a pygame window
uv run python game.py --headless   # no window; prints per-game status to the CLI (faster)
```

This is a [uv](https://docs.astral.sh/uv/) project: dependencies live in `pyproject.toml`
(pinned in `uv.lock`). Add deps with `uv add <pkg>`.

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
- **Device.** A global `DEVICE` is selected at import: `--device cpu|cuda|mps`
  overrides, else auto-detect (CUDA → MPS → CPU). Any new tensor must be created
  with `device=DEVICE` (or `.to(DEVICE)`), and `torch.load` must pass
  `map_location=DEVICE`, or you'll get device-mismatch errors.
- **The network is tiny, so a big GPU is the wrong tool.** It's 11 → 256 → 3 and
  trains one sample per step, so a datacenter GPU sits near-idle (~1% util) and CPU
  is usually *faster* (no host↔device overhead). `--device cpu` to compare.
- **Keep `train_step` vectorized.** The Bellman target uses a single batched forward
  over `next_state` plus advanced indexing — do NOT reintroduce a per-sample Python
  loop with `self.model(next_state[idx])` / `.item()`. On GPU that becomes thousands
  of tiny kernel launches + host↔device syncs and was the original perf bug.
- **Persistence.** `model.pth` holds weights; `training_stats.txt` holds games-played
  and the high score. Both are loaded on startup and saved only when a new record
  is hit (and on Ctrl+C). `model.pth` is committed to the repo.
- **Rendering throttles training.** In windowed mode `play_step` calls
  `pygame.time.Clock().tick(120)`, capping the loop at 120 FPS. The `--headless`
  flag sets a global `HEADLESS` (parsed from `argv` at import) that skips
  `pygame.init()`, the font, `_update_ui`, and the FPS cap — so training runs as
  fast as the GPU/CPU allows and status is printed per game to the CLI.
