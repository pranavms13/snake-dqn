"""Vectorized, GPU-resident Snake DQN trainer (Ape-X-lite).

Runs N Snake games in parallel as batched tensors on the GPU so the device
actually has work to do (the original game.py feeds one sample per step, which
leaves a big GPU ~idle). There is no pygame and no per-step Python game loop:
the entire batch of games is advanced with a handful of tensor ops.

Each parallel env uses its OWN fixed exploration epsilon (Ape-X style), so the
population explores at many noise levels at once while a single shared network
learns from all of their experience via a replay buffer.

Weight-compatible with game.py: the network is the same 11 -> hidden -> 3 MLP
with layers named linear1/linear2, so model.pth and training_stats.txt are
shared between the two trainers.

Usage:
    uv run python train_vectorized.py                       # auto device, 1024 envs
    uv run python train_vectorized.py --num-envs 4096
    uv run python train_vectorized.py --device cpu          # force CPU
    uv run python train_vectorized.py --max-games 50000     # stop after N games

State vector (11 ints, matches game.py exactly):
    [danger_straight, danger_right, danger_left,
     dir_left, dir_right, dir_up, dir_down,
     food_left, food_right, food_up, food_down]
Action index: 0 = straight, 1 = right turn, 2 = left turn (relative to heading).
"""

import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

MODEL_FILE = 'model.pth'
STATS_FILE = 'training_stats.txt'


def select_device():
    """--device cpu|cuda|mps overrides; else auto-detect (CUDA -> MPS -> CPU)."""
    if '--device' in sys.argv:
        i = sys.argv.index('--device')
        if i + 1 < len(sys.argv):
            return torch.device(sys.argv[i + 1])
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = select_device()


class VectorizedSnake:
    """B independent Snake games advanced together as batched tensors.

    Directions are clockwise integers: 0=right, 1=down, 2=left, 3=up, so a
    "right turn" is +1 and a "left turn" is -1 (mod 4) — identical to game.py.

    The snake body is a per-env circular buffer of flattened cell indices
    (y*W + x) between tail_ptr and head_ptr; `occ` is a (B,H,W) occupancy grid
    for O(1) self-collision tests.
    """

    def __init__(self, num_envs, w=32, h=24, device=DEVICE):
        self.B = num_envs
        self.W = w
        self.H = h
        self.device = device
        self.CAP = w * h + 8  # circular-buffer capacity (max body + slack)

        self.DX = torch.tensor([1, 0, -1, 0], device=device)  # per-direction dx
        self.DY = torch.tensor([0, 1, 0, -1], device=device)  # per-direction dy
        self.ar = torch.arange(self.B, device=device)         # [0..B-1] helper

        z = lambda: torch.zeros(self.B, dtype=torch.long, device=device)
        self.head = torch.zeros(self.B, 2, dtype=torch.long, device=device)  # (x, y)
        self.dir = z()
        self.food = torch.zeros(self.B, 2, dtype=torch.long, device=device)
        self.score = z()
        self.steps = z()
        self.length = z()
        self.head_ptr = z()
        self.tail_ptr = z()
        self.body = torch.zeros(self.B, self.CAP, dtype=torch.long, device=device)
        self.occ = torch.zeros(self.B, self.H, self.W, dtype=torch.bool, device=device)

        self.reset(torch.ones(self.B, dtype=torch.bool, device=device))

    def reset(self, mask):
        """Reset every env where mask is True to a fresh length-3 snake."""
        di = mask.nonzero(as_tuple=True)[0]
        if di.numel() == 0:
            return
        cx, cy = self.W // 2, self.H // 2

        self.occ[di] = False
        self.dir[di] = 0
        self.score[di] = 0
        self.steps[di] = 0
        self.length[di] = 3
        self.head[di, 0] = cx
        self.head[di, 1] = cy

        # Body laid out left-to-right: tail=(cx-2,cy) .. head=(cx,cy)
        self.body[di, 0] = cy * self.W + (cx - 2)
        self.body[di, 1] = cy * self.W + (cx - 1)
        self.body[di, 2] = cy * self.W + cx
        self.tail_ptr[di] = 0
        self.head_ptr[di] = 2
        self.occ[di, cy, cx - 2] = True
        self.occ[di, cy, cx - 1] = True
        self.occ[di, cy, cx] = True

        self._place_food(mask)

    def _place_food(self, mask):
        """Sample a uniformly random free cell as food for masked envs."""
        mi = mask.nonzero(as_tuple=True)[0]
        if mi.numel() == 0:
            return
        free = (~self.occ[mi]).reshape(mi.numel(), self.H * self.W).float()
        free += 1e-6  # guard against an all-occupied row (board full)
        flat = torch.multinomial(free, 1).squeeze(1)
        self.food[mi, 0] = flat % self.W
        self.food[mi, 1] = flat // self.W

    def _danger(self, d):
        """For each env, is the cell one step in direction d a collision?"""
        cx = self.head[:, 0] + self.DX[d]
        cy = self.head[:, 1] + self.DY[d]
        oob = (cx < 0) | (cx >= self.W) | (cy < 0) | (cy >= self.H)
        cxc = cx.clamp(0, self.W - 1)
        cyc = cy.clamp(0, self.H - 1)
        body = self.occ[self.ar, cyc, cxc] & ~oob
        return oob | body

    def get_states(self):
        """(B, 11) float state batch, matching game.py's feature order."""
        d = self.dir
        hx, hy = self.head[:, 0], self.head[:, 1]
        fx, fy = self.food[:, 0], self.food[:, 1]
        state = torch.stack([
            self._danger(d),            # danger straight
            self._danger((d + 1) % 4),  # danger right
            self._danger((d - 1) % 4),  # danger left
            d == 2,                     # dir left
            d == 0,                     # dir right
            d == 3,                     # dir up
            d == 1,                     # dir down
            fx < hx,                    # food left
            fx > hx,                    # food right
            fy < hy,                    # food up
            fy > hy,                    # food down
        ], dim=1).float()
        return state

    def step(self, action):
        """Advance all envs by one action.

        action: (B,) long in {0=straight, 1=right, 2=left}.
        Returns (reward, done, final_score) each shaped (B,); final_score is the
        score reached just before any death-triggered reset (valid where done).
        """
        turn = torch.where(action == 1, 1, torch.where(action == 2, -1, 0))
        self.dir = (self.dir + turn) % 4

        nx = self.head[:, 0] + self.DX[self.dir]
        ny = self.head[:, 1] + self.DY[self.dir]
        wall = (nx < 0) | (nx >= self.W) | (ny < 0) | (ny >= self.H)
        nxc = nx.clamp(0, self.W - 1)
        nyc = ny.clamp(0, self.H - 1)
        hit_body = self.occ[self.ar, nyc, nxc] & ~wall

        self.steps += 1
        timeout = self.steps > 100 * self.length
        done = wall | hit_body | timeout
        alive = ~done
        ate = (nx == self.food[:, 0]) & (ny == self.food[:, 1]) & alive

        reward = torch.zeros(self.B, device=self.device)
        reward = torch.where(ate, torch.full_like(reward, 10.0), reward)
        reward = torch.where(done, torch.full_like(reward, -10.0), reward)

        self.score += ate.long()
        final_score = self.score.clone()

        # Insert the new head for every surviving env.
        ai = alive.nonzero(as_tuple=True)[0]
        if ai.numel():
            new_hp = (self.head_ptr[ai] + 1) % self.CAP
            self.body[ai, new_hp] = ny[ai] * self.W + nx[ai]
            self.head_ptr[ai] = new_hp
            self.occ[ai, ny[ai], nx[ai]] = True
            self.head[ai, 0] = nx[ai]
            self.head[ai, 1] = ny[ai]
            self.length[ai] += 1

        # Drop the tail unless the env just ate (then the snake grows).
        remove = alive & ~ate
        ri = remove.nonzero(as_tuple=True)[0]
        if ri.numel():
            tptr = self.tail_ptr[ri]
            tail_flat = self.body[ri, tptr]
            self.occ[ri, tail_flat // self.W, tail_flat % self.W] = False
            self.tail_ptr[ri] = (tptr + 1) % self.CAP
            self.length[ri] -= 1

        # New food where eaten, fresh game where dead.
        self._place_food(ate)
        self.reset(done)

        return reward, done, final_score


class QNet(nn.Module):
    """Same architecture/param names as game.py's Linear_QNet (11 -> h -> 3)."""

    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

    def save(self, file_name=None):
        torch.save(self.state_dict(), file_name or MODEL_FILE)

    def load(self, file_name=None):
        self.load_state_dict(torch.load(file_name or MODEL_FILE, map_location=DEVICE))


class ReplayBuffer:
    """Pre-allocated GPU ring buffer of transitions (no Python-object churn)."""

    def __init__(self, capacity, state_size=11, device=DEVICE):
        self.capacity = capacity
        self.device = device
        self.s = torch.zeros(capacity, state_size, device=device)
        self.a = torch.zeros(capacity, dtype=torch.long, device=device)
        self.r = torch.zeros(capacity, device=device)
        self.s2 = torch.zeros(capacity, state_size, device=device)
        self.d = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.pos = 0
        self.size = 0

    def push(self, s, a, r, s2, d):
        n = s.shape[0]
        idx = (torch.arange(n, device=self.device) + self.pos) % self.capacity
        self.s[idx] = s
        self.a[idx] = a
        self.r[idx] = r
        self.s2[idx] = s2
        self.d[idx] = d
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


def apex_epsilons(num_envs, base=0.5, alpha=8.0, device=DEVICE):
    """Per-env fixed exploration rates: eps_i = base ** (1 + i/(N-1) * alpha).

    Env 0 explores most (~base); the last env is near-greedy. This gives the
    population a wide spread of exploration noise simultaneously (Ape-X).
    """
    i = torch.arange(num_envs, device=device).float()
    denom = max(num_envs - 1, 1)
    return base ** (1.0 + i / denom * alpha)


def load_checkpoint(model):
    games, record = 0, 0
    try:
        model.load()
        with open(STATS_FILE, 'r') as f:
            lines = f.readlines()
            games = int(lines[0].split(':')[1].strip())
            record = int(lines[1].split(':')[1].strip())
        print(f'Loaded model + stats. Games: {games}, Record: {record}')
    except FileNotFoundError:
        print('No saved model/stats found; starting from scratch.')
    return games, record


def save_checkpoint(model, games, record):
    model.save()
    with open(STATS_FILE, 'w') as f:
        f.write(f'Number of games: {games}\n')
        f.write(f'Highest score: {record}\n')


def train(args):
    print(f'Using device: {DEVICE}')
    print(f'Envs: {args.num_envs} | batch: {args.batch_size} | '
          f'buffer: {args.buffer_size} | grid: {args.width}x{args.height}')

    env = VectorizedSnake(args.num_envs, w=args.width, h=args.height, device=DEVICE)
    model = QNet(11, args.hidden, 3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    buffer = ReplayBuffer(args.buffer_size, device=DEVICE)
    epsilons = apex_epsilons(args.num_envs, device=DEVICE)

    base_games, record = load_checkpoint(model)

    # GPU-side counters so the hot loop never syncs to host.
    games_t = torch.zeros((), dtype=torch.long, device=DEVICE)
    score_sum_t = torch.zeros((), device=DEVICE)
    record_t = torch.tensor(float(record), device=DEVICE)

    step = 0
    t0 = time.time()
    last_log_step = 0
    print('Training... (Ctrl+C to stop and save)')
    try:
        while True:
            states = env.get_states()

            with torch.no_grad():
                greedy = model(states).argmax(dim=1)
            rand_a = torch.randint(0, 3, (env.B,), device=DEVICE)
            explore = torch.rand(env.B, device=DEVICE) < epsilons
            actions = torch.where(explore, rand_a, greedy)

            reward, done, final_score = env.step(actions)
            next_states = env.get_states()
            buffer.push(states, actions, reward, next_states, done)

            # One gradient step per env-step (vanilla DQN target).
            if buffer.size >= args.batch_size:
                s, a, r, s2, d = buffer.sample(args.batch_size)
                q = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = model(s2).max(dim=1).values
                    target = r + args.gamma * q_next * (~d)
                loss = criterion(q, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Accumulate finished-game stats on-device (no host sync here).
            n_done = done.sum()
            if n_done > 0:
                games_t += n_done
                fs = final_score[done]
                score_sum_t += fs.sum()
                record_t = torch.maximum(record_t, fs.max())

            step += 1
            if step % args.log_every == 0:
                games = base_games + int(games_t.item())
                cur_record = int(record_t.item())
                mean = (score_sum_t.item() / max(int(games_t.item()), 1))
                now = time.time()
                eps_per_s = env.B * (step - last_log_step) / max(now - t0, 1e-6)
                last_log_step, t0 = step, now
                print(f'step {step:>7} | games {games:>9} | record {cur_record:>3} | '
                      f'mean {mean:5.2f} | {eps_per_s/1e6:5.2f}M env-steps/s')

                if cur_record > record:
                    record = cur_record
                    save_checkpoint(model, games, record)

                if args.max_games and games >= args.max_games:
                    print('Reached --max-games; saving and exiting.')
                    save_checkpoint(model, games, record)
                    return
    except KeyboardInterrupt:
        games = base_games + int(games_t.item())
        cur_record = int(record_t.item())
        record = max(record, cur_record)
        print('\nInterrupted. Saving model and stats...')
        save_checkpoint(model, games, record)
        sys.exit()


def parse_args():
    p = argparse.ArgumentParser(description='Vectorized GPU Snake DQN trainer (Ape-X-lite).')
    p.add_argument('--num-envs', type=int, default=1024, help='Parallel games (default 1024).')
    p.add_argument('--batch-size', type=int, default=2048, help='Replay minibatch size.')
    p.add_argument('--buffer-size', type=int, default=200_000, help='Replay capacity.')
    p.add_argument('--hidden', type=int, default=256, help='Hidden units (keep 256 for model.pth compat).')
    p.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate.')
    p.add_argument('--gamma', type=float, default=0.9, help='Discount factor.')
    p.add_argument('--width', type=int, default=32, help='Grid width in cells (640/20).')
    p.add_argument('--height', type=int, default=24, help='Grid height in cells (480/20).')
    p.add_argument('--log-every', type=int, default=100, help='Steps between status lines.')
    p.add_argument('--max-games', type=int, default=0, help='Stop after this many games (0 = forever).')
    p.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=None,
                   help='Force compute device. Default: auto-detect.')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
