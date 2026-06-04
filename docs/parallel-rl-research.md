# Parallel "branches with different noise, learn from the best" — research notes

**Question:** While training on CUDA, can we run many simulations in parallel
(different branches with different noise) at the same time and learn from the best
branch? And would it help the DGX Spark's ~1% GPU utilization?

**Short answer:** Yes — this is a well-studied family of methods, and it splits into
two philosophies. One of them ("learn from *all* branches with diverse exploration")
is also the correct fix for low GPU utilization. The other ("select the *best* branch
and discard the rest") is what people usually mean by "learn from the best," and maps
to evolutionary methods.

---

## Two interpretations, two method families

### A. Learn from ALL branches (diverse exploration, shared learner) — *recommended*

Run N copies of the environment in parallel, each with a **different exploration noise**
(e.g. a different epsilon), batch every step's states through the network in one GPU
forward pass, and feed **all** their experience into one shared replay buffer / one
network. You don't throw branches away — you prioritize the most useful transitions.

- **Ape-X / Distributed Prioritized Experience Replay** (Horgan et al., 2018):
  hundreds of actors, each with a *different* epsilon-greedy value, all writing to a
  shared prioritized replay buffer sampled by a single learner. Actors with different
  epsilon "collect different kinds of experience," which broadens coverage and solves
  hard-exploration tasks. This is almost exactly "many branches at different noise
  levels, learn from the best *experience*." → arXiv:1803.00933
- **Vectorized / GPU environments** (EnvPool, Brax, Isaac Gym, Pgx, PufferLib): run
  thousands of envs at once so the accelerator always has a full batch of work. This is
  THE technique that fixes a near-idle GPU — the small-net DQN here stalls precisely
  because it feeds the GPU one sample at a time. Brax/Isaac/Pgx put the *environment
  itself* on the GPU as tensor ops (no Python game loop), reaching millions of
  steps/sec. → EnvPool arXiv:2206.10558, Pgx arXiv:2303.17503, PufferLib arXiv:2406.12905

**Trade-off:** highest sample efficiency, keeps standard DQN, directly raises GPU
utilization. Main cost: you must vectorize the env (run B games as a batch).

### B. Learn from the BEST branch (selection / evolution) — literal "best branch"

Perturb the *policy parameters* with N different noise vectors, run a rollout per
perturbation, then move toward the ones that scored best. Embarrassingly parallel and
GPU-friendly.

- **Evolution Strategies (ES)** (Salimans et al., 2017): sample N Gaussian parameter
  perturbations ("branches with different noise"), evaluate each by episode return,
  and update the parameters as a **reward-weighted average of all perturbations**
  (not just the single argmax). No backprop — forward pass only, ~2–3× faster per
  step, scales to 1000+ workers via common-random-numbers. Solved MuJoCo humanoid in
  ~10 min on 1,440 workers; competitive on Atari. → arXiv:1703.03864
- **Population-Based Training (PBT)** (Jaderberg et al., 2017): a population trains in
  parallel; periodically the worst ~20% **copy the weights of the best ~20%**
  ("exploit") and perturb their hyperparameters ("explore"). This is the most literal
  "learn from the best branch," and it also auto-tunes LR/epsilon/gamma on a schedule.
  → arXiv:1711.09846
- **CEM / CMA-ES:** sample a population, keep the elite set, refit the sampling
  distribution. Same select-the-best idea, classic black-box optimizers.

**Trade-off:** great for exploration, sparse/deceptive rewards, and non-differentiable
objectives; needs no value function. But it's **sample-inefficient** — it discards most
rollouts and learns only a scalar (return) per branch.

---

## Important nuance: don't *naively* keep only the single best branch

A greedy "every step, keep the argmax branch and drop the others" loop tends to
**collapse exploration** and overfit to early noise. That's why the real methods avoid
it: ES uses a **reward-weighted average over all** perturbations, and PBT only
**periodically** exploits (every few thousand steps) while continuing to explore. If you
build "learn from the best branch," copy these guardrails.

---

## Recommendation for this project (Snake DQN on a DGX Spark)

1. **Biggest win, do this first — vectorized environments (family A).** Refactor
   `SnakeGame` so B games (e.g. 256–4096) step in parallel as a batched tensor/array
   state, and run one batched network forward per step. This is what actually fills the
   Spark's GPU and fixes the 1% utilization — the current bottleneck is tiny per-step
   ops, not the device. Keep the standard DQN; give each parallel env a **different
   epsilon** (Ape-X-lite) so you get "branches with different noise" for free.
2. **To truly saturate the GPU**, make the env itself tensor-resident (Brax/Pgx style):
   represent the board as integer tensors and implement move/collision/fruit as vector
   ops, with **no pygame** in the training path. Snake is a grid, so this is very doable.
3. **If you specifically want "pick the best branch"**, use **ES** (simplest to
   parallelize on one GPU, no backprop) or **PBT** (if you also want hyperparameter
   tuning). Expect worse sample efficiency than vectorized DQN, but strong robustness.
4. Realistic expectation: an 11→256→3 network is tiny. Even vectorized, you won't
   "max out" a Blackwell GPU — but you'll go from ~1% to a useful fraction and train
   far faster in wall-clock by stepping thousands of games at once.

---

## Sources

- Horgan et al., *Distributed Prioritized Experience Replay (Ape-X)* — https://arxiv.org/abs/1803.00933
- Salimans et al., *Evolution Strategies as a Scalable Alternative to RL* — https://arxiv.org/abs/1703.03864 · https://openai.com/index/evolution-strategies/
- Jaderberg et al., *Population Based Training of Neural Networks* — https://arxiv.org/abs/1711.09846
- Weng et al., *EnvPool* — https://arxiv.org/abs/2206.10558
- Koyamada et al., *Pgx: Hardware-Accelerated Parallel Game Simulators* — https://arxiv.org/abs/2303.17503
- Suarez, *PufferLib* — https://arxiv.org/pdf/2406.12905
