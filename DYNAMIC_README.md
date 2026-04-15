# Dynamic PyQuaticus — Setup, Training & Deployment

## 1. Setup

From the project root:

```bash
conda activate env-full
# or: pip install -e .[torch,ray]
# Robin: conda activate ./env-full
```

---

## 2. Training — quick start

**New run (Blue vs random Red):**
```bash
python rl_test/train_dynamic.py
```

**Resume from a checkpoint with no opponents:**
```bash
python rl_test/train_dynamic.py --resume ./ray_dynamic/iter_360 --red-dummy
```

**Watch while training** (one window, brief freezes each iteration):
```bash
python rl_test/train_dynamic.py --resume ./ray_dynamic/iter_360 --red-dummy --render
```

Checkpoints save to `./ray_dynamic/iter_N/`. Progress is logged to `./ray_dynamic/train.log`.

---

## 3. Deploy / watch a policy

```bash
python rl_test/deploy_dynamic.py ./ray_dynamic/iter_360 --red-dummy
```

Add `--no-render` to run headless. For fixed 3v3: `--team-size-min 3 --team-size-max 3`.

---

## 4. Training defaults

| Setting | Default | Notes |
|---------|---------|-------|
| Iterations | 2000 | Stops after this many iterations |
| Runners | 8 | Parallel envs (0 when `--render`) |
| Speedup | 8× | Sim runs 8× real time |
| Episode time | 600 s | Max seconds per game |
| Score limit | 3 caps | First to 3 captures wins |
| Save every | 12 iters | Checkpoint written every 12 iterations |
| Batch size | 4000 steps | PPO update size (500 when `--render`) |
| Team sizes | 1–3 | Random per-team size each episode |
| Red opponent | random | Use flags below to change |

---

## 5. Training phases

Recommended order: dummy → heuristic → self-play. Use `--resume ./ray_dynamic/iter_N` to continue between phases.

### Phase 1 — Dummy: learn to capture

No Red opponents. Blue learns to navigate and capture. Run ~100–300 iters, then pick a checkpoint for Phase 2.

```bash
python rl_test/train_dynamic.py \
  --red-dummy --max-time 600 --max-score 3 --save-every 12
```

### Phase 2 — Heuristic Red: learn vs an opponent

Start easy, then increase difficulty.

```bash
# Easy
python rl_test/train_dynamic.py \
  --resume ./ray_dynamic/iter_300 --red-heuristic --red-heuristic-mode easy --save-every 12

# Medium
python rl_test/train_dynamic.py \
  --resume ./ray_dynamic/iter_500 --red-heuristic --red-heuristic-mode medium --save-every 12

# Hard
python rl_test/train_dynamic.py \
  --resume ./ray_dynamic/iter_700 --red-heuristic --red-heuristic-mode hard --save-every 12
```

### Phase 3 — Self-play: vs a previous iteration

Red uses an older Blue checkpoint. Resume from latest, pass the older one as `--red-from-checkpoint`.

```bash
python rl_test/train_dynamic.py \
  --resume ./ray_dynamic/iter_700 --red-from-checkpoint ./ray_dynamic/iter_688
```

**Alternative — random Red** (omit Red flags for diversity):
```bash
python rl_test/train_dynamic.py --resume ./ray_dynamic/iter_700
```

### Quick reference

| Phase | Goal | Key flags |
|-------|------|-----------|
| Dummy | Learn to capture | `--red-dummy --max-time 600 --max-score 3` |
| Heuristic | Learn vs opponent | `--red-heuristic --red-heuristic-mode easy` (then `medium` / `hard`) |
| Self-play | Vs previous iteration | `--red-from-checkpoint ./ray_dynamic/iter_M` |
| Random | Generalize | _(no Red flags)_ |

---

## 6. All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--resume PATH` | — | Resume from checkpoint (e.g. `./ray_dynamic/iter_360`) |
| `--red-dummy` | off | No Red opponents — Blue plays alone |
| `--red-heuristic` | off | Red uses built-in heuristic |
| `--red-heuristic-mode` | `easy` | `easy`, `medium`, or `hard` (requires `--red-heuristic`) |
| `--red-from-checkpoint PATH` | — | Red uses Blue policy from this checkpoint (self-play) |
| `--render` | off | Show one game window while training |
| `--iters N` | 2000 | Stop after N iterations |
| `--save-every N` | 12 | Save checkpoint every N iterations |
| `--out-dir PATH` | `./ray_dynamic/` | Where checkpoints and `train.log` are written |
| `--runners N` | 8 | Parallel env runners (ignored when `--render`) |
| `--speedup N` | 8 | Sim speedup factor |
| `--max-time S` | 600 | Max episode time in seconds |
| `--max-score N` | 3 | Score limit per team to end episode |
| `--team-size-min N` | 1 | Min agents per team at episode start (1–3) |
| `--team-size-max N` | 3 | Max agents per team at episode start (1–3) |
| `--tag-removes-agent` | off | Tagged agents are disabled until reinforcement spawns |
| `--reinforcement-interval N` | 0 | Steps between reinforcement checks (0 = off) |
| `--reinforcement-prob F` | 0.5 | Probability of spawning one reinforcement at each check |
| `--fixed-spawn` | off | Deterministic spawn-line placement (default: random on own side) |
| `--no-log-file` | off | Do not write progress to `train.log` |

> **Save on demand:** create an empty file named `SAVE_NOW` in the output dir. The next finished iteration saves a checkpoint and deletes it.

> **Only one Red flag at a time:** `--red-dummy`, `--red-heuristic`, and `--red-from-checkpoint` are mutually exclusive.

---

## 7. Fresh run (no prior checkpoints)

Use a dedicated `--out-dir` to keep checkpoints and logs separate from other runs.

### Phase 1 — Dummy (fresh)
```bash
python rl_test/train_dynamic.py \
  --out-dir ./ray_dynamic_random/ --red-dummy --max-time 600 --max-score 3 --save-every 12
```

### Phase 2 — Heuristic (fresh)
```bash
python rl_test/train_dynamic.py \
  --resume ./ray_dynamic_random/iter_300 --red-heuristic --red-heuristic-mode easy --save-every 12

python rl_test/train_dynamic.py \
  --resume ./ray_dynamic_random/iter_500 --red-heuristic --red-heuristic-mode medium --save-every 12
```

### Phase 3 — Self-play (fresh)
```bash
python rl_test/train_dynamic.py \
  --resume ./ray_dynamic_random/iter_700 --red-from-checkpoint ./ray_dynamic_random/iter_688
```

### Deploy (fresh)
```bash
python rl_test/deploy_dynamic.py ./ray_dynamic_random/iter_360 --red-dummy
```

---

## 8. ray_dynamic_v2 — hard heuristic from scratch

Trains directly against hard Red with no warm-up phases, using the current `caps_and_grabs` reward.
Checkpoints go to `./ray_dynamic_v2/` (gitignored).

**Start from scratch:**
```bash
python rl_test/train_dynamic.py \
  --out-dir ./ray_dynamic_v2/ --red-heuristic --red-heuristic-mode hard \
  --max-time 600 --max-score 3 --save-every 12 --runners 16
```

**Continue:**
```bash
python rl_test/train_dynamic.py \
  --resume ./ray_dynamic_v2/iter_N --red-heuristic --red-heuristic-mode hard \
  --save-every 12 --runners 16
```

**Deploy:**
```bash
python rl_test/deploy_dynamic.py ./ray_dynamic_v2/iter_N --red-heuristic --red-heuristic-mode hard
```

---

## 9. Dynamic settings

All training commands use **1–3 agents per team** by default. On each episode reset, the number of active agents is chosen randomly in that range — extra agents are present but disabled and never move. Active agents spawn at a random location on their own side of the scrimmage line each episode.

**To enable mid-game agent removal and reinforcement:**
```bash
python rl_test/train_dynamic.py --tag-removes-agent --reinforcement-interval 500 --reinforcement-prob 0.5
```

---

## 10. File reference

| File | Purpose |
|------|---------|
| `rl_test/train_dynamic.py` | Main training script — GNN policy, graph obs |
| `rl_test/deploy_dynamic.py` | Run a saved policy — watch only, no training |
| `pyquaticus/envs/dynamic_pyquaticus.py` | Dynamic env — variable teams, dummy mode, reinforcements |
| `pyquaticus/envs/graph_obs_wrapper.py` | Graph observation wrapper |
| `pyquaticus/models/gnn_model.py` | GNN model — message passing, self-node embedding |
| `pyquaticus/utils/rewards.py` | Reward functions including `caps_and_grabs` |