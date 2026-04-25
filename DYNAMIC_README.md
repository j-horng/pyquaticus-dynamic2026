# Dynamic PyQuaticus — Setup, Training & Deployment

## 1. Setup

From the project root:

```bash
conda activate env-full
# or: pip install -e .[torch,ray]
# or: conda activate ./env-full (for Robin)
```

---

## 2. Training — quick start

**New run (Blue vs random Red, 1–6v1–6):**
```bash
python rl_test/train_dynamic.py
```

**Resume from a checkpoint:**
```bash
python rl_test/train_dynamic.py --resume ./training/iter_360 --red-dummy
```

**Watch while training** (one window, brief freezes each iteration):
```bash
python rl_test/train_dynamic.py --resume ./training/iter_360 --red-dummy --render
```

Checkpoints save to `./training/iter_N/`. Progress is logged to `./training/train.log`.

---

## 3. Watch a policy (same CLI as training, no PPO)

```bash
python rl_test/train_dynamic.py --watch --resume ./training/iter_360 --red-dummy
```

Uses the exact same `make_env` and flags as training. All team-size, red-mode, and spawn flags work identically. Omit `--resume` for random Blue.

```bash
# Fixed 4v4, heuristic red
python rl_test/train_dynamic.py --watch --resume ./training/iter_500 \
  --red-heuristic --red-heuristic-mode medium --team-size-min 4 --team-size-max 4
```

---

## 4. Training defaults

| Setting | Default | Notes |
|---------|---------|-------|
| Iterations | 2000 | `--iters` |
| Runners | 8 | Parallel envs; 0 when `--render` |
| Speedup | 8× | Sim runs 8× real time |
| Episode time | 600 s | Max seconds per game |
| Score limit | 3 caps | `--max-score`; episodes **end** when a team hits max score unless you pass `--no-score-end` |
| Save every | 5 iters | `--save-every` |
| Batch size | 0 (auto) | PPO: **16000** steps when headless, **500** when `--render` (`--train-batch-size` overrides) |
| Learning rate | 3e-4 | `--lr` |
| GNN hidden | 128 | `--gnn-hidden` |
| GNN layers | 2 | `--gnn-layers` |
| Team sizes | 1–6 | Random per-team size each episode |
| Red opponent | random | Use flags below to change |
| Spawn | Fixed line | `--random-spawn` for random side placement |

---

## 5. Training phases

Recommended order: dummy → heuristic → self-play. Use `--resume` to continue between phases.

### Phase 1 — Dummy: learn to capture

No Red opponents. Blue learns to navigate and capture the flag alone. Run ~200–400 iters.

```bash
python rl_test/train_dynamic.py --red-dummy
```

### Phase 2 — Heuristic Red: learn vs an opponent

Start easy, increase difficulty as win rate improves. Use per-matchup win rate logs to decide when to advance.

```bash
# Easy
python rl_test/train_dynamic.py \
  --resume ./training/iter_300 --red-heuristic --red-heuristic-mode easy

# Medium
python rl_test/train_dynamic.py \
  --resume ./training/iter_500 --red-heuristic --red-heuristic-mode medium

# Hard
python rl_test/train_dynamic.py \
  --resume ./training/iter_700 --red-heuristic --red-heuristic-mode hard
```

### Phase 3 — Self-play: vs a previous iteration

Red uses an older Blue checkpoint. Good for closing the gap against adaptive opponents.

```bash
python rl_test/train_dynamic.py \
  --resume ./training/iter_900 --red-from-checkpoint ./training/iter_888
```

### Quick reference

| Phase | Goal | Key flags |
|-------|------|-----------|
| Dummy | Learn to capture | `--red-dummy` |
| Heuristic | Learn vs opponent | `--red-heuristic --red-heuristic-mode easy/medium/hard` |
| Self-play | Vs previous iteration | `--red-from-checkpoint ./training/iter_M` |
| Random | Generalize | _(no Red flags)_ |

---

## 6. All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--resume PATH` | — | Resume from checkpoint |
| `--watch` | off | Render behavior instead of training; loads Blue from `--resume` if provided |
| `--render` | off | Show game window while training (0 workers, slow) |
| `--iters N` | 2000 | Stop after N iterations |
| `--save-every N` | 5 | Checkpoint every N iterations |
| `--out-dir PATH` | `./training/` | Checkpoint and log output directory |
| `--runners N` | 8 | Parallel env runners |
| `--envs-per-runner N` | 1 | Env copies per runner (ignored with `--render`) |
| `--speedup N` | 8 | Sim speedup factor |
| `--train-batch-size N` | 0 | PPO batch (0 = auto: **16000** headless, **500** with `--render**) |
| `--lr F` | 3e-4 | PPO learning rate (lower to 1e-4 for fine-tuning) |
| `--gnn-hidden N` | 128 | GNN hidden dimension |
| `--gnn-layers N` | 2 | GNN message passing layers |
| `--entropy-coeff F` | 0.05 | PPO entropy coefficient (0.01 for later phases) |
| `--max-time S` | 600 | Max episode time in seconds |
| `--max-score N` | 3 | Score cap per team |
| `--no-score-end` | off | Keep playing after `--max-score` (default: episode ends on max score) |
| `--team-size-min N` | 1 | Min active agents per team (1–6) |
| `--team-size-max N` | 6 | Max active agents per team (1–6) |
| `--random-spawn` | off | Random position on own side (default: fixed spawn line) |
| `--red-dummy` | off | No Red opponents |
| `--red-stationary` | off | Stationary Red (no-op); default spawn line (see `--stationary-red`) |
| `--stationary-red` | off | Alias for `--red-stationary` |
| `--red-stationary-midfield` | off | Stationary Red on the **center** pair of slots on Red’s default forward spawn row |
| `--stationary-red-midfield` | off | Alias for `--red-stationary-midfield` |
| `--red-stationary-topfield` | off | Stationary Red on the **upper** pair of spawn-row slots (same row as default init) |
| `--stationary-red-topfield` | off | Alias for `--red-stationary-topfield` |
| `--red-stationary-bottomfield` | off | Stationary Red on the **lower** pair of spawn-row slots |
| `--stationary-red-bottomfield` | off | Alias for `--red-stationary-bottomfield` |
| `--red-stationary-block-random` | off | Each episode picks **center / upper / lower** spawn-row pair uniformly |
| `--stationary-red-block-random` | off | Alias for `--red-stationary-block-random` |
| `--red-heuristic` | off | Red uses built-in heuristic |
| `--red-heuristic-mode` | `easy` | `easy`, `medium`, or `hard` |
| `--red-attack-hard` | off | One hard attacker Red, rest disabled |
| `--red-all-attack` | off | All Red use hard attack heuristic |
| `--red-all-defend` | off | All Red use hard defend heuristic |
| `--red-from-checkpoint PATH` | — | Red uses Blue policy from this checkpoint |
| `--tag-removes-agent` | off | Tagged agents are disabled until reinforcement |
| `--reinforcement-interval N` | 0 | Steps between reinforcement checks (0 = off) |
| `--reinforcement-prob F` | 0.5 | Probability of spawning reinforcement at each check |
| `--no-log-file` | off | Skip writing to `train.log` |

> **Save on demand:** `touch training/SAVE_NOW` (Unix) or create `SAVE_NOW` in `out_dir` (Windows) — next completed iteration saves a checkpoint.

> **Only one Red mode at a time.** `--red-dummy`, `--red-stationary`, the stationary **block** flags (`--red-stationary-midfield`, `--red-stationary-topfield`, `--red-stationary-bottomfield`, `--red-stationary-block-random`), `--red-heuristic`, `--red-attack-hard`, `--red-all-attack`, `--red-all-defend`, and `--red-from-checkpoint` are mutually exclusive. **At most one** fixed block flag (mid/top/bottom) may be set; **block-random** cannot combine with those three.

---

## 7. Curriculum example (fresh run, 1–6v1–6)

```bash
# Phase 1: dummy, no opponents
python rl_test/train_dynamic.py --out-dir ./run_a/ --red-dummy

# Phase 2: easy heuristic
python rl_test/train_dynamic.py --resume ./run_a/iter_300 \
  --out-dir ./run_a/ --red-heuristic --red-heuristic-mode easy

# Phase 2b: medium
python rl_test/train_dynamic.py --resume ./run_a/iter_550 \
  --out-dir ./run_a/ --red-heuristic --red-heuristic-mode medium

# Phase 3: self-play
python rl_test/train_dynamic.py --resume ./run_a/iter_800 \
  --out-dir ./run_a/ --red-from-checkpoint ./run_a/iter_788

# Watch the result
python rl_test/train_dynamic.py --watch --resume ./run_a/iter_900 \
  --red-heuristic --red-heuristic-mode hard
```

---

## 8. Dynamic team sizing

All training commands use **1–6 agents per team** by default (up to **6v6**, 12 total). Each episode reset randomly picks how many agents are active per team — extra slots stay disabled and no-op.

Spawn layout with fixed spawn (the training default unless `--random-spawn`):

- Agents 1–3: forward spawn line  
- Agents 4–6: second row, 8 m back toward own flag  

**Fixed team size** (e.g. always 4v4):

```bash
python rl_test/train_dynamic.py --team-size-min 4 --team-size-max 4
```

**Mid-game agent removal and reinforcement:**

```bash
python rl_test/train_dynamic.py --tag-removes-agent --reinforcement-interval 500
```

---

## 9. Metrics

Per-matchup stats are emitted via `DynamicPyQuaticusCallbacks` (`episode.custom_metrics` keys like `{Nb}v{Nr}/win`, `.../ep_len`, `.../blue_caps`, etc.). After each training iteration, a **`matchup_distribution`** line is printed (episode counts per `NvM` since the last iter). TensorBoard / aggregated metrics may show these with a `_mean` suffix depending on RLlib version.

---

## 10. File reference

| File | Purpose |
|------|---------|
| `rl_test/train_dynamic.py` | Training, `--watch`, all CLI flags |
| `pyquaticus/envs/dynamic_pyquaticus.py` | Dynamic env — variable teams, OOB tracking, reinforcements |
| `pyquaticus/envs/graph_obs_wrapper.py` | Graph observation wrapper (`MAX_TEAM_SIZE` knob for 4/5/6v6) |
| `pyquaticus/models/gnn_model.py` | GNN policy — message passing, self-node embedding |
| `pyquaticus/utils/rewards.py` | `caps_and_grabs` reward function |
| `pyquaticus/base_policies/` | Heuristic policies (easy/medium/hard; disabled-agent aware) |
| `test/test_dynamic_env.py` | Smoke tests for all team sizes **1–6** (`pytest` or `python test/test_dynamic_env.py`) |
