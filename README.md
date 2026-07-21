# Boids GNN — Next-Step Velocity Prediction

A PyTorch Geometric model that learns to predict each boid's next-step
velocity from the current simulation graph, trained on `boids_log.bin`
produced by `boids.py`.

## Task

Given the graph at step *t* (each boid's position/velocity as node
features, plus the separation/cohesion/alignment edges logged at that
step), predict each boid's `[x_vel, y_vel]` at step *t+1*.

This is the standard starting task for GNNs on flocking/particle data —
it uses every column already being logged, has an unambiguous ground
truth (the next row), and is a natural stepping stone toward other tasks
(leader/follower classification, link prediction, longer-horizon rollout)
once it's working.

## Files

| File | Purpose |
|---|---|
| `dataset.py` | Parses `boids_log.bin` and builds one PyG `Data` graph per step |
| `model.py` | `BoidsGNN` — a small relational GCN |
| `train.py` | Trains the model, saves `boids_gnn.pt` and `loss_curve.png` |
| `requirements.txt` | `pip install -r requirements.txt` |

## Data → graph mapping

- **Node features** (`x`): `[x, y, x_vel, y_vel]`, normalized by the
  known world size (900×650) and `MAX_SPEED` (4) from `boids.py`.
- **Edges**: kept as **three separate edge sets** — `edge_index_sep`,
  `edge_index_coh`, `edge_index_ali` — rather than collapsed into one
  graph, since that's the whole point of logging three relation types.
  An edge `j -> i` exists whenever boid `j` appeared in boid `i`'s
  neighbor list for that relation at that step (i.e. `i` "receives" a
  message from each of its detected neighbors).
- **Target** (`y`): the same boid's `[x_vel, y_vel]` at the *next*
  logged step, also normalized.
- The last step in the log has no "next step," so it's dropped — 500
  logged steps become 499 training graphs.

## Model

`BoidsGNN` is a simplified relational GCN: each layer runs a separate
`GCNConv` for separation, cohesion, and alignment edges (plus a linear
self-loop term) and sums the results, so the model can in principle learn
different behavior per relation type rather than treating all neighbors
identically. Two message-passing layers, hidden size 32, followed by a
linear layer down to `[x_vel, y_vel]`.

## Training

```bash
pip install -r requirements.txt
python3 train.py /path/to/boids_log.bin
```

- Split is **by time**, not shuffled — the first 80% of steps train, the
  last 20% validate. A random split would leak future information into
  training (steps close in time look very similar), so this keeps the
  eval honest.
- Batches multiple consecutive-step graphs together for efficiency; PyG's
  batching automatically offsets our custom `edge_index_sep/coh/ali`
  fields since they contain "index" in the name.
- Loss: MSE on normalized velocity. 60 epochs, Adam, lr 1e-3 — all easy
  to change at the top of `train.py`.
- Outputs `boids_gnn.pt` (model weights) and `loss_curve.png`
  (train/val loss over epochs).

## Result on your data

Trained on your 500-step, 36-boid log (499 usable graphs → 400 train /
99 val): validation MSE dropped from ~0.014 to ~0.0007 (normalized
velocity units) over 60 epochs, with train and val loss tracking closely
— no sign of overfitting at this size/epoch count.

## Ideas for next steps

- **Longer-horizon prediction**: instead of 1 step ahead, predict *k*
  steps ahead, or do autoregressive rollout (feed predictions back in as
  input) and see how far it stays stable.
- **Position instead of / in addition to velocity**: trickier because of
  screen-wrap discontinuities — would need to predict a bounded delta
  and handle wrap explicitly, or switch the sim to a bounded (non-wrapping)
  world first.
- **Edge features**: currently edges are just "present/absent" per
  relation. Distance itself could be an edge feature (e.g. via
  `GCNConv`'s edge_weight, or switching to `NNConv`/`GATConv` with edge
  attributes).
- **Leader/follower classification**: once `pre_planned` boids are
  actually implemented in the sim (currently always `False`), this
  becomes a node classification task on the same graphs.
- **Link prediction**: predict which separation/cohesion/alignment edges
  will exist at *t+1*, rather than (or in addition to) velocity.