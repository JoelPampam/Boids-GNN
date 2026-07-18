# Boids Simulation with Kinematic + Adjacency Logging

A 2D boids flocking simulation (Pygame) that logs per-timestep kinematic
state and neighbor-graph information for every boid, intended as a data
source for downstream graph neural network (GNN) work.

## Contents

- `boids.py` — the simulation. Run it to open an interactive window and
  simultaneously generate `boids_log.csv` and `boids_log.bin` in the
  working directory.

## Requirements

- Python 3.9+
- `pygame` (`pip install pygame`)

## Running

```bash
python3 boids.py
```

A 900x650 window opens with 100 boids. Two output files are created (or
overwritten) in the current directory the moment the script starts:
`boids_log.csv` and `boids_log.bin`.

### Controls

| Input | Action |
|---|---|
| `Space` | Pause / resume the simulation |
| `Left click` | Place a circular obstacle at the cursor |
| `Right click` | Remove the obstacle nearest the cursor |
| `X` | Clear all obstacles |
| `Esc` or close window | Stop the simulation and cleanly close the log files |

Logging only advances while the simulation is unpaused — pausing does not
write new rows.

## Simulation model

Each boid updates its velocity every step based on three classic flocking
rules, plus obstacle avoidance:

| Rule | Radius (px) | Effect |
|---|---|---|
| Separation | 35 | Steer away from boids that are too close |
| Cohesion | 100 | Steer toward the average position of nearby boids |
| Alignment | 100 | Match velocity with nearby boids |

Boids also avoid user-placed obstacles and wrap around the screen edges.
Speed is capped at 4 units/step.

**Note:** Cohesion and alignment currently use the same radius (100), so
their neighbor sets are identical at every step. If your GNN work needs
these as distinct relation types with distinct topology, consider giving
them different radii in `boids.py` (`COHESION_RADIUS` /
`ALIGNMENT_RADIUS` constants near the top of the file).

## Output: `boids_log.csv`

One row per boid per simulation step. Header:

```
step,boid_id,x,y,x_vel,y_vel,pre_planned,separation,cohesion,alignment
```

| Column | Description |
|---|---|
| `step` | Simulation timestep index (increments once per unpaused frame) |
| `boid_id` | Stable integer ID for the boid this row belongs to |
| `x`, `y` | Position |
| `x_vel`, `y_vel` | Velocity components |
| `pre_planned` | `1` if this boid is following a precomputed path, else `0`. Currently always `0` — reserved for a not-yet-implemented feature. |
| `separation` | IDs of boids within the separation radius of this boid, `-` delimited (e.g. `12-47-88`). Empty string if none. |
| `cohesion` | IDs of boids within the cohesion radius, `-` delimited |
| `alignment` | IDs of boids within the alignment radius, `-` delimited |

### These columns as graphs

`separation`, `cohesion`, and `alignment` are each an **adjacency list**
for an undirected graph at that timestep: an edge connects boid *i* and
boid *j* whenever their distance is under that rule's radius. Since
distance is symmetric, each edge is currently listed on both endpoints'
rows (i.e. the adjacency is stored redundantly, not as a deduplicated
edge list). `x, y, x_vel, y_vel` double as node features for that same
timestep.

This is **not** yet in the `edge_index` COO format that PyTorch Geometric
/ DGL expect. A conversion step (CSV/binary → per-step `edge_index`
tensors) is needed before feeding this into most GNN libraries.

## Output: `boids_log.bin`

Same information as the CSV, packed as binary records for compactness.
Little-endian. **Variable-length** — there is no fixed record size, so
the file must be parsed sequentially from the start (read each count
before reading its ID list).

Per boid per step:

```
uint32   step
uint32   boid_id
float32  x
float32  y
float32  x_vel
float32  y_vel
uint8    pre_planned          (0 or 1)
uint16   separation_count
uint32[] separation_ids       (separation_count entries)
uint16   cohesion_count
uint32[] cohesion_ids
uint16   alignment_count
uint32[] alignment_ids
```

Example read loop (Python):

```python
import struct

HEADER_FMT = "<IIffffB"
header_size = struct.calcsize(HEADER_FMT)

with open("boids_log.bin", "rb") as f:
    data = f.read()

offset = 0
while offset < len(data):
    step, boid_id, x, y, x_vel, y_vel, pre_planned = struct.unpack_from(
        HEADER_FMT, data, offset
    )
    offset += header_size

    neighbor_lists = []
    for _ in range(3):  # separation, cohesion, alignment
        (count,) = struct.unpack_from("<H", data, offset)
        offset += 2
        ids = struct.unpack_from(f"<{count}I", data, offset) if count else ()
        offset += 4 * count
        neighbor_lists.append(ids)

    separation_ids, cohesion_ids, alignment_ids = neighbor_lists
```

## Known limitations / next steps

- Cohesion and alignment share a radius, making their graphs identical.
- Adjacency is stored per-node rather than as a deduplicated edge list or
  COO `edge_index` array — a conversion step is needed for direct use
  with PyG/DGL.
- No fixed episode length; the simulation logs indefinitely until the
  window is closed.
- `pre_planned` is a placeholder field, not yet driven by any actual
  precomputed-path logic.