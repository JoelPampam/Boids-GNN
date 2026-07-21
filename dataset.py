"""
Loads boids_log.bin and turns it into a sequence of PyTorch Geometric
Data objects -- one graph per simulation step.

Each graph:
  - x            : node features [x, y, x_vel, y_vel] (normalized)
  - edge_index_sep / edge_index_coh / edge_index_ali : one edge_index per
                    relation type (separation / cohesion / alignment),
                    built from the adjacency lists logged at that step
  - y            : target = next step's [x_vel, y_vel] (normalized), used
                    for the next-step velocity prediction task
  - pre_planned  : per-node flag, carried through in case it's useful later

The binary format (must match boids.py's pack_record / RECORD_HEADER_FMT):
  uint32 step, uint32 boid_id, float32 x, float32 y, float32 x_vel,
  float32 y_vel, uint8 pre_planned, then 3x (uint16 count + uint32[count] ids)
  for separation / cohesion / alignment.
"""
import struct
import torch
from torch_geometric.data import Data

HEADER_FMT = "<IIffffB"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# Known simulation constants (see boids.py) -- used only to normalize
# features into a friendlier range for training.
WORLD_W, WORLD_H = 900.0, 650.0
MAX_SPEED = 4.0


def parse_bin(path):
    """Returns {step: {boid_id: record_dict}}, sorted by step."""
    with open(path, "rb") as f:
        data = f.read()

    offset = 0
    steps = {}
    while offset < len(data):
        step, boid_id, x, y, vx, vy, pre_planned = struct.unpack_from(
            HEADER_FMT, data, offset
        )
        offset += HEADER_SIZE

        neighbor_lists = []
        for _ in range(3):
            (count,) = struct.unpack_from("<H", data, offset)
            offset += 2
            ids = struct.unpack_from(f"<{count}I", data, offset) if count else ()
            offset += 4 * count
            neighbor_lists.append(ids)
        sep, coh, ali = neighbor_lists

        steps.setdefault(step, {})[boid_id] = {
            "x": x, "y": y, "vx": vx, "vy": vy,
            "pre_planned": pre_planned,
            "separation": sep, "cohesion": coh, "alignment": ali,
        }

    return dict(sorted(steps.items()))


def _edges_from_adjacency(boid_records, boid_ids, relation_key):
    """Build a directed edge_index (2, E) tensor: edge j->i whenever j is in
    boid i's neighbor list for `relation_key`. (i "receives" a message from
    each neighbor j it detected within that behavior's radius.)"""
    src, dst = [], []
    for i in boid_ids:
        for j in boid_records[i][relation_key]:
            src.append(j)
            dst.append(i)
    if not src:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def build_graph_sequence(steps):
    """Turns the parsed {step: {boid_id: record}} dict into a list of PyG
    Data objects, one per step (except the last, which has no "next step"
    to use as a training target)."""
    sorted_steps = sorted(steps.keys())
    boid_ids = sorted(steps[sorted_steps[0]].keys())
    n_boids = len(boid_ids)
    assert boid_ids == list(range(n_boids)), (
        "expected boid ids to be a contiguous 0..N-1 range so they can be "
        "used directly as graph node indices"
    )

    graphs = []
    for t_idx in range(len(sorted_steps) - 1):
        step = sorted_steps[t_idx]
        next_step = sorted_steps[t_idx + 1]
        records = steps[step]
        next_records = steps[next_step]

        feats, targets, pre_planned = [], [], []
        for i in boid_ids:
            r = records[i]
            feats.append([
                r["x"] / WORLD_W,
                r["y"] / WORLD_H,
                r["vx"] / MAX_SPEED,
                r["vy"] / MAX_SPEED,
            ])
            nr = next_records[i]
            targets.append([nr["vx"] / MAX_SPEED, nr["vy"] / MAX_SPEED])
            pre_planned.append(r["pre_planned"])

        data = Data(
            x=torch.tensor(feats, dtype=torch.float),
            y=torch.tensor(targets, dtype=torch.float),
            edge_index_sep=_edges_from_adjacency(records, boid_ids, "separation"),
            edge_index_coh=_edges_from_adjacency(records, boid_ids, "cohesion"),
            edge_index_ali=_edges_from_adjacency(records, boid_ids, "alignment"),
            pre_planned=torch.tensor(pre_planned, dtype=torch.float),
        )
        data.num_nodes = n_boids
        graphs.append(data)

    return graphs


def load_boids_graphs(bin_path):
    steps = parse_bin(bin_path)
    return build_graph_sequence(steps)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "boids_log.bin"
    graphs = load_boids_graphs(path)
    print(f"Loaded {len(graphs)} graphs (steps) from {path}")
    g = graphs[0]
    print("Example graph:", g)
    print("  edges (separation):", g.edge_index_sep.shape[1])
    print("  edges (cohesion):  ", g.edge_index_coh.shape[1])
    print("  edges (alignment): ", g.edge_index_ali.shape[1])