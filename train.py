"""
Trains BoidsGNN to predict each boid's next-step velocity from the current
graph (positions, velocities, and separation/cohesion/alignment edges).

Usage:
    python3 train.py /path/to/boids_log.bin
"""
import sys
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from dataset import load_boids_graphs
from model import BoidsGNN

EPOCHS = 60
BATCH_SIZE = 8
LR = 1e-3
VAL_FRACTION = 0.2  # last 20% of *time*, not a random split -- avoids
                     # training on the future and validating on the past


def main(bin_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs = load_boids_graphs(bin_path)
    n_val = int(len(graphs) * VAL_FRACTION)
    train_graphs = graphs[: len(graphs) - n_val]
    val_graphs = graphs[len(graphs) - n_val :]
    print(f"{len(graphs)} total steps -> {len(train_graphs)} train / "
          f"{len(val_graphs)} val (split by time, not shuffled)")

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)

    model = BoidsGNN(in_channels=4, hidden_channels=32, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_examples = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_nodes
            total_examples += batch.num_nodes
        train_loss = total_loss / total_examples

        model.eval()
        total_loss, total_examples = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                loss = F.mse_loss(pred, batch.y)
                total_loss += loss.item() * batch.num_nodes
                total_examples += batch.num_nodes
        val_loss = total_loss / total_examples

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"epoch {epoch:3d}/{EPOCHS}  "
                  f"train_mse={train_loss:.5f}  val_mse={val_loss:.5f}")

    torch.save(model.state_dict(), "boids_gnn.pt")
    print("Saved model weights to boids_gnn.pt")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(history["train_loss"], label="train")
        plt.plot(history["val_loss"], label="val")
        plt.xlabel("epoch")
        plt.ylabel("MSE (normalized velocity)")
        plt.title("BoidsGNN next-step velocity prediction")
        plt.legend()
        plt.tight_layout()
        plt.savefig("loss_curve.png", dpi=150)
        print("Saved loss_curve.png")
    except ImportError:
        print("matplotlib not installed -- skipping loss_curve.png")

    return model, history


if __name__ == "__main__":
    bin_path = sys.argv[1] if len(sys.argv) > 1 else "boids_log.bin"
    main(bin_path)