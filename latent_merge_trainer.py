# latent_merge_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

@torch.no_grad()
def collect_latents(model, exemplar_memory, dna_memory, adapters, device, target_dim=None):
    """
    Collects:
      - Real exemplars (x_ex → μ_ex)
      - Prototype-based generative samples (z → decode → μ_replay)
    Returns:
        mu_all  : (N, latent_dim)
        y_all   : (N,)
    """

    model.eval()
    all_mu = []
    all_y = []

    # ---------------------------
    # Collect real exemplar μ
    # ---------------------------
    for cls in range(exemplar_memory.num_classes):
        xs = exemplar_memory.storage[cls]
        if len(xs) == 0:
            continue

        xs = xs.to(device)
        ys = torch.full((xs.size(0),), cls, device=device, dtype=torch.long)

        mu = model.encode_to_mu(xs, ys)
        all_mu.append(mu)
        all_y.append(ys)

    # ---------------------------
    # Collect replay μ via prototypes
    # ---------------------------
    if dna_memory.total_prototypes() > 0:
        z_p, y_p, v_p = dna_memory.sample_latents(
            num_samples=min(2000, dna_memory.total_prototypes()),
            device=device
        )
        if z_p is not None:
            with torch.no_grad():
                x_p = model.decode(z_p, y_p)
                mu_p = model.encode_to_mu(x_p, y_p)

            # Optionally project μ back using adapters (safety)
            if adapters is not None:
                if isinstance(v_p, torch.Tensor):
                    proj = []
                    for i, vid in enumerate(v_p):
                        proj.append(adapters.project_to_task(mu_p[i:i+1], int(vid.item())))
                    mu_p = torch.cat(proj, dim=0)
                else:
                    mu_p = adapters.project_to_task(mu_p, int(v_p))

            all_mu.append(mu_p)
            all_y.append(y_p)

    mu_all = torch.cat(all_mu, dim=0)
    y_all = torch.cat(all_y, dim=0)

    print(f"[MERGE] collected {mu_all.size(0)} latent vectors.")
    return mu_all, y_all


class LatentMerger(nn.Module):
    """
    Simple linear merge layer: mu_out = W * mu
    latent_dim -> target_dim
    You can set target_dim == latent_dim (no compression) or smaller.
    """
    def __init__(self, latent_dim, target_dim):
        super().__init__()
        self.W = nn.Linear(latent_dim, target_dim, bias=False)

    def forward(self, mu):
        return self.W(mu)


def train_latent_merger(
    model,
    exemplar_memory,
    dna_memory,
    adapters,
    latent_dim,
    target_dim=None,
    batch_size=256,
    epochs=10,
    lr=1e-3,
    device=None
):
    """
    Trains a merger W to unify all task-specific subspaces into a single cohesive space.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if target_dim is None:
        target_dim = latent_dim  # no compression

    mu_all, y_all = collect_latents(model, exemplar_memory, dna_memory, adapters, device)

    # Build classifier training set using μ and labels
    merger = LatentMerger(latent_dim, target_dim).to(device)

    ds = TensorDataset(mu_all, y_all)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # freeze everything except the merger + final classifier
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # New classifier head for merged space
    classifier = nn.Linear(target_dim, 10).to(device)

    optim = torch.optim.Adam(list(merger.parameters()) + list(classifier.parameters()), lr=lr)

    # ---------------------------
    # Train small merged classifier
    # ---------------------------
    for epoch in range(epochs):
        total_loss = 0
        correct, total = 0, 0

        for mu, y in loader:
            mu, y = mu.to(device), y.to(device)

            mu_m = merger(mu)
            logits = classifier(mu_m)

            loss = F.cross_entropy(logits, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item() * mu.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += mu.size(0)

        print(f"[MERGE] Epoch {epoch+1}/{epochs}  loss={total_loss/total:.4f}  acc={correct/total:.4f}")

    print("[MERGE] merge training complete.")
    return merger, classifier
