# hybrid_dna_dce_cvae_main.py
import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from hybrid_dna_dce_cvae_train import hybrid_dna_dce_cvae_continual_learning
from latent_merge_trainer import train_latent_merger, LatentMerger
from hybrid_memory import AdapterManager

# -------------------------
# Split-MNIST builder
# -------------------------
def build_split_mnist(batch_size=128, num_workers=None, pin_memory=True):
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    splits = [(0,1),(2,3),(4,5),(6,7),(8,9)]
    tasks = []
    for a, b in splits:
        train_subset = [(x,y) for (x,y) in train_dataset if int(y) in (a,b)]
        test_subset  = [(x,y) for (x,y) in test_dataset if int(y) in (a,b)]

        if num_workers is None:
            cpu_count = os.cpu_count() or 4
            num_workers = min(4, max(0, cpu_count // 2))

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
        test_loader  = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                                  num_workers=max(0, num_workers//2), pin_memory=pin_memory)
        tasks.append((train_loader, test_loader, (a,b)))
    return tasks

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    print("Initializing Pure Latent Replay Hybrid System…")

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    pin_memory = True if use_cuda else False

    cpu_count = os.cpu_count() or 4
    num_workers = min(4, max(0, cpu_count // 2))

    tasks = build_split_mnist(batch_size=128, num_workers=num_workers, pin_memory=pin_memory)

    # Continual training (this trains model in-place and returns acc matrix)
    acc_matrix = hybrid_dna_dce_cvae_continual_learning(
        tasks,
        latent_dim=128,
        lr=5e-4,
        beta_vae=0.20,                # stronger latent regularization
        epochs_per_task=6,
        hot_buffer_size=50000,
        exemplar_capacity=1000,
        min_per_class_for_dna=80,
        stability_gamma=0.20,
    )

    print("\nFinal Accuracy Matrix (per task evaluation):")
    for i, row in enumerate(acc_matrix):
        print(f"Task {i+1}: {row}")

    # Save plots (already attempted by training, but ensure an accuracy_matrix plot)
    try:
        plt.figure(figsize=(8,6))
        for i, row in enumerate(acc_matrix):
            plt.plot(range(1, len(row)+1), row, marker='o', label=f'Train task {i+1}')
        plt.xlabel("Evaluation Task")
        plt.ylabel("Accuracy")
        plt.title("Pure Latent Replay — SplitMNIST (generative-refined)")
        plt.legend()
        plt.grid(True)
        plt.savefig("accuracy_matrix_merged.png", dpi=150)
        plt.close()
    except Exception:
        pass

    # -------------------------
    # Latent merger: train unified merger + classifier
    # -------------------------
    # The training function collects exemplars & prototype-derived latents internally
    # and trains a merger -> merged_classifier
    from latent_merge_trainer import train_latent_merger

    # Determine latent dim and adapters (these must match training)
    latent_dim = 128
    # adapters instance is created inside training; create a placeholder adapter instance to pass in.
    # If you want to reuse the exact adapter instance from training, adjust to import/return it from the training function.
    adapters = AdapterManager(latent_dim=latent_dim).to(device)

    # NOTE: exemplar_memory and dna_memory are managed inside the continual training function.
    # we expect them to be accessible or returned; since the training function does not currently return them,
    # we'll try to import them from the module namespace if present. Otherwise, user should load saved memories.
    try:
        # Try to access exemplar_memory and dna_memory created during training via the training module's globals
        import hybrid_dna_dce_cvae_train as trainer_mod
        exemplar_memory = getattr(trainer_mod, "exemplar_memory", None)
        dna_memory = getattr(trainer_mod, "dna_memory", None)
        model = getattr(trainer_mod, "model", None)
    except Exception:
        exemplar_memory = None
        dna_memory = None
        model = None

    # Fallback: try loading model from file if training module didn't keep objects
    if model is None:
        # attempt to load model saved by training (if you saved it); else instruct user
        raise RuntimeError("Model object not found in training module. Please ensure `hybrid_dna_dce_cvae_continual_learning` returns or exposes `model`, `exemplar_memory`, and `dna_memory`, or adjust this main script accordingly.")

    if exemplar_memory is None or dna_memory is None:
        raise RuntimeError("ExemplarMemory and/or DNAMemory not found in training module namespace. Ensure they are accessible for merger training.")

    # Train merger
    merger, merged_classifier = train_latent_merger(
        model=model,
        exemplar_memory=exemplar_memory,
        dna_memory=dna_memory,
        adapters=adapters,
        latent_dim=latent_dim,
        target_dim=64,   # compress 128 -> 64 (change to latent_dim to keep same size)
        epochs=12,
        lr=1e-3,
        device=device
    )

    # Save merger + classifier
    torch.save(merger.state_dict(), "merger.pt")
    torch.save(merged_classifier.state_dict(), "merged_classifier.pt")
    print("Saved merger.pt and merged_classifier.pt")

    # -------------------------
    # Evaluate merged model on all test sets
    # -------------------------
    merger.eval(); merged_classifier.eval(); model.eval()
    def eval_merged_on_loader(loader):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                mu = model.encode_to_mu(x, y)  # encode (note: encode_to_mu ignores y in current model)
                mu_m = merger(mu)
                logits = merged_classifier(mu_m)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total>0 else 0.0

    merged_acc_matrix = []
    for t_eval, (_, test_l, _) in enumerate(tasks):
        acc = eval_merged_on_loader(test_l)
        merged_acc_matrix.append(acc)
        print(f"Merged Eval Task {t_eval+1}: {acc:.4f}")

    # Save merged accuracy report
    with open("merged_accuracy.txt","w") as f:
        for i, a in enumerate(merged_acc_matrix):
            f.write(f"Task {i+1}: {a:.6f}\n")
    print("Saved merged_accuracy.txt")

    print("Done.")
