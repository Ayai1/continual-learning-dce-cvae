# hybrid_memory.py
"""
Memory utilities for hybrid latent-replay continual learning.

Contains:
- AdapterManager (orthogonal subspace projector)
- ExemplarMemory (per-class exemplar storage with version tracking)
- NOTE: HotLatentBuffer and DNAMemory are expected to be present in your repo.
  This file only defines/overwrites AdapterManager and ExemplarMemory to avoid
  accidental changes to the other memories you already rely on.
"""

from collections import deque
import random
import torch
import torch.nn as nn

# -------------------------
# AdapterManager: orthogonal subspace projector
# -------------------------
# hybrid_memory.py
"""
Memory utilities for hybrid latent-replay continual learning.

Contains:
- AdapterManager (orthogonal subspace projector)
- ExemplarMemory (per-class exemplar storage with version tracking)
- NOTE: HotLatentBuffer and DNAMemory are expected to be present in your repo.
  This file only defines/overwrites AdapterManager and ExemplarMemory to avoid
  accidental changes to the other memories you already rely on.
"""

from collections import deque
import random
import torch
import torch.nn as nn

# -------------------------
# AdapterManager: orthogonal subspace projector
# -------------------------
class AdapterManager(nn.Module):
    """
    Orthogonal subspace adapter manager.
    - Creates fixed orthonormal basis P_t (latent_dim x per_task_dim) for each task t.
    - project_to_task(mu, t): projects full-latent mu into task t subspace: mu_proj = (P_t P_t^T) mu
    - to_current(z, versions): identity mapping by default (stored z are projected already).
    """
    def __init__(self, latent_dim: int = 128, max_tasks: int = 10):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.max_tasks = int(max_tasks)
        self.per_task_dim = max(1, self.latent_dim // max(1, self.max_tasks))

        # Build fixed orthonormal bases for each task (latent_dim x per_task_dim)
        mats = []
        for t in range(self.max_tasks):
            A = torch.randn(self.latent_dim, self.per_task_dim)
            # Use linalg.qr (works for modern PyTorch); fallback to svd if needed
            try:
                q, r = torch.linalg.qr(A, mode='reduced')
            except Exception:
                u, s, v = torch.linalg.svd(A)
                q = u[:, :self.per_task_dim]
            mats.append(q)  # latent_dim x per_task_dim

        P = torch.stack(mats, dim=0)  # (max_tasks, latent_dim, per_task_dim)
        self.register_buffer("P", P)

        # Precompute P P^T projector for each task: shape (max_tasks, latent_dim, latent_dim)
        PPt = torch.matmul(P, P.transpose(1, 2))
        self.register_buffer("PPt", PPt)

        self.current_version = 0

    def set_current_version(self, v: int):
        self.current_version = int(v)

    def project_to_task(self, mu: torch.Tensor, task_id: int):
        """
        Project mu (B, latent_dim) into the task-specific subspace for task_id.
        If mu is (latent_dim,) treat as single-row.
        """
        if mu is None:
            return None
        if not (0 <= int(task_id) < self.max_tasks):
            # out of range: return mu unchanged
            return mu
        Pmat = self.PPt[int(task_id)].to(mu.device)  # (latent_dim, latent_dim)
        # support both 1D and 2D mu
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)
            single = True
        else:
            single = False
        mu_dev = mu.to(Pmat.device)
        # projected = mu @ Pmat
        mu_proj = torch.matmul(mu_dev, Pmat)
        mu_proj = mu_proj.to(mu.device)
        return mu_proj[0] if single else mu_proj

    def to_current(self, z: torch.Tensor, versions: torch.Tensor):
        """
        Identity mapping by default. We expect stored latents to already live in their
        projected original subspace. This method exists so training loop can call adapters.to_current(z, versions).
        If in the future you want to transform old latents into a new canonical form, implement it here.
        """
        return z

    def forward(self, x):
        return x


# -------------------------
# ExemplarMemory: per-class image storage with version tracking
# -------------------------
class ExemplarMemory:
    """
    Per-class exemplar storage (raw images) with version/task tracking.

    Storage format: for each class c, a deque of (x_tensor_cpu, version_id)
    x_tensor_cpu: CPU tensor shaped (C,H,W) or (1,C,H,W) depending on add API
    version_id: integer task version (0,1,2,...). Use -1 if unknown.
    """

    def __init__(self, capacity_per_class: int = 200, num_classes: int = 10):
        self.capacity = int(capacity_per_class)
        self.num_classes = int(num_classes)
        self.store = {c: deque() for c in range(self.num_classes)}

    def add_batch(self, x_batch, y_batch):
        """
        Backwards-compatible add. x_batch: CPU tensor (B,C,H,W) or list of tensors.
        y_batch: CPU tensor (B,)
        Stores with version = -1 (unknown). Prefer add_batch_with_version().
        """
        self.add_batch_with_version(x_batch, y_batch, version=-1)

    def add_batch_with_version(self, x_batch, y_batch, version: int):
        """
        Add a batch of examples to exemplar memory with a version id.
        x_batch: CPU tensor (B,C,H,W) or single example tensor (1,C,H,W)
        y_batch: CPU tensor (B,) or single-element tensor
        version: int
        """
        # accept lists as well
        if isinstance(x_batch, list):
            for i in range(len(x_batch)):
                x = x_batch[i].cpu().detach().clone()
                y = int(y_batch[i].item()) if hasattr(y_batch[i], 'item') else int(y_batch[i])
                dq = self.store[y]
                dq.append( (x, int(version)) )
                if len(dq) > self.capacity:
                    dq.popleft()
            return

        # If x_batch is a tensor
        if x_batch is None:
            return
        if x_batch.dim() == 4:
            B = x_batch.size(0)
            for i in range(B):
                x = x_batch[i].cpu().detach().clone()
                y = int(y_batch[i].item())
                dq = self.store[y]
                dq.append( (x, int(version)) )
                if len(dq) > self.capacity:
                    dq.popleft()
        elif x_batch.dim() == 3:
            # single example (C,H,W)
            x = x_batch.cpu().detach().clone()
            y = int(y_batch.item()) if hasattr(y_batch, 'item') else int(y_batch)
            dq = self.store[y]
            dq.append( (x, int(version)) )
            if len(dq) > self.capacity:
                dq.popleft()
        else:
            raise ValueError("Unsupported x_batch shape for add_batch_with_version")

    def total_count(self):
        return sum(len(self.store[c]) for c in range(self.num_classes))

    def clear(self):
        self.store = {c: deque() for c in range(self.num_classes)}

    def sample(self, n, device, return_versions: bool = False):
        """
        Random sampling across classes (not balanced).
        Returns:
            xs (tensor B C H W), ys (tensor B), optionally versions (tensor B)
        """
        classes = [c for c in range(self.num_classes) if len(self.store[c]) > 0]
        if len(classes) == 0:
            if return_versions:
                return None, None, None
            return None, None

        per_class = max(1, n // len(classes))
        xs, ys, vs = [], [], []
        for c in classes:
            entries = list(self.store[c])
            k = min(per_class, len(entries))
            chosen = random.sample(entries, k)
            for (x, v) in chosen:
                xs.append(x)
                ys.append(c)
                vs.append(v)
            if len(xs) >= n:
                break

        if len(xs) == 0:
            if return_versions:
                return None, None, None
            return None, None

        xs = torch.stack(xs[:n], dim=0).to(device)
        ys = torch.tensor(ys[:n], dtype=torch.long, device=device)
        if return_versions:
            vs = torch.tensor(vs[:n], dtype=torch.long, device=device)
            return xs, ys, vs
        return xs, ys

    def sample_balanced(self, n, device, return_versions: bool = False):
        """
        Class-balanced sampling: attempts to draw roughly n/num_classes per class.
        """
        total = self.total_count()
        if total == 0:
            if return_versions:
                return None, None, None
            return None, None

        classes = [c for c in range(self.num_classes) if len(self.store[c]) > 0]
        per_class = max(1, n // len(classes))

        xs, ys, vs = [], [], []
        for c in classes:
            entries = list(self.store[c])
            k = min(per_class, len(entries))
            if k <= 0:
                continue
            idxs = random.sample(range(len(entries)), k)
            for idx in idxs:
                x, v = entries[idx]
                xs.append(x)
                ys.append(c)
                vs.append(v)
            if len(xs) >= n:
                break

        if len(xs) == 0:
            if return_versions:
                return None, None, None
            return None, None

        xs = torch.stack(xs[:n], dim=0).to(device)
        ys = torch.tensor(ys[:n], dtype=torch.long, device=device)
        if return_versions:
            vs = torch.tensor(vs[:n], dtype=torch.long, device=device)
            return xs, ys, vs
        return xs, ys

    # small helper to inspect per-class counts
    def counts(self):
        return {c: len(self.store[c]) for c in range(self.num_classes)}
    
class HotLatentBuffer:
    def __init__(self, max_size=50000):
        self.max_size = max_size
        self.storage = []   # list of (z_cpu, y_cpu, version_int)

    def __len__(self):
        return len(self.storage)

    def add_batch(self, z_cpu, y_cpu, version):
        z_cpu = z_cpu.detach().cpu()
        y_cpu = y_cpu.detach().cpu()
        version = int(version)

        for i in range(z_cpu.size(0)):
            self.storage.append((z_cpu[i], int(y_cpu[i].item()), version))
            if len(self.storage) > self.max_size:
                self.storage.pop(0)

    def sample(self, n, device):
        if len(self.storage) == 0:
            return None, None, None

        items = random.sample(self.storage, min(n, len(self.storage)))

        z_list, y_list, v_list = [], [], []
        for z, y, v in items:
            z_list.append(z)
            y_list.append(y)
            v_list.append(v)

        z = torch.stack(z_list).to(device)
        y = torch.tensor(y_list, dtype=torch.long, device=device)
        v = torch.tensor(v_list, dtype=torch.long, device=device)

        return z, y, v
    
class DNAMemory:
    def __init__(self, latent_dim=128, num_classes=10):
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.storage = {c: [] for c in range(num_classes)}  # list of (mu, version)

    def total_prototypes(self):
        return sum(len(self.storage[c]) for c in range(self.num_classes))

    def prototype_coverage_ratio(self, total_classes):
        covered = sum(1 for c in range(total_classes) if len(self.storage[c]) > 0)
        return covered / total_classes

    def ingest_from_hot(self, hot_buffer, version, min_per_class=20):
        counts = {c: 0 for c in range(self.num_classes)}

        for z, y, v in hot_buffer.storage:
            c = int(y)
            if counts[c] < min_per_class:
                self.storage[c].append((z.clone(), int(v)))
                counts[c] += 1

    def sample_latents(self, n, device):
        available = []
        for c in range(self.num_classes):
            for (mu, v) in self.storage[c]:
                available.append((mu, c, v))

        if len(available) == 0:
            return None, None, None

        chosen = random.sample(available, min(n, len(available)))

        z, y, v = zip(*chosen)
        z = torch.stack(z).to(device)
        y = torch.tensor(y, dtype=torch.long, device=device)
        v = torch.tensor(v, dtype=torch.long, device=device)

        return z, y, v

    def get_class_stats_safe(self, y_batch):
        """
        Returns prototype mean for each label in y_batch (if exists)
        """
        device = y_batch.device
        B = y_batch.size(0)

        mu_old = torch.zeros(B, self.latent_dim, device=device)
        valid_mask = torch.zeros(B, dtype=torch.bool, device=device)

        for i in range(B):
            c = int(y_batch[i].item())
            if len(self.storage[c]) > 0:
                # simple mean of stored prototypes
                mus = torch.stack([mu for (mu, _) in self.storage[c]], dim=0).to(device)
                mu_old[i] = mus.mean(dim=0)
                valid_mask[i] = True

        return mu_old, None, valid_mask