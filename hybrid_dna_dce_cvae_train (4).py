# hybrid_dna_dce_cvae_train.py
import os
import random
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List

from dce_cvae_model import DCECVAEModule, dce_cvae_loss
from hybrid_memory import HotLatentBuffer, DNAMemory, AdapterManager, ExemplarMemory

# Safety & tuning constants
MAX_HOT_RATIO = 0.40
MAX_COLD_RATIO = 0.20
DEFAULT_EXEMPLAR_WEIGHT = 1.5
PRINT_BATCH_EVERY = 250

# VAE replay weighting (tuneable)
REPLAY_VAE_WEIGHT = 0.4

@torch.no_grad()
def test_hybrid_dna(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mu = model.encode_to_mu(x, y)
        logits = model.classifier(mu)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0

class ReplayController:
    def __init__(self, num_classes_total: int = 10):
        self.num_classes_total = num_classes_total

    def task_level_params(self, task_id: int, dna_memory: DNAMemory):
        # base desired values (controller)
        hot_ratio = 0.30
        cold_ratio = 0.10
        base_align = 0.25
        coverage = dna_memory.prototype_coverage_ratio(self.num_classes_total)
        warmup_epochs = 1 if (coverage < 0.08 and task_id > 0) else 0

        # clamp to globals
        hot_ratio = min(hot_ratio, MAX_HOT_RATIO)
        cold_ratio = min(cold_ratio, MAX_COLD_RATIO)
        return float(hot_ratio), float(cold_ratio), float(base_align), int(warmup_epochs)

def freeze_early_conv(model: DCECVAEModule):
    for name in ("encoder_img", "encoder_cnn", "encoder_fc"):
        if hasattr(model, name):
            module = getattr(model, name)
            for p in module.parameters():
                p.requires_grad = False

def unfreeze_all(model: DCECVAEModule):
    for p in model.parameters():
        p.requires_grad = True

def _project_batch_by_versions(adapters: AdapterManager, mu_batch: torch.Tensor, versions: torch.Tensor):
    """
    Project each row of mu_batch into its corresponding task subspace using adapters.project_to_task.
    versions: tensor of ints (batch,)
    returns projected tensor on same device/dtype as mu_batch
    """
    if mu_batch is None or versions is None:
        return mu_batch
    proj_list = []
    for i, vid in enumerate(versions):
        proj_mu_i = adapters.project_to_task(mu_batch[i : i+1], int(vid.item()))
        proj_list.append(proj_mu_i)
    return torch.cat(proj_list, dim=0)

def train_one_task_hybrid(
    model,
    train_loader,
    optimizer,
    device,
    hot_buffer: HotLatentBuffer,
    dna_memory: DNAMemory,
    adapters: AdapterManager,
    exemplar_memory: ExemplarMemory,
    current_version: int,
    controller: ReplayController,
    task_id: int,
    beta_vae: float = 0.05,
    epochs_per_task: int = 8,
    stability_gamma: float = 0.40,
    exemplar_weight: float = DEFAULT_EXEMPLAR_WEIGHT,
):
    model.train()
    adapters.set_current_version(current_version)

    hot_ratio, cold_ratio, base_align, warmup_epochs = controller.task_level_params(task_id, dna_memory)
    if task_id == 0:
        hot_ratio = 0.0; cold_ratio = 0.0; base_align = 0.0; warmup_epochs = 0

    print(f"  HOT={hot_ratio:.3f}, COLD={cold_ratio:.3f}, BASE_ALIGN={base_align:.3f}, WARMUP_EPOCHS={warmup_epochs}")

    exemplar_local = exemplar_memory

    for epoch in range(epochs_per_task):
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            if epoch < warmup_epochs:
                n_hot, n_cold = 0, 0
            else:
                n_hot = int(batch_size * hot_ratio) if len(hot_buffer) > 0 else 0
                n_cold = int(batch_size * cold_ratio) if dna_memory.total_prototypes() > 0 else 0

            # -----------------------------------------------------------------
            # Forward real images
            # -----------------------------------------------------------------
            logits_real, z_real, x_recon, mu_real, logvar_real = model(x, y)

            # Project encoder output into the current task subspace
            try:
                mu_real = adapters.project_to_task(mu_real, current_version)
                logvar_real = adapters.project_to_task(logvar_real, current_version)
            except Exception:
                pass

            z_real_for_class = mu_real

            # -----------------------------------------------------------------
            # Exemplars: balanced sample if available (encode & project)
            # -----------------------------------------------------------------
            z_ex, y_ex = None, None
            if exemplar_local is not None and exemplar_local.total_count() > 0:
                cur_ex_count = max(1, int(batch_size * 0.45))
                x_ex, y_ex = exemplar_local.sample_balanced(cur_ex_count, device)
                if x_ex is not None:
                    try:
                        z_ex = model.encode_to_mu(x_ex, y_ex)
                        # project exemplar mus into current_version subspace for classifier consistency
                        z_ex = adapters.project_to_task(z_ex, current_version)
                    except Exception:
                        z_ex = None

            # -----------------------------------------------------------------
            # HOT replay: sample stored mus, map to current, decode & re-encode,
            # then project re-encoded mus BACK to their original task subspace
            # -----------------------------------------------------------------
            z_hot, y_hot = None, None
            if n_hot > 0:
                z_hot, y_hot, v_hot = hot_buffer.sample(n_hot, device)
                if z_hot is not None:
                    try:
                        # ensure on device and let adapter map if needed (identity or transform)
                        z_hot = adapters.to_current(z_hot.to(device), v_hot)

                        with torch.no_grad():
                            x_hot_recon = model.decode(z_hot, y_hot)
                            mu_reenc = model.encode_to_mu(x_hot_recon, y_hot)

                        # project per-sample back to original task subspace
                        if isinstance(v_hot, torch.Tensor):
                            z_hot = _project_batch_by_versions(adapters, mu_reenc, v_hot)
                        else:
                            z_hot = adapters.project_to_task(mu_reenc, int(v_hot))

                    except Exception:
                        z_hot = None

            # -----------------------------------------------------------------
            # COLD prototypes: same treatment
            # -----------------------------------------------------------------
            z_cold, y_cold = None, None
            if n_cold > 0:
                z_cold, y_cold, v_cold = dna_memory.sample_latents(n_cold, device)
                if z_cold is not None:
                    try:
                        z_cold = adapters.to_current(z_cold.to(device), v_cold)
                        with torch.no_grad():
                            x_cold_recon = model.decode(z_cold, y_cold)
                            mu_reenc = model.encode_to_mu(x_cold_recon, y_cold)
                        if isinstance(v_cold, torch.Tensor):
                            z_cold = _project_batch_by_versions(adapters, mu_reenc, v_cold)
                        else:
                            z_cold = adapters.project_to_task(mu_reenc, int(v_cold))
                    except Exception:
                        z_cold = None

            # -----------------------------------------------------------------
            # Build classifier batch (mu only) — ensure all mus are on same device
            # -----------------------------------------------------------------
            z_list = [z_real_for_class]; y_list = [y]
            if z_ex is not None: z_list.append(z_ex); y_list.append(y_ex)
            if z_hot is not None: z_list.append(z_hot); y_list.append(y_hot)
            if z_cold is not None: z_list.append(z_cold); y_list.append(y_cold)

            z_all = torch.cat(z_list, dim=0)
            y_all = torch.cat(y_list, dim=0)

            n_new = z_real_for_class.size(0)
            n_ex = z_ex.size(0) if z_ex is not None else 0
            n_hot_batch = z_hot.size(0) if z_hot is not None else 0
            n_cold_batch = z_cold.size(0) if z_cold is not None else 0
            n_replay = n_ex + n_hot_batch + n_cold_batch

            logits_all = model.classifier(z_all)
            # safety
            if logits_all.size(0) != y_all.size(0):
                raise RuntimeError(f"Mismatch logits {logits_all.size()} vs labels {y_all.size()}")

            if n_replay > 0:
                ce_new = F.cross_entropy(logits_all[:n_new], y_all[:n_new])
                ce_replay = F.cross_entropy(logits_all[n_new:n_new + n_replay], y_all[n_new:n_new + n_replay])
                ce_loss = ce_new + exemplar_weight * ce_replay
            else:
                ce_loss = F.cross_entropy(logits_all, y_all)

            # -----------------------------------------------------------------
            # VAE loss: split real vs replay and weight replay contribution
            # -----------------------------------------------------------------
            # real-only VAE loss
            vae_real = dce_cvae_loss(x, x_recon, mu_real, logvar_real)

            # build replay tensors for VAE (if present)
            x_replay_list = []
            x_recon_replay_list = []
            mu_replay_list = []
            logvar_replay_list = []

            if z_hot is not None:
                x_hot_gen = model.decode(z_hot, y_hot)
                _, _, x_h_recon, mu_h, logvar_h = model(x_hot_gen, y_hot)
                # project re-encoded mus into their original task subspaces (safety)
                try:
                    if isinstance(v_hot, torch.Tensor):
                        mu_h = _project_batch_by_versions(adapters, mu_h, v_hot)
                    else:
                        mu_h = adapters.project_to_task(mu_h, int(v_hot))
                except Exception:
                    pass
                x_replay_list.append(x_hot_gen); x_recon_replay_list.append(x_h_recon); mu_replay_list.append(mu_h); logvar_replay_list.append(logvar_h)

            if z_cold is not None:
                x_cold_gen = model.decode(z_cold, y_cold)
                _, _, x_c_recon, mu_c, logvar_c = model(x_cold_gen, y_cold)
                try:
                    if isinstance(v_cold, torch.Tensor):
                        mu_c = _project_batch_by_versions(adapters, mu_c, v_cold)
                    else:
                        mu_c = adapters.project_to_task(mu_c, int(v_cold))
                except Exception:
                    pass
                x_replay_list.append(x_cold_gen); x_recon_replay_list.append(x_c_recon); mu_replay_list.append(mu_c); logvar_replay_list.append(logvar_c)

            if len(x_replay_list) > 0:
                x_replay = torch.cat(x_replay_list, dim=0)
                x_recon_replay = torch.cat(x_recon_replay_list, dim=0)
                mu_replay = torch.cat(mu_replay_list, dim=0)
                logvar_replay = torch.cat(logvar_replay_list, dim=0)
                vae_replay = dce_cvae_loss(x_replay, x_recon_replay, mu_replay, logvar_replay)
            else:
                vae_replay = torch.tensor(0.0, device=device)

            vae_loss = vae_real + REPLAY_VAE_WEIGHT * vae_replay

            # -----------------------------------------------------------------
            # Alignment / stability: compare to prototype aggregates (coarse)
            # -----------------------------------------------------------------
            L_align = torch.tensor(0.0, device=device)
            L_stability = torch.tensor(0.0, device=device)
            if dna_memory.total_prototypes() > 0 and base_align > 0.0:
                try:
                    mu_old, _, valid_mask = dna_memory.get_class_stats_safe(y)
                    if valid_mask.any():
                        L_align = ((mu_real - mu_old) ** 2).mean()
                except Exception:
                    L_align = torch.tensor(0.0, device=device)
                try:
                    mu_old_s, _, valid_mask_s = dna_memory.get_class_stats_safe(y)
                    if valid_mask_s.any():
                        mu_new_local = mu_real[valid_mask_s]
                        mu_old_local = mu_old_s[valid_mask_s]
                        if mu_new_local.numel() > 0:
                            L_stability = ((mu_new_local - mu_old_local) ** 2).mean()
                except Exception:
                    L_stability = torch.tensor(0.0, device=device)

            # -----------------------------------------------------------------
            # Final loss and optimization
            # -----------------------------------------------------------------
            loss = ce_loss + beta_vae * vae_loss + 0.5 * base_align * L_align + stability_gamma * L_stability

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Throttled debug print (safe — inside the batch loop)
            if (batch_idx % PRINT_BATCH_EVERY) == 0:
                # ----- SAFE DEBUG LOGGING -----
                def fmt(x):
                    if isinstance(x, (float, int)):
                        return f"{x:.4f}"
                    if isinstance(x, torch.Tensor):
                        # scalar tensor
                        try:
                            return f"{x.item():.4f}"
                        except Exception:
                            return "tensor"
                    return "n/a"

                ce_new_val     = fmt(ce_new if 'ce_new' in locals() else None)
                ce_replay_val  = fmt(ce_replay if 'ce_replay' in locals() else None)
                vae_real_val   = fmt(vae_real if 'vae_real' in locals() else None)
                vae_replay_val = fmt(vae_replay if 'vae_replay' in locals() else None)

                print(
                    f"[DBG] task={task_id} epoch={epoch} batch={batch_idx} "
                    f"bs={batch_size} n_hot={n_hot} n_cold={n_cold} "
                    f"hot_buf={len(hot_buffer)} exemplars={exemplar_local.total_count() if exemplar_local else 0}"
                )

                print(
                    f"      [DBG-Loss] ce_new={ce_new_val} ce_replay={ce_replay_val} "
                    f"vae_real={vae_real_val} vae_replay={vae_replay_val} "
                    f"L_align={float(L_align):.6f} L_stab={float(L_stability):.6f}"
                )

# --------------------------------

            # Update memories after step (store CPU)
            with torch.no_grad():
                try:
                    hot_buffer.add_batch(mu_real.detach().cpu(), y.detach().cpu(), current_version)
                except Exception:
                    pass
                if exemplar_local is not None:
                    try:
                        exemplar_local.add_batch(x.detach().cpu(), y.detach().cpu())
                    except Exception:
                        pass

def hybrid_dna_dce_cvae_continual_learning(
    tasks,
    latent_dim: int = 128,
    lr: float = 5e-4,
    beta_vae: float = 0.2,                 # increased from 0.05 to strengthen latent regularization
    epochs_per_task: int = 8,
    hot_buffer_size: int = 50000,
    exemplar_capacity: int = 1000,
    min_per_class_for_dna: int = 80,       # slightly larger requirement
    device_override: str = None,
    stability_gamma: float = 0.20,         # stronger stability weight
):
    device = device_override if device_override is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    model = DCECVAEModule(latent_dim=latent_dim, num_classes=10).to(device)

    hot_buffer = HotLatentBuffer(max_size=hot_buffer_size)
    dna_memory = DNAMemory(latent_dim=latent_dim)
    adapters = AdapterManager(latent_dim).to(device)

    exemplar_per_class = max(20, exemplar_capacity // 10)
    exemplar_memory = ExemplarMemory(capacity_per_class=exemplar_per_class, num_classes=10)

    controller = ReplayController(num_classes_total=10)

    acc_matrix = []
    current_version = 0
    total_tasks = len(tasks)

    for task_id, (train_loader, test_loader, labels) in enumerate(tasks):
        print(f"\n=== Training Task {task_id+1}/{total_tasks}, labels={labels} ===")

        # Keep encoder unfrozen while training this task
        unfreeze_all(model)

        # slower LR for encoder, faster for classifier/decoder
        enc_params = list(model.encoder_cnn.parameters()) + list(model.encoder_fc.parameters()) + list(model.fc_mu.parameters()) + list(model.fc_logvar.parameters())
        other_params = [p for n,p in model.named_parameters() if not any(n.startswith(epn) for epn in ("encoder_cnn","encoder_fc","fc_mu","fc_logvar")) and p.requires_grad]

        optimizer = torch.optim.Adam([
            {'params': enc_params, 'lr': lr * 0.10},   # encoder learning rate reduced further
            {'params': other_params, 'lr': lr}         # classifier/decoder use base lr
        ], lr=lr)

        train_one_task_hybrid(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            hot_buffer=hot_buffer,
            dna_memory=dna_memory,
            adapters=adapters,
            exemplar_memory=exemplar_memory,
            current_version=current_version,
            controller=controller,
            task_id=task_id,
            beta_vae=beta_vae,
            epochs_per_task=epochs_per_task,
            stability_gamma=stability_gamma,
            exemplar_weight=DEFAULT_EXEMPLAR_WEIGHT,
        )

        # Evaluate seen tasks
        row = []
        for t_eval, (_, test_l, _) in enumerate(tasks[: task_id + 1]):
            acc = test_hybrid_dna(model, test_l, device)
            row.append(acc)
            print(f"  Test Task {t_eval+1}: {acc:.4f}")
        acc_matrix.append(row)

        # Ingest prototypes from hot buffer and freeze early conv
        dna_memory.ingest_from_hot(hot_buffer, current_version, min_per_class=min_per_class_for_dna)
        freeze_early_conv(model)

        current_version += 1

    # Plot forgetting curves (task-by-task)
    try:
        plt.figure(figsize=(8,6))
        for i, row in enumerate(acc_matrix):
            plt.plot(row, marker='o', label=f'Task {i+1}')
        plt.xlabel("Evaluation Task")
        plt.ylabel("Accuracy")
        plt.title("Continual Learning — Pure Latent Replay (generative-refined, orthogonal-subspaces)")
        plt.legend()
        plt.grid(True)
        plt.savefig("forgetting_curves.png")
        plt.close()
    except Exception:
        pass

    return acc_matrix, model, exemplar_memory, dna_memory, adapters

