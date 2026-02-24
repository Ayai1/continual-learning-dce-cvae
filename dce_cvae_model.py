# dce_cvae_model.py — FIXED DCE-CVAE for CL (robust handling of y=None)
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCECVAEModule(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256), nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        # classifier on mu
        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        h = self.encoder_cnn(x)
        h = self.encoder_fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @torch.no_grad()
    def encode_to_mu(self, x, y=None):
        mu, _ = self.encode(x)
        return mu

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        # y can be None (unconditional) -> use zeros class vector
        if y is None:
            y_oh = torch.zeros(z.size(0), self.num_classes, device=z.device)
        else:
            y_oh = F.one_hot(y.long(), num_classes=self.num_classes).float().to(z.device)
        z_in = torch.cat([z, y_oh], dim=1)
        return self.decoder(z_in)

    def forward(self, x, y):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.classifier(mu)
        # If y is None, decode uses zeros vector (safe)
        x_recon = self.decode(z, y)
        return logits, z, x_recon, mu, logvar

def dce_cvae_loss(x, x_recon, mu, logvar):
    x_recon = torch.clamp(x_recon, 1e-6, 1 - 1e-6)
    x = torch.clamp(x, 0.0, 1.0)
    recon = F.binary_cross_entropy(x_recon, x, reduction='sum') / max(1, x.size(0))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / max(1, x.size(0))
    return recon + kl

def project_latent(self, z):
    return z 