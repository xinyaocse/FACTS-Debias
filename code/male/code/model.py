import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class Encoder(nn.Module):
    def __init__(self, token_dim=768, latent_dim=768):
        super().__init__()
        self.shared = nn.Linear(token_dim, latent_dim)
        self.specific = nn.Linear(token_dim, latent_dim)

    def forward(self, x):
        z_shared = self.shared(x)
        z_specific = self.specific(x)
        return z_shared, z_specific


class Classifier(nn.Module):
    def __init__(self, latent_dim=768, num_classes=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, num_classes)
        )

    def forward(self, z):
        return self.classifier(z)


class AdversarialClassifier(nn.Module):
    def __init__(self, latent_dim=768, num_classes=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, num_classes)
        )

    def forward(self, z):
        return self.classifier(z)


class Decoder(nn.Module):
    """
    Use [z_shared, z_specific] to reconstruct x.
    """
    def __init__(self, latent_dim=768, token_dim=768):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, token_dim)
        )

    def forward(self, z_shared, z_specific):
        z = torch.cat([z_shared, z_specific], dim=-1)
        x_recon = self.decoder(z)
        return x_recon


def orthogonality_loss(a, b):
    """
    Encourage z_shared and z_specific to be orthogonal.
    """
    return torch.mean(torch.abs(torch.sum(a * b, dim=-1)))


def variance_loss(z):
    """
    Prevent z_specific from collapsing.
    """
    if z.size(0) <= 1:
        return torch.zeros((), device=z.device)
    return -torch.var(z, dim=0, unbiased=False).mean()


def class_specific_direction_loss(z_specific, y, num_classes=3, eps=1e-8):
    """
    Encourage z_specific to form class-specific directions.

    1. Intra-class compactness:
       Samples from the same class should be close to their class mean.

    2. Inter-class separation:
       Class mean directions should be separated.
    """
    device = z_specific.device

    mus = []
    present = []

    for c in range(num_classes):
        mask = y == c
        if mask.any():
            mu_c = z_specific[mask].mean(dim=0)
            mus.append(mu_c)
            present.append(c)

    if len(mus) == 0:
        zero = torch.tensor(0.0, device=device)
        return zero, zero

    loss_intra = torch.tensor(0.0, device=device)

    for mu_c, c in zip(mus, present):
        mask = y == c
        loss_intra = loss_intra + F.mse_loss(
            z_specific[mask],
            mu_c.unsqueeze(0).expand_as(z_specific[mask])
        )

    loss_intra = loss_intra / len(mus)

    if len(mus) <= 1:
        loss_inter = torch.tensor(0.0, device=device)
        return loss_intra, loss_inter

    mus = torch.stack(mus, dim=0)
    mus = F.normalize(mus, dim=-1, eps=eps)

    sim = mus @ mus.t()

    idx = torch.triu_indices(
        sim.size(0),
        sim.size(1),
        offset=1,
        device=device
    )

    loss_inter = (sim[idx[0], idx[1]] ** 2).mean()

    return loss_intra, loss_inter


def train_model(
    encoder,
    classifier,
    adv_classifier,
    decoder,
    dataloader,
    opt_main,
    opt_adv,
    device,
    class_weights,
    num_classes=3,
    adv_alpha=5.0,
    recon_alpha=0.1,
    proto_intra_alpha=0.05,
    proto_inter_alpha=0.10,
    proto_warmup_epochs=3,
    epochs=30,
):
    encoder.train()
    classifier.train()
    adv_classifier.train()
    decoder.train()

    for epoch in range(epochs):
        total_loss = 0.0

        grl_alpha = min(
            1.0,
            0.1 + 0.9 * epoch / max(1, epochs // 3)
        )

        if proto_warmup_epochs > 0:
            w_proto = min(1.0, (epoch + 1) / float(proto_warmup_epochs))
        else:
            w_proto = 1.0

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            # Step A: train adversarial classifier with detached z_shared
            z_shared, z_specific = encoder(x)

            adv_logits_clf = adv_classifier(z_shared.detach())
            loss_adv_clf = F.cross_entropy(adv_logits_clf, y)

            opt_adv.zero_grad()
            loss_adv_clf.backward()

            torch.nn.utils.clip_grad_norm_(
                adv_classifier.parameters(),
                max_norm=1.0
            )

            opt_adv.step()

            # Step B: train encoder, classifier, and decoder
            z_shared, z_specific = encoder(x)

            logits = classifier(z_specific)
            loss_cls = F.cross_entropy(
                logits,
                y,
                weight=class_weights
            )

            grl = GradientReversalLayer(alpha=grl_alpha)
            adv_logits_grl = adv_classifier(grl(z_shared))
            loss_adv = F.cross_entropy(adv_logits_grl, y)

            x_recon = decoder(z_shared, z_specific)
            loss_recon = F.mse_loss(x_recon, x)

            loss_ortho = orthogonality_loss(
                z_shared,
                z_specific
            )

            loss_var = variance_loss(z_specific)

            loss_proto_intra, loss_proto_inter = class_specific_direction_loss(
                z_specific,
                y,
                num_classes=num_classes
            )

            loss = (
                loss_cls
                + adv_alpha * loss_adv
                + recon_alpha * loss_recon
                + 0.2 * loss_ortho
                + 0.01 * loss_var
                + w_proto * proto_intra_alpha * loss_proto_intra
                + w_proto * proto_inter_alpha * loss_proto_inter
            )

            opt_main.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) +
                list(classifier.parameters()) +
                list(decoder.parameters()),
                max_norm=1.0
            )

            opt_main.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))

        print(
            f"[Epoch {epoch + 1}] "
            f"grl_alpha: {grl_alpha:.3f}, "
            f"L_cls: {loss_cls.item():.4f}, "
            f"L_adv_clf: {loss_adv_clf.item():.4f}, "
            f"L_adv_grl: {loss_adv.item():.4f}, "
            f"L_recon: {loss_recon.item():.4f}, "
            f"L_ortho: {loss_ortho.item():.4f}, "
            f"L_var: {loss_var.item():.4f}, "
            f"L_proto_intra: {loss_proto_intra.item():.4f}, "
            f"L_proto_inter: {loss_proto_inter.item():.4f}, "
            f"Total: {loss.item():.4f}, "
            f"Avg: {avg_loss:.4f}"
        )