# healnet_early_fusion_multibackbone.py
# Early Fusion with selectable CXR backbones (ResNet/DenseNet/EfficientNet/ConvNeXt/VGG)
# Keeps the latent-token + cross-attention fusion intact. No gating added.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import lightning.pytorch as pl


# ================================
# Helper: CXR backbone factory
# -------------------------------
# Returns a module that produces a *pooled feature vector* for a CXR image
# and the dimensionality of that vector. We wrap each torchvision model so
# the forward() returns a flat tensor (B, out_dim).
# ================================
def make_cxr_backbone(name: str):
    name = name.lower()

    # --- ResNet50 (out: 2048) ---
    if name == "resnet50":
        m = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        out_dim = 2048

        class Wrap(nn.Module):
            def __init__(self, res):
                super().__init__()
                self.backbone = res
                self.pool = nn.AdaptiveAvgPool2d((1, 1))  # spatial → (1,1)

            def forward(self, x):
                # replicate forward but stop before FC, then pool
                x = self.backbone.conv1(x); x = self.backbone.bn1(x); x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                x = self.backbone.layer1(x); x = self.backbone.layer2(x)
                x = self.backbone.layer3(x); x = self.backbone.layer4(x)
                return self.pool(x).flatten(1)            # (B, 2048)

        return Wrap(m), out_dim

    # --- DenseNet121 (out: 1024) ---
    if name == "densenet121":
        m = tv_models.densenet121(weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1)
        out_dim = 1024

        class Wrap(nn.Module):
            def __init__(self, dnet):
                super().__init__()
                self.features = dnet.features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = self.features(x)
                x = F.relu(x, inplace=True)
                return self.pool(x).flatten(1)            # (B, 1024)

        return Wrap(m), out_dim

    # --- EfficientNet-B2 (out: 1408) ---
    if name == "efficientnet_b2":
        m = tv_models.efficientnet_b2(weights=tv_models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        out_dim = 1408

        class Wrap(nn.Module):
            def __init__(self, eff):
                super().__init__()
                self.features = eff.features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = self.features(x)
                return self.pool(x).flatten(1)            # (B, 1408)

        return Wrap(m), out_dim

    # --- ConvNeXt-Tiny (out: 768) ---
    if name == "convnext_tiny":
        m = tv_models.convnext_tiny(weights=tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        out_dim = 768

        class Wrap(nn.Module):
            def __init__(self, cnx):
                super().__init__()
                self.features = cnx.features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = self.features(x)
                return self.pool(x).flatten(1)            # (B, 768)

        return Wrap(m), out_dim

    # --- VGG16-BN (out: 512) ---
    if name == "vgg16_bn":
        m = tv_models.vgg16_bn(weights=tv_models.VGG16_BN_Weights.IMAGENET1K_V1)
        out_dim = 512

        class Wrap(nn.Module):
            def __init__(self, vgg):
                super().__init__()
                self.features = vgg.features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = self.features(x)
                return self.pool(x).flatten(1)            # (B, 512)

        return Wrap(m), out_dim

    raise ValueError(f"Unknown CXR backbone: {name}")


# ================================
# ECG encoder (tiny 1D CNN → linear)
# -------------------------------
# Encodes (B, 12, T) ECG into a fixed 128‑D vector.
# ================================
class ECGEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # global pool over time
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.cnn(x)                # (B, 64, 1)
        x = x.view(x.size(0), -1)      # (B, 64)
        return self.fc(x)              # (B, 128)


# ================================
# Labs encoder (simple MLP)
# -------------------------------
# Encodes tabular labs (B, D_labs) into a 128‑D embedding.
# ================================
class LabEncoder(nn.Module):
    def __init__(self, input_dim, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


# ================================
# Cross‑attention: latent tokens query each modality
# -------------------------------
# Given a set of latent tokens S (B, L, D) and a modality feature f (B, d_in),
# project f → (B, 1, D) and apply MHA with S as Query.
# The output updates S (residual) and returns attention weights for vis.
# ================================
class CrossAttention(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.kv_proj = nn.Linear(input_dim, latent_dim)
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, batch_first=True)

    def forward(self, S, feat):
        kv = self.kv_proj(feat).unsqueeze(1)      # (B, 1, D)
        out, weights = self.attn(S, kv, kv)       # weights: (B, L, 1)
        return out, weights


# ================================
# Fusion block: iterate cross‑attn over available modalities
# -------------------------------
# For each modality present, we let the latent tokens attend to it and
# add the result back to S (residual accumulation). We can record weights.
# ================================
class FusionBlock(nn.Module):
    def __init__(self, latent_dim, input_dims):
        super().__init__()
        self.attn_blocks = nn.ModuleDict({
            mod: CrossAttention(latent_dim, input_dims[mod]) for mod in input_dims
        })

    def forward(self, S, feats, store_weights=False):
        attn_weights = {}
        for mod, feat in feats.items():
            if feat is None:           # allow missing modalities gracefully
                continue
            out, w = self.attn_blocks[mod](S, feat)
            S = S + out                # residual update to latent tokens
            if store_weights:
                attn_weights[mod] = w.detach()
        return S, (attn_weights if store_weights else None)


# ================================
# Early Fusion model (with selectable CXR backbone)
# -------------------------------
# Pipeline:
#   CXR -> backbone(pool) -> cxr_feat (d_cxr)
#   ECG -> tiny CNN -> ecg_feat (128)
#   Labs -> MLP -> lab_feat (128)
#   Shared latent tokens S (L tokens, dim D)
#   Repeat N fusion layers: for mod in {CXR,ECG,Labs}:
#       S += CrossAttention(S, mod_feat)
#   Pool S across tokens -> classifier -> logits (14 labels)
# ================================
class EarlyFusionModel(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 14,
        lab_input_dim: int = 40,
        latent_tokens: int = 4,
        latent_dim: int = 128,
        n_fusion_layers: int = 3,
        lr: float = 1e-4,
        cxr_backbone: str = "resnet50",   # NEW: choose CXR encoder
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- CXR encoder (selectable) ---
        # Produces flat vector (B, d_cxr); we keep d_cxr for the fusion projection.
        self.cxr_encoder, self.cxr_out_dim = make_cxr_backbone(cxr_backbone)

        # --- ECG & Lab encoders (both to 128‑D) ---
        self.ecg_encoder = ECGEncoder(out_dim=128)
        self.lab_encoder = LabEncoder(input_dim=lab_input_dim, out_dim=128)

        # --- Learnable latent tokens S (B, L, D) ---
        # Shared across fusion layers; copied across the batch on forward.
        self.latent_tokens = nn.Parameter(torch.randn(1, latent_tokens, latent_dim))

        # --- Stack of fusion blocks ---
        # Each block lets S attend to each modality (projected to latent_dim).
        input_dims = {"cxr": self.cxr_out_dim, "ecg": 128, "labs": 128}
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(latent_dim, input_dims) for _ in range(n_fusion_layers)
        ])

        # --- Classifier head on pooled S ---
        # We average tokens (mean-pool) and classify into CheXpert labels.
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # --- Optimizer hyper‑params (kept here for Lightning) ---
        self.lr = lr

    # ------------- Encoders -------------
    def encode_modalities(self, cxr, ecg, labs):
        feats = {
            "cxr": self.cxr_encoder(cxr) if cxr is not None else None,  # (B, d_cxr)
            "ecg": self.ecg_encoder(ecg) if ecg is not None else None,  # (B, 128)
            "labs": self.lab_encoder(labs) if labs is not None else None  # (B, 128)
        }
        return feats

    # ------------- Forward ------------
    def forward(self, cxr, ecg, labs, return_attention: bool = False):
        # 1) Batch size (avoid boolean evaluation of tensors)
        def _batch_size(*tensors):
            for t in tensors:
                if isinstance(t, torch.Tensor):
                    return t.shape[0]
            raise ValueError("All input modalities are None; expected at least one tensor.")
        B = _batch_size(cxr, ecg, labs)

        # 2) Prepare latent tokens for this batch
        S = self.latent_tokens.expand(B, -1, -1)   # (B, L, D) on the correct device (since it's a Parameter)

        # 3) Encode modalities → fixed-size vectors
        feats = self.encode_modalities(cxr, ecg, labs)

        # 4) Run stacked fusion blocks (cross‑attention per modality)
        attn_record = [] if return_attention else None
        for block in self.fusion_blocks:
            S, attn_w = block(S, feats, store_weights=return_attention)
            if return_attention:
                attn_record.append(attn_w)

        # 5) Pool tokens and classify
        pooled = S.mean(dim=1)                 # (B, D)
        logits = self.classifier(pooled)       # (B, num_classes)

        return (logits, attn_record) if return_attention else logits


    # ------------- Masked BCE (CheXpert‑style) -------------
    @staticmethod
    def masked_bce_loss(preds, targets):
        mask = (targets != -1)
        return F.binary_cross_entropy_with_logits(preds[mask], targets[mask].float())

    # ------------- Lightning hooks -------------
    def training_step(self, batch, batch_idx):
        cxr, ecg, labs, y = batch
        logits = self.forward(cxr.float(), ecg.float(), labs.float())
        loss = self.masked_bce_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, batch_size=cxr.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        cxr, ecg, labs, y = batch
        logits = self.forward(cxr.float(), ecg.float(), labs.float())
        loss = self.masked_bce_loss(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=cxr.size(0))
        return {"logits": logits, "labels": y}

    def test_step(self, batch, batch_idx):
        cxr, ecg, labs, y = batch
        logits = self.forward(cxr.float(), ecg.float(), labs.float())
        return {"logits": logits, "labels": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
