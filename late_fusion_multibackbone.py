# === late_fusion_multibackbone.py ===
# Late-fusion: modality encoders → cross-attn → gated fusion → residual concat → classifier
# Now supports multiple CXR backbones: resnet50, densenet121, efficientnet_b2, convnext_tiny, vgg16_bn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv
import lightning.pytorch as pl


# -------------------------------------------------------------
# CXR backbone factory (returns pooled feature & output dim)
# -------------------------------------------------------------
def make_cxr_backbone(name: str):
    """
    Build a torchvision CXR backbone and return (module, out_dim).
    The module outputs a flat pooled feature vector of size out_dim.
    """
    name = name.lower()

    # --- ResNet50 ---
    if name == "resnet50":
        m = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        out_dim = 2048

        class Wrap(nn.Module):
            def __init__(self, res):
                super().__init__()
                self.backbone = res
                self.pool = nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                # keep explicit path to be robust with freezing/BN later if needed
                x = self.backbone.conv1(x); x = self.backbone.bn1(x); x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                x = self.backbone.layer1(x); x = self.backbone.layer2(x)
                x = self.backbone.layer3(x); x = self.backbone.layer4(x)
                x = self.pool(x).flatten(1)
                return x

        return Wrap(m), out_dim

    # --- DenseNet121 ---
    if name == "densenet121":
        m = tv.densenet121(weights=tv.DenseNet121_Weights.IMAGENET1K_V1)
        out_dim = 1024

        class Wrap(nn.Module):
            def __init__(self, dnet):
                super().__init__()
                self.features = dnet.features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = self.features(x)
                x = F.relu(x, inplace=True)
                x = self.pool(x).flatten(1)
                return x

        return Wrap(m), out_dim

    # --- EfficientNet-B2 ---
    if name == "efficientnet_b2":
        m = tv.efficientnet_b2(weights=tv.EfficientNet_B2_Weights.IMAGENET1K_V1)
        out_dim = 1408

        class Wrap(nn.Module):
            def __init__(self, eff):
                super().__init__()
                self.features = eff.features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = self.features(x)
                x = self.pool(x).flatten(1)
                return x

        return Wrap(m), out_dim

    # --- ConvNeXt-Tiny ---
    if name == "convnext_tiny":
        m = tv.convnext_tiny(weights=tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        out_dim = 768

        class Wrap(nn.Module):
            def __init__(self, cnx):
                super().__init__()
                self.features = cnx.features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = self.features(x)
                x = self.pool(x).flatten(1)
                return x

        return Wrap(m), out_dim

    # --- VGG16-BN ---
    if name == "vgg16_bn":
        m = tv.vgg16_bn(weights=tv.VGG16_BN_Weights.IMAGENET1K_V1)
        out_dim = 512

        class Wrap(nn.Module):
            def __init__(self, vgg):
                super().__init__()
                self.features = vgg.features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = self.features(x)
                x = self.pool(x).flatten(1)
                return x

        return Wrap(m), out_dim

    raise ValueError(f"Unknown CXR backbone: {name}")


# -------------------------------------------------------------
# ECG & Lab encoders
# -------------------------------------------------------------
class ECGEncoder(nn.Module):
    """12‑lead 1D CNN → global avg pool → linear to 128-dim."""
    def __init__(self, out_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.cnn(x)           # (B, 64, 1)
        x = x.view(x.size(0), -1) # (B, 64)
        return self.fc(x)         # (B, 128)


class LabEncoder(nn.Module):
    """Simple MLP for tabular labs → 128-dim."""
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


# -------------------------------------------------------------
# Cross-attention block (single-token query; residual + LN)
# -------------------------------------------------------------
class CrossAttention(nn.Module):
    """
    Given a (B, D) query and a (B, D) key/value, run 1-head MHA in (B, 1, D) space.
    You used this to nudge one modality with another (e.g., ECG → CXR).
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query_vec, key_vec):
        q = query_vec.unsqueeze(1)  # (B,1,D)
        k = key_vec.unsqueeze(1)    # (B,1,D)
        out, _ = self.attn(q, k, k) # (B,1,D)
        out = self.norm(out + q)
        return out.squeeze(1)       # (B,D)


# -------------------------------------------------------------
# Gated fusion (3-way softmax gate over 128-dim features)
# -------------------------------------------------------------
class GatedFusion(nn.Module):
    """Compute weights over [ECG, CXR, Labs] and weighted-sum the three 128-d features."""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, ecg_f, cxr_f, lab_f):
        concat = torch.cat([ecg_f, cxr_f, lab_f], dim=1)  # (B, 384)
        w = self.gate(concat)                             # (B, 3)
        fused = w[:, 0:1] * ecg_f + w[:, 1:2] * cxr_f + w[:, 2:3] * lab_f
        return fused                                      # (B, 128)


# -------------------------------------------------------------
# Late Fusion Classifier (multi-backbone)
# -------------------------------------------------------------
class LateFusionClassifier(pl.LightningModule):
    """
    Architecture (your late fusion variant):
      1) CXR encoder (selectable backbone) → reduce to 128
      2) ECG encoder (128) and Lab encoder (128)
      3) Two cross-attn hops:
           - ECG → CXR (use ECG to inform CXR)
           - Labs → ECG (use Labs to inform ECG)
      4) Gated fusion over [ECG*, CXR*, Labs] to produce a fused 128-d vector
      5) Residual concat [fused, ecg_raw, cxr_raw, labs_raw] → classifier
    """
    def __init__(
        self,
        num_classes: int = 14,
        lr: float = 1e-4,
        lab_input_dim: int = 40,
        cxr_backbone: str = "resnet50",
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- (1) CXR encoder: selectable backbone, pooled feature + linear to 128 ---
        self.cxr_encoder, cxr_out = make_cxr_backbone(cxr_backbone)  # pooled feat
        self.reduce_cxr = nn.Linear(cxr_out, 128)

        # --- (2) ECG & Lab encoders (128 each) ---
        self.ecg_encoder = ECGEncoder(out_dim=128)
        self.lab_encoder = LabEncoder(input_dim=lab_input_dim, out_dim=128)

        # --- (3) Cross-attention hops (single-token, residual+LN) ---
        self.cross_cxr_ecg  = CrossAttention(embed_dim=128)  # ECG informs CXR
        self.cross_ecg_labs = CrossAttention(embed_dim=128)  # Labs inform ECG

        # --- (4) Gated fusion over the 3 modality features (128) ---
        self.gmf = GatedFusion(dim=128)

        # --- (5) Classification head on residual concat [fused, ecg_raw, cxr_raw, labs_raw] ---
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    # ---------------------------
    # Forward
    # ---------------------------
    def forward(self, cxr, ecg, labs):
        """
        cxr:  (B, 3, H, W)
        ecg:  (B, 12, T)       # squeeze/permute should be done by dataloader or caller
        labs: (B, D_labs)
        """
        # --- encode raw modalities ---
        cxr_feat_raw = self.reduce_cxr(self.cxr_encoder(cxr))  # (B,128)
        ecg_feat_raw = self.ecg_encoder(ecg)                   # (B,128)
        lab_feat_raw = self.lab_encoder(labs)                  # (B,128)

        # --- cross-attention nudges (producing attended views) ---
        cxr_attn  = self.cross_cxr_ecg(cxr_feat_raw, ecg_feat_raw)  # ECG → CXR
        ecg_attn  = self.cross_ecg_labs(ecg_feat_raw, lab_feat_raw) # Labs → ECG
        labs_attn = lab_feat_raw                                    # (no third hop)

        # --- gated fusion of the attended features ---
        fused = self.gmf(ecg_attn, cxr_attn, labs_attn)             # (B,128)

        # --- residual concat with the raw features (keeps original info) ---
        final = torch.cat([fused, ecg_feat_raw, cxr_feat_raw, lab_feat_raw], dim=1)  # (B,512)
        logits = self.classifier(final)                                              # (B,C)
        return logits

    # ---------------------------
    # Loss: CheXpert masked BCE
    # ---------------------------
    @staticmethod
    def masked_bce_loss(preds, targets):
        mask = (targets != -1)
        return F.binary_cross_entropy_with_logits(preds[mask], targets[mask].float())

    # ---------------------------
    # Lightning hooks
    # ---------------------------
    def training_step(self, batch, batch_idx):
        cxr, ecg, labs, y = batch
        # Ensure ECG is (B,12,T)
        if ecg.ndim == 4 and ecg.shape[1] == 1:
            ecg = ecg.squeeze(1)
        if ecg.shape[1] != 12 and ecg.shape[-1] == 12:
            ecg = ecg.permute(0, 2, 1)

        logits = self.forward(cxr.float(), ecg.float(), labs.float())
        loss = self.masked_bce_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=cxr.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        cxr, ecg, labs, y = batch
        if ecg.ndim == 4 and ecg.shape[1] == 1:
            ecg = ecg.squeeze(1)
        if ecg.shape[1] != 12 and ecg.shape[-1] == 12:
            ecg = ecg.permute(0, 2, 1)

        logits = self.forward(cxr.float(), ecg.float(), labs.float())
        loss = self.masked_bce_loss(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=cxr.size(0))
        return {"logits": logits, "labels": y}

    def test_step(self, batch, batch_idx):
        cxr, ecg, labs, y = batch
        if ecg.ndim == 4 and ecg.shape[1] == 1:
            ecg = ecg.squeeze(1)
        if ecg.shape[1] != 12 and ecg.shape[-1] == 12:
            ecg = ecg.permute(0, 2, 1)

        logits = self.forward(cxr.float(), ecg.float(), labs.float())
        return {"logits": logits, "labels": y}

    def configure_optimizers(self):
        # default single-lr Adam; in training script we’ll split LR so backbone can have a scaled LR
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
