import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import lightning.pytorch as pl


# ================================
# Helper: CXR backbone factory
# ================================
def make_cxr_backbone(name: str):
    """
    Returns (backbone_module, out_dim).
    The module outputs a flat pooled feature vector of size out_dim.
    """
    name = name.lower()

    # --- ResNet50 ---
    if name == "resnet50":
        m = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        out_dim = 2048

        class Wrap(nn.Module):
            def __init__(self, res):
                super().__init__()
                self.backbone = res
                self.pool = nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = self.backbone.conv1(x); x = self.backbone.bn1(x); x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                x = self.backbone.layer1(x); x = self.backbone.layer2(x)
                x = self.backbone.layer3(x); x = self.backbone.layer4(x)
                x = self.pool(x).flatten(1)
                return x

        return Wrap(m), out_dim

    # --- DenseNet121 ---
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
                x = self.pool(x).flatten(1)
                return x

        return Wrap(m), out_dim

    # --- EfficientNet-B2 ---
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
                x = self.pool(x).flatten(1)
                return x

        return Wrap(m), out_dim

    # --- ConvNeXt-Tiny ---
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
                x = self.pool(x).flatten(1)
                return x

        return Wrap(m), out_dim

    # --- VGG16-BN ---
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
                x = self.pool(x).flatten(1)
                return x

        return Wrap(m), out_dim

    raise ValueError(f"Unknown CXR backbone: {name}")


# ================================
# Utility: structured modality dropout
# ================================
def structured_modality_dropout(ecg_feat, cxr_feat, lab_feat, p: float):
    """
    Drops at most ONE modality per sample with total probability p.
    Prevents early random nuking of multiple modalities that can push
    gates into bad basins.
    """
    if p <= 0:
        return ecg_feat, cxr_feat, lab_feat

    B = ecg_feat.size(0)
    with torch.no_grad():
        # Categories: 0=drop ECG, 1=drop CXR, 2=drop Labs, 3=drop NONE
        probs = torch.tensor([p/3, p/3, p/3, 1-p], device=ecg_feat.device)
        choice = torch.multinomial(probs, num_samples=B, replacement=True)

    def maybe_drop(t, idx):
        return t * (choice != idx).float().unsqueeze(1)

    return maybe_drop(ecg_feat, 0), maybe_drop(cxr_feat, 1), maybe_drop(lab_feat, 2)


# ================================
# ECG Encoder
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
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.cnn(x)                # (B, 64, 1)
        x = x.view(x.size(0), -1)      # (B, 64)
        return self.fc(x)              # (B, 128)


# ================================
# Lab Encoder
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
# Adaptive Gated Fusion (temperature-softmax)
# ================================
class AdaptiveGatedFusion(nn.Module):
    def __init__(self, dim, init_temp=1.5, min_temp=0.7, anneal=0.995, floor_eps: float = 0.02):
        super().__init__()
        self.g1 = nn.Linear(dim * 3, 64)
        self.g2 = nn.Linear(64, 3)
        # tiny, near‑zero init for last layer to avoid early collapse
        nn.init.zeros_(self.g2.bias)
        nn.init.uniform_(self.g2.weight, a=-1e-3, b=1e-3)

        # Temperature starts smooth, anneals (but not too low)
        self.register_buffer("temperature", torch.tensor(float(init_temp)))
        self.min_temp = float(min_temp)
        self.anneal = float(anneal)
        self.floor_eps = float(floor_eps)

    def step_temperature(self):
        new_t = max(self.min_temp, float(self.temperature) * self.anneal)
        self.temperature.fill_(new_t)

    def forward(self, ecg, cxr, labs):
        x = torch.cat([ecg, cxr, labs], dim=1)            # (B, 384)
        x = F.relu(self.g1(x))
        logits = self.g2(x) / self.temperature
        w = F.softmax(logits, dim=1)                      # (B, 3)
        if self.floor_eps > 0.0:
            w = (1.0 - 3.0 * self.floor_eps) * w + self.floor_eps
        fused = w[:, 0:1] * ecg + w[:, 1:2] * cxr + w[:, 2:3] * labs
        return fused, w


# ================================
# LightningModule
# ================================
class AdaptiveFusionClassifier(pl.LightningModule):
    def __init__(self,
        num_classes: int = 14,
        lr: float = 1e-4,
        lab_input_dim: int = 100,
        dropout_prob: float = 0.2,
        cxr_backbone: str = "resnet50",
        moddrop_warm_epochs: int = 5,
        gate_init_temp: float = 1.5,
        gate_min_temp: float = 0.7,
        gate_anneal: float = 0.995,
        lambda_gate_entropy: float = 0.005,
        lambda_gate_diversity: float = 0.005,
        normalization: str = "none",
        # --- legacy args (accepted but unused) ---
        lambda_gate_prior: float | None = None,
        gate_prior: tuple | None = None,
        gate_floor_eps: float | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        # ignore lambda_gate_prior/gate_prior; gate_floor_eps is handled inside fusion already


        # allow manual control of epoch when training without Lightning Trainer
        self._manual_epoch = 0
        self.set_epoch = lambda epoch: setattr(self, "_manual_epoch", int(epoch))

        # ---------------------------
        # CXR encoder (selectable backbone)
        # ---------------------------
        self.cxr_encoder, cxr_out = make_cxr_backbone(cxr_backbone)
        self.reduce_cxr = nn.Linear(cxr_out, 128)

        # ---------------------------
        # Other modality encoders
        # ---------------------------
        self.ecg_encoder = ECGEncoder(out_dim=128)
        self.lab_encoder = LabEncoder(input_dim=lab_input_dim, out_dim=128)

        # ---------------------------
        # Gated fusion with temperature
        # ---------------------------
        self.fusion = AdaptiveGatedFusion(
            dim=128,
            init_temp=gate_init_temp,
            min_temp=gate_min_temp,
            anneal=gate_anneal,
            floor_eps=0.00,
        )

        # ---------------------------
        # Final classifier (compact head)
        # ---------------------------
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128 * 3, 256),  # [fused, ecg, cxr, labs] → 256
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # ---------------------------
        # Regularization / scheduling knobs
        # ---------------------------
        self.modality_dropout = dropout_prob         # total prob to drop at most one modality
        self.moddrop_warm_epochs = moddrop_warm_epochs
        self.lambda_ent = float(lambda_gate_entropy)
        self.lambda_div = float(lambda_gate_diversity)
        self.normalization = normalization

    # ---------------------------
    # Forward pass
    # ---------------------------
    def forward(
        self,
        cxr, ecg, labs,
        return_gates: bool = False,
        ablate_cxr: bool = False,
        ablate_ecg: bool = False,
        ablate_labs: bool = False
    ):
        # sanity check for lab dim
        assert labs.shape[1] == self.hparams.lab_input_dim, \
            f"Expected labs dim {self.hparams.lab_input_dim}, got {labs.shape[1]}"

        # === Encode modalities ===
        cxr_feat = self.reduce_cxr(self.cxr_encoder(cxr))  # (B, 128)
        ecg_feat = self.ecg_encoder(ecg)                   # (B, 128)
        lab_feat = self.lab_encoder(labs)                  # (B, 128)

        # === Manual ablations (eval-time diagnostics) ===
        if ablate_cxr:  cxr_feat = torch.zeros_like(cxr_feat)
        if ablate_ecg:  ecg_feat = torch.zeros_like(ecg_feat)
        if ablate_labs: lab_feat = torch.zeros_like(lab_feat)

        # === Structured, annealed modality dropout (training only) ===
        if self.training and self.modality_dropout > 0:
            p0, p_min, warm = self.modality_dropout, 0.0, max(1, self.moddrop_warm_epochs)
            epoch_idx = getattr(self, "_manual_epoch", getattr(self, "current_epoch", 0))
            p = max(p_min, p0 * (1 - min(1.0, epoch_idx / warm)))
            ecg_feat, cxr_feat, lab_feat = structured_modality_dropout(ecg_feat, cxr_feat, lab_feat, p=p)

        # === Optional normalization before gating ===
        if self.normalization == "layernorm":
            ecg_feat = F.layer_norm(ecg_feat, ecg_feat.size()[1:])
            cxr_feat = F.layer_norm(cxr_feat, cxr_feat.size()[1:])
            lab_feat = F.layer_norm(lab_feat, lab_feat.size()[1:])
        elif self.normalization == "l2":
            ecg_feat = F.normalize(ecg_feat, dim=1)
            cxr_feat = F.normalize(cxr_feat, dim=1)
            lab_feat = F.normalize(lab_feat, dim=1)
        # else: "none"

        # === Gated fusion (temperature-softmax) ===
        fused, gates = self.fusion(ecg_feat, cxr_feat, lab_feat)

        # === Residual concat → classifier ===
        combined = torch.cat([fused, ecg_feat, cxr_feat, lab_feat], dim=1)  # (B, 512)
        logits = self.classifier(combined)                                  # (B, C)

        return (logits, gates) if return_gates else logits

    # ---------------------------
    # CheXpert-style masked BCE
    # ---------------------------
    @staticmethod
    def masked_bce_loss(preds, targets):
        mask = (targets != -1)
        return F.binary_cross_entropy_with_logits(preds[mask], targets[mask].float())

    # ---------------------------
    # Tiny gate regularizers
    # ---------------------------
    @staticmethod
    def _gate_entropy(g):
        eps = 1e-8
        return (-(g * (g + eps).log()).sum(dim=1)).mean()

    @staticmethod
    def _gate_batch_var(g):
        return g.var(dim=0).mean()

    # ---------------------------
    # Training loop
    # ---------------------------
    def training_step(self, batch, batch_idx):
        cxr, ecg, labs, y = batch
        logits, gates = self.forward(cxr.float(), ecg.float(), labs.float(), return_gates=True)

        loss = self.masked_bce_loss(logits, y)
        # Encourage exploration and diversity slightly
        loss = loss + self.lambda_ent * self._gate_entropy(gates) - self.lambda_div * self._gate_batch_var(gates)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=cxr.size(0))
        self.log_dict({
            "train_gate_ecg": gates[:, 0].mean(),
            "train_gate_cxr": gates[:, 1].mean(),
            "train_gate_labs": gates[:, 2].mean()
        }, prog_bar=False, on_step=False, on_epoch=True, batch_size=cxr.size(0))

        # smooth → sharper over time
        self.fusion.step_temperature()
        return loss

    # ---------------------------
    # Validation loop
    # ---------------------------
    def validation_step(self, batch, batch_idx):
        cxr, ecg, labs, y = batch
        logits, gates = self.forward(cxr.float(), ecg.float(), labs.float(), return_gates=True)

        loss = self.masked_bce_loss(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=cxr.size(0))
        self.log_dict({
            "val_gate_ecg": gates[:, 0].mean(),
            "val_gate_cxr": gates[:, 1].mean(),
            "val_gate_labs": gates[:, 2].mean()
        }, prog_bar=True, on_step=False, on_epoch=True, batch_size=cxr.size(0))

        return {"logits": logits, "labels": y}

    # ---------------------------
    # Test loop
    # ---------------------------
    def test_step(self, batch, batch_idx):
        cxr, ecg, labs, y = batch
        logits, gates = self.forward(cxr.float(), ecg.float(), labs.float(), return_gates=True)
        return {"logits": logits, "labels": y, "gates": gates}

    # ---------------------------
    # Early smoothing for gates
    # ---------------------------
    def on_train_epoch_start(self):
        # Keep gates extra-smooth for the first 2 epochs to avoid thrashing
        if self.current_epoch < 2:
            self.fusion.temperature.fill_(max(self.hparams.gate_init_temp, 2.0))

    # ---------------------------
    # Optimizer
    # ---------------------------
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
