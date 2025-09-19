import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path


def load_tensor_data(data_dir: str, split: str = "train"):
    """
    Load CXR, ECG, Lab Features (percentiles + missingness), and Labels for a given split from .pt files.
    """
    data_path = Path(data_dir) / split

    if split == "test":
        cxr    = torch.load(data_path / "cxr_test_labeled.pt")
        ecg    = torch.load(data_path / "ecg_test_labeled.pt")
        labs_p = torch.load(data_path / "labs_percentiles_test_labeled.pt")
        labs_m = torch.load(data_path / "labs_missingness_test_labeled.pt")
        labels = torch.load(data_path / "label_test_labeled.pt")
    else:
        cxr    = torch.load(data_path / f"cxr_{split}.pt")
        ecg    = torch.load(data_path / f"ecg_{split}.pt")
        labs_p = torch.load(data_path / f"labs_percentiles_{split}.pt")
        labs_m = torch.load(data_path / f"labs_missingness_{split}.pt")
        labels = torch.load(data_path / f"label_{split}.pt")

    labs = torch.cat([labs_p, labs_m], dim=1)  # (N, L1 + L2)

    return TensorDataset(cxr, ecg, labs, labels)


def get_dataloaders(data_dir: str, batch_size: int = 16, num_workers: int = 4):
    """
    Return DataLoaders for train, val, and test splits for CXR + ECG + Labs input.

    Parameters:
        data_dir (str): Directory containing .pt files.
        batch_size (int): Batch size.
        num_workers (int): Number of parallel workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_ds = load_tensor_data(data_dir, "train")
    val_ds   = load_tensor_data(data_dir, "val")
    test_ds  = load_tensor_data(data_dir, "test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
