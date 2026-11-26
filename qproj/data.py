"""CIFAR-10 loading, splits (45k train / 5k val / 10k test) and the 512-image calibration set.

Works two ways:
  * from torchvision (Colab):  load_cifar(root, source="torchvision")
  * from a pre-built cifar10.npz (offline):  load_cifar(root, source="npz")
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_cifar(root="data", source="auto"):
    """Returns dict of uint8 arrays: xtr (50000,32,32,3), ytr, xte (10000,32,32,3), yte."""
    npz = os.path.join(root, "cifar10.npz")
    if source == "npz" or (source == "auto" and os.path.exists(npz)):
        d = np.load(npz)
        return {k: d[k] for k in ["xtr", "ytr", "xte", "yte"]}
    import torchvision
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root, train=False, download=True)
    return dict(xtr=tr.data, ytr=np.array(tr.targets), xte=te.data, yte=np.array(te.targets))


def splits(d, split_seed=0, n_val=5000, n_calib=512, calib_seed=0):
    """45k train / 5k val (from the official train set), official 10k test.
    Calibration images are drawn from the 45k train part (never from val/test)."""
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(len(d["ytr"]))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    crng = np.random.RandomState(1000 + calib_seed)
    calib_idx = tr_idx[crng.permutation(len(tr_idx))[:n_calib]]
    return dict(
        train=(d["xtr"][tr_idx], d["ytr"][tr_idx]),
        val=(d["xtr"][val_idx], d["ytr"][val_idx]),
        test=(d["xte"], d["yte"]),
        calib=(d["xtr"][calib_idx], d["ytr"][calib_idx]),
    )


def to_tensor(x):
    """uint8 NHWC -> normalized float NCHW."""
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255)
    m = torch.tensor(MEAN).view(1, 3, 1, 1)
    s = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - m) / s


class ArrayDS(Dataset):
    def __init__(self, x, y, transform=None):
        self.x, self.y, self.t = x, y, transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.fromarray(self.x[i])
        if self.t is not None:
            img = self.t(img)
        return img, int(self.y[i])


def train_loader(x, y, batch_size=128, randaug=False, workers=2):
    import torchvision.transforms as T
    tf = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if randaug:
        tf.append(T.RandAugment(num_ops=2, magnitude=9))
    tf += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return DataLoader(ArrayDS(x, y, T.Compose(tf)), batch_size=batch_size, shuffle=True,
                      num_workers=workers, drop_last=True, persistent_workers=workers > 0)


@torch.no_grad()
def evaluate(model, x, y, batch_size=500, device="cpu"):
    """Top-1 accuracy (%) on uint8 arrays."""
    model.eval()
    correct = 0
    for i in range(0, len(y), batch_size):
        xb = to_tensor(x[i:i + batch_size]).to(device)
        correct += (model(xb).argmax(1).cpu().numpy() == y[i:i + batch_size]).sum()
    return 100.0 * correct / len(y)
