import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def data_prep(X):
    tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return loader

class SnapshotDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.file_names = sorted(os.listdir(folder_path))  # ensures time order
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.file_names[idx])
        image = Image.open(img_path).convert("RGB")  # "L" for grayscale, "RGB" if colored

        if self.transform:
            image = self.transform(image)

        return image