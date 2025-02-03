import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class BodyFatDataset(Dataset):
    def __init__(self, data):
        self.features = data.drop(columns=["BodyFat"]).values
        self.targets = data["BodyFat"].values
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
        }
