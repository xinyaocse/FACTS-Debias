
import torch
from torch.utils.data import Dataset

class BiasFactDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], torch.tensor(self.labels[idx], dtype=torch.long)
