from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class TrainDataset(Dataset):
    def __init__(self):
        super(TrainDataset, self).__init__()
        
        df = pd.read_csv("7m_trn.csv")
        self.x = df["infusionoffset"].values
        self.y = df["y"].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return np.array(self.x[idx]), self.y[idx]
    
    
class TestDataset(Dataset):
    def __init__(self):
        super(TestDataset, self).__init__()
        
        df = pd.read_csv("7m_tes.csv")
        df = df.sort_values(by='infusionoffset')
        self.x = df["infusionoffset"].values
        self.y = df["y"].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return np.array(self.x[idx]), self.y[idx]