import torch
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer


# Dataset

class NewsCategoryDataset(Dataset):
    def __init__(self, path, model='train', balance=[0.7,0.15,0.15]):
        self.df = pd.read_json(path)
        train_num = int(len(self.df)*balance[0])
        val_num = int(len(self.df)*balance[1])
        test_num = int(len(self.df)*balance[2])
        if model == 'train':
            self.df = self.df[:train_num]
        elif model == 'val':
            self.df = self.df[train_num:train_num + val_num]
        elif model == 'test':
            self.df = self.df[-test_num:]

        self.tokenizer =


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {'label': self.df['category'][idx],
                'headline': self.df['headline'][idx],
                'description': self.df['description'][idx]}