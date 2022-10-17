import torch
from torch.utils.data import Dataset
#定义pytorch可处理的数据集Dataset
class traffic_dataset(Dataset):
    def __init__(self, X,Y):
        self.X = X
        self.Y = Y
        self.length = len(X)

    def __getitem__(self, mask):
        Y = torch.Tensor(self.Y[mask])
        X = torch.Tensor(self.X[mask])
        if torch.cuda.is_available():
            Y = Y.cuda()
            X = X.cuda()
        return X, Y

    def __len__(self):
        return self.length