from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, x, transform=None):
        self.transform = transform
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        curr_x = self.x[idx, ...]
        if self.transform:
            curr_x = self.transform(curr_x)
        return curr_x