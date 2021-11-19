from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, features, targets, masks, train=True):
        super(TrainDataset, self).__init__()
        self.features = features
        self.targets = targets
        self.masks = masks

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (self.features[index].astype('float32'),
                self.targets[index].astype('float32'),
                self.masks[index].astype('bool'))

class TestDataset(Dataset):
    def __init__(self, features): #HDKIM 100
        super(TestDataset, self).__init__()
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):

        return self.features[index].astype('float32')