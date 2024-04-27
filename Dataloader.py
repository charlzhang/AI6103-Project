from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import PIL

class FaceSwapDataset(Dataset):
    def __init__(self, source_dir, target_dir, predictor, transform=None):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.transform = transform
        self.predictor = predictor
        self.filenames_s = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
        self.filenames_t = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
    def __len__(self):
        return len(self.filenames_s)

    def __getitem__(self, idx):
        source_path = os.path.join(self.source_dir, self.filenames_s[idx])
        target_path = os.path.join(self.target_dir, self.filenames_t[idx])

        # source_image = align_face(source_path, self.predictor)
        # target_image = align_face(target_path, self.predictor)

        source_image = PIL.Image.open(source_path)
        target_image = PIL.Image.open(target_path)
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        return source_image, target_image

def create_dataloader(source_dir, target_dir, transform, predictor,batch_size=4, shuffle=True):
    dataset = FaceSwapDataset(source_dir, target_dir,predictor, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 可以为train 和 val定义不同的transform
def get_dataloader(data_dir,train_transforms,common_transforms,predictor,batch_size):
    train_loader = create_dataloader(data_dir+'/train/source', data_dir+'/train/target', train_transforms, predictor,batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(data_dir+'/val/source', data_dir+'/val/target', common_transforms, predictor,batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(data_dir+'/test/source', data_dir+'/test/target', common_transforms, predictor,batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


