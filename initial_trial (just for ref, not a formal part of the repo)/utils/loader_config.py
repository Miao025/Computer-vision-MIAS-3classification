import os
import pandas as pd
from torchvision.transforms import functional
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def get_labels(csv_path):

    # create lable col in csv
    df = pd.read_csv(csv_path)
    def map_label(row):
        if row['CLASS'] == 'NORM': # Normal
            return 0
        elif row['SEVERITY'] == 'Benign': # Benign
            return 1 
        else: # Malignant
            return 2
    df['label'] = df.apply(map_label, axis=1)
    
    # create df_labels which is like mdb001.png, 0 mdb002.png, 2 ...
    df['imgid'] = df['REFNUM'] + '.png'

    # keep only the row with the highest-priority label for each imgid
    df = df.sort_values('label', ascending=False)  # highest first
    df_labels = df.groupby('imgid', as_index=False).first()[['imgid', 'label']]

    return df_labels


class CustomImageDataset(Dataset): # the standard class for pytorch accessing imgs and labels, see https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def __init__(self, df_labels, img_dir, transform=None, target_transform=None):
        self.img_labels = df_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]

        # Flip images with odd indices
        if idx % 2 == 1:
            image = functional.hflip(image)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def weight_sampler(df):
    df = df.copy()
    custom_weights = {0: 0.53, 1: 1.67, 2: 1.97}
    sample_weights = df['label'].map(custom_weights).values
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


def get_loader(train_subset, val_subset, train_transforms, val_test_transforms):
    train_dataset = CustomImageDataset(train_subset, 'Dataset/MIAS', transform=train_transforms)
    val_dataset = CustomImageDataset(val_subset, 'Dataset/MIAS', transform=val_test_transforms)
    train_loader = DataLoader(train_dataset, batch_size=4,
                              # shuffle=True,
                              sampler=weight_sampler(train_subset), # use sampler for train to upsample some classes
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader