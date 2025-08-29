import os
import pandas as pd
from torchvision import transforms
from torchvision.io import decode_image
from torch.utils.data import Dataset


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
    
    # Aggregate labels per image, choosing the most severe: Malignant > Benign > Normal
    def aggregate_labels(labels):
        if 2 in labels.values:
            return 2
        elif 1 in labels.values:
            return 1
        else:  # Normal
            return 0
    df = df.groupby('REFNUM').agg({'label': aggregate_labels}).reset_index()

    # create df_labels which is like mdb001.png, 0 mdb002.png, 2 ...
    df['imgid'] = df['REFNUM'] + '.png'
    df_labels = df[['imgid','label']]

    return df_labels


class CustomImageDataset(Dataset):
    def __init__(self, df_labels, img_dir, transform=None, target_transform=None):
        self.img_labels = df_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image_tensor = decode_image(img_path) # get the tensor

        if int(self.img_labels.iloc[idx, 0][3:6])%2 == 1: # flip odd-numbered images (left breast to right)
            image_tensor = transforms.functional.hflip(image_tensor)

        if self.transform:
            image_tensor = self.transform(image_tensor)
        label = self.img_labels.iloc[idx, 1]
        if self.target_transform:
            label = self.target_transform(label)

        return image_tensor, label