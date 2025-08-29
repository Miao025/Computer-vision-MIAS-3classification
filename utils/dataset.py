from torch.utils.data import Dataset
from torchvision.transforms import v2


class MIASDataset(Dataset):
    def __init__(self, img_dict, label_dict, indices=None):
        self.img_dict = img_dict
        self.label_dict = label_dict
        self.indices = indices if indices is not None else list(img_dict.keys())
        self.transform = v2.Compose([
            v2.ToPILImage(),
            v2.Lambda(lambda x: x.convert('RGB')),
            v2.Resize((224, 224)),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) # Convert grayscale to 3-channel, resize, normalize for the imagenet-pretrained models
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img_id = self.indices[idx]
        img = self.img_dict[img_id]  # NumPy array (H, W)
        img = self.transform(img)
        label = self.label_dict[img_id]  # Integer label
        
        return img, label