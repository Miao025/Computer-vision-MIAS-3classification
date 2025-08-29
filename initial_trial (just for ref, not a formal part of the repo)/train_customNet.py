import torch
import warnings
from sklearn.model_selection import StratifiedKFold
from torchvision.transforms import v2
from utils.loader_config import get_labels, get_loader
from utils.train_config import init_model_config, train_one_epoch, validate

warnings.filterwarnings('ignore')

# Run this to see the results of Customized Simple CNNs


# Get the df
df_labels = get_labels('Dataset/mias_derived_info.csv')

# Define train and val-test transformer
train_transforms = v2.Compose([
    v2.Resize((256, 256)),
    v2.RandomRotation(10), # data augumentation
    v2.ColorJitter(brightness=0.1, contrast=0.1), # data augumentation
    v2.GaussianBlur(3), # data augumentation
    v2.ToDtype(torch.float32, scale=True), # convert to float32 for normalization
    v2.Normalize(mean=[0.5], std=[0.5])
])
val_transforms = v2.Compose([
    v2.Resize((256, 256)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5], std=[0.5])
])

# 5-Fold cross validation to evaluate model's general performance as the dataset is small
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_acc_all_folds = []
for fold, (train_idx, val_idx) in enumerate(skf.split(df_labels, df_labels['label'])):
    print(f"Fold {fold+1}")
    
    # Get train and val subset
    train_subset = df_labels.iloc[train_idx].reset_index(drop=True)
    val_subset = df_labels.iloc[val_idx].reset_index(drop=True)
    
    # Data loader for model
    train_loader, val_loader = get_loader(train_subset, val_subset, train_transforms, val_transforms)
    
    # Init model (fine-tune the whole model), set criteria, use adam optimizer
    model, criteria, optimizer = init_model_config(model="custom")
    
    # Train in epochs with mini-batches
    for epoch in range(1000):
        train_acc = train_one_epoch(model, train_loader, optimizer, criteria)
        val_acc, report = validate(model, val_loader)
        print(f"Epoch {epoch+1} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
        print(report)
        # Early stop
        if val_acc >= 0.9:
            print("Early stopping triggered: Reached 0.9 validation accuracy.")
            break
    
    # Use the final epoch's val_acc as the fold's val acc
    val_acc_all_folds.append(val_acc)
print(f"Mean val accuracy of all folds: {sum(val_acc_all_folds)/len(val_acc_all_folds):.4f}")