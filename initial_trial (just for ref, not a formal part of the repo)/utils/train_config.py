import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from utils.customNet_config import customNet

def init_model_config(model):
    if model == "custom":
        model = customNet(num_classes=3).to('cpu')
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    else:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True) # 11M params model
        for param in model.parameters(): # freeze all layers
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 3) # replace the final fully connected layer for 3 classes
        for name, param in model.named_parameters(): # unfreeze last block + fc
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
        optimizer = torch.optim.Adam([{'params': model.layer4.parameters(), 'lr': 0.0001}, {'params': model.fc.parameters(), 'lr': 0.0005}])

    criteria = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 1.2, 1.3], dtype=torch.float32).to('cpu')
        )

    return model.to('cpu'), criteria, optimizer


def train_one_epoch(model, loader, optimizer, criteria):
    model.train() # train mode with Dropout layer, BatchNorm layers, etc.
    correct, total = 0, 0
    for imgs, labels_batch in loader: # loop all the mini-batches in the epoch
        imgs, labels_batch = imgs.to('cpu'), labels_batch.to('cpu')
        optimizer.zero_grad() # clear old gradients as pytorch automately memorize
        logits = model(imgs)
        loss = criteria(logits, labels_batch)
        loss.backward() # backpropagation
        optimizer.step() # update weights

        preds = logits.argmax(dim=1)
        correct += (preds == labels_batch).sum().item()
        total += labels_batch.size(0)
    train_acc = correct/total # the total accuracy of one epoch's train
    return train_acc


def validate(model, loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad(): # disables gradient calculation in val
        for imgs, labels_batch in loader:
            imgs, labels_batch = imgs.to('cpu'), labels_batch.to('cpu')
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)
            all_preds.extend(preds.cpu().numpy() if preds.is_cpu else preds.numpy())
            all_labels.extend(labels_batch.cpu().numpy() if labels_batch.is_cpu else labels_batch.numpy())
    val_acc = correct/total # the total accuracy of one epoch's validation
    report = classification_report(all_labels, all_preds, target_names=["Normal", "Benign", "Malignant"])
    return val_acc, report