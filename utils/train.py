import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from utils.dataset import MIASDataset
from utils.models import HybridCNN
from utils.evaluate import evaluate_model


def perf_curve(train_loss_list, val_loss_list, train_acc_list, val_acc_list):
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    epochs = list(range(1, len(train_loss_list)+1))

    # Plot Loss
    ax1.plot(epochs, train_loss_list, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, val_loss_list, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 15)

    # Plot Accuracy
    ax2.plot(epochs, train_acc_list, 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    ax2.plot(epochs, val_acc_list, 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 15)
    ax2.set_ylim(0, 1)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def train_one_fold(model, train_loader, val_loader, device, num_epochs, lr=0.001): # train one fold in multi epoches, save the model state with the highest val acc
    # Weighted CE
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 1.2, 1.3], dtype=torch.float32).to(device)
        )
    # Other train configs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # step size means decay at n epochs
    
    best_val_acc = 0.0
    best_model_state = None
    train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Val
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()*inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct/val_total
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        
        # Add scheduler's step
        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
    
    # Plot the loss and acc curve of train and val for this fold
    perf_curve(train_loss_list, val_loss_list, train_acc_list, val_acc_list)
    
    return best_model_state, best_val_acc


def train(img_dict, label_dict, num_classes=3, num_folds=5, batch_size=32, drop_last=True, num_epochs=15): # train in n-fold and calculate the average performance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use device: {device}")
    
    # Prepare data indices
    indices = list(img_dict.keys())
    labels = [label_dict[idx] for idx in indices]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    confusion_matrices = []
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels)):
        print(f"\nFold {fold+1}/{num_folds}")
        
        # Split indices
        train_indices = [indices[i] for i in train_idx]
        test_indices = [indices[i] for i in test_idx]
        
        # Create datasets
        train_dataset = MIASDataset(img_dict, label_dict, train_indices)
        test_dataset = MIASDataset(img_dict, label_dict, test_indices)
        
        # Create data loaders
        def weight_sampler(dataset): # use weight sampler to upsample abnormal classes
            class_weights = {0: 1.97, 1: 1.67, 2: 0.53}
            
            # Extract labels from dataset and map each label to its weight
            labels = []
            for _, label in dataset:
                labels.append(label)
            labels = torch.tensor(labels)
            sample_weights = torch.tensor([class_weights[int(l)] for l in labels], dtype=torch.float32)
            
            # Create sampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            return sampler
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=drop_last, # use drop_last = True for n-fold, as some fold train-test split may result in a batch has only one sample in the train dataset, this will cause error  
                              # shuffle=True, # if no weight sampler, then use this
                              sampler=weight_sampler(train_dataset)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = HybridCNN(num_classes=num_classes).to(device)
        
        # Train model
        best_model_state, best_val_acc = train_one_fold(model, train_loader, test_loader, device, num_epochs=num_epochs)
        
        # Load the best model of this fold
        model.load_state_dict(best_model_state)
        
        # Evaluate the best model and check its confusion matrix
        conf_matrix, test_acc = evaluate_model(model, test_loader, device, num_classes)
        confusion_matrices.append(conf_matrix)
        accuracies.append(test_acc)
        
        print(f"Fold {fold+1} Test Accuracy: {test_acc:.4f}")
        print(f"Fold {fold+1} Confusion Matrix:\n{conf_matrix}")
    
    # Compute average confusion matrix
    avg_conf_matrix = np.mean(confusion_matrices, axis=0)
    avg_accuracy = np.mean(accuracies)
    
    print("\nAverage Results Across Folds:")
    print(f"Average Test Accuracy: {avg_accuracy:.4f}")
    print(f"Average Confusion Matrix:\n{avg_conf_matrix}")