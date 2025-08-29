import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate_model(model, test_loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    
    return conf_matrix, accuracy