import torch 


def compute_accuracy(logits, y_true, device='cuda:0'):
    y_pred = torch.argmax(logits, dim=1)
    y_true_on_device = y_true
    accuracy = (y_pred == y_true_on_device).float().mean()
    return accuracy