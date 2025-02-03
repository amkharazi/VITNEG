import torch

def topk_accuracy(outputs, targets, topk=(1,)):
    maxk = max(topk)
    _, topk_indices = torch.topk(input=outputs, k=maxk, dim=1, largest=True, sorted=True)
    correct = topk_indices.eq(targets.view(-1, 1).expand_as(topk_indices))
    accuracies = {}
    for k in topk:
        correct_k = correct[:,:k].float().sum() 
        accuracies[k] = {'correct':correct_k, 'accuracy': (correct_k / outputs.shape[0]) * 100.0}
    return accuracies