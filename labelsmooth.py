import numpy as np
import torch
import torch.nn as nn



class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.3, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs.cuda()).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        # return loss


if __name__ == '__main__':

    # parser.add_argument('--smooth', type=float, default=0.1)
    loss_fn = nn.CrossEntropyLoss()
    classifier_loss = CrossEntropyLabelSmooth(num_classes=2)#(outputs_source, labels_source)
    input = torch.tensor([[0.6, 0.4], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6]])
    target = torch.tensor([1, 0, 0, 0])
    loss=classifier_loss(input,target)
    loss1=loss_fn(input,target)
    print(loss)
    print(loss1)