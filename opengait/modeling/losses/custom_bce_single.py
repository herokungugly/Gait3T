
import torch
import torch.nn as nn
from .base import BaseLoss

class ClipBinaryCrossEntropyLoss(BaseLoss):
    def __init__(self, temperature=1, B=0):
        super(ClipBinaryCrossEntropyLoss, self).__init__()
        self.B = nn.Parameter(torch.ones(1))
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, projections, targets):

        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")
        mask_similar_class_bool = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_similar_class = mask_similar_class_bool.int() - 1 + torch.eye(targets.shape[0]).to(device)
        
        dot_product_tempered = projections / self.temperature - self.B
        sigmoid_dot_product = torch.sigmoid(dot_product_tempered*mask_similar_class)

        log_prob = -torch.log(sigmoid_dot_product)
        log_prob = torch.sum(log_prob, dim = 1)
        binary_crossentropy_loss_per_sample = torch.sum(log_prob, dim=1, keepdim=True) / targets.shape[0]  # does not consider the masked same id samples
        binary_crossentropy_loss = torch.mean(binary_crossentropy_loss_per_sample)


        self.info.update({
            'clip_bce_loss': binary_crossentropy_loss,
            'B': self.B,
            'temp': self.temperature
        })

        return binary_crossentropy_loss, self.info
