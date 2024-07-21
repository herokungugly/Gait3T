
import torch
from .base import BaseLoss

class ClipBinaryCrossEntropyLoss(BaseLoss):
    def __init__(self, temperature=0.07, beta=1.0):
        super(ClipBinaryCrossEntropyLoss, self).__init__()
        self.temperature = temperature
        self.beta = beta

    def forward(self, projections, targets):
        # Compute similarity matrix
        similarities = projections / self.temperature
        
        # Create mask for positive and negative pairs
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).float()
        mask_dissimilar_class = 1 - mask_similar_class
        m_ij = mask_similar_class - mask_dissimilar_class
        
        # Compute the loss
        loss_matrix = torch.log(1 + torch.exp(m_ij * (-similarities + self.beta)))
        loss = loss_matrix.sum() / targets.shape[0]
        self.info.update({
            'clip_bce_loss': loss
        })

        return loss, self.info
