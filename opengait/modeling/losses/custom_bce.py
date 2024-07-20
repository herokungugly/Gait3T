
import torch
from .base import BaseLoss

class ClipBinaryCrossEntropyLoss(BaseLoss):
    def __init__(self, temperature=0.07, B=0.29):
        super(ClipBinaryCrossEntropyLoss, self).__init__()
        self.B = B
        self.temperature = temperature

    def forward(self, projections, targets):

        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)

        dot_product_tempered = projections / self.temperature - self.B
        sigmoid_dot_product = torch.sigmoid(projections*mask_similar_class)
        
        # sigmoid_dot_product = torch.sigmoid(projections)
        cardinality_per_samples = torch.sum(mask_similar_class, dim = 1)

        log_prob = -torch.log(sigmoid_dot_product)
        # binary_crossentropy_loss_per_sample = torch.sum(log_prob * mask_similar_class, dim=1, keepdim=True) / cardinality_per_samples
        binary_crossentropy_loss_per_sample = torch.sum(log_prob, dim=1, keepdim=True) / cardinality_per_samples
        binary_crossentropy_loss = torch.mean(binary_crossentropy_loss_per_sample)


        self.info.update({
            'clip_bce_loss': binary_crossentropy_loss
        })

        return binary_crossentropy_loss, self.info
