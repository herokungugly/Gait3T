
import torch
from .base import BaseLoss

class ClipBinaryCrossEntropyLoss(BaseLoss):
    def __init__(self, temperature=1, B=0):
        super(ClipBinaryCrossEntropyLoss, self).__init__()
        self.B = B
        self.temperature = temperature

    def forward(self, projections, targets, B, temperature):

        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")
        mask_similar_class_bool = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_similar_class = mask_similar_class_bool.int() * 2 - 1
        
        dot_product_tempered = projections / temperature - B
        sigmoid_dot_product = torch.sigmoid(dot_product_tempered*mask_similar_class)
        
        cardinality_per_samples = torch.sum(mask_similar_class_bool, dim = 1)

        log_prob = -torch.log(sigmoid_dot_product)
        log_prob = torch.sum(log_prob, dim = 1)
        binary_crossentropy_loss_per_sample = torch.sum(log_prob, dim=1, keepdim=True) / (cardinality_per_samples*targets.shape[0])
        binary_crossentropy_loss = torch.mean(binary_crossentropy_loss_per_sample)


        self.info.update({
            'clip_bce_loss': binary_crossentropy_loss
        })

        return binary_crossentropy_loss, self.info
