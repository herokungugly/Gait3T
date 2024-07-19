
import torch
from .base import BaseLoss

class ClipBinaryCrossEntropyLoss(BaseLoss):
    def __init__(self, temperature=0.07):
        super(ClipBinaryCrossEntropyLoss, self).__init__()
        self.temperature = temperature

    def custom_sigmoid(self, x):
        """
        Custom implementation of the sigmoid function.
        :param x: torch.Tensor
        :return: torch.Tensor
        """
        return 1 / (1 + torch.exp(-x))

    def forward(self, projections, targets):

        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = projections / self.temperature
        # exp_dot_tempered = (torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5)  # softmax for supconloss
        sigmoid_dot_product = self.custom_sigmoid(dot_product_tempered)
  
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        cardinality_per_samples = torch.sum(mask_similar_class, dim = 1)

        log_prob = -torch.log(sigmoid_dot_product + self.eps)
        binary_crossentropy_loss_per_sample = torch.sum(log_prob * mask_similar_class, dim=1, keepdim=True) / cardinality_per_samples
        binary_crossentropy_loss = torch.mean(binary_crossentropy_loss_per_sample)


        self.info.update({
            'clip_bce_loss': binary_crossentropy_loss
        })

        return binary_crossentropy_loss, self.info
