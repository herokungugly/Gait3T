
import torch
from .base import BaseLoss

class SupervisedContrastiveLossSingle(BaseLoss):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLossSingle, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim] 
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")
      
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) != targets).to(device)
        mask_similar_class = mask_similar_class + torch.eye(targets.shape[0]).to(device)
        dot_product_tempered = projections * mask_similar_class / self.temperature
        
      
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        log_prob = -torch.log(exp_dot_tempered / torch.sum(exp_dot_tempered, dim=1, keepdim=True))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * torch.eye(targets.shape[0]).to(device), dim=1, keepdim=True)
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        self.info.update({
            'super_con_loss': supervised_contrastive_loss
        })

        return supervised_contrastive_loss, self.info
