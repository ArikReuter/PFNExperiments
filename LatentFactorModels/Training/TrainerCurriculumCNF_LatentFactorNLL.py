from PFNExperiments.Training.TrainerCurriculumCNF import TrainerCurriculumCNF
import torch

class TrainerCurriculumCNF_LatentFactorNLL(TrainerCurriculumCNF):
    """
    Trainer class for Latent Factor Models
    """


    def batch_to_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the loss from a batch       
        Args:
            batch: dict[str, torch.Tensor]: the batch
        Returns:
            torch.Tensor: the loss
        """
        x = batch["x"]

        z_1 = batch["beta"]     # sample from the ground truth distribution
        z_0 = batch['base_sample_beta']  # sample from the base distribution
        t = batch["time"]  # sample the time
        t = t.unsqueeze(-1) # add a dimension to the time tensor to give it shape (batch_size, 1)

        t = t.float()

        t = torch.ones_like(t) # replace t with ones to avoid the model to learn the time dependency

        p = z_1.shape[1] 
        zt = torch.ones(z_1.shape[0], p*p + 2*p).to(self.device) # create a tensor of ones with the same shape as z_1


        model_pred = self.model(zt, x, t)  # compute the vector field prediction by the model

        loss = self.loss_function(model_pred, z_1)  # compute the loss
    
        return loss