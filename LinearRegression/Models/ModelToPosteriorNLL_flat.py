import torch 
from tqdm import tqdm

from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint

from PFNExperiments.LinearRegression.Evaluation.CompareComparisonModels import PosteriorComparisonModel
from PFNExperiments.Training.FlowMatching.DDPMLossDiffusionVP import DDPMLossDiffusionVP

class ModelToPosteriorNLL_flat(PosteriorComparisonModel):
    """
    A class to compute posterior samples from a model trained with NLL loss
    """


    def __init__(self,
                 model: torch.nn.Module,
                 sample_shape: torch.Size,
                 sample_name: str = "beta",
                 n_samples: int = 1000,
                 batch_size: int = 256,
                 device: str = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                 target_device: str = torch.device("cpu"),
                 ) -> None:
        """
        Args:
            model: torch.nn.Module: the model that is trained with flow matching and takes in the forward pass a triple of inputs (z, x, t) and returns a vector field
            sample_shape: torch.Size: the shape of the samples that are generated
            sample_name: str: the name of the sample
            n_samples: int: the number of samples to generate
            batch_size: int: the batch size for the ODE solver
            device: str: the device to use for the computation
            tarfet_device: str: the device to use for the final output samples
        """
        self.model = model.to(device)
        self.sample_name = sample_name
        self.sample_shape = sample_shape
        self.device = device
        self.target_device = target_device

        self.n_samples = n_samples
        self.batch_size = batch_size



    def sample_posterior_x_batch(self, x: torch.tensor, n_samples: int) -> torch.Tensor:
        """
        Return samples from the posterior by solving the ODE with the learned vector field
        this processes a single batch 
        Args:
            x: torch.tensor: the input to the model that conditions the distribution
            n_samples: int: the number of samples to generate
        """


        # duplicate the input x to match the number of samples

        p = self.sample_shape[0]
        batch_size = x.shape[0]

        zt = torch.ones(batch_size, 2*p + p**2).to(self.device) # create a tensor of ones with the same shape as the input x
        t = torch.ones(batch_size, 1).to(self.device) # create a tensor of ones with the same shape as the input 
        x = x.to(self.device) # move the input x to the device

        zt = zt.unsqueeze(0)
        t = t.unsqueeze(0)
        x = x.unsqueeze(0)

        pred = self.model(zt, x, t)

        mean = pred[:, :p]
        cov_diag = pred[:, p:2*p]
        cov_factor = pred[:, 2*p:]
        

        cov_factor = cov_factor.reshape(batch_size, p, p)
        cov_diag = cov_diag **2 + 1e-5
        dist = torch.distributions.LowRankMultivariateNormal(
            loc = mean,
            cov_factor = cov_factor,
            cov_diag = cov_diag
        )

        # sample from the distribution
        samples = dist.rsample((n_samples,))

        assert samples.shape[1] == self.sample_shape, f"the shape of the samples is not in the right shape, it should be {self.sample_shape} but it is {samples.shape}"


        samples = samples.to(self.target_device)

        return samples[-1]

    def sample_posterior_x(self, X: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Return samples from the posterior by solving the ODE with the learned vector field
        Args:
            X: torch.Tensor: the input to the model that conditions the distribution
            n_samples: int: the number of samples to generate
        """

        n_batches = int(n_samples / self.batch_size)
        res = []

        for i in tqdm(list(range(n_batches))):
            x = X[i * self.batch_size: (i + 1) * self.batch_size]
            res.append(self.sample_posterior_x_batch(x, self.batch_size))

        if n_samples % self.batch_size != 0:
            x = X[n_batches * self.batch_size: n_samples]
            res.append(self.sample_posterior_x_batch(x, n_samples % self.batch_size))

        return torch.cat(res, dim=0)

    
    def sample_posterior(self,  
                X: torch.Tensor,
                y: torch.Tensor = None) -> torch.Tensor:
        """
        A method that samples from the posterior distribution
        Args:
            X: torch.Tensor: the covariates
            y: torch.Tensor: the response variable
        Returns:
            torch.Tensor
        """
        if y is not None:
            if len(y.shape) == 1:
                y = y.unsqueeze(-1)


            X_y = torch.cat([X, y], dim = -1) # concatenate the x and y values to one data tensor
        else:
            X_y = X
            
        samples = self.sample_posterior_x(X_y, self.n_samples)

        res = {
            self.sample_name: samples,
            "X": X,
            "y": y
        }

        return res