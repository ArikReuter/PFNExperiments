import pyro 
import torch

import pyro.infer
import pyro.optim
from pyro.infer import SVI, Trace_ELBO

from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import ppgram_linear_model_return_y
from PFNExperiments.LinearRegression.ComparisonModels.PosteriorComparisonModel import PosteriorComparisonModel

from pyro.infer.autoguide import AutoDiagonalNormal

class Variational_InferenceAutoguide(PosteriorComparisonModel):
    """
    Perform variational inference on a given probabilistic program
    """

    def __init__(self, 
                 pprogram: ppgram_linear_model_return_y,
                 make_guide_fun: callable = AutoDiagonalNormal,
                 additional_make_guide_args: dict = {},
                 n_steps:int = 2000,
                 n_samples:int = 200,
                 lr: float = 1e-3) -> None:
        """
        Args:
            pprogram: ppgram_linear_model_return_y: the probabilistic program
            make_guide_fun : callable: a function that generates the guide
            n_steps: int: the number of steps to take in the optimization
            lr: float: the learning rate of the optimizer

        Returns:
            None
        """
        self.pprogram = pprogram
        self.make_guide_fun = make_guide_fun
        self.additional_make_guide_args = additional_make_guide_args
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.lr = lr
    

    def generate_guide(self):
        """
        generate the guide
        """

        self.guide = self.make_guide_fun(self.pprogram, **self.additional_make_guide_args)


    def do_inference(self,  
                X: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """
        A method that performs variational inference
        Args:
            X: torch.Tensor: the covariates
            y: torch.Tensor: the response variable
        Returns:
            torch.Tensor: the samples from the posterior distribution
        """
        self.guide = self.make_guide_fun(self.pprogram)


        self.optimizer = pyro.optim.Adam({"lr": self.lr})

        self.svi = SVI(self.pprogram, self.guide, self.optimizer, loss=Trace_ELBO())

        pyro.clear_param_store()
        for step in range(self.n_steps):
            self.loss = self.svi.step(X, y)
            if step % 100 == 0:
                print('.', end='')
        print()
        return self.loss
    
    def sample_posterior(self,
                X: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """
        A method that samples from the posterior distribution
        Args:
            X: torch.Tensor: the covariates
            y: torch.Tensor: the response variable
        Returns:
            torch.Tensor: the samples from the posterior distribution
        """
        self.do_inference(X, y)
        posterior_samples = pyro.infer.Predictive(self.guide, num_samples=self.n_samples)(X, y)
        return posterior_samples
    
    def __repr__(self) -> str:
        self.guide = self.make_guide_fun(self.pprogram)
        return "Variational Inference with guide: {}".format(self.guide)