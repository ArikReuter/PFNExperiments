import pyro
import torch

import pyro.infer
import pyro.optim
from pyro.infer import SVI, Trace_ELBO

from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import ppgram_linear_model_return_y
from PFNExperiments.LinearRegression.ComparisonModels.PosteriorComparisonModel import PosteriorComparisonModel

from pyro.infer.autoguide import AutoDiagonalNormal

from pyro.optim import PyroOptim
import torch.optim as optim

class Variational_InferenceAutoguide(PosteriorComparisonModel):
    """
    Perform variational inference on a given probabilistic program
    """

    def __init__(self,
                 pprogram: ppgram_linear_model_return_y,
                 make_guide_fun: callable = AutoDiagonalNormal,
                 additional_make_guide_args: dict = {},
                 n_steps:int = 100,
                 n_samples:int = 200,
                 lr: float = 1,
                 print_lr: bool = False) -> None:
        """
        Args:
            pprogram: ppgram_linear_model_return_y: the probabilistic program
            make_guide_fun : callable: a function that generates the guide
            n_steps: int: the number of steps to take in the optimization
            lr: float: the learning rate of the optimizer
            print_lr: bool: whether to print the learning rate

        Returns:
            None
        """
        self.pprogram = pprogram
        self.make_guide_fun = make_guide_fun
        self.additional_make_guide_args = additional_make_guide_args
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.lr = lr
        self.print_lr = print_lr



        self.guide = self.make_guide_fun(self.pprogram)

        self.optimizer = PyroOptim(optim.LBFGS, {"lr": lr, "max_iter": n_steps})

        self.svi = SVI(self.pprogram, self.guide, self.optimizer, loss=Trace_ELBO())



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
        rep = "Variational Inference with guide: {}".format(self.guide)

        if self.print_lr:
            rep = f"Variational Inference with learning rate: {self.lr} and guide: {self.guide}"

        return rep