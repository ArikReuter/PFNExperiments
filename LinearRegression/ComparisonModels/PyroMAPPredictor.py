from pyro.infer.autoguide import  AutoDelta
from PFNExperiments.LinearRegression.ComparisonModels.Variational_InferenceAutoguide import Variational_InferenceAutoguide
import torch

class PyroMAPPredictor():
    """
    Class that takes a pprogram and then becomes a MAP predictor.
    """

    def __init__(
            self,
            pprogram_y,
            lr:float = 1e-2,
            n_steps:int = 2000
            ):
        """
        Args:
            pprogram_y: a probabilistic program for the response variable
            lr: float: the learning rate
            n_steps: int: the number of steps
        """
        self.pprogram_y = pprogram_y

        self.vi_MAP = Variational_InferenceAutoguide(
        pprogram=pprogram_y,
        make_guide_fun=AutoDelta,
        n_steps=n_steps,
        n_samples=1,
        lr=lr
        )
    
    def fit(self,
            X: torch.Tensor,
            y: torch.Tensor
            ):
        """
        fit the model
        """
        self.map_estimates = self.vi_MAP.sample_posterior(X,y)

    def predict(self,
                X: torch.Tensor
                ):
        """
        predict the MAP estimate
        """
        beta = self.map_estimates["beta"]
        beta0 = self.map_estimates["beta0"]

        return torch.matmul(X,beta) + beta0


        