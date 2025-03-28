from pyro.infer.autoguide import  AutoDelta
from PFNExperiments.LinearRegression.ComparisonModels.Variational_InferenceAutoguide import Variational_InferenceAutoguide
import torch

from pyro.infer.autoguide import  AutoDelta
from PFNExperiments.LinearRegression.ComparisonModels.Variational_InferenceAutoguide import Variational_InferenceAutoguide
import torch

class PyroMAPPredictor():
    """
    Class that takes a pprogram and then becomes a MAP predictor.
    """

    ppgrogram_name2response_function = {
    "ig": lambda x: x,
    "ig_intercept": lambda x: x,
    "Laplace_ig": lambda x: x,
    "Laplace_ig_intercept": lambda x: x,
    "Gamma_ig": lambda x: x,
    "logreg_ig": lambda x: torch.sigmoid(x),
    "ig_gamma_response_reparam": lambda x: torch.exp(x),
    }

    ppgrogram_name2useintercept = {
        "ig": False,
        "ig_intercept": True,
        "Laplace_ig": False,
        "Laplace_ig_intercept": True,
        "Gamma_ig": False,
        "logreg_ig": False,
        "ig_gamma_response_reparam": False,
    }

    def __init__(
            self,
            pprogram_y,
            pprogram_name: str,
            lr:float = 1e-2,
            n_steps:int = 2000
            ):
        """
        Args:
            pprogram_y: a probabilistic program for the response variable
            pprogram_name: str: the name of the probabilistic program
            lr: float: the learning rate
            n_steps: int: the number of steps
        """
        self.pprogram_y = pprogram_y

        self.response_function = self.ppgrogram_name2response_function[pprogram_name]
        self.use_intercept = self.ppgrogram_name2useintercept[pprogram_name]


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

        try: 
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
        except:
            pass
        self.map_estimates = self.vi_MAP.sample_posterior(X,y)

    def predict(self,
                X: torch.Tensor
                ):
        """
        predict the MAP estimate
        """
        beta = self.map_estimates["beta"].squeeze()

        try: 
            X = torch.tensor(X, dtype=torch.float32)
        except:
            pass
        
        if self.use_intercept:
            intercept = self.map_estimates["beta0"].squeeze()

            return self.response_function(torch.matmul(X,beta) + intercept)
        else:
            return self.response_function(torch.matmul(X,beta))
        
        


        