import torch 
import pyro
from tqdm import tqdm
from typing import List, Dict, Tuple
try:
    from LM_abstract import pprogram_linear_model_return_dict, return_only_y, pprogram_X
    from GenerateX import simulate_X_uniform
except:
    from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import pprogram_linear_model_return_dict, return_only_y, pprogram_X
    from PFNExperiments.LinearRegression.GenerativeModels.GenerateX import simulate_X_uniform


class GenerateData:
    """
    Class to generate data for (linear) regression
    """

    def __init__(self,
                 pprogram: pprogram_linear_model_return_dict,
                 pprogram_covariates: pprogram_X = simulate_X_uniform,
                 seed:int = 42
                 ):
        """
        Constructor for GenerateData
        Args:
            pprogram: pprogram_linear_model_return_dict: a linear model probabilistic program
            ppriogram_covariates: pprogram_X: a probabilistic program that simulates covariates
            seed: int: the seed for the random number generator
        """
        self.pprogram = pprogram
        self.pprogram_covariates = pprogram_covariates
        self.seed = seed
        
        #torch.manual_seed(seed)
        #pyro.set_rng_seed(seed)


    def simulate_data(self, 
                     n:int = 100, 
                     p:int = 5, 
                     n_batch:int = 10_000
                    ) -> Tuple[list[dict], int]:
        """
        Simulate data for a linear model with the same beta for each batch.
        Discard samples where the linear model is not finite to avoid numerical issues.
        Args:
            n: int: the number of observations per batch 
            p: int: the number of covariates
            n_batch: int: the number of batches
        Returns:
            Tuple[list[dict], int]: a list of dictionaries containing the simulated data and the number of discarded samples
        """

        n_discarded = 0

        data = []
        for i in tqdm(list(range(n_batch))):
            x = self.pprogram_covariates(n,p)

            while True:
                lm_res = self.pprogram(x)

                # if anything is nan or inf, sample again
                if all([torch.isfinite(lm_res[key]).all() for key in lm_res.keys()]):
                    break
                else:
                    n_discarded += 1

            data.append(lm_res)

        if n_discarded > 0:
            print(f"Warning: {n_discarded} samples were discarded because the linear model was not finite.")

        return data, n_discarded
    
    def render_model(self, p:int = 10):
        """
        Render the model
        """
        Xt = self.pprogram_covariates(1, p)
        yt = return_only_y(self.pprogram)(Xt)

        r = pyro.render_model(self.pprogram, model_args=(Xt, yt), render_distributions=True)
        return r
    
    def check_model(self, p:int = 3, n:int = 100, n_batch:int = 1_000):
        """
        Check the model for a few batches
        Args: 
            p: int: the number of covariates
            n: int: the number of observations
            n_batch: int: the number of batches
        """

        sample_data, discarded = self.simulate_data(n, p, n_batch)

        print(f"Discarded {discarded} samples")
        r = check_and_plot_data(sample_data)

        return r

    def make_dataloaders_static(self,
                         n:int = 100,
                         p:int = 5,
                        n_batch:int = 10_000,
                        batch_size:int = 256,
                        train_frac = 0.7,
                        val_frac = 0.15,
                        shuffle: bool = True
                        ) -> Tuple[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader], list[dict]]:
        """
        Make a dataloader from a generated dataset
        Args:
            n: int: the number of observations per batch 
            p: int: the number of covariates
            n_batch: int: the number of batches
            batch_size: int: the batch size
            train_frac: float: the fraction of the data to use for training
            val_frac: float: the fraction of the data to use for validation
            shuffle: bool: whether to shuffle the data
        Returns:
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: a tuple of dataloaders for the training, test and validation data
            list[dict]: the simulated data
        """
        sample_data, discarded = self.simulate_data(n, p, n_batch)
        dataloaders = make_dataloaders_static_data(sample_data, batch_size, train_frac, val_frac, shuffle)
        return dataloaders, sample_data
    
    def make_dataloaders_dynamic(self,
                                 n:int = 100,
                                 p:int = 5,
                                 n_batch:int = 10_000,
                                 batch_size:int = 256,
                                 train_frac = 0.7,
                                 val_frac = 0.15,
                                 shuffle: bool = True
                                 ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Make a dataloader where the data is generated on the fly
        Args:
            n: int: the number of observations per batch 
            p: int: the number of covariates
            n_batch : int: the number of batches
            batch_size: int: the batch size
            train_frac: float: the fraction of the data to use for training
            val_frac: float: the fraction of the data to use for validation
            shuffle: bool: whether to shuffle the data
        Returns:
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: a tuple of dataloaders for the training, test and validation data
        """

        train_size = int(train_frac * n_batch)
        val_size = int(val_frac * n_batch)
        test_size = n_batch - train_size - val_size

        dataset_train = SyntheticData(n = n, 
                                p = p, 
                                n_batch = train_size, 
                                pprogram = self.pprogram, 
                                pprogram_covariates = self.pprogram_covariates, 
                                seed = self.seed)
        dataset_val = SyntheticData(n = n,
                                p = p,
                                n_batch = val_size,
                                pprogram = self.pprogram,
                                pprogram_covariates = self.pprogram_covariates,
                                seed = self.seed)
        dataset_test = SyntheticData(n = n,
                                p = p,
                                n_batch = test_size,
                                pprogram = self.pprogram,
                                pprogram_covariates = self.pprogram_covariates,
                                seed = self.seed)
        
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=shuffle)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle)

        return train_loader, test_loader, val_loader

class SyntheticData(torch.utils.data.Dataset):
    """
    A class to represent synthetic data that is generated on the fly
    Note that sampling is always random and the same seed is used for all batches
    """
    def __init__(
                self,
                n:int = 100,
                p:int = 5,
                n_batch:int = 10_000,
                pprogram: pprogram_linear_model_return_dict = None,
                pprogram_covariates: pprogram_X = simulate_X_uniform,
                seed:int = None
                ):
        """
        a  torch.utils.data.Dataset that generates synthetic data on the fly
        Args:
            n: int: the number of observations per batch 
            p: int: the number of covariates
            n_batch: int: the number of batches
            pprogram: pprogram_linear_model_return_dict: a linear model probabilistic program
            pprogram_covariates: pprogram_X: a probabilistic program that simulates covariates
            seed: int: the seed for the random number generator
        """
        self.n = n
        self.p = p
        self.n_batch = n_batch
        self.pprogram = pprogram
        self.pprogram_covariates = pprogram_covariates

        if seed is None:
            self.seed = torch.randint(0, 100000, (1,)).item()
        self.seed = seed

        #torch.manual_seed(seed)
        #pyro.set_rng_seed(seed)
    
    def __len__(self):
        return self.n_batch

    def __getitem__(self, idx):
        x = self.pprogram_covariates(self.n, self.p)

        while True:
            lm_res = self.pprogram(x)

            # if anything is nan or inf, sample again
            if all([torch.isfinite(lm_res[key]).all() for key in lm_res.keys()]):
                break

        return lm_res

def check_data(data: List[Dict[str, torch.tensor]], batched_input:bool = False) -> dict:
    """
    Check statistics about the simulated data
    Args:
        data: List[Dict[str, torch.tensor]]: the simulated data in the form of a list of dictionaries where each element of the list is a dictionary containing the data for one batch
        batched_input: bool: whether the input is batched
    
    Returns:
        dict: a dictionary containing the means and variances of the simulated data
    """
    if not batched_input:
    # check for non-finite values
        for i, d in enumerate(data):
            for key in d.keys():
                if torch.isfinite(d[key]).all():
                    pass
                else:
                    raise ValueError(f"Data point {i} has non-finite values for {key}")
                
        stacked_data = {key: torch.stack([d[key] for d in data]) for key in data[0].keys()}

    else:
        stacked_data = {key: torch.cat([d[key] for d in data]) for key in data[0].keys()}

        for key in stacked_data.keys():
            if torch.isfinite(stacked_data[key]).all():
                pass
            else:
                raise ValueError(f"Data has non-finite values for {key}")

    means = {key: torch.mean(stacked_data[key], dim=0) for key in stacked_data.keys()}
    variances = {key: torch.var(stacked_data[key], dim=0) for key in stacked_data.keys()}
    minimums = {key: torch.min(stacked_data[key], dim=0).values for key in stacked_data.keys()}
    maximums = {key: torch.max(stacked_data[key], dim=0).values for key in stacked_data.keys()}

    return {
        "means": means,
        "variances": variances,
        "minimums": minimums,
        "maximums": maximums
    }

def check_and_plot_data(data: List[Dict[str, torch.tensor]], batched_input = False) -> None:
    """
    Check statistics about the simulated data and plot the data
    Args:
        data: List[Dict[str, torch.tensor]]: the simulated data in the form of a list of dictionaries where each element of the list is a dictionary containing the data for one batch
        batched_input: bool: whether the input is batched
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    stats = check_data(data)

    if not batched_input:
        stacked_data = {key: torch.stack([d[key] for d in data]) for key in data[0].keys()} 
    else:
        stacked_data = {key: torch.cat([d[key] for d in data]) for key in data[0].keys()}

    stats = check_data(data, batched_input=batched_input)

    X_overall_mean = stats["means"]["x"].mean()
    X_overall_variance = stats["variances"]["x"].mean()
    X_overall_min = stats["minimums"]["x"].min()
    X_overall_max = stats["maximums"]["x"].max()

    y_overall_mean = stats["means"]["y"].mean()
    y_overall_variance = stats["variances"]["y"].mean()
    y_overall_min = stats["minimums"]["y"].min()
    y_overall_max = stats["maximums"]["y"].max()


    overall_agg_stats = {
        'X': {
            'mean': X_overall_mean,
            'variance': X_overall_variance,
            'min': X_overall_min,
            'max': X_overall_max
        },
        'y': {
            'mean': y_overall_mean,
            'variance': y_overall_variance,
            'min': y_overall_min,
            'max': y_overall_max
        },
        'beta': {
            'mean': stats["means"]["beta"],
            'variance': stats["variances"]["beta"],
            'min': stats["minimums"]["beta"],
            'max': stats["maximums"]["beta"]
        }
    }

    print(overall_agg_stats)

    # also print the statistics for the other keys in the dictionary
    for key in stacked_data.keys():
        if key not in ["x", "y", "beta"]:
            print(f"Statistics for {key}:")
            print(f"Mean: {stats['means'][key]}")
            print(f"Variance: {stats['variances'][key]}")
            print(f"Min: {stats['minimums'][key]}")
            print(f"Max: {stats['maximums'][key]}")
            print("\n")

    # plot the data
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.histplot(stacked_data["y"].flatten(), ax=ax[0])
    ax[0].set_title("Histogram of y")
    sns.histplot(stacked_data["x"].flatten(), ax=ax[1])
    ax[1].set_title("Histogram of x")
    plt.show()

    # plot each of the beta values in a histogram 
    p = stacked_data["beta"].shape[1]
    fig, ax = plt.subplots(1, p, figsize=(15, 5))
    for i in range(p):
        sns.histplot(stacked_data["beta"][:, i], ax=ax[i])
        ax[i].set_title(f"Histogram of beta_{i}")
    
    plt.show()

    # also plot all other keys in the dictionary

    for key in stacked_data.keys():
        if key not in ["x", "y", "beta"]:
            data = stacked_data[key]

            if len(data.shape) == 1 or (len(data.shape) == 2 and data.shape[1] == 1):
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                sns.histplot(data, ax=ax)
                ax.set_title(f"Histogram of {key}")
                plt.show()
            else:
                p = data.shape[1]
                fig, ax = plt.subplots(1, p, figsize=(15, 5))
                for i in range(p):
                    sns.histplot(data[:, i], ax=ax[i])
                    ax[i].set_title(f"Histogram of {key}_{i}")
                plt.show()

    

    return overall_agg_stats


def make_dataloaders_static_data(
                            data: List[Dict[str, torch.tensor]], 
                            batch_size: int = 256,
                            train_frac = 0.7,
                            val_frac = 0.15,
                            shuffle: bool = True
                            ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Make a dataloader from a generated dataset
    Args:
        data: List[Dict[str, torch.tensor]]: the simulated data in the form of a list of dictionaries where each element of the list is a dictionary containing the data for one batch
        batch_size: int: the batch size
        train_frac: float: the fraction of the data to use for training
        val_frac: float: the fraction of the data to use for validation
        shuffle: bool: whether to shuffle the data
    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: a tuple of dataloaders for the training, test and validation data
    """
    val_frac = 0.15  

    train_size = int(train_frac * len(data))
    val_size = int(val_frac * len(data))
    test_size = len(data) - train_size - val_size

    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    val_data = data[train_size + test_size:]

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader, val_loader