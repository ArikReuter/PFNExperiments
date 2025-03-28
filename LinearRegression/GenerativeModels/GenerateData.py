import torch 
import pyro
from tqdm import tqdm
from typing import List, Dict, Tuple
import os
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
                 pprogram_covariates_train: pprogram_X = None,
                 pprogram_covariates_val: pprogram_X = None,
                 pprogram_covariates_test: pprogram_X = None,
                 seed:int = 42
                 ):
        """
        Constructor for GenerateData
        Args:
            pprogram: pprogram_linear_model_return_dict: a linear model probabilistic program
            ppriogram_covariates: pprogram_X: a probabilistic program that simulates covariates
            pprogram_covariates_train: pprogram_X: a probabilistic program that simulates covariates for the training set
            pprogram_covariates_val: pprogram_X: a probabilistic program that simulates covariates for the validation set
            pprogram_covariates_test: pprogram_X: a probabilistic program that simulates covariates for the test set

            seed: int: the seed for the random number generator
        """

        if pprogram_covariates_train is None:
            pprogram_covariates_train = pprogram_covariates
            print("pprogram_covariates_train is None, using pprogram_covariates instead")
        if pprogram_covariates_val is None:
            pprogram_covariates_val = pprogram_covariates
            print("pprogram_covariates_val is None, using pprogram_covariates instead")
        if pprogram_covariates_test is None:
            pprogram_covariates_test = pprogram_covariates
            print("pprogram_covariates_test is None, using pprogram_covariates instead")

        if pprogram_covariates is not None and pprogram_covariates_train is not None and pprogram_covariates_val is not None and pprogram_covariates_test is not None:
            print("Warning: pprogram_covariates, pprogram_covariates_train, pprogram_covariates_val, and pprogram_covariates_test are all not None. Ignoring argument pprogram_covariates")
        

        self.pprogram = pprogram
        self.pprogram_covariates = pprogram_covariates
        self.seed = seed
        
        self.pprogram_covariates_train = pprogram_covariates_train
        self.pprogram_covariates_val = pprogram_covariates_val
        self.pprogram_covariates_test = pprogram_covariates_test



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
            x = self.pprogram_covariates_train(n,p)

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
        Xt = self.pprogram_covariates_train(1, p)
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
                                pprogram_covariates = self.pprogram_covariates_train, 
                                seed = self.seed)
        dataset_val = SyntheticData(n = n,
                                p = p,
                                n_batch = val_size,
                                pprogram = self.pprogram,
                                pprogram_covariates = self.pprogram_covariates_val,
                                seed = self.seed)
        dataset_test = SyntheticData(n = n,
                                p = p,
                                n_batch = test_size,
                                pprogram = self.pprogram,
                                pprogram_covariates = self.pprogram_covariates_test,
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

def check_data(data: List[Dict[str, torch.tensor]], batched_input:bool = False, consider_average_variance_statistics: bool = False) -> dict:
    """
    Check statistics about the simulated data
    Args:
        data: List[Dict[str, torch.tensor]]: the simulated data in the form of a list of dictionaries where each element of the list is a dictionary containing the data for one batch
        batched_input: bool: whether the input is batched
        consider_average_variance_statistics: bool: whether to consider the average variance statistics
    
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

    if consider_average_variance_statistics:
        mean_means = {}
        mean_variances = {}
        mean_minimums = {}
        mean_maximums = {}

        median_means = {}
        median_variances = {}
        median_minimums = {}
        median_maximums = {}


        for key in stacked_data.keys():
            try:
                mean_means[key] = torch.mean(torch.mean(stacked_data[key], dim=1), dim = 0)
            except:
                mean_means[key] = None 
            try:
                mean_variances[key] = torch.mean(torch.var(stacked_data[key], dim=1), dim = 0)
            except:
                mean_variances[key] = None
            try:
                mean_minimums[key] = torch.mean(torch.min(stacked_data[key], dim=1), dim = 0)
            except:
                mean_minimums[key] = None
            try:
                mean_maximums[key] = torch.mean(torch.max(stacked_data[key], dim=1), dim = 0)
            except:
                mean_maximums[key] = None

            try:
                median_means[key] = torch.median(torch.mean(stacked_data[key], dim=1), dim = 0)
            except:
                median_means[key] = None
            try:
                median_variances[key] = torch.median(torch.var(stacked_data[key], dim=1), dim = 0)
            except:
                median_variances[key] = None
            try:
                median_minimums[key] = torch.median(torch.min(stacked_data[key], dim=1), dim = 0)
            except:
                median_minimums[key] = None
            try:
                median_maximums[key] = torch.median(torch.max(stacked_data[key], dim=1), dim = 0)
            except:
                median_maximums[key] = None
            

        

        return {
            "means": means,
            "variances": variances,
            "minimums": minimums,
            "maximums": maximums,
            "mean_means": mean_means,
            "mean_variances": mean_variances,
            "mean_minimums": mean_minimums,
            "mean_maximums": mean_maximums,
            "median_means": median_means,
            "median_variances": median_variances,
            "median_minimums": median_minimums,
            "median_maximums": median_maximums
            
        }
    else:
        return {
            "means": means,
            "variances": variances,
            "minimums": minimums,
            "maximums": maximums
        }

def check_and_plot_data(data: List[Dict[str, torch.tensor]], 
                        batched_input = False,
                        consider_average_variance_statistics: bool = False,
                        save_path_plots: str = None
                        ) -> None:
    """
    Check statistics about the simulated data and plot the data
    Args:
        data: List[Dict[str, torch.tensor]]: the simulated data in the form of a list of dictionaries where each element of the list is a dictionary containing the data for one batch
        batched_input: bool: whether the input is batched
        consider_average_variance_statistics: bool: whether to consider the average variance statistics
        save_path_plots: str: the path to save the plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if save_path_plots is not None:
        if not os.path.exists(save_path_plots):
            os.makedirs(save_path_plots)

    if not batched_input:
        stacked_data = {key: torch.stack([d[key] for d in data]) for key in data[0].keys()} 
    else:
        stacked_data = {key: torch.cat([d[key] for d in data]) for key in data[0].keys()}

    stats = check_data(data, batched_input=batched_input, consider_average_variance_statistics = consider_average_variance_statistics)

    overall_agg_stats = {}
    if "x" in stacked_data.keys():
        X_overall_mean = stats["means"]["x"].mean()
        X_overall_variance = stats["variances"]["x"].mean()
        X_overall_min = stats["minimums"]["x"].min()
        X_overall_max = stats["maximums"]["x"].max()

        overall_agg_stats['X'] = {
            'mean': X_overall_mean,
            'variance': X_overall_variance,
            'min': X_overall_min,
            'max': X_overall_max
        }

    if "y" in stacked_data.keys():
        y_overall_mean = stats["means"]["y"].mean()
        y_overall_variance = stats["variances"]["y"].mean()
        y_overall_min = stats["minimums"]["y"].min()
        y_overall_max = stats["maximums"]["y"].max()

        overall_agg_stats['y'] = {
            'mean': y_overall_mean,
            'variance': y_overall_variance,
            'min': y_overall_min,
            'max': y_overall_max
        }

    if "beta" in stacked_data.keys():
        beta_overall_mean = stats["means"]["beta"].mean()
        beta_overall_variance = stats["variances"]["beta"].mean()
        beta_overall_min = stats["minimums"]["beta"].min()
        beta_overall_max = stats["maximums"]["beta"].max()

        overall_agg_stats['beta'] = {
            'mean': beta_overall_mean,
            'variance': beta_overall_variance,
            'min': beta_overall_min,
            'max': beta_overall_max
        }


    if consider_average_variance_statistics and "x" in stacked_data.keys():
        overall_agg_stats['X']['mean_mean'] = stats['mean_means']['x']
        overall_agg_stats['X']['variance_mean'] = stats['mean_variances']['x']
        overall_agg_stats['X']['min_mean'] = stats['mean_minimums']['x']
        overall_agg_stats['X']['max_mean'] = stats['mean_maximums']['x']

    if consider_average_variance_statistics and "y" in stacked_data.keys():
        overall_agg_stats['y']['mean_mean'] = stats['mean_means']['y']
        overall_agg_stats['y']['variance_mean'] = stats['mean_variances']['y']
        overall_agg_stats['y']['min_mean'] = stats['mean_minimums']['y']
        overall_agg_stats['y']['max_mean'] = stats['mean_maximums']['y']

    if consider_average_variance_statistics and "beta" in stacked_data.keys():
        overall_agg_stats['beta']['mean_mean'] = stats['mean_means']['beta']
        overall_agg_stats['beta']['variance_mean'] = stats['mean_variances']['beta']
        overall_agg_stats['beta']['min_mean'] = stats['mean_minimums']['beta']
        overall_agg_stats['beta']['max_mean'] = stats['mean_maximums']['beta']


    print(overall_agg_stats)

    # also print the statistics for the other keys in the dictionary
    for key in stacked_data.keys():
        if key not in ["x", "y", "beta"]:
            print(f"Statistics for {key}:")
            print(f"Mean: {stats['means'][key]}")
            print(f"Variance: {stats['variances'][key]}")
            print(f"Min: {stats['minimums'][key]}")
            print(f"Max: {stats['maximums'][key]}")

            if consider_average_variance_statistics:
                if key in stats['mean_means'].keys():
                    print(f"Mean of means: {stats['mean_means'][key]}")
                    print(f"Mean of variances: {stats['mean_variances'][key]}")
                    print(f"Mean of minimums: {stats['mean_minimums'][key]}")
                    print(f"Mean of maximums: {stats['mean_maximums'][key]}")
            print("\n")
    """
    # plot the data
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    if "y" in stacked_data.keys():
        
        sns.histplot(stacked_data["y"].flatten(), ax=ax[0])
        ax[0].set_title("Histogram of y")

    if "x" in stacked_data.keys():
        sns.histplot(stacked_data["x"].flatten(), ax=ax[1])
        ax[1].set_title("Histogram of x")
    
    if save_path_plots is not None:
        plt.savefig(save_path_plots + "Histograms.png")

    plt.show()


    # plot each of the beta values in a histogram 
    p = stacked_data["beta"].shape[1]
    fig, ax = plt.subplots(1, p, figsize=(15, 5))
    for i in range(p):
        sns.histplot(stacked_data["beta"][:, i], ax=ax[i])
        ax[i].set_title(f"Histogram of beta_{i}")
    
    if save_path_plots is not None:
        plt.savefig(save_path_plots + "Beta_Histograms.png")
    
    plt.show()

    # also plot all other keys in the dictionary

    for key in stacked_data.keys():
        if key not in ["x", "y", "beta"]:
            data = stacked_data[key]

            if len(data.shape) == 1 or (len(data.shape) == 2 and data.shape[1] == 1):
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                sns.histplot(data, ax=ax)
                ax.set_title(f"Histogram of {key}")
                if save_path_plots is not None:
                    plt.savefig(save_path_plots + f"{key}_Histogram.png")
                plt.show()
            else:
                p = data.shape[1]
                fig, ax = plt.subplots(1, p, figsize=(15, 5))
                for i in range(p):
                    sns.histplot(data[:, i], ax=ax[i])
                    ax[i].set_title(f"Histogram of {key}_{i}")
                
                if save_path_plots is not None:
                    plt.savefig(save_path_plots + f"{key}_Histograms.png")
                plt.show()

        """

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