import pyro 
import torch
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from PFNExperiments.Evaluation.BasicMetrics import compare_Wasserstein, compare_basic_statistics, compare_covariance, compare_marginals
from PFNExperiments.Evaluation.ClassifcationBasedComparison import compare_samples_classifier_based_RF, compare_samples_classifier_based_NN_Sklearn_Lueck, compare_samples_classifier_based_NN_RealMLP
from PFNExperiments.Evaluation.MMD import compare_samples_mmd


def try_otherwise_return_nan(fun: callable) -> callable:
    """
    A wrapper that modifies a fuction to return torch.nan if the function fails
    Args:
        fun: callable: the function
    Returns:
        callable: a function that returns torch.nan if the original function fails
    """
    def wrapper(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except:
            return torch.nan
    return wrapper
    

def compare_all_metrics(P: torch.tensor, Q: torch.tensor, methods: list[callable] = [compare_samples_classifier_based_RF, 
                                                                                    compare_samples_classifier_based_NN_Sklearn_Lueck,
                                                                                    compare_samples_classifier_based_NN_RealMLP,
                                                                                     #compare_Wasserstein, 
                                                                                     #compare_basic_statistics, 
                                                                                     #compare_covariance, 
                                                                                     #compare_marginals
                                                                                     ], 
                        fun_wrapper = try_otherwise_return_nan) -> dict:
    """
    A method that compares two sets of samples using a list of comparison methods.
    Args:
        P: torch.tensor: the first set of samples
        Q: torch.tensor: the second set of samples
        methods: list[callable]: a list of comparison methods to use
        fun_wrapper: callable: a wrapper that ensures a function returns a value or torch.nan if an exception is raised
    Returns:
        dict: a dictionary containing the results of the comparison methods
    """
    results = {}
    for method in methods:
        method = fun_wrapper(method)
        results.update(method(P, Q))
    return results

def marginal_plots_hist_parallel(P: torch.tensor, Q: torch.tensor):
    """
    A method that plots the marginals of two sets of samples.
    Args:
        P: torch.tensor: the first set of samples
        Q: torch.tensor: the second set of samples
    """
    

    P_df = pd.DataFrame(P.numpy(), columns=[f"Dim_{i}" for i in range(P.shape[1])])
    Q_df = pd.DataFrame(Q.numpy(), columns=[f"Dim_{i}" for i in range(Q.shape[1])])

    for i in range(P.shape[1]):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        sns.histplot(P_df[f"Dim_{i}"], ax=ax[0])
        ax[0].set_title(f"Marginal dim {i} P")
        sns.histplot(Q_df[f"Dim_{i}"], ax=ax[1])
        ax[1].set_title(f"Marginal dim {i} Q")

        # also plot the means as vertical lines
        ax[0].axvline(P_df[f"Dim_{i}"].mean(), color='r', linestyle='--')
        ax[1].axvline(Q_df[f"Dim_{i}"].mean(), color='r', linestyle='--')
        
        plt.show()

def marginal_plots_kde_parallel(P: torch.tensor, Q: torch.tensor):
    """
    A method that plots the marginals of two sets of samples using kernel density estimation.
    Args:
        P: torch.tensor: the first set of samples
        Q: torch.tensor: the second set of samples
    """
    P_df = pd.DataFrame(P.numpy(), columns=[f"Dim_{i}" for i in range(P.shape[1])])
    Q_df = pd.DataFrame(Q.numpy(), columns=[f"Dim_{i}" for i in range(Q.shape[1])])

    for i in range(P.shape[1]):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        sns.kdeplot(P_df[f"Dim_{i}"], ax=ax[0])
        ax[0].set_title(f"Marginal dim {i} P")
        sns.kdeplot(Q_df[f"Dim_{i}"], ax=ax[1])
        ax[1].set_title(f"Marginal dim {i} Q")

        # also plot the means as vertical lines
        ax[0].axvline(P_df[f"Dim_{i}"].mean(), color='r', linestyle='--')
        ax[1].axvline(Q_df[f"Dim_{i}"].mean(), color='r', linestyle='--')

        plt.show()

def marginal_plots_kde_together(P: torch.tensor, Q: torch.tensor):
    """
    A method that plots the marginals of two sets of samples using kernel density estimation.
    Plot the densities for each dimension of the samples in the same plot.
    Args:
        P: torch.tensor: the first set of samples
        Q: torch.tensor: the second set of samples
    """
    P_df = pd.DataFrame(P.numpy(), columns=[f"Dim_{i}" for i in range(P.shape[1])])
    Q_df = pd.DataFrame(Q.numpy(), columns=[f"Dim_{i}" for i in range(Q.shape[1])])

    n_dims = P.shape[1]
    fig, ax = plt.subplots(1, n_dims, figsize=(5*n_dims, 5))

    for i in range(n_dims):
        sns.kdeplot(P_df[f"Dim_{i}"], ax=ax[i], label="P", color='b')
        sns.kdeplot(Q_df[f"Dim_{i}"], ax=ax[i], label="Q", color='r')
        ax[i].set_title(f"Marginal dim {i}")
        ax[i].legend()

        # also plot the means as vertical lines
        ax[i].axvline(P_df[f"Dim_{i}"].mean(), color='b', linestyle='--', label="P mean")
        ax[i].axvline(Q_df[f"Dim_{i}"].mean(), color='r', linestyle='--', label="Q mean")

    plt.show()

def sample_plot_2d(P: torch.tensor, Q: torch.tensor, method: str = "PCA"):
    """
    A method that plots the samples P and Q together in a scatterplot after projecting them into the same 2d space
    Args:
        P: torch.tensor: the first set of samples
        Q: torch.tensor: the second set of samples
        method: str: dimensionality reduction method to use ('PCA' or 't-SNE')
    """
    # Convert tensors to numpy arrays for compatibility with sklearn
    P_np = P.numpy()
    Q_np = Q.numpy()

    # Combine P and Q for dimensionality reduction
    combined_data = np.vstack((P_np, Q_np))

    # Apply dimensionality reduction
    if method.lower() == "pca":
        reducer = PCA(n_components=2)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=2, learning_rate='auto', init='random')
    else:
        raise ValueError("Unsupported dimensionality reduction method. Use 'PCA' or 't-SNE'.")

    transformed_data = reducer.fit_transform(combined_data)

    # Split the transformed data back into P and Q
    P_transformed = transformed_data[:len(P), :]
    Q_transformed = transformed_data[len(P):, :]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(P_transformed[:, 0], P_transformed[:, 1], label='P samples', alpha=0.5)
    plt.scatter(Q_transformed[:, 0], Q_transformed[:, 1], label='Q samples', alpha=0.5)
    plt.title(f"{method} projection of P and Q samples")
    plt.legend()
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.show()

def plot(P: torch.tensor, Q:torch.tensor, plot_funs: list[callable] = [marginal_plots_hist_parallel, marginal_plots_kde_together, sample_plot_2d]):
    """
    A method that plots the comparison of two sets of samples using a list of plotting methods.
    Args:
        P: torch.tensor: the first set of samples
        Q: torch.tensor: the second set of samples
        plot_funs: list[callable]: a list of plotting methods to use
    """
    for fun in plot_funs:
        fun(P, Q)