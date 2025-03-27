import torch

from PFNExperiments.Evaluation.Evaluate import Evaluate
from IPython.display import display


from PFNExperiments.LinearRegression.ComparisonModels.PosteriorComparisonModel import PosteriorComparisonModel 

from PFNExperiments.Evaluation.CompareTwoModels import CompareTwoModels
from PFNExperiments.Evaluation.Plot import Plot

import pandas  as pd
import numpy as np
import pickle
import os
from scipy.stats import mannwhitneyu, wilcoxon
from tqdm import tqdm

def just_return_results(result:dict) -> dict:
    """
    Just return the results
    """
    return result


def results_dict_to_data_x_y_tuple(result:dict) -> (torch.tensor, torch.tensor):
    """
    Take the dictionary with results and return the data x and y
    """
    x = result["x"]
    y = result["y"]

    x = x.squeeze(0)
    
    y = torch.tensor(y, dtype = torch.float32)
    if len(y.shape) == 2:
      y = y.squeeze(0)
    return x, y


def results_dict_to_latent_variable_beta(result:dict) ->  torch.tensor:
    """
    Take the dictionary with results and return the latent variable
    """
    return result["beta"]

def results_dict_to_latent_variable_beta0_and_beta(result:dict) ->  torch.tensor:
    """
    Take the dictionary with results and return the beta coefficient combined with the intercept
    """
    beta0 = result["beta0"]
    beta = result["beta"]

    beta0 = beta0.unsqueeze(-1)

    if len(beta.shape) == 3 and len(beta0.shape) == 4:
        beta0 = beta0.squeeze(-1).squeeze(-1)
        beta = beta.squeeze(-1)
    
    if len(beta0.shape) == 3 and len(beta.shape) == 2:
        beta0 = beta0.squeeze(-1)
  
    beta_combined = torch.cat([beta0, beta], dim=-1)

    
    result["beta"] = beta_combined
    return result

class EvaluateRealWorld(Evaluate):
    """
    A class to evaluate the performance of different models on real world data
    """

    def __init__(
            self,
            posterior_model: PosteriorComparisonModel,
            evaluation_datasets: list[dict],
            comparison_models: list[PosteriorComparisonModel] = [],
            model_names: list[str] = None,
            n_evaluation_cases: int = 1,
            results_dict_to_latent_variable_posterior_model: callable = just_return_results,
            results_dict_to_latent_variable_comparison_models: callable = results_dict_to_latent_variable_beta0_and_beta,
            results_dict_to_data_for_model: callable = results_dict_to_data_x_y_tuple,
            result_dict_to_data_for_comparison_models: callable = None,
            compare_two_models: CompareTwoModels = CompareTwoModels(),
            verbose = True,
            compare_comparison_models_among_each_other = True,
            save_path: str = None,
            overwrite_results: bool = False
            
    ):
        """
        Args:
            posterior_model: PosteriorComparisonModel: the posterior model to evaluate
            evaluation_loader: torch.utils.data.DataLoader: the dataloader used for evaluation
            comparison_models: list[PosteriorComparisonModel]: the comparison models
            n_evaluation_cases: int: the number of evaluation cases to use, corresponds to the number of datasets to evaluate on
            n_posterior_samples: int: the number of posterior samples to draw from the posterior model and the comparison models
            results_dict_to_latent_variable: callable: a function that takes the results dictionary and returns the latent variable
            results_dict_to_data_for_model: callable: a function that takes the results dictionary and returns the data
            results_dict_to_data_for_model: callable: a function that takes the results dictionary and returns the data
            compare_to_gt: CompareModelToGT: the class to compare the model to the ground truth
            compare_two_models: CompareTwoModels: the class to compare two models
            verbose: bool: whether to print the results
            compare_comparison_models_among_each_other: bool: whether to compare the comparison models among each other
            save_path: str: the path to save the results
            overwrite_results: bool: whether to overwrite the results if they already exist
        """

        assert len(evaluation_datasets) >= n_evaluation_cases, f"The number of evaluation cases is larger than the number of datasets. But got {len(evaluation_datasets)} and {n_evaluation_cases}"


        self.posterior_model = posterior_model
        self.evaluation_list = evaluation_datasets[:n_evaluation_cases]
        self.comparison_models = comparison_models
        self.n_evaluation_cases = n_evaluation_cases
        #self.n_posterior_samples = n_posterior_samples
        self.results_dict_to_latent_variable_posterior_model = results_dict_to_latent_variable_posterior_model
        self.results_dict_to_latent_variable_comparison_models = results_dict_to_latent_variable_comparison_models
        self.results_dict_to_data_for_model = results_dict_to_data_for_model
        self.compare_two_models = compare_two_models
        self.verbose = verbose
        self.compare_comparison_models_among_each_other = compare_comparison_models_among_each_other
        self.save_path = save_path
        self.overwrite_results = overwrite_results

        if result_dict_to_data_for_comparison_models is None:
            self.result_dict_to_data_for_comparison_models = self.results_dict_to_data_for_model
        else:
            self.result_dict_to_data_for_comparison_models = result_dict_to_data_for_comparison_models

        if model_names is None:
            comparison_models_names = [str(model) for model in comparison_models]
            model_names = [str(posterior_model)] + comparison_models_names

        assert len(model_names) == len(comparison_models) +1, f"The number of model names must be equal to the number of comparison models + 1. But got {len(model_names)} and {len(comparison_models) +1}"
        model_names_dict = {model: name for model, name in zip([posterior_model] + comparison_models, model_names)}
        self.model_names_dict = model_names_dict

        # check if the save path exists, if not create it. If it exists and is not empty, check if the overwrite flag is set

        if self.save_path is not None:
            # append "real_world" to save path
            

            if not os.path.exists(self.save_path):
                print(f"The save path {self.save_path} does not exist, creating it")
                os.makedirs(self.save_path)
            else:
                if len(os.listdir(self.save_path)) > 0:
                    if not self.overwrite_results:
                        raise ValueError("The save path is not empty and the overwrite flag is not set")
                    else:
                        print("The save path is not empty and the overwrite flag is set, the results will be overwritten")

        if self.save_path is not None:
            plot_save_path = f"{self.save_path}/plots" if self.save_path is not None else None
            if not os.path.exists(plot_save_path):
                os.makedirs(plot_save_path)

        else:
            plot_save_path = None
        self.plot = Plot(save_path=plot_save_path)

    def sample_posterior_model(self, model: PosteriorComparisonModel, is_comparison_model:bool = False) -> list[dict]:
        """
        Sample the posterior model
        Args:
            model: PosteriorComparisonModel: the model to sample from
            is_comparison_model: bool: whether the model is a comparison model
        Returns:
            list[dict]: a list of dictionaries containing the posterior samples
        """

        posterior_samples = []
        for case in tqdm(self.evaluation_list, desc="Sampling posterior"):

            if is_comparison_model:
                data = self.result_dict_to_data_for_comparison_models(case)

            else:
                data = self.results_dict_to_data_for_model(case) 
            
            samples = model.sample_posterior(*data)
            for key in case.keys():
                if key not in samples.keys():
                    samples[key] = case[key]

            if is_comparison_model:
                samples = self.results_dict_to_latent_variable_comparison_models(samples)
            else:
                samples = self.results_dict_to_latent_variable_posterior_model(samples)
            
            posterior_samples.append(samples)

        return posterior_samples

    def _run_eval_raw_results(self) -> tuple:
        """
        Run the evaluation
        Returns:
            dict: a dictionary containing the results in form of dataframes
            dict: a dictionary containing the results in form of raw results
        """

      
        posterior_model_samples = self.sample_posterior_model(self.posterior_model, is_comparison_model=False)
        comparison_model_samples = [self.sample_posterior_model(model, is_comparison_model=True) for model in self.comparison_models]

        #print(posterior_model_samples)
      
        self.posterior_model_samples = posterior_model_samples
        self.comparison_model_samples = comparison_model_samples

    
        posterior_model_vs_comparison_models = {
            (self.model_names_dict[self.posterior_model], self.model_names_dict[model]): self.compare_two_models.compare_model_samples(posterior_model_samples, model_samples) for model, model_samples in zip(self.comparison_models, comparison_model_samples)
        }

        comparison_models_vs_comparison_models = {}
        for i in range(len(self.comparison_models)):
            for j in range(i+1, len(self.comparison_models)):
                comparison_models_vs_comparison_models[(self.model_names_dict[self.comparison_models[i]], self.model_names_dict[self.comparison_models[j]])] = self.compare_two_models.compare_model_samples(comparison_model_samples[i], comparison_model_samples[j])
        

        res_raw = {
            "posterior_model_vs_comparison_models": posterior_model_vs_comparison_models
        }

        if self.compare_comparison_models_among_each_other: 
            res_raw["comparison_models_vs_comparison_models"] = comparison_models_vs_comparison_models


        model_comparison_among_each_other = {**posterior_model_vs_comparison_models, **comparison_models_vs_comparison_models}
        model_comparison_among_each_other_df = {key: pd.DataFrame(value) for key, value in model_comparison_among_each_other.items()}
        
        res_df = {
            "model_comparison_among_each_other": model_comparison_among_each_other_df
        }

        if self.save_path is not None:
            with open(f"{self.save_path}/res_raw.pkl", "wb") as f:
                pickle.dump(res_raw, f)
            
            with open(f"{self.save_path}/res_df.pkl", "wb") as f:
                pickle.dump(res_df, f)

            
        self.res_df = res_df
        self.res_raw = res_raw

        return res_df, res_raw
    

    def summarize_results(self, res_df: dict) -> dict:
        """
        Summarize the results of the benchmark
        Args:
            res_df: dict: a dictionary containing the results in form of dataframes, as returned by _run_eval_raw_results

        Returns:
            dict: a dictionary containing the summarized results
        """

       
        model_comparison_among_each_other_df = res_df["model_comparison_among_each_other"]

    
        model_comparison_among_each_other_summarized_df_list = []

        for key, df in model_comparison_among_each_other_df.items():
            model1, model2 = key

            values_mean = df.mean(axis=0).to_dict()
            column_names = df.columns
            values_mean2 = {}
            for column_name, (key, value) in zip(column_names, values_mean.items()):
                values_mean2[f"Mean_{column_name}"] = value

            values_std = df.std(axis=0).to_dict()
            values_std2 = {}
            for column_name, (key, value) in zip(column_names, values_std.items()):
                values_std2[f"Std_{column_name}"] = value


            row_df = {
                "Model 1": model1,
                "Model 2": model2,
                **values_mean2,
                **values_std2
            }

            model_comparison_among_each_other_summarized_df_list.append(row_df)

        res = {
            "model_comparison_among_each_other": pd.DataFrame(model_comparison_among_each_other_summarized_df_list)
        }

        if self.save_path is not None:
            res["model_comparison_among_each_other"].to_csv(f"{self.save_path}/model_comparison_among_each_other_summarized.csv")

        return res
    

    def run_tests(self, res_df: dict, test_paired = mannwhitneyu, test_unpaired = wilcoxon) -> dict:
        """
        Run an unpaired test to test if the results against the ground truth are statistically significantly different
        Run a paired test to test if the results between the models are statistically significantly different
        Args: 
            res_df: dict: a dictionary containing the results in form of dataframes, as returned by _run_eval_raw_results
            test_paired: callable: the paired test to use
            test_unpaired: callable: the unpaired test to use
        Returns:
            dict: a dictionary containing the results of the tests
        """

        model_comparison_among_each_other_df = res_df["model_comparison_among_each_other"]


        res_dict_list = []

        model_comparison_among_each_other_df_keys = list(model_comparison_among_each_other_df.keys())

        for i in range(len(model_comparison_among_each_other_df_keys)):
            for j in range(i+1, len(model_comparison_among_each_other_df_keys)):
                k1 = model_comparison_among_each_other_df_keys[i]
                k2 = model_comparison_among_each_other_df_keys[j]

                if k1[0] == k1[1] or k1[0] == k2[0] or k1[0] == k2[1] or k1[1] == k2[0] or k1[1] == k2[1] or k2[0] == k2[1]:  # only consider cases where one partner if the same 

                    df1 = model_comparison_among_each_other_df[k1]
                    df2 = model_comparison_among_each_other_df[k2]

                    test_paired_res = test_paired(df1, df2)[1]

                    assert np.all(df1.columns == df2.columns), "The columns of the dataframes must be equal"

                    metric_names = df1.columns

                    metric_names = [f"{metric_name}_p-value" for metric_name in metric_names]

                    test_res = {
                        metric_name: test_paired_res[i] for i, metric_name in enumerate(metric_names)
                    }

                    
                    r = {
                        "Pair 1: Model A": k1[0],
                        "Pair 1: Model B": k1[1],
                        "Pair 2: Model A": k2[0],
                        "Pair 2: Model B": k2[1],
                        **test_res
                    }

                    res_dict_list.append(r)

        pvals_model_comparison_among_each_other = pd.DataFrame(res_dict_list)

        res = {
            "model_comparison_among_each_other": pvals_model_comparison_among_each_other
        }

        if self.save_path is not None:
            pvals_model_comparison_among_each_other.to_csv(f"{self.save_path}/pvals_model_comparison_among_each_other.csv")

        return res
    
    def plot_results(self, max_number_plots:int = 5, fontsize: int = 12, suptitle: bool = True, bbox_to_anchor=(-0.17, 0.5), width = 5, height = 7, n_rows = 1, plot_gt = False) -> None:
        """
        Plot the results
        Args:
            max_number_plots: int: the maximum number of plots to show
            fontsize: int: the fontsize of the plots
            suptitle: bool: whether to show the suptitle
            bbox_to_anchor: tuple: the position of the legend
        
        """
        self.plot.fontsize = fontsize

        assert self.posterior_model_samples is not None, "You need to run the evaluation first"
        assert self.comparison_model_samples is not None, "You need to run the evaluation first"

        assert len(self.comparison_models) == len(self.comparison_model_samples), "The number of comparison models and comparison model samples must be equal"
        comparison_model_samples_dict = {
            self.model_names_dict[model]: samples for model, samples in zip(self.comparison_models, self.comparison_model_samples)
        }

        model_samples_dict = {
            self.model_names_dict[self.posterior_model]: self.posterior_model_samples,
            **comparison_model_samples_dict
        }

        self.plot.density_plot_marginals(
            model_samples= model_samples_dict,
            gt_samples=None,
            plot_gt=plot_gt,
            max_number_plots = max_number_plots,
            suptitle=suptitle,
            bbox_to_anchor=bbox_to_anchor,
            width=width,
            height=height,
            n_rows=n_rows
        )

    def run_evaluation(self, print_results: bool = True) -> dict:
        """
        Run the entire evaluation
        Args:
            print_results: bool: whether to print the results
        Returns:
            dict: a dictionary containing the summarized results and results of the tests:
                "summarized_results": dict: a dictionary containing the summarized results
                "test_results": dict: a dictionary containing the results of the tests
                "res_df": dict: a dictionary containing the results in form of dataframes
        """

        res_df, res_raw = self._run_eval_raw_results()
        summarized_results = self.summarize_results(res_df)
        test_results = self.run_tests(res_df)

        if print_results:
            print("Comparison to ground truth:")
            print()
            print("Summarized results:")
          
            print()
            print("Comparison among models:")
            print()
            print("Summarized results:")
            try:
                display(summarized_results["model_comparison_among_each_other"])
            except:
                print(summarized_results["model_comparison_among_each_other"])

            print("P-values:")
            try:
                display(test_results["model_comparison_among_each_other"])
            except:
                print(test_results["model_comparison_among_each_other"])

        return {
            "summarized_results": summarized_results,
            "test_results": test_results,
            "res_df": res_df,
        }
    
    def only_sample_posterior(self):
        """
        Only sample the posterior
        """
        self.posterior_model_samples = self.sample_posterior_model(self.posterior_model, is_comparison_model=False)
        self.comparison_model_samples = [self.sample_posterior_model(model, is_comparison_model=True) for model in self.comparison_models]

        if self.save_path is not None:
            with open(f"{self.save_path}/posterior_model_samples.pkl", "wb") as f:
                pickle.dump(self.posterior_model_samples, f)
            
            with open(f"{self.save_path}/comparison_model_samples.pkl", "wb") as f:
                pickle.dump(self.comparison_model_samples, f)