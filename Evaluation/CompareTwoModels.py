import torch

from PFNExperiments.Evaluation.BasicMetrics import compare_Wasserstein
from PFNExperiments.Evaluation.ClassifcationBasedComparison import compare_samples_classifier_based_RF, compare_samples_classifier_based_NN_Sklearn_Lueck, compare_samples_classifier_based_NN_RealMLP
from PFNExperiments.Evaluation.MMD import compare_samples_mmd
from PFNExperiments.Evaluation.CompareModelToGT import results_dict_to_data_x_y, results_dict_to_latent_variable_beta, flatten_dict_list

from PFNExperiments.Evaluation.ComparePosteriorSamples import compare_all_metrics
from PFNExperiments.Evaluation.CompareModelToGT import try_otherwise_return_error

class CompareTwoModels():
    """
    A class to compare the samples by two models
    """

    def __init__(self,
                 results_dict_to_latent_variable: callable = results_dict_to_latent_variable_beta,
                 results_dict_to_data: callable = results_dict_to_data_x_y,
                 metrics = [
                    compare_samples_classifier_based_RF, 
                    compare_samples_classifier_based_NN_Sklearn_Lueck,
                    #compare_samples_classifier_based_NN_RealMLP,
                 ]
                    ) -> None:
        """
        Args:
            results_dict_to_latent_variable: callable: a function that takes the results dictionary and returns the latent variable
            results_dict_to_data: callable: a function that takes the results dictionary and returns the data
            metrics: list[callable]: a list of metrics to compare the two models
        """

        self.results_dict_to_latent_variable = results_dict_to_latent_variable
        self.results_dict_to_data = results_dict_to_data
        self.metrics = metrics


    def _compare_model_samples_one_example(
            self,
            model1_samples: dict,
            model2_samples: dict,
             ) -> dict:
        
        """
        Compare the samples by two models for one example
        Args:
            model1_samples: dict: the samples by the first model
            model2_samples: dict: the samples by the second model

        Returns:
            dict: the results of the comparison
        """
        data1 = self.results_dict_to_data(model1_samples)
        data2 = self.results_dict_to_data(model2_samples)

        #assert torch.all(data1 == data2), "The data is not the same"

        if not torch.all(data1 == data2):
            print("Warning: The data is not the same for the two models")

        latent1 = self.results_dict_to_latent_variable(model1_samples)
        latent2 = self.results_dict_to_latent_variable(model2_samples)

        results = compare_all_metrics(latent1, latent2, methods=self.metrics)

        return results


    def compare_model_samples(
            self,
            model1_samples: list[dict],
            model2_samples: list[dict]
             ) -> list[dict]:
        
        """
        Compare the samples by two models
        Args:
            model1_samples: list[dict]: the samples by the first model
            model2_samples: list[dict]: the samples by the second model
        
        Returns:
            list[dict]: the results of the comparison
        """
        assert len(model1_samples) == len(model2_samples), "The number of samples by the two models should match. but got {} and {}".format(len(model1_samples), len(model2_samples))

        results = []

        for model1_sample, model2_sample in zip(model1_samples, model2_samples):
            results.append(self._compare_model_samples_one_example(model1_sample, model2_sample))

        #results = flatten_dict_list(results)
        
        return results
