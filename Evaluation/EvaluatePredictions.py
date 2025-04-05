import torch 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import sklearn
import pandas as pd
import os

class EvaluatePredictions:
    """
    Class to evaluate the preditive performance based on the samples of the models. 
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
            pprogram_name: str,
            posterior_model_samples: list[dict],
            comparison_model_samples: list[list[dict]],
            baselines_regression: list[sklearn.base.BaseEstimator] = [LinearRegression()],
            baselines_classification: list[sklearn.base.BaseEstimator] = [LogisticRegression()],
            save_path: str = None
    ):
        """
        Args:
            pprogram_name: str: the name of the probabilistic program.
            posterior_model_samples: list of dictionaries containing the samples of the posterior model.
            comparison_model_samples: list of lists of dictionaries containing the samples of the comparison models.
            baselines: list of dictionaries containing the baselines.
            save_path: str: the path to save the results.
        """

        self.pprogram_name = pprogram_name
        self.use_intercept = self.ppgrogram_name2useintercept[pprogram_name]
        self.response_function = self.ppgrogram_name2response_function[pprogram_name]
        self.posterior_model_samples = posterior_model_samples
        self.comparison_model_samples = comparison_model_samples
        self.evaluation_results = None
        self.is_regression = False if pprogram_name in ["logreg_ig"] else True
        self.baselines_regression = baselines_regression
        self.baselines_classification = baselines_classification
        self.save_path = save_path

    def rmse(self, y_true, y_pred):
        """
        Compute the root mean squared error.
        Args:
            y_true: torch.Tensor: the true values
            y_pred: torch.Tensor: the predicted values
        Returns:
            rmse: float: the root mean squared error
        """
        assert y_true.shape == y_pred.shape, "The shapes of y_true and y_pred must be equal. But got {} and {}.".format(y_true.shape, y_pred.shape)
        assert len(y_true.shape) == 1, "The shapes of y_true and y_pred must be 1-dimensional. But got {} and {}.".format(y_true.shape, y_pred.shape)

        return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).squeeze().detach().item()
    
    def r2(self, y_true, y_pred):
        """
        Compute the R2 score.
        Args:
            y_true: torch.Tensor: the true values
            y_pred: torch.Tensor: the predicted values
        Returns:
            r2: float: the R2 score
        """
        assert y_true.shape == y_pred.shape, "The shapes of y_true and y_pred must be equal. But got {} and {}.".format(y_true.shape, y_pred.shape)
        assert len(y_true.shape) == 1, "The shapes of y_true and y_pred must be 1-dimensional. But got {} and {}.".format(y_true.shape, y_pred.shape)

        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        return (1 - ss_res / ss_tot).squeeze().detach().item()
    
    def accuracy(self, y_true, y_pred):
        """
        Compute the accuracy score.
        Args:
            y_true: torch.Tensor: the true values
            y_pred: torch.Tensor: the predicted values
        Returns:
            accuracy: float: the accuracy score
        """
        assert y_true.shape == y_pred.shape, "The shapes of y_true and y_pred must be equal. But got {} and {}.".format(y_true.shape, y_pred.shape)
        assert len(y_true.shape) == 1, "The shapes of y_true and y_pred must be 1-dimensional. But got {} and {}.".format(y_true.shape, y_pred.shape)

        class_preds = (y_pred > 0.5).float()
        acc = torch.mean((class_preds == y_true).float()).squeeze()
        return acc.detach().item()
    

    def compute_posterior_mean_predictions(
            self,
            samples_beta: torch.Tensor,
            x_test: torch.Tensor,
    ):
        """
        Use posterior samples for beta and optionally beta0 (intercept) to predict y_test for x_test 
        Args:
            samples_beta: torch.Tensor: the samples for beta. Have shape (n_posterior_samples, p)
            x_test: torch.Tensor: the test data. Have shape (n_test_samples, p)
        Returns:
            predictions: torch.Tensor: the predictions for y_test. Have shape (n_test_samples,)
        """

        if not self.use_intercept:
            assert samples_beta.shape[1] == x_test.shape[1], "The number of features in the samples_beta and x_test must be equal. But got {} and {}.".format(samples_beta.shape[1], x_test.shape[1])
        else:
            assert samples_beta.shape[1] == x_test.shape[1] + 1, "The number of features in the samples_beta and x_test must be equal. But got {} and {}.".format(samples_beta.shape[1], x_test.shape[1])        

        if self.use_intercept:
            x_test = torch.cat([torch.ones(x_test.shape[0], 1), x_test], dim=1)
        
        samples_beta = samples_beta.unsqueeze(1)

        pred_raw = torch.matmul(x_test, samples_beta.transpose(1, 2)).squeeze()
        
        preds = self.response_function(pred_raw)

        preds_mean = torch.mean(preds, dim=0)

        return preds_mean

        
    
    def evaluate_instance_posterior_samples(
            self,
            posterior_samples: dict,
    ):
        """
        Evaluate the predictions for a single model instance.
        Args:
            posterior_samples: dict: the samples of the posterior model
        """

        # extract the samples
        samples_beta = posterior_samples["beta"].squeeze()

        x_test = posterior_samples["x_test"].squeeze()
        y_test = posterior_samples["y_test"].squeeze()

        # compute the predictions
        posterior_mean = self.compute_posterior_mean_predictions(samples_beta, x_test)

        # compute the evaluation metrics

        if self.is_regression:
            rmse = self.rmse(y_test, posterior_mean)
            r2 = self.r2(y_test, posterior_mean)
            metrics = {
                "rmse": rmse,
                "r2": r2,
            }

        else:
            accuracy = self.accuracy(y_test, posterior_mean)
            metrics = {
                "accuracy": accuracy,
            }

        # Also get results on training data

        x_train = posterior_samples["x"].squeeze()
        y_train = posterior_samples["y"].squeeze()
        # compute the predictions
        posterior_mean_train = self.compute_posterior_mean_predictions(samples_beta, x_train)

        # compute the evaluation metrics
        if self.is_regression:
            rmse_train = self.rmse(y_train, posterior_mean_train)
            r2_train = self.r2(y_train, posterior_mean_train)
            metrics["rmse_train"] = rmse_train
            metrics["r2_train"] = r2_train
        else:
            accuracy_train = self.accuracy(y_train, posterior_mean_train)
            metrics["accuracy_train"] = accuracy_train
            metrics["accuracy_train"] = accuracy_train          

        return metrics
    
    def evaluate_instance_baseline(
            self,
            baseline: sklearn.base.BaseEstimator,
            x_test: torch.Tensor,
            y_test: torch.Tensor,
            x_train: torch.Tensor = None,
            y_train: torch.Tensor = None,
    ):
        """
        Evaluate the predictions for a single baseline instance.
        Args:
            baseline: sklearn.base.BaseEstimator: the baseline model
            x_test: torch.Tensor: the test data
            y_test: torch.Tensor: the true values
        """
        x_noise = torch.randn_like(x_test) * 1e-6 # avoid issues with TabPFN
        x_train = x_train + x_noise
        if self.is_regression:
            baseline.fit(x_train.numpy(), y_train.numpy())
            y_pred = torch.tensor(baseline.predict(x_test.numpy()))
            rmse = self.rmse(y_test, y_pred)
            r2 = self.r2(y_test, y_pred)
            metrics = {
                "rmse": rmse,
                "r2": r2,
            }
        else:
            try:
                baseline.fit(x_train.numpy(), y_train.numpy())
                y_pred = torch.tensor(baseline.predict(x_test.numpy()))
                accuracy = self.accuracy(y_test, y_pred)
                metrics = {
                    "accuracy": accuracy,
                }
            except Exception as e:
                print("Error in baseline model fitting: ", e)
                metrics = {
                    "accuracy": 0.5,
                }

        # Also get results on training data
        if x_train is not None and y_train is not None:
            x_noise = torch.randn_like(x_train) * 1e-6
            x_train = x_train + x_noise
            if self.is_regression:
                y_pred_train = torch.tensor(baseline.predict(x_train.numpy()))
                rmse_train = self.rmse(y_train, y_pred_train)
                r2_train = self.r2(y_train, y_pred_train)
                metrics["rmse_train"] = rmse_train
                metrics["r2_train"] = r2_train
            else:
                y_pred_train = torch.tensor(baseline.predict(x_train.numpy()))
                accuracy_train = self.accuracy(y_train, y_pred_train)
                metrics["accuracy_train"] = accuracy_train

        

        return metrics

    def run_evaluation(
            self,
    ):
        """
        Run the evaluation of the predictions.
        """
        evaluation_results = {}

        res_icl = []
        for posterior_samples in self.posterior_model_samples:
            metrics = self.evaluate_instance_posterior_samples(posterior_samples)
            res_icl.append(metrics)

        evaluation_results["ICL"] = res_icl

        for i, comparison_model_samples in enumerate(self.comparison_model_samples):
            res_comparison = []
            for comparison_samples in comparison_model_samples:
                metrics = self.evaluate_instance_posterior_samples(comparison_samples)
                res_comparison.append(metrics)

            evaluation_results["ComparisonModel{}".format(i)] = res_comparison

        # evaluate the baselines
        if self.is_regression:
            for i, baseline in enumerate(self.baselines_regression):
                res_baseline = []
                for posterior_samples in self.posterior_model_samples:
                    x_test = torch.tensor(posterior_samples["x_test"].squeeze())
                    y_test = torch.tensor(posterior_samples["y_test"].squeeze())
                    x_train = torch.tensor(posterior_samples["x"].squeeze())
                    y_train = torch.tensor(posterior_samples["y"].squeeze())
                    metrics = self.evaluate_instance_baseline(baseline, torch.tensor(x_test), torch.tensor(y_test), torch.tensor(x_train), torch.tensor(y_train))

                    res_baseline.append(metrics)
                evaluation_results["Baseline{}".format(i)] = res_baseline

        else:
            for i, baseline in enumerate(self.baselines_classification):
                res_baseline = []
                for posterior_samples in self.posterior_model_samples:
                    x_test = torch.tensor(posterior_samples["x_test"].squeeze())
                    y_test = torch.tensor(posterior_samples["y_test"].squeeze())
                    x_train = torch.tensor(posterior_samples["x"].squeeze())
                    y_train = torch.tensor(posterior_samples["y"].squeeze())
                    metrics = self.evaluate_instance_baseline(baseline, torch.tensor(x_test), torch.tensor(y_test), torch.tensor(x_train), torch.tensor(y_train))
                    res_baseline.append(metrics)
                evaluation_results["Baseline{}".format(i)] = res_baseline



        # convert the results to a DataFrame
        self.evaluation_results_raw = evaluation_results

        # compute std and mean for each metric for each model. Put them into different rows (!) in the same DataFrame. Use one Data Frame for all models
        rows = []

        for key, res_list in self.evaluation_results_raw.items():
            res = pd.DataFrame(res_list)
            res_mean = res.mean()
            res_std = res.std()
            
            # Combine mean and std into one row with suffixed column names
            combined = {}
            for col in res.columns:
                combined[f"{col}_mean"] = res_mean[col]
                combined[f"{col}_std"] = res_std[col]
            
            combined["model"] = key
            rows.append(combined)

        # Create final DataFrame
        evaluation_results = pd.DataFrame(rows).set_index("model")

        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
                            
            evaluation_results.to_csv(f"{self.save_path}/prediction_evaluation.csv")        


        self.evaluation_results = evaluation_results

        return evaluation_results

        





        