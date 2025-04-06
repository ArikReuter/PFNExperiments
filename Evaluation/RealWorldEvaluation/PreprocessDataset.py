import torch 
from sklearn.preprocessing import PowerTransformer
from scipy.stats import boxcox

def scale_features_01_total(X):
    """
    Scale the features between 0 and 1 using the min and max of the entire dataset
    """
    return 0.95*(X - X.min()) / (X.max() - X.min()) + 0.025  # scale between 0.025 and 0.975

def scale_features_01(X, eps = 1e-3):
    """
    Scale the features between 0 and 1 using the min and max of each feature
    Args: 
        X: torch.Tensor: the features of shape (N,P)
        eps: float: a small number to avoid division by zero
    """
    mins = torch.min(X, dim = 1, keepdim=True) # shpe (N,1)
    maxs = torch.max(X, dim = 1, keepdim=True) # shape (N,1)

    X_scaled = (X - mins.values) / (maxs.values - mins.values + eps)

    return X_scaled


def scale_features_01_power_transform(X):
    """
    Scale the features between 0 and 1 using the min and max of each feature. Apply the power-transform per feature before.
    Args: 
        X: torch.Tensor: the features of shape (N,P)
    """
    pt = PowerTransformer()
    X = pt.fit_transform(X)
    X = torch.tensor(X, dtype = torch.float)

    mins = torch.min(X, dim = 1, keepdim=True) # shpe (N,1)
    maxs = torch.max(X, dim = 1, keepdim=True) # shape (N,1)

    X_scaled = (X - mins.values) / (maxs.values - mins.values)

    return X_scaled


def make_target_scaler(mu: float = 0.0, var: float = 1.0, power_transform:bool = True) -> callable:
    """
    Make a target scaler that scales the target to have mean mu and variance var averages the target
    Args:
        mu: float: the mean of the target
        var: float: the variance of the target
        power_transform: bool: whether to apply a power transform to the target
    Returns:
        callable: a target scaler
    """

    def target_scaler(y: torch.Tensor) -> torch.Tensor:
        """
        Scale the target to have mean mu and variance var
        Args:
            y: torch.Tensor: the target
        Returns:
            torch.Tensor: the scaled target
        """
        assert y.dim() == 1, "The target should be 1D"
        if power_transform:
            pt = PowerTransformer()
            y = pt.fit_transform(y.reshape(-1,1)).squeeze()
            y = torch.tensor(y, dtype = torch.float).unsqueeze(0)

        
        return mu + (var**0.5) * (y - y.mean()) / y.std()

    return target_scaler



class Preprocessor():

    def __init__(
        self,
        N_datapoints: int,
        P_features: int,
        scale_features: callable = scale_features_01_power_transform,
        target_mean: float = 0.0,
        target_var: float = 1.0,
        power_transform_y: bool = True,
        seed: int = 0,
        additive_noise_std: float = 0.0
    ):
        """
        Args:
            N_datapoints (int): The number of datapoints to use.
            P_features (int): The number of features to use.
            scale_features (callable): A callable that scales the features.
            target_mean (float): The mean of the target.
            target_var (float): The variance of the target.
            power_transform_y (bool): Whether to apply a power transform to the target.
            seed (int): The seed to use.
            additive_noise_var (float): The variance of the additive noise.
        """
        self.N_datapoints = N_datapoints
        self.P_features = P_features
        self.scale_features = scale_features
        self.target_scaler = make_target_scaler(target_mean, target_var, power_transform = power_transform_y)

        self.seed = seed
        self.additive_noise_std = additive_noise_std

        # set the torch seed
        torch.manual_seed(seed)


    def _subsample_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_test: bool = True
    ) -> dict[str, torch.Tensor]:
        """
        Subsample the data
        Args:
            x: torch.Tensor: the features
            y: torch.Tensor: the target
        Returns:
            x, y
        """

        assert len(x) >= self.N_datapoints, "The number of datapoints is larger than the number of datapoints in the dataset"
            
        indices = torch.randperm(x.shape[0])[:self.N_datapoints]

        x_train = x[indices]
        y_train = y[indices]

        test_indices = torch.randperm(x.shape[0])[self.N_datapoints:2*self.N_datapoints]
        x_test = x[test_indices]
        y_test = y[test_indices]

        if return_test:
            return x_train, y_train, x_test, y_test
        
        else:
            return x_train, y_train
    
    def _identify_numerical_features(
            self,
            x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Identify the numerical features in the dataset
        Args:
            x: torch.Tensor: the features
        Returns:
            torch.Tensor: the features sorted by the number of different values
            n_diff_values: torch.Tensor: the number of different values for each feature
        """
        
        n_diff_values = torch.tensor([len(torch.unique(x[:, i])) for i in range(x.shape[1])])
        _, indices = torch.sort(n_diff_values, descending = True)

        x = x[:, indices]
        n_diff_values = n_diff_values[indices]

        return x, n_diff_values


    def preprocess(
            self,
            dataset: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Scale the features and the target of the dataset
        Args:
            dataset: dict[str, torch.Tensor]: the dataset
        Returns:
            dict[str, torch.Tensor]: the preprocessed dataset
        """

        x = dataset["x"]
        y = dataset["y"]

        try:
            x = torch.tensor(x, dtype = torch.float)
            y = torch.tensor(y, dtype = torch.float)
        except Exception as e:
            raise ValueError("The features and target should be convertible to a torch tensor") from e

        assert len(x) == len(y), "The number of features and targets is different. got {} and {}".format(len(x), len(y))

        x, n_diff_values = self._identify_numerical_features(x)

        assert x.shape[1] >= self.P_features, "The number of features is smaller than the number of features to use"
        x = x[:, :self.P_features] # select the first P_features
        n_diff_values = n_diff_values[:self.P_features]
        

        x, y, x_test, y_test = self._subsample_data(x, y)   

        #x = self.scale_features(x)
        #y = self.target_scaler(y)

        ##

        if self.additive_noise_std > 0:
            y = y + torch.randn(y.shape) * self.additive_noise_std
            x = x + torch.randn(x.shape) * self.additive_noise_std

        x = self.scale_features(x)
        y = self.target_scaler(y) 

        x_test = self.scale_features(x_test)
        y_test = self.target_scaler(y_test)

    
        new_dataset = {
            "x": x,
            "y": y,
            "x_test": x_test,
            "y_test": y_test,
            "n_diff_values": n_diff_values
        }

        return new_dataset
    
class PreprocessorClassification(Preprocessor):

    def __init__(
        self,
        N_datapoints: int,
        P_features: int,
        scale_features: callable = scale_features_01_power_transform,
        seed: int = 0,
        additive_noise_std: float = 0.0
    ):
        """
        Args:
            N_datapoints (int): The number of datapoints to use.
            P_features (int): The number of features to use.
            scale_features (callable): A callable that scales the features.
            seed (int): The seed to use.
            additive_noise_var (float): The variance of the additive noise.
        """
        self.N_datapoints = N_datapoints
        self.P_features = P_features
        self.scale_features = scale_features

        self.seed = seed
        self.additive_noise_std = additive_noise_std

        # set the torch seed
        torch.manual_seed(seed)


    def preprocess(
            self,
            dataset: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Scale the features and the target of the dataset
        Args:
            dataset: dict[str, torch.Tensor]: the dataset
        Returns:
            dict[str, torch.Tensor]: the preprocessed dataset
        """

        x = dataset["x"]
        y = dataset["y"]

        try:
            x = torch.tensor(x, dtype = torch.float)
            y = torch.tensor(y, dtype = torch.float)
        except Exception as e:
            raise ValueError("The features and target should be convertible to a torch tensor") from e

        assert len(x) == len(y), "The number of features and targets is different. got {} and {}".format(len(x), len(y))


        x, n_diff_values = self._identify_numerical_features(x)

        assert x.shape[1] >= self.P_features, "The number of features is smaller than the number of features to use"
        x = x[:, :self.P_features] # select the first P_features
        n_diff_values = n_diff_values[:self.P_features]
        if self.additive_noise_std > 0:
            x = x + torch.randn(x.shape) * self.additive_noise_std
        

        x, y, x_test, y_test = self._subsample_data(x, y)   

        #x = self.scale_features(x)
        #y = self.target_scaler(y)

        y_med = y.median()
        y = (y > y_med).float()

        y_test_med = y_test.median()
        y_test = (y_test > y_test_med).float()

    
        x = self.scale_features(x)
        x_test = self.scale_features(x_test)

        new_dataset = {
            "x": x,
            "y": y,
            "x_test": x_test,
            "y_test": y_test,
            "n_diff_values": n_diff_values
        }

        return new_dataset
    

class PreprocessorGammaResponse():

    def __init__(
        self,
        N_datapoints: int,
        P_features: int,
        scale_features: callable = scale_features_01_power_transform,
        target_mean: float = 0.0,
        target_var: float = 1.0,
        target_lambda: float = 1.0,
        power_transform_y: bool = True,
        seed: int = 0,
        additive_noise_std: float = 0.0
    ):
        """
        Args:
            N_datapoints (int): The number of datapoints to use.
            P_features (int): The number of features to use.
            scale_features (callable): A callable that scales the features.
            target_mean (float): The mean of the target.
            target_var (float): The variance of the target.
            target_lambda (float): The lambda parameter of the boxcox transformation,
            power_transform_y (bool): Whether to apply a power transform to the target.
            seed (int): The seed to use.
            additive_noise_var (float): The variance of the additive noise.
        """
        self.N_datapoints = N_datapoints
        self.P_features = P_features
        self.scale_features = scale_features
        self.target_scaler = make_target_scaler(target_mean, target_var, power_transform = power_transform_y)

        self.seed = seed
        self.additive_noise_std = additive_noise_std
        self.target_lambda = target_lambda

        # set the torch seed
        torch.manual_seed(seed)


    def _subsample_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_test: bool = True
    ) -> dict[str, torch.Tensor]:
        """
        Subsample the data
        Args:
            x: torch.Tensor: the features
            y: torch.Tensor: the target
        Returns:
            x, y
        """

        assert len(x) >= self.N_datapoints, "The number of datapoints is larger than the number of datapoints in the dataset"
            
        indices = torch.randperm(x.shape[0])[:self.N_datapoints]

        x_train = x[indices]
        y_train = y[indices]

        test_indices = torch.randperm(x.shape[0])[self.N_datapoints:2*self.N_datapoints]
        x_test = x[test_indices]
        y_test = y[test_indices]

        if return_test:
            return x_train, y_train, x_test, y_test
        
        else:
            return x_train, y_train
        
    def _identify_numerical_features(
            self,
            x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Identify the numerical features in the dataset
        Args:
            x: torch.Tensor: the features
        Returns:
            torch.Tensor: the features sorted by the number of different values
            n_diff_values: torch.Tensor: the number of different values for each feature
        """
        
        n_diff_values = torch.tensor([len(torch.unique(x[:, i])) for i in range(x.shape[1])])
        _, indices = torch.sort(n_diff_values, descending = True)

        x = x[:, indices]
        n_diff_values = n_diff_values[indices]

        return x, n_diff_values


    def preprocess(
            self,
            dataset: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Scale the features and the target of the dataset
        Args:
            dataset: dict[str, torch.Tensor]: the dataset
        Returns:
            dict[str, torch.Tensor]: the preprocessed dataset
        """

        x = dataset["x"]
        y = dataset["y"]

        try:
            x = torch.tensor(x, dtype = torch.float)
            y = torch.tensor(y, dtype = torch.float)
        except Exception as e:
            raise ValueError("The features and target should be convertible to a torch tensor") from e

        assert len(x) == len(y), "The number of features and targets is different. got {} and {}".format(len(x), len(y))

        x, n_diff_values = self._identify_numerical_features(x)

        assert x.shape[1] >= self.P_features, "The number of features is smaller than the number of features to use"
        x = x[:, :self.P_features] # select the first P_features
        n_diff_values = n_diff_values[:self.P_features]

        x, y, x_test, y_test = self._subsample_data(x, y)   

        #x = self.scale_features(x)
        #y = self.target_scaler(y)

        if self.additive_noise_std > 0:
            y = y + torch.randn(y.shape) * self.additive_noise_std
            x = x + torch.randn(x.shape) * self.additive_noise_std
            x_test = x_test + torch.randn(x_test.shape) * self.additive_noise_std
            y_test = y_test + torch.randn(y_test.shape) * self.additive_noise_std


        x = self.scale_features(x)
        x_test = self.scale_features(x_test)
        
        y = self.target_scaler(y)
        y_test = self.target_scaler(y_test)

        #y = torch.exp(y) # the target is the log of the response
        #y = boxcox(y, self.target_lambda)

        y = torch.tensor(y, dtype = torch.float)

        # ensure y is positive
        y = y - y.min() + 1e-5

        # bring vriance 
        y = y / (y.std() + 1e-5)	

        y_test = torch.exp(y_test)
        #y_test = boxcox(y_test, self.target_lambda)
        y_test = torch.tensor(y_test, dtype = torch.float)
        y_test = y_test - y_test.min() + 1e-5
        y_test = y_test / (y_test.std() + 1e-5)

        new_dataset = {
            "x": x,
            "y": y,
            "x_test": x_test,
            "y_test": y_test,
            "n_diff_values": n_diff_values
        }

        return new_dataset
    

