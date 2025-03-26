import torch 
import pyro
from tqdm import tqdm
from typing import List, Dict, Tuple
import inspect

try:
    from LM_abstract import pprogram_linear_model_return_dict, return_only_y, pprogram_X
    from GenerateX import simulate_X_uniform
    from Curriculum import Curriculum
    from GenerateData import GenerateData
    from EpochLoader import EpochLoader
    from GenerateData import check_data, check_and_plot_data
except:
    from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import pprogram_linear_model_return_dict, return_only_y, pprogram_X
    from PFNExperiments.LinearRegression.GenerativeModels.GenerateX import simulate_X_uniform
    from PFNExperiments.LinearRegression.GenerativeModels.Curriculum import Curriculum
    from PFNExperiments.LinearRegression.GenerativeModels.GenerateData import GenerateData
    from PFNExperiments.LinearRegression.GenerativeModels.EpochLoader import EpochLoader
    from PFNExperiments.LinearRegression.GenerativeModels.GenerateData import check_data, check_and_plot_data



def get_function_parameter_names(func):
    # Get the signature of the function
    sig = inspect.signature(func)
    # Extract the parameter names from the signature
    param_names = list(sig.parameters.keys())
    return param_names

    

class GenerateDataCurriculum(GenerateData):
    """
    Class to generate data for (linear) regression with a curriculum
    Attributes:
        pprogram_maker: callable: a function that returns a probabilistic program
        curriculum: Curriculum: the curriculum that determines the way the samples are generated during training
    """

    def __init__(self,
                 pprogram_maker: callable,
                 curriculum: Curriculum,
                 pprogram_covariates: pprogram_X = simulate_X_uniform,
                 pprogram_covariates_train: pprogram_X = None,
                 pprogram_covariates_val: pprogram_X = None,
                 pprogram_covariates_test: pprogram_X = None,
                 seed: int = None):
        """
        Args:
            pprogram_maker: callable: a function that returns a probabilistic program
            curriculum: Curriculum: the curriculum
            pprogram_covariates: pprogram_X = simulate_X_uniform: a function that returns the covariates
            pprogram_covariates_train: pprogram_X: a function that returns the covariates for the training set. If None, use pprogram_covariates
            pprogram_covariates_val: pprogram_X: a function that returns the covariates for the validation set. If None, use pprogram_covariates
            pprogram_covariates_test: pprogram_X: a function that returns the covariates for the test set. If None, use pprogram_covariates
            seed: int: the seed for the random number generator, default 42
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
            print("Warning: pprogram_covariates, pprogram_covariates_train, pprogram_covariates_val, and pprogram_covariates_test are all not None. This most likely doesn't make sense")
        
    
        pprogram_maker_param_names = get_function_parameter_names(pprogram_maker)
        curriculum_param_names = list(curriculum.generation_params.keys())

        if seed is None:
            self.seed = torch.randint(0, 100000, (1,)).item()
        else:
            self.seed = seed

        if not pprogram_maker_param_names == curriculum_param_names:
            print("Warning: pprogram_covariates, pprogram_covariates_train, pprogram_covariates_val, and pprogram_covariates_test are all not None. Ignoring argument pprogram_covariates")

        self.pprogram_maker = pprogram_maker
        self.pprogram_covariates = pprogram_covariates
        self.curriculum = curriculum
        self.pprogram_covariates_train = pprogram_covariates_train
        self.pprogram_covariates_val = pprogram_covariates_val
        self.pprogram_covariates_test = pprogram_covariates_test

        self.program = pprogram_maker(**curriculum.get_params(0))  # get the program for the first iteration. Also use this for plotting

        super().__init__(pprogram = self.program, 
                         pprogram_covariates = self.pprogram_covariates, 
                         pprogram_covariates_train = self.pprogram_covariates_train,
                         pprogram_covariates_val = self.pprogram_covariates_val,
                         pprogram_covariates_test = self.pprogram_covariates_test,
                         seed = self.seed)


    def simulate_data(self, 
                     n:int = 100, 
                     p:int = 5, 
                     n_batch:int = 10_000,
                     epoch: int = 0
                    ) -> Tuple[list[dict], int]:
        """
        Simulate data for a linear model with the same beta for each batch.
        Discard samples where the linear model is not finite to avoid numerical issues.
        Args:
            n: int: the number of observations per batch 
            p: int: the number of covariates
            n_batch: int: the number of batches
            epoch: int: the epoch
        Returns:
            Tuple[list[dict], int]: a list of dictionaries containing the simulated data and the number of discarded samples
        """

        n_discarded = 0

        data = []
        for i in tqdm(list(range(n_batch))):
            pprogram = self.pprogram_maker(**self.curriculum.get_params(i + epoch * n_batch))
            x = self.pprogram_covariates(n,p)

            while True:
                lm_res = pprogram(x)

                # if anything is nan or inf, sample again
                if all([torch.isfinite(lm_res[key]).all() for key in lm_res.keys()]):
                    break
                else:
                    n_discarded += 1

            data.append(lm_res)

        if n_discarded > 0:
            print(f"Warning: {n_discarded} samples were discarded because the linear model was not finite.")

        return data, n_discarded
    
    def check_model(self, 
                    p:int = 3, 
                    n:int = 100, 
                    n_samples_per_epoch:int = 1_000, 
                    batch_size: int = 256, 
                    epochs_to_check:List[int] = [0, 10], 
                    used_batch_samples:int = 1000,
                    consider_average_variance_statistics: bool = True,
                    save_path_plots: str = None
                    
    )-> List:
        """
        Check the model for a few batches
        Args: 
            p: int: the number of covariates
            n: int: the number of observations
            n_samples_per_epoch: int: the number of batches
            batch_size: int: the batch size
            epochs_to_check: List[int]: the epochs to check
            used_batch_samples: int: the number of batches to use for checking
            consider_average_variance_statistics: bool: whether to consider the average variance statistics
            save_path_plots: str: the path to save the plots

        Returns:
            List: a list of results
        """

        results = []
        for epoch in epochs_to_check:
            print("#"*100)
            print(f"Epoch {epoch}")
            loader = self.make_dataloaders_for_epoch_dynamic(
                epoch = epoch, 
                n = n, 
                p = p,
                n_samples_per_epoch = n_samples_per_epoch,
                batch_size = batch_size, 
                train_frac = 1, 
                val_frac = 0, 
                shuffle = False, 
                use_seed = False)[0]

            sample_data = []
            for i in tqdm(list(range(used_batch_samples))):
                data = next(iter(loader))
                if i >= used_batch_samples:
                    break
                sample_data.append(data)

            if save_path_plots is not None:
                save_path_plots = save_path_plots + "/"
            
            r = check_and_plot_data(sample_data, batched_input=True, consider_average_variance_statistics = consider_average_variance_statistics, save_path_plots = save_path_plots)

            results.append(r)

        return results




    def make_dataloaders_for_epoch_dynamic(self,
                                 epoch:int = 0,
                                 n:int = 100,
                                 p:int = 5,
                                 n_samples_per_epoch:int = 10_000,
                                 batch_size:int = 256,
                                 train_frac = 0.7,
                                 val_frac = 0.15,
                                 shuffle: bool = True,
                                 use_seed: bool = False,
                                 n_samples_to_generate_at_once:int = 10_000
                                 ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Make a dataloader where the data is generated on the fly
        Args:
            epoch: int: the epoch
            n: int: the number of observations per batch 
            p: int: the number of covariates
            n_samples_per_epoch : int: the number of samples per epoch
            batch_size: int: the batch size
            train_frac: float: the fraction of the data to use for training
            val_frac: float: the fraction of the data to use for validation
            shuffle: bool: whether to shuffle the data
            use_seed: bool: whether to use the seed for the random number generator
            n_samples_to_generate_at_once: int: the number of samples to generate at once
        Returns:
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: a tuple of dataloaders for the training, test and validation data
        """

        train_size = int(train_frac * n_samples_per_epoch)
        val_size = int(val_frac * n_samples_per_epoch)
        test_size = n_samples_per_epoch - train_size - val_size

        if use_seed:
            seed = self.seed
        else:
            seed = torch.randint(0, 100000, (1,)).item()

        self.seed = seed

        
        dataset_train = SyntheticDataCurriculumBatched(n = n, 
                                p = p, 
                                n_samples_per_epoch = train_size, 
                                epoch = epoch,
                                pprogram_maker = self.pprogram_maker,
                                curriculum = self.curriculum,
                                pprogram_covariates = self.pprogram_covariates_train,
                                seed = None,
                                n_samples_to_generate_at_once = n_samples_to_generate_at_once)
        
        dataset_val = SyntheticDataCurriculumBatched(n = n,
                                p = p,
                                n_samples_per_epoch = val_size,
                                epoch = epoch,
                                pprogram_maker = self.pprogram_maker,
                                curriculum = self.curriculum,
                                pprogram_covariates = self.pprogram_covariates_val,
                                seed = None,
                                n_samples_to_generate_at_once = n_samples_to_generate_at_once)   
        
        dataset_test = SyntheticDataCurriculumBatched(n = n,
                                p = p,
                                n_samples_per_epoch = test_size,
                                epoch = epoch,
                                pprogram_maker = self.pprogram_maker,
                                curriculum = self.curriculum,
                                pprogram_covariates = self.pprogram_covariates_test,
                                seed = None, 
                                n_samples_to_generate_at_once = n_samples_to_generate_at_once)
        
        """
        dataset_train = SyntheticDataCurriculum(n = n, 
                                p = p, 
                                n_samples_per_epoch = train_size, 
                                epoch = epoch,
                                pprogram_maker = self.pprogram_maker,
                                curriculum = self.curriculum,
                                pprogram_covariates = self.pprogram_covariates,
                                seed = None)
        dataset_val = SyntheticDataCurriculum(n = n,
                                p = p,
                                n_samples_per_epoch = val_size,
                                epoch = epoch,
                                pprogram_maker = self.pprogram_maker,
                                curriculum = self.curriculum,
                                pprogram_covariates = self.pprogram_covariates,
                                seed = None)
        dataset_test = SyntheticDataCurriculum(n = n,
                                               p = p,
                                                  n_samples_per_epoch = test_size,
                                                    epoch = epoch,
                                                    pprogram_maker = self.pprogram_maker,
                                                    curriculum = self.curriculum,
                                                    pprogram_covariates = self.pprogram_covariates,
                                                    seed = None)
        """


        
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = shuffle)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = batch_size, shuffle = shuffle)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size, shuffle = shuffle)

        return dataloader_train, dataloader_val, dataloader_test
                                               

    def make_epoch_loader(
            self, 
            n:int = 100,
            p:int = 5,
            number_of_batches_per_epoch:int = 10_000,
            n_epochs:int = 100,
            batch_size:int = 256,
            train_frac = 0.7,
            val_frac = 0.15,
            shuffle: bool = False,
            n_samples_to_generate_at_once:int = 10_000
            ) -> EpochLoader:
        """
        Make a loader where for each epoch the data is generated on the fly
        Args:
            n: int: the number of observations per batch 
            p: int: the number of covariates
            number_of_batches_per_epoch : int: the number of batches
            n_epochs: int: the number of epochs
            batch_size: int: the batch size
            train_frac: float: the fraction of the data to use for training
            val_frac: float: the fraction of the data to use for validation
            shuffle: bool: whether to shuffle the data
        
        Returns:
            EpochLoader: an epoch loader
        """

        if not abs(number_of_batches_per_epoch * n_epochs * batch_size - self.curriculum.max_iter) < batch_size*n_epochs:
            print(f"The number of batches times the number of epochs must be equal to the total number of iterations in the curriculum. But got {number_of_batches_per_epoch * n_epochs * batch_size} and {self.curriculum.max_iter} respectively")

        epoch_loader = EpochLoader(
            GenerateDataCurriculum = self,
            n_epochs = n_epochs,
            n = n,
            p = p,
            n_batches_per_epoch= number_of_batches_per_epoch,
            batch_size = batch_size,
            train_frac = train_frac,
            val_frac = val_frac,
            shuffle = shuffle,
            n_samples_to_generate_at_once = n_samples_to_generate_at_once
        )
        
        
        return epoch_loader




class SyntheticDataCurriculum(torch.utils.data.Dataset):
    """
    A class to represent synthetic data that is generated on the fly
    This class uses a curriculum to determine the way the samples are generated
    Note that sampling is always random and the same seed is used for all batches
    """
    def __init__(
                self,
                n:int = 100,
                p:int = 5,
                n_samples_per_epoch:int = 10_000,
                epoch:int = 0,
                pprogram_maker:callable = None,
                curriculum: Curriculum = None,
                pprogram_covariates: pprogram_X = simulate_X_uniform,
                seed:int = None,
                check_data:bool = False,
                ):
        """
        a  torch.utils.data.Dataset that generates synthetic data on the fly
        Args:
            n: int: the number of observations per batch 
            p: int: the number of covariates
            n_samples_per_epoch: int: the number of SAMPLES per epoch
            epoch: int: the epoch
            pprogram_maker: callable: a function that returns a probabilistic program
            curriculum: Curriculum: the curriculum that determines the way the samples are generated during training
            pprogram: pprogram_linear_model_return_dict: a linear model probabilistic program
            pprogram_covariates: pprogram_X: a probabilistic program that simulates covariates
            seed: int: the seed for the random number generator
            check_data: bool: whether to check the data for numerical issues
        """
        self.n = n
        self.p = p
        self.n_samples_per_epoch = n_samples_per_epoch
        self.epoch = epoch
        self.pprogram_maker = pprogram_maker
        self.curriculum = curriculum
        self.pprogram_covariates = pprogram_covariates
        
        self.check_data = check_data

        if seed is None:
            seed = torch.randint(0, 100000, (1,)).item()
        self.seed = seed
        #torch.manual_seed(seed)
        #pyro.set_rng_seed(seed)
    
    def __len__(self) -> int:
        return self.n_samples_per_epoch 

    def __getitem__(self, idx) -> dict:
        total_iteration = self.epoch * self.n_samples_per_epoch + idx

        pprogram = self.pprogram_maker(**self.curriculum(total_iteration))
        x = self.pprogram_covariates(self.n, self.p)

        while True:
            lm_res = pprogram(x)
            # if anything is nan or inf, sample again
            if self.check_data:
                if all([torch.isfinite(lm_res[key]).all() for key in lm_res.keys()]):
                    break
            else:
                break

        return lm_res
    
class SyntheticDataCurriculumBatched(SyntheticDataCurriculum):
    """
    A class to represent synthetic data that is generated on the fly
    This class uses a curriculum to determine the way the samples are generated
    Note that sampling is always random and the same seed is used for all batches
    Here, we aim to speedup the generation by generating the covariates only once for a specified number of samples (e.g. a batch)
    This means, a ppgroram needs to be supplied that can handle a batch of covariates
    """

    def __init__(
                self,
                n:int = 100,
                p:int = 5,
                n_samples_per_epoch:int = 10_000,
                epoch:int = 0,
                pprogram_maker:callable = None,
                curriculum: Curriculum = None,
                pprogram_covariates: pprogram_X = simulate_X_uniform,
                seed:int = None,
                check_data:bool = False,
                n_samples_to_generate_at_once:int = 10_000,
                generate_test_data_in_nonmeta_datasets:bool = True
                ):
        """
        a  torch.utils.data.Dataset that generates synthetic data on the fly
        Args:
            n: int: the number of observations per batch 
            p: int: the number of covariates
            n_samples_per_epoch: int: the number of SAMPLES per epoch
            epoch: int: the epoch
            pprogram_maker: callable: a function that returns a probabilistic program
            curriculum: Curriculum: the curriculum that determines the way the samples are generated during training
            pprogram: pprogram_linear_model_return_dict: a linear model probabilistic program
            pprogram_covariates: pprogram_X: a probabilistic program that simulates covariates
            seed: int: the seed for the random number generator
            check_data: bool: whether to check the data for numerical issues
            n_samples_to_generate_at_once: int: the number of samples to generate at once
            generate_test_data_in_nonmeta_datasets: bool: whether to generate test data in non-meta datasets
        """
        if n_samples_to_generate_at_once > n_samples_per_epoch:
            print(f"Warning: n_samples_to_generate_at_once should be smaller than n_samples_per_epoch, but got {n_samples_to_generate_at_once} and {n_samples_per_epoch} respectively. This most likely won't make sense")

        super().__init__(
            n = n,
            p = p,
            n_samples_per_epoch = n_samples_per_epoch,
            epoch = epoch,
            pprogram_maker = pprogram_maker,
            curriculum = curriculum,
            pprogram_covariates = pprogram_covariates,
            seed = seed,
            check_data = check_data
        )
        self.batch_size = n_samples_to_generate_at_once
        self.n_samples_to_generate_at_once = n_samples_to_generate_at_once
        self.stored_samples = None
        self.generate_test_data_in_nonmeta_datasets = generate_test_data_in_nonmeta_datasets

    def __repr__(self) -> str:
        representation = f"""SyntheticDataCurriculumBatched(
        n = {self.n},
        p = {self.p},
        n_samples_per_epoch = {self.n_samples_per_epoch},
        epoch = {self.epoch},
        pprogram_maker = {self.pprogram_maker},
        curriculum = {self.curriculum},
        pprogram_covariates = {self.pprogram_covariates},
        seed = {self.seed},
        check_data = {self.check_data},
        n_samples_to_generate_at_once = {self.n_samples_to_generate_at_once}
        gerate_test_data_in_nonmeta_datasets = {self.generate_test_data_in_nonmeta_datasets}
        )"""
        return representation

    def compute_samples_parallel(self, start_idx:int, n_samples:int) -> dict:
        """
        Compute the samples in parallel
        Args:
            start_idx: int: the start index
            n_samples: int: the number of samples to generate
        Returns:
            dict: the samples
        """
        pprogram = self.pprogram_maker(**self.curriculum(start_idx))

        if not self.generate_test_data_in_nonmeta_datasets:
            x = self.pprogram_covariates(self.n, self.p, n_samples)
        else:
            x = self.pprogram_covariates(2 * self.n, self.p, n_samples)

        while True:
            lm_res = pprogram(x)
            # if anything is nan or inf, sample again
            if self.check_data:
                if all([torch.isfinite(lm_res[key]).all() for key in lm_res.keys()]):
                    break
            else:
                break

        if self.generate_test_data_in_nonmeta_datasets:
            for key, value in lm_res.items():
                value_shape = list(value.shape)
                # split the tensor into train and test if the axis has length 2* self.n
                if len(value_shape) > 1 and value_shape[0] == 2 * self.n:
                    lm_res[key] = value[:self.n]
                    lm_res[f"{key}_test"] = value[self.n:]
                elif len(value_shape) == 1 and value_shape[0] == 2 * self.n:
                    lm_res[key] = value[:self.n]
                    lm_res[f"{key}_test"] = value[self.n:]
                elif len(value_shape) > 1 and value_shape[1] == 2 * self.n:
                    lm_res[key] = value[:, :self.n]
                    lm_res[f"{key}_test"] = value[:, self.n:]

                
        self.stored_samples = {
            "start_idx": start_idx,
            "end_idx": start_idx + n_samples,
            "used_samples": 0,
            "samples": lm_res
        }

    def __getitem__(self, idx) -> dict:
        total_iteration = self.epoch * self.n_samples_per_epoch + idx

        if self.stored_samples is None:
            self.compute_samples_parallel(total_iteration, self.n_samples_to_generate_at_once)

        if total_iteration >= self.stored_samples["end_idx"] or total_iteration < self.stored_samples["start_idx"]:
            self.compute_samples_parallel(total_iteration, self.n_samples_to_generate_at_once)

        idx_to_acess_stored_samples = self.stored_samples["used_samples"]

        if idx_to_acess_stored_samples >= self.n_samples_to_generate_at_once:
            self.compute_samples_parallel(total_iteration, self.n_samples_to_generate_at_once)
            idx_to_acess_stored_samples = 0

        res = {}

        for key, value in self.stored_samples["samples"].items():
            if len(value.shape) == 1:
                res[key] = value
            else:
                res[key] = value[idx_to_acess_stored_samples]

        self.stored_samples["used_samples"] += 1
        return res