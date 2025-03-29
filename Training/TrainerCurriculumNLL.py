import torch 
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
from time import strftime
from PFNExperiments.Training.FlowMatching.CFMLossOT import CFMLossOT
from PFNExperiments.Training.Losses import nll_loss_full_gaussian_flattened_pred

try:
    from Trainer import batch_to_model_lm, visualize_training_results

except:
    from PFNExperiments.Training.Trainer import batch_to_model_lm, visualize_training_results

from PFNExperiments.Training.TrainerCurriculum import TrainerCurriculum

from PFNExperiments.Training.FlowMatching.Couplings import MiniBatchOTCoupling

class TrainerCurriculumNLL(TrainerCurriculum):
    """
    A custom class for training neural networks
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 epoch_loader: callable,
                 loss_function: torch.nn.modules.loss._Loss = nll_loss_full_gaussian_flattened_pred,
                 valset: torch.utils.data.Dataset = None,
                 testset: torch.utils.data.Dataset = None,
                 scheduler: torch.optim.lr_scheduler = None,
                 schedule_step_on: str = "epoch",
                 target_key_in_batch: str = "beta",
                 evaluation_functions: dict[str, callable] = {},
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 n_epochs: int = 100,
                 save_path: str = "../experiments/",
                 make_new_folder: bool = True,
                 early_stopping_patience: int = 10,
                 verbose:int = 100,
                 summary_writer_path: str = "runs/",
                 use_same_timestep_per_batch: bool = False,
                 use_train_mode_during_validation: bool = False,
                 using_OTLossGaussianBase: bool = False,
                 loss_aggregation_functions: dict = {
                     "mean": torch.mean,
                     "median": torch.median,
                     "std": torch.std
                 },
                 sub_losses_names = [],
                 max_gradient_norm: float = 1.0
    ):
        """
        A custom class for training neural networks
        Args:
            model: torch.nn.Module: the model
            optimizer: torch.optim.Optimizer: the optimizer
            loss_function: torch.nn.modules.loss._Loss: the loss function
            epoch_loader: callable: a function that returns the training, testing and validation set for each epoch according to the curriculum
            valset: torch.utils.data.Dataset: the validation set
            testset: torch.utils.data.Dataset: the test set
            scheduler: torch.optim.lr_scheduler: the scheduler
            schedule_step_on: str: the step on which to schedule the learning rate. Can be "epoch" or "batch"
            batch_to_model_function: callable: a function that maps a batch to the model input
            target_key_in_batch: str: the key of the target in the batch
            evaluation_functions: dict[str, callable]: the evaluation functions in a dictionary where the key is the name of the evaluation function and the value is the evaluation function
            device: torch.device: the device
            n_epochs: int: the number of epochs
            save_path: str: the path to save the model
            make_new_folder: bool: whether to make a new folder
            early_stopping_patience: int: the patience for early stopping
            verbose: int: how much to print
            summary_writer_path: str: the path to save the summary writer
            use_same_timestep_per_batch: bool: whether to use the same timestep for all elements in the batch
            coupling: MiniBatchOTCoupling: the coupling to use to align z_0 and z_t. Can be None
            use_train_mode_during_validation: bool: whether to use the train mode during validation
            using_OTLossGaussianBase: bool: whether to use the OT loss with a Gaussian base distribution
            loss_aggregation_functions: dict: a dictionary containing the loss aggregation functions
            sub_losses_names: list: a list containing the names of the sub losses
            max_gradient_norm: float: the maximum gradient norm
        """

        assert schedule_step_on in ["epoch", "batch"], "schedule_step_on must be either 'epoch' or 'batch'"

        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epoch_loader = epoch_loader
        self.valset = valset
        self.testset = testset
        self.scheduler = scheduler
        self.schedule_step_on = schedule_step_on
        self.evaluation_functions = evaluation_functions
        self.target_key_in_batch = target_key_in_batch
        self.device = device
        self.n_epochs = n_epochs
        
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.summary_writer_path = summary_writer_path
        self.use_train_mode_during_validation = use_train_mode_during_validation


        self.use_same_timestep_per_batch = use_same_timestep_per_batch
        self.using_OTLossGaussianBase = using_OTLossGaussianBase
        self.loss_aggregation_functions = loss_aggregation_functions
        self.sub_losses_names = sub_losses_names
        self.max_gradient_norm = max_gradient_norm

        if self.valset is None:
            self.valset = self.epoch_loader(n_epochs)[1]  #load the validation set for the last epoch from the epoch_loader
        
        if self.testset is None:
            self.testset = self.epoch_loader(self.n_epochs)[2]

        self.model = self.model.to(self.device)

        if make_new_folder:
            time = strftime("%Y_%m_%d_%H_%M_%S")
            folder_name = f"experiment_{time}"
            self.save_path = f"{save_path}/{folder_name}"
        
        else:
            self.save_path = save_path

        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))


        self.writer = SummaryWriter(self.summary_writer_path)

        # write the model setup to the save_path 
        self.model_save_path = f"{self.save_path}/model.pth"

        self.setup_save_path = f"{self.save_path}/setup.txt"

        if not os.path.exists(os.path.dirname(self.setup_save_path)):
            os.makedirs(os.path.dirname(self.setup_save_path))
            
        with open(self.setup_save_path, "w") as f:
            f.write(str(self))

        self.tensorboard_save_path = f"{self.save_path}/tensorboard"

        if not os.path.exists(os.path.dirname(self.tensorboard_save_path)):
            os.makedirs(os.path.dirname(self.tensorboard_save_path))

        self.second_writer = SummaryWriter(self.tensorboard_save_path)

    def __repr__(self) -> str:
        repr_str = f"""TrainerCurriculumCNF(
    model = {self.model},
    optimizer = {self.optimizer},
    loss_function = {self.loss_function},
    epoch_loader = {self.epoch_loader},
    valset = {self.valset},
    testset = {self.testset},
    scheduler = {self.scheduler},
    schedule_step_on = {self.schedule_step_on},
    evaluation_functions = {self.evaluation_functions},
    target_key_in_batch = {self.target_key_in_batch},
    device = {self.device},
    n_epochs = {self.n_epochs},
    save_path = {self.save_path},
    early_stopping_patience = {self.early_stopping_patience},
    verbose = {self.verbose},
    summary_writer_path = {self.summary_writer_path}
    )"""
        return repr_str


    def batch_to_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the loss from a batch       
        Args:
            batch: dict[str, torch.Tensor]: the batch
        Returns:
            torch.Tensor: the loss
        """
        x = batch["x"]
        y = batch["y"]

        X_y = torch.cat([x, y.unsqueeze(-1)], dim = -1) # concatenate the x and y values to one data tensor

        z_1 = batch["beta"]     # sample from the ground truth distribution
        z_0 = batch['base_sample_beta']  # sample from the base distribution
        t = batch["time"]  # sample the time
        t = t.unsqueeze(-1) # add a dimension to the time tensor to give it shape (batch_size, 1)

        t = t.float()

        t = torch.ones_like(t) # replace t with ones to avoid the model to learn the time dependency

        p = z_1.shape[1] 
        zt = torch.ones(z_1.shape[0], p*p + 2*p).to(self.device) # create a tensor of ones with the same shape as z_1

        model_pred = self.model(zt, X_y, t)  # compute the vector field prediction by the model

        loss = self.loss_function(model_pred, z_1)  # compute the loss
    
        return loss
    
    def validate_loader(self, loader: torch.utils.data.DataLoader) -> dict[str, float]:
        """
        Validate the model
        Args:
            loader: torch.nn.utils.data.DataLoader: the loader
        Returns
            dict[str, float]: the validation results where the key is the name of the evaluation function and the value is the result of the evaluation function
        """
        self.model.eval()

        if self.use_train_mode_during_validation:
            self.model.train()

        loss_lis = []
        sub_losses_lis = []

        with torch.no_grad():
            for batch in tqdm(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss_total = self.batch_to_loss(batch)

                if self.sub_losses_names:
                    loss = loss_total[0]
                    sub_losses = loss_total[1:]

                    sub_losses = [l.detach().cpu() for l in sub_losses]

                    sub_losses_lis.append(sub_losses)

                else:
                    loss = loss_total

                
                loss_lis.append(loss.detach().cpu())



        results = {}

        for agg_name, agg_fun in self.loss_aggregation_functions.items():
            results[f"loss_{agg_name}"] = agg_fun(torch.stack(loss_lis)).item()

        if self.sub_losses_names:
            for i, name in enumerate(self.sub_losses_names):
                for agg_name, agg_fun in self.loss_aggregation_functions.items():
                    results[f"{name}_{agg_name}"] = agg_fun(torch.stack([l[i] for l in sub_losses_lis])).item()

        return results

    def validate(self) -> dict[str, float]:
        """
        Validate the model
        Returns
            dict[str, float]: the validation results where the key is the name of the evaluation function and the value is the result of the evaluation function
        """
        self.model.eval()

        if self.use_train_mode_during_validation:
            self.model.train()

        return self.validate_loader(self.valset)
    
    def test(self) -> dict[str, float]:
        """
        Test the model
        Returns
            dict[str, float]: the test results where the key is the name of the evaluation function and the value is the result of the evaluation function
        """

        self.model.eval()

        if self.use_train_mode_during_validation:
            self.model.train()
        
        return self.validate_loader(self.testset)
    
                
    def train(self) -> tuple[list[dict[str, float]], list[dict[str, float]], list[float]]:
        """
        Train the model
        Returns:
            tuple[list[dict[str, float]], list[dict[str, float]], list[float]]: a tuple containing the training results, the validation results and the time for each epoch
        """
        best_val_loss = float("inf")
        patience = 0

        self.model = self.model.to(self.device)

        all_results_training = []
        all_results_validation = []
        all_results_validation_curriculum = []  # the validation results for the current curriculum
        all_results_time = []
        
        start_time_epoch = time.time()
        overall_iter = 0

        batch_size = 0
        for epoch in range(self.n_epochs):
            self.model.train()
            #predictions = []
            #targets = []

            loss_lis = []
            sub_losses_lis = []

            trainset_epoch = self.epoch_loader(epoch)[0]
            valset_epoch = self.epoch_loader(epoch)[1]

            
            if self.verbose > 1: 
                current_params = self.epoch_loader.GenerateDataCurriculum.curriculum.get_params(iter = overall_iter * batch_size)
                print(f"Curriculum parameters: {current_params} at iteration { overall_iter * batch_size}")

            for batch in tqdm(trainset_epoch):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                batch_size = batch["x"].shape[0]


                loss = self.batch_to_loss(batch)

                if self.sub_losses_names:
                    sub_losses = loss[1:]
                    loss = loss[0]
                    

                self.optimizer.zero_grad()
                loss.backward()

                if self.max_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)

                self.optimizer.step()

                loss_lis.append(loss.detach().cpu())

                if self.sub_losses_names:
                    sub_losses = [l.detach().cpu() for l in sub_losses]
                    sub_losses_lis.append(sub_losses)

                

                self.writer.add_scalar("Loss/train", loss, overall_iter)
                self.second_writer.add_scalar("Loss/train", loss, overall_iter)

                for curriculum_param_name, curriculum_param in current_params.items():
                    self.writer.add_scalar(f"Curriculum/{curriculum_param_name}", curriculum_param, overall_iter)
                    self.second_writer.add_scalar(f"Curriculum/{curriculum_param_name}", curriculum_param, overall_iter)

                if self.sub_losses_names:
                    for i, name in enumerate(self.sub_losses_names):
                        self.writer.add_scalar(f"{name}/train", sub_losses[i], overall_iter)
                        self.second_writer.add_scalar(f"{name}/train", sub_losses[i], overall_iter)

                if self.scheduler is not None and self.schedule_step_on == "batch":
                    try: 
                        self.scheduler.step()
                    except:
                        self.scheduler.step(loss)

                    lr = self.scheduler.get_last_lr()
                    lr = lr[0]

                    self.writer.add_scalar("Learning rate", lr, overall_iter)
                    self.second_writer.add_scalar("Learning rate", lr, overall_iter)

                #predictions.append([elem.detach().cpu() for elem in pred])
                #targets.append(target.detach().cpu())

                overall_iter += 1

        
            validation_results = self.validate()

            for name, result in validation_results.items():
                self.writer.add_scalar(f"{name}/validation", result, epoch)
                self.second_writer.add_scalar(f"{name}/validation", result, epoch)

            validation_results_current_curriculum = self.validate_loader(valset_epoch)
            for name, result in validation_results_current_curriculum.items():
                self.writer.add_scalar(f"{name}/validation_curriculum", result, epoch)
                self.second_writer.add_scalar(f"{name}/validation_curriculum", result, epoch)

            if self.scheduler is not None and self.schedule_step_on == "epoch":
                try: 
                    self.scheduler.step()
                except:
                    self.scheduler.step(validation_results["loss_median"])
            training_results = {}
            #for name, fun in self.evaluation_functions.items():
            #    training_results[name] = fun(targets, predictions)

            for agg_name, agg_fun in self.loss_aggregation_functions.items():
                training_results[f"loss_{agg_name}"] = agg_fun(torch.stack(loss_lis)).item()

            if self.sub_losses_names:
                for i, name in enumerate(self.sub_losses_names):
                    for agg_name, agg_fun in self.loss_aggregation_functions.items():
                        training_results[f"{name}_{agg_name}"] = agg_fun(torch.stack([l[i] for l in sub_losses_lis])).item()


            end_time_epoch = time.time()
            time_epoch = end_time_epoch - start_time_epoch

            all_results_training.append(training_results)
            all_results_validation.append(validation_results)
            all_results_validation_curriculum.append(validation_results_current_curriculum)
            all_results_time.append(time_epoch)

            for name, result in training_results.items():
                self.writer.add_scalar(f"{name}/train", result, epoch)
                self.second_writer.add_scalar(f"{name}/train", result, epoch)

            for name, result in validation_results.items():
                self.writer.add_scalar(f"{name}/validation", result, epoch)
                self.second_writer.add_scalar(f"{name}/validation", result, epoch)

            print(f"Epoch {epoch}:")
            print(f"Training: {training_results}")
            print(f"Validation: {validation_results}")
            print(f"Validation curriculum: {validation_results_current_curriculum}")
            print(f"Time: {time_epoch}")
            if self.scheduler is not None:
                try:
                    print(f"Learning rate: {self.scheduler.get_last_lr()}")
                except:
                    try: 
                        print(f"Learning rate: {self.scheduler._last_lr}")
                    except:
                        print("Could not get learning rate")
                
            print("\n")
            print(100*"-")

            val_loss = validation_results_current_curriculum["loss_median"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Saving model")
                torch.save(self.model.state_dict(), self.model_save_path)
                patience = 0
            else:
                patience += 1

            if patience > self.early_stopping_patience:
                print("Early stopping")
                break

            start_time_epoch = time.time()

        
        if self.testset is not None:
            test_results = self.test()
            print(f"Test: {test_results}")
            return all_results_training, all_results_validation, all_results_time, test_results
        

        return all_results_training, all_results_validation, all_results_time
    

    