import configparser
import ast
from copy import deepcopy
import os
import sys

from PFNExperiments.LinearRegression.GenerativeModels.LM_abstract import return_only_y, print_code
from PFNExperiments.LinearRegression.Models.Transformer_CNF import TransformerCNFConditionalDecoder, TransformerCNFConditionalDecoderSequenceZ
from PFNExperiments.Training.FlowMatching.CFMLossOT2 import CFMLossOT2
from PFNExperiments.LinearRegression.Models.ModelToPosteriorCNF import ModelToPosteriorCNF
from PFNExperiments.LinearRegression.GenerativeModels.Curriculum import Curriculum
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataCurriculumCFM import GenerateDataCurriculumCFM
from PFNExperiments.Training.TrainerCurriculumCNF import TrainerCurriculumCNF
from PFNExperiments.Evaluation.Evaluate import Evaluate
from PFNExperiments.Evaluation.RealWorldEvaluation.EvaluateRealWorld import EvaluateRealWorld, just_return_results, results_dict_to_latent_variable_beta0_and_beta
from PFNExperiments.LinearRegression.GenerativeModels.GenerateX_TabPFN.MakeGenerator import MakeGenerator
from torch.optim.lr_scheduler import OneCycleLR
from PFNExperiments.LinearRegression.ComparisonModels.MakeDefaultListComparison import make_default_list_comparison, make_reduced_list_comparison
from PFNExperiments.Evaluation.RealWorldEvaluation.PreprocessDataset import Preprocessor
from PFNExperiments.Evaluation.RealWorldEvaluation.GetDataOpenML import GetDataOpenML
from PFNExperiments.LinearRegression.GenerativeModels.Name2Pprogram import name2pprogram_maker
from PFNExperiments.Training.Trainer import visualize_training_results
from PFNExperiments.Evaluation.EvaluatePredictions import EvaluatePredictions
import torch
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from tabpfn import TabPFNClassifier, TabPFNRegressor  
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from PFNExperiments.LinearRegression.ComparisonModels.PyroMAPPredictor import PyroMAPPredictor

from PFNExperiments.Training.FlowMatching.CFMLossDiffusionVP import CFMLossDiffusionVP

def string2bool(s: str) -> bool:
    """
    Convert a string to a boolean.
    Args:
    s: str: the string to convert
    Returns:
    bool: the boolean value
    """
    if s.lower() in ["true", "1"]:
        return True
    elif s.lower() in ["false", "0"]:
        return False
    else:
        raise ValueError(f"String {s} not recognized as boolean!")

class RunExperiments():
    """
    A class to run experiments in Colab.
    """
        
    def __init__(
                self,
                config_path: str = "./config.ini",
        ):
        """
        Constructor of the RunExperimentsColab class.
        Args:
        config_path: str: the path to save the configuration file
        """
        self.config_path = config_path
        self.config = configparser.ConfigParser() # Create a ConfigParser object
        
        self.config.read(f"{self.config_path}")

        # save config at the save path
        # if target forlder does not exist, create it
        if not os.path.exists(self.config["BASIC"]["Save_path"]):
            os.makedirs(self.config["BASIC"]["Save_path"])

        with open(f"{self.config['BASIC']['Save_path']}/config.ini", "w") as configfile:
            self.config.write(configfile)

    def setup_data_generation(self):
        """
        Setup the data generation."""
        gen_config = self.config["DATA_GENERATION"]
        N_EPOCHS = int(self.config["BASIC"]["N_epochs"])
        N_BATCHES_PER_EPOCH = int(self.config["BASIC"]["N_batches_per_epoch"])
        BATCH_SIZE = int(self.config["BASIC"]["Batch_size"])
        P = int(self.config["BASIC"]["P"])
        N = int(self.config["BASIC"]["N"])
        N_SAMPLES_PER_EPOCH = int(self.config["BASIC"]["N_samples_per_epoch"])

        pprogram_params = ast.literal_eval(gen_config["pprogram_params"])

        self.curriculum = Curriculum(max_iter=int(N_EPOCHS*N_BATCHES_PER_EPOCH*BATCH_SIZE*0.5))
        param_list = [(name, self.curriculum.constant_scheduler(float(value))) for name, value in pprogram_params.items()]

        self.curriculum.add_param_list(
            param_list
        )
        self.curriculum.plot_all_schedules()

        if gen_config["Generate_X_behaviour"] == "TabPFNX_extended1":
            X_files_paths_str = gen_config["X_data_files"]
            # parse a string of paths to a list of paths
            X_files_paths = ast.literal_eval(X_files_paths_str)
            self.generate_X = MakeGenerator(
                paths = X_files_paths
            ).make_generator()

        pprogram_batched, pprogram = name2pprogram_maker[gen_config["pprogram"]]

        self.data_generator = GenerateDataCurriculumCFM(
            pprogram_maker = pprogram_batched,
            curriculum= self.curriculum,
            pprogram_covariates = self.generate_X,
        )

        self.check_model_res = self.data_generator.check_model(
            n_samples_per_epoch=N_SAMPLES_PER_EPOCH,
            epochs_to_check = [0, N_EPOCHS-1],
            p = P,
            n = N,
            used_batch_samples = 3,
            save_path_plots=self.config["BASIC"]["Save_path"] + "/check_model"
        )
        self.epoch_loader = self.data_generator.make_epoch_loader(
            n = N,
            p = P,
            number_of_batches_per_epoch = N_BATCHES_PER_EPOCH,
            n_epochs = N_EPOCHS,
            batch_size= BATCH_SIZE,
            train_frac= float(self.config["BASIC"]["Train_frac"]),
            val_frac= float(self.config["BASIC"]["Val_frac"]),
            shuffle= string2bool(self.config["BASIC"]["Shuffle"]),
            n_samples_to_generate_at_once = int(self.config["BASIC"]["N_samples_to_generate_at_once"]),
        )

        sample_batch = next(iter(self.epoch_loader[0][0])) # try if loader works

    def setup_model(self):
        """
        Setup the model.
        """
        model_kwargs = deepcopy(self.config["MODEL"])

        if model_kwargs["type"] == "TransformerCNFConditionalDecoder":
            self.model = TransformerCNFConditionalDecoder(
                n_input_features_encoder = int(model_kwargs["n_input_features_encoder"]),
                n_input_features_decoder = int(model_kwargs["n_input_features_decoder"]),
                d_model_encoder = int(model_kwargs["d_model_encoder"]),
                d_model_decoder = int(model_kwargs["d_model_decoder"]),
                n_heads_encoder = int(model_kwargs["n_heads_encoder"]),
                n_heads_decoder = int(model_kwargs["n_heads_decoder"]),
                d_ff_encoder = int(model_kwargs["d_ff_encoder"]),
                d_ff_decoder = int(model_kwargs["d_ff_decoder"]),
                dropout_encoder = float(model_kwargs["dropout_encoder"]),
                dropout_decoder = float(model_kwargs["dropout_decoder"]),
                n_conditional_input_features =  int(model_kwargs["n_conditional_input_features"]),
                n_condition_features = int(model_kwargs["n_condition_features"]),
                n_layers_condition_embedding = int(model_kwargs["n_layers_condition_embedding"]),
                n_layers_encoder = int(model_kwargs["n_layers_encoder"]),
                n_layers_decoder = int(model_kwargs["n_layers_decoder"]),
                use_positional_encoding_encoder = string2bool(model_kwargs["use_positional_encoding_encoder"]),
                use_positional_encoding_decoder = string2bool(model_kwargs["use_positional_encoding_decoder"]),
                use_self_attention_decoder = string2bool(model_kwargs["use_self_attention_decoder"]),
                output_dim = int(model_kwargs["output_dim"]),
                d_final_processing = int(model_kwargs["d_final_processing"]),
                n_final_layers = int(model_kwargs["n_final_layers"]),
                dropout_final = float(model_kwargs["dropout_final"]),
                treat_z_as_sequence =  string2bool(model_kwargs["treat_z_as_sequence"]),
            )

        if model_kwargs["type"] == "TransformerCNFConditionalDecoderSequenceZ":
            self.model = TransformerCNFConditionalDecoderSequenceZ(
                n_input_features_encoder = int(model_kwargs["n_input_features_encoder"]),
                n_input_features_decoder = int(model_kwargs["n_input_features_decoder"]),
                d_model_encoder = int(model_kwargs["d_model_encoder"]),
                d_model_decoder = int(model_kwargs["d_model_decoder"]),
                n_heads_encoder = int(model_kwargs["n_heads_encoder"]),
                n_heads_decoder = int(model_kwargs["n_heads_decoder"]),
                d_ff_encoder = int(model_kwargs["d_ff_encoder"]),
                d_ff_decoder = int(model_kwargs["d_ff_decoder"]),
                dropout_encoder = float(model_kwargs["dropout_encoder"]),
                dropout_decoder = float(model_kwargs["dropout_decoder"]),
                n_conditional_input_features =  int(model_kwargs["n_conditional_input_features"]),
                n_condition_features = int(model_kwargs["n_condition_features"]),
                n_layers_condition_embedding = int(model_kwargs["n_layers_condition_embedding"]),
                n_layers_encoder = int(model_kwargs["n_layers_encoder"]),
                n_layers_decoder = int(model_kwargs["n_layers_decoder"]),
                use_positional_encoding_encoder = string2bool(model_kwargs["use_positional_encoding_encoder"]),
                use_positional_encoding_decoder = string2bool(model_kwargs["use_positional_encoding_decoder"]),
                use_self_attention_decoder = string2bool(model_kwargs["use_self_attention_decoder"]),
                output_dim = int(model_kwargs["output_dim"]),
            )

    def setup_training(self):
        """
        Setup the training.
        """
        
        train_config = self.config["TRAINING"]

        if train_config["Loss_function"] == "CFMLossOT2":
            self.loss_function = CFMLossOT2(
                sigma_min = float(train_config["Sigma_min"])
            )

        elif train_config["Loss_function"] == "CFMLossDiffusionVP":
            self.loss_function = CFMLossDiffusionVP(
                epsilon_for_t = float(train_config["epsilon_for_t"]),
                beta_min = float(train_config["beta_min"]),
                beta_max = float(train_config["beta_max"]),
            )
        else:
            raise ValueError(f"Loss function {train_config['Loss_function']} not implemented yet!")
        
        self.optimizer = torch.optim.Adam(
                                          self.model.parameters(), 
                                          lr=float(self.config["TRAINING"]["Learning_rate"]),
                                          weight_decay=float(self.config["TRAINING"]["Weight_decay"]))
        
        scheduler_params = ast.literal_eval(train_config["Scheduler_params"])
        self.scheduler = OneCycleLR(
            optimizer = self.optimizer,
            max_lr = float(scheduler_params["max_lr"]),
            epochs = int(scheduler_params["epochs"]),
            steps_per_epoch = int(scheduler_params["steps_per_epoch"]),
            pct_start = float(scheduler_params["pct_start"]),
            div_factor = float(scheduler_params["div_factor"]),
            final_div_factor = float(scheduler_params["final_div_factor"])

        )

        

        self.trainer = TrainerCurriculumCNF(
            model = self.model,
            optimizer = self.optimizer,
            scheduler = self.scheduler,
            loss_function = self.loss_function,
            epoch_loader = self.epoch_loader,
            evaluation_functions = {},
            n_epochs = int(self.config["BASIC"]["N_epochs"]),
            early_stopping_patience = int(self.config["TRAINING"]["Early_stopping_patience"]),
            schedule_step_on = "batch",
            save_path = self.config["BASIC"]["Save_path"],
            coupling = None,
            use_same_timestep_per_batch = False,
            use_train_mode_during_validation = False,
            max_gradient_norm = float(self.config["TRAINING"]["max_grad_norm"]),

        )

        initial_val_loss = self.trainer.validate()

        print(f"Initial validation loss: {initial_val_loss}")

        r = self.trainer.train()

        visualize_training_results(r, loglog=False)

        self.trainer.load_best_model()
        self.model = self.trainer.model
        self.model.eval()

        test_res = self.trainer.test()

        print(f"Test results: {test_res}")

    def load_model_from_path(self, new_save_path: str, validate = True):
        """
        Load the model from a path.
        Args:
        new_save_path: str: the path to load the model from
        validate: bool: whether to validate the model 
        """
        train_config = self.config["TRAINING"]

        if train_config["Loss_function"] == "CFMLossOT2":
            self.loss_function = CFMLossOT2(
                sigma_min = float(train_config["Sigma_min"])
            )

        elif train_config["Loss_function"] == "CFMLossDiffusionVP":
            self.loss_function = CFMLossDiffusionVP(
                epsilon_for_t = float(train_config["epsilon_for_t"]),
                beta_min = float(train_config["beta_min"]),
                beta_max = float(train_config["beta_max"]),
            )
        else:
            raise ValueError(f"Loss function {train_config['Loss_function']} not implemented yet!")
        
        self.optimizer = torch.optim.Adam(
                                          self.model.parameters(), 
                                          lr=float(self.config["TRAINING"]["Learning_rate"]),
                                          weight_decay=float(self.config["TRAINING"]["Weight_decay"]))
        
        scheduler_params = ast.literal_eval(train_config["Scheduler_params"])
        self.scheduler = OneCycleLR(
            optimizer = self.optimizer,
            max_lr = float(scheduler_params["max_lr"]),
            epochs = int(scheduler_params["epochs"]),
            steps_per_epoch = int(scheduler_params["steps_per_epoch"]),
            pct_start = float(scheduler_params["pct_start"]),
            div_factor = float(scheduler_params["div_factor"]),
            final_div_factor = float(scheduler_params["final_div_factor"])

        )

        

        self.trainer = TrainerCurriculumCNF(
            model = self.model,
            optimizer = self.optimizer,
            scheduler = self.scheduler,
            loss_function = self.loss_function,
            epoch_loader = self.epoch_loader,
            evaluation_functions = {},
            n_epochs = int(self.config["BASIC"]["N_epochs"]),
            early_stopping_patience = int(self.config["TRAINING"]["Early_stopping_patience"]),
            schedule_step_on = "batch",
            save_path = self.config["BASIC"]["Save_path"],
            coupling = None,
            use_same_timestep_per_batch = False,
            use_train_mode_during_validation = False,
            max_gradient_norm = float(self.config["TRAINING"]["max_grad_norm"]),

        )
        self.trainer.set_new_save_path(new_save_path)

        self.trainer.load_best_model()
        self.model = self.trainer.model
        self.model.eval()

        if validate:
            initial_val_loss = self.trainer.validate()

            print(f"Initial validation loss: {initial_val_loss}")

    def setup_full_model(self):
        """
        Setup the full model.
        """
        full_model_kwargs = self.config["FULL_MODEL"]

        if self.config["TRAINING"]["Loss_function"] == "CFMLossDiffusionVP":
            epsilon_for_t = float(self.config["TRAINING"]["epsilon_for_t"])
            print("epsilon_for_t: ", epsilon_for_t)

        else:
            epsilon_for_t = 0.0

        self.full_model = ModelToPosteriorCNF(
            model = self.model,
            sample_shape= ast.literal_eval(full_model_kwargs["sample_shape"]),
            sample_name= full_model_kwargs["sample_name"],
            n_samples= int(full_model_kwargs["n_samples"]),
            batch_size= int(full_model_kwargs["batch_size"]),
            solve_adjoint= string2bool(full_model_kwargs["solve_adjoint"]),
            atol = float(full_model_kwargs["atol"]),
            rtol = float(full_model_kwargs["rtol"]),
            epsilon_for_t = epsilon_for_t
        )

    def setup_evaluation(self):
        """
        Setup the evaluation.
        """

        benchmark_params_ppgrogram = self.data_generator.curriculum.get_params(-1)
        print(f"params for pprogram: {benchmark_params_ppgrogram}")

        self.pprogram1 = name2pprogram_maker[self.config["DATA_GENERATION"]["Pprogram"]][1](**benchmark_params_ppgrogram)

        assert self.pprogram1 is not None, "pprogram1 is None!"

        self.pprogram1_y = return_only_y(self.pprogram1)
        
        print_code(self.pprogram1)

        if string2bool(self.config["EVALUATION"]["do_full_evaluation"]) is True: 
            self.comparison_models = make_default_list_comparison(
                pprogram_y=self.pprogram1_y,
                n_samples=int(self.config["EVALUATION"]["N_samples_per_model"])
            )
        else:
            self.comparison_models = make_reduced_list_comparison(
                pprogram_y=self.pprogram1_y,
                n_samples=int(self.config["EVALUATION"]["N_samples_per_model"])
            )

        
    def evaluate_synthetic(self):
        """
        Evaluate the synthetic data.
        """
        use_intercept = string2bool(self.config["DATA_GENERATION"]["Use_intercept"])

        run_posterior_eval = string2bool(self.config["EVALUATION"]["run_posterior_sample_eval"])

        self.evaluator = Evaluate(
            posterior_model=self.full_model,
            evaluation_loader=self.trainer.testset,
            comparison_models=self.comparison_models,
            n_evaluation_cases = int(self.config["EVALUATION"]["N_synthetic_cases"]),
            save_path = self.config["BASIC"]["Save_path"] + "/synthetic_evaluation",
            results_dict_to_latent_variable_comparison_models = just_return_results if not use_intercept else results_dict_to_latent_variable_beta0_and_beta,
            overwrite_results=True
        )
        if run_posterior_eval:
            self.eval_res_synthetic = self.evaluator.run_evaluation_no_tests()
        else:
            self.evaluator.only_sample_posterior()

        self.map_predictor = PyroMAPPredictor(
            pprogram_y=self.pprogram1_y,
            pprogram_name=self.config["DATA_GENERATION"]["Pprogram"],
        )
        """

        self.evaluator_predictions_synthetic = EvaluatePredictions(
            pprogram_name = self.config["DATA_GENERATION"]["Pprogram"],
            posterior_model_samples = self.evaluator.posterior_model_samples,
            comparison_model_samples = self.evaluator.comparison_model_samples,
            baselines_regression = [self.map_predictor, LinearRegression(), RandomForestRegressor(), TabPFNRegressor()],
            baselines_classification = [self.map_predictor, LogisticRegression(), RandomForestClassifier(), TabPFNClassifier()],
            save_path=self.config["BASIC"]["Save_path"] + "/synthetic_evaluation_predictions",
        )

        
        r_pred = self.evaluator_predictions_synthetic.run_evaluation()

        print("Results of the predictions synthetic data:")
        print(r_pred)
        """

    def evaluate_real_world(self):
        """
        Evaluate the real world data.
        """
        self.map_predictor = PyroMAPPredictor(
            pprogram_y=self.pprogram1_y,
            pprogram_name=self.config["DATA_GENERATION"]["Pprogram"],
        )

        run_posterior_eval = string2bool(self.config["EVALUATION"]["run_posterior_sample_eval"])

        target_mean = self.check_model_res[1]["y"]['mean_mean']
        target_var = self.check_model_res[1]["y"]['variance_mean']

        self.getdata = GetDataOpenML(
            preprocessor = Preprocessor(
                N_datapoints = int(self.config["BASIC"]["N"]),
                P_features = int(self.config["BASIC"]["P"]),
                target_mean = target_mean,
                target_var = target_var
            ),
            save_path = self.config["EVALUATION"]["save_path_data_real_world_eval"],
            benchmark_id = self.config["EVALUATION"]["real_world_benchmark_id"]
        )
        self.datasets = self.getdata.get_data()

        use_intercept = string2bool(self.config["DATA_GENERATION"]["Use_intercept"])

        if self.config["EVALUATION"]["n_evaluation_cases_real_world"] == "All":
            n_evaluation_cases = len(self.datasets)
        else:
            n_evaluation_cases = int(self.config["EVALUATION"]["n_evaluation_cases_real_world"])

        self.eval_rw = EvaluateRealWorld(
            posterior_model = self.full_model,
            evaluation_datasets = self.datasets,
            comparison_models = self.comparison_models,
            results_dict_to_latent_variable_comparison_models = just_return_results if not use_intercept else results_dict_to_latent_variable_beta0_and_beta,
            n_evaluation_cases = n_evaluation_cases,
            save_path = self.config["BASIC"]["Save_path"] + "/real_world_evaluation",
            overwrite_results = True
        )

        if run_posterior_eval:
            self.eval_res_real_world = self.eval_rw.run_evaluation_no_tests()
        else:
            self.eval_rw.only_sample_posterior()

        """

        self.evaluator_predictions_rw = EvaluatePredictions(
            pprogram_name = self.config["DATA_GENERATION"]["Pprogram"],
            posterior_model_samples = self.eval_rw.posterior_model_samples,
            comparison_model_samples = self.eval_rw.comparison_model_samples,
            baselines_regression = [self.map_predictor, LinearRegression(), RandomForestRegressor(), TabPFNRegressor()],
            baselines_classification = [self.map_predictor, LogisticRegression(), RandomForestClassifier(), TabPFNClassifier()],
            save_path=self.config["BASIC"]["Save_path"] + "/realworld_evaluation_predictions",
        )

        r_pred = self.evaluator_predictions_rw.run_evaluation()

        print("Results of the predictions snthetic data:")
        print(r_pred)
        """

    def run(self):
        """
        Run the experiments.
        """
        stdout = sys.stdout
        log_file = open(self.config["BASIC"]["Save_path"] + "/log.txt", "w")
        sys.stdout = log_file
        sys.stderr = log_file

        try:
            print("Starting experiments")

            self.setup_data_generation()
            self.setup_model()
            self.setup_training()
            self.setup_full_model()
            self.setup_evaluation()
            self.evaluate_synthetic()
            self.evaluate_real_world()

            print("Experiments finished")
        finally:
            log_file.flush()
            os.fsync(log_file.fileno())
            log_file.close()
            sys.stdout = stdout
            sys.stderr = stdout


        
        