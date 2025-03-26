"""
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataLM_Examples import make_lm_program_ig_batched, make_lm_program_ig
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataLM_ExamplesIntercept import make_lm_program_ig_intercept_batched, make_lm_program_ig_intercept
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataLM_ExamplesSparsity import make_lm_program_Laplace_ig_batched, make_lm_program_Laplace_ig, make_lm_program_Laplace_ig_intercept_batched, make_lm_program_Laplace_ig_intercept
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataLM_SkewedPrior import make_lm_program_gamma_prior_batched, make_lm_program_gamma_prior, make_lm_program_gamma_prior_intercept_batched, make_lm_program_gamma_prior_intercept
from PFNExperiments.LinearRegression.GenerativeModels.GenerateData_LogReg import make_logreg_program_ig_batched, make_logreg_program_ig, make_logreg_program_ig_intercept_batched, make_logreg_program_ig_intercept
from PFNExperiments.LatentFactorModels.GenerativeModels.Clustering.GMMs import make_gmm_program_univariate_batched, make_gmm_program_univariate
from PFNExperiments.LatentFactorModels.GenerativeModels.Clustering.GMMs import make_gmm_program_spherical_batched, make_gmm_program_spherical
from PFNExperiments.LatentFactorModels.GenerativeModels.Clustering.GMMs import make_gmm_program_diagonal_batched, make_gmm_program_diagonal
from PFNExperiments.LatentFactorModels.GenerativeModels.FactorAnalysis.BasicFA import make_fa_program_normal_weight_prior_batched, make_fa_program_normal_weight_prior
from PFNExperiments.LatentFactorModels.GenerativeModels.FactorAnalysis.BasicFA import make_fa_program_laplace_weight_prior_batched, make_fa_program_laplace_weight_prior
from PFNExperiments.LatentFactorModels.GenerativeModels.FactorAnalysis.BasicFA import make_fa_program_normal_weight_prior_laplace_z_prior_batched, make_fa_program_normal_weight_prior_laplace_z_prior
from PFNExperiments.LatentFactorModels.GenerativeModels.Numpyro_Versions.LDA_NumPyro import make_lda_program, make_lda_program_batched
from PFNExperiments.PFN_GLM.GenerativeModels.GenerateData_LM_TabpfnNoise import make_lm_program_ig_intercept_tabpfn_noise_batched


This file provides the mapping from the name of the program to the probabilistic program.


name2pprogram_maker = {
    "ig": (make_lm_program_ig_batched, make_lm_program_ig),
    "ig_intercept": (make_lm_program_ig_intercept_batched, make_lm_program_ig_intercept),
    "Laplace_ig": (make_lm_program_Laplace_ig_batched, make_lm_program_Laplace_ig),
    "Laplace_ig_intercept": (make_lm_program_Laplace_ig_intercept_batched, make_lm_program_Laplace_ig_intercept),
    "Gamma_ig": (make_lm_program_gamma_prior_batched, make_lm_program_gamma_prior),
    "Gamma_ig_intercept": (make_lm_program_gamma_prior_intercept_batched, make_lm_program_gamma_prior_intercept),
    "logreg_ig": (make_logreg_program_ig_batched, make_logreg_program_ig),
    "logreg_ig_intercept": (make_logreg_program_ig_intercept_batched, make_logreg_program_ig_intercept),
    "gmm_univariate": (make_gmm_program_univariate_batched, make_gmm_program_univariate),
    "gmm_spherical": (make_gmm_program_spherical_batched, make_gmm_program_spherical),
    "gmm_diagonal": (make_gmm_program_diagonal_batched, make_gmm_program_diagonal),
    "fa_basic": (make_fa_program_normal_weight_prior_batched, make_fa_program_normal_weight_prior),
    "fa_laplace": (make_fa_program_laplace_weight_prior_batched, make_fa_program_laplace_weight_prior),
    "fa_normal_laplace": (make_fa_program_normal_weight_prior_laplace_z_prior_batched, make_fa_program_normal_weight_prior_laplace_z_prior),
    "lda_basic": (make_lda_program_batched, make_lda_program),
}   
"""

from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataLM_Examples import make_lm_program_ig_batched, make_lm_program_ig
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataLM_ExamplesIntercept import make_lm_program_ig_intercept_batched, make_lm_program_ig_intercept
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataLM_ExamplesSparsity import make_lm_program_Laplace_ig_batched, make_lm_program_Laplace_ig, make_lm_program_Laplace_ig_intercept_batched, make_lm_program_Laplace_ig_intercept
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataLM_SkewedPrior import make_lm_program_gamma_prior_batched, make_lm_program_gamma_prior
from PFNExperiments.LinearRegression.GenerativeModels.GenerateData_LogReg import make_logreg_program_ig_batched, make_logreg_program_ig
from PFNExperiments.LatentFactorModels.GenerativeModels.Clustering.GMMs import make_gmm_program_univariate_batched, make_gmm_program_univariate
from PFNExperiments.LatentFactorModels.GenerativeModels.Clustering.GMMs import make_gmm_program_spherical_batched, make_gmm_program_spherical
from PFNExperiments.LatentFactorModels.GenerativeModels.Clustering.GMMs import make_gmm_program_diagonal_batched, make_gmm_program_diagonal
from PFNExperiments.LatentFactorModels.GenerativeModels.FactorAnalysis.BasicFA import make_fa_program_normal_weight_prior_batched, make_fa_program_normal_weight_prior
from PFNExperiments.LatentFactorModels.GenerativeModels.FactorAnalysis.BasicFA import make_fa_program_laplace_weight_prior_batched, make_fa_program_laplace_weight_prior
from PFNExperiments.LatentFactorModels.GenerativeModels.FactorAnalysis.BasicFA import make_fa_program_normal_weight_prior_laplace_z_prior_batched, make_fa_program_normal_weight_prior_laplace_z_prior
from PFNExperiments.LinearRegression.GenerativeModels.GenerateDataLM_TargetDist import make_lm_program_ig_gamma_response_reparam_batched, make_lm_program_ig_gamma_response_reparam
from PFNExperiments.LatentFactorModels.GenerativeModels.Numpyro_Versions.LDA_NumPyro import make_lda_program, make_lda_program_batched
from PFNExperiments.PFN_GLM.GenerativeModels.GenerateData_LM_TabpfnNoise import make_lm_program_ig_intercept_tabpfn_noise_batched


"""
This file provides the mapping from the name of the program to the probabilistic program.
"""

name2pprogram_maker = {
    "ig": (make_lm_program_ig_batched, make_lm_program_ig),
    "ig_intercept": (make_lm_program_ig_intercept_batched, make_lm_program_ig_intercept),
    "Laplace_ig": (make_lm_program_Laplace_ig_batched, make_lm_program_Laplace_ig),
    "Laplace_ig_intercept": (make_lm_program_Laplace_ig_intercept_batched, make_lm_program_Laplace_ig_intercept),
    "Gamma_ig": (make_lm_program_gamma_prior_batched, make_lm_program_gamma_prior),
    "logreg_ig": (make_logreg_program_ig_batched, make_logreg_program_ig),
    "ig_gamma_response_reparam": (make_lm_program_ig_gamma_response_reparam_batched, make_lm_program_ig_gamma_response_reparam),
    "gmm_univariate": (make_gmm_program_univariate_batched, make_gmm_program_univariate),
    "gmm_spherical": (make_gmm_program_spherical_batched, make_gmm_program_spherical),
    "gmm_diagonal": (make_gmm_program_diagonal_batched, make_gmm_program_diagonal),
    "fa_basic": (make_fa_program_normal_weight_prior_batched, make_fa_program_normal_weight_prior),
    "fa_laplace": (make_fa_program_laplace_weight_prior_batched, make_fa_program_laplace_weight_prior),
    "fa_normal_laplace": (make_fa_program_normal_weight_prior_laplace_z_prior_batched, make_fa_program_normal_weight_prior_laplace_z_prior),
    "lda_basic": (make_lda_program_batched, make_lda_program),
    "ig_intercept_tabpfn_noise": (make_lm_program_ig_intercept_tabpfn_noise_batched, None),

}   