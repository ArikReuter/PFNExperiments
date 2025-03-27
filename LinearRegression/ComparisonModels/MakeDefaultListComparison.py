from PFNExperiments.LinearRegression.ComparisonModels.Hamiltionion_MC import Hamiltionian_MC
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoLaplaceApproximation, AutoIAFNormal, AutoStructured, AutoDelta
from PFNExperiments.LinearRegression.ComparisonModels.Variational_InferenceAutoguide import Variational_InferenceAutoguide


def make_default_list_comparison(
        pprogram_y,
        n_samples: int = 1000
        ):
    """
    Create the default list of comparison models.
    Args:
        pprogram_y: a probabilistic program for the response variable
        n_samples: int: the number of samples to draw
    """

    hmc_sampler = Hamiltionian_MC(pprogram=pprogram_y, n_warmup=n_samples//2, n_samples=n_samples)

    vi_diag = Variational_InferenceAutoguide(
        pprogram=pprogram_y,
        make_guide_fun=AutoDiagonalNormal,
        n_steps=5000,
        n_samples=n_samples,
        lr=10
    )

    vi_multivariate_normal = Variational_InferenceAutoguide(
        pprogram=pprogram_y,
        make_guide_fun=AutoMultivariateNormal,
        n_steps=5000,
        n_samples=n_samples,
        lr=10
    )
    vi_laplace = Variational_InferenceAutoguide(
        pprogram=pprogram_y,
        make_guide_fun=AutoLaplaceApproximation,
        n_steps=5000,
        n_samples=n_samples,
        lr=10
    )

    vi_autoIAF = Variational_InferenceAutoguide(
        pprogram=pprogram_y,
        make_guide_fun=AutoIAFNormal,
        n_steps=5000,
        n_samples=n_samples,
        lr=10
    )

    vi_autostrucured = Variational_InferenceAutoguide(
        pprogram=pprogram_y,
        make_guide_fun=AutoStructured,
        n_steps=5000,
        n_samples=n_samples,
        lr=10
    )

    model_list = [
        hmc_sampler,
        vi_diag,
        vi_multivariate_normal,
        vi_laplace,
        vi_autoIAF,
        vi_autostrucured
    ]

    return model_list


def make_reduced_list_comparison(
        pprogram_y,
        n_samples: int = 1000
        ):
    """
    Create the reduced list of comparison models.
    Args:
        pprogram_y: a probabilistic program for the response variable
        n_samples: int: the number of samples to draw
    """

    vi_diag = Variational_InferenceAutoguide(
        pprogram=pprogram_y,
        make_guide_fun=AutoDiagonalNormal,
        n_steps=5000,
        n_samples=n_samples,
        lr=10
    )

    vi_multivariate_normal = Variational_InferenceAutoguide(
        pprogram=pprogram_y,
        make_guide_fun=AutoMultivariateNormal,
        n_steps=5000,
        n_samples=n_samples,
        lr=10
    )

    model_list = [
        vi_diag,
        vi_multivariate_normal,
    ]

    return model_list


def make_vi_list_lr(
        pprogram_y,
        n_samples: int = 1000,
        n_steps: int = 2000,
        lr: float = 1e-3,
        print_lr: bool = False
        ):
    """
    Create the default list of comparison models.
    Args:
        pprogram_y: a probabilistic program for the response variable
        n_steps : int: the number of steps to take in the optimization
        n_samples: int: the number of samples to draw
        lr: float: the learning rate of the optimizer
        print_lr: bool: whether to print the learning rate
    """

    vi_diag = Variational_InferenceAutoguide(
        pprogram=pprogram_y,
        make_guide_fun=AutoDiagonalNormal,
        n_steps=n_steps,
        n_samples=n_samples,
        lr=lr,
        print_lr=print_lr   
    )

    vi_multivariate_normal = Variational_InferenceAutoguide(
        pprogram=pprogram_y,
        make_guide_fun=AutoMultivariateNormal,
        n_steps=n_steps,
        n_samples=n_samples,
        lr=lr,
        print_lr=print_lr
    )
    vi_laplace = Variational_InferenceAutoguide(
        pprogram=pprogram_y,
        make_guide_fun=AutoLaplaceApproximation,
        n_steps=n_steps,
        n_samples=n_samples,
        lr=lr,
        print_lr=print_lr
    )

    vi_autoIAF = Variational_InferenceAutoguide(
        pprogram=pprogram_y,
        make_guide_fun=AutoIAFNormal,
        n_steps=n_steps,
        n_samples=n_samples,
        lr=lr,
        print_lr=print_lr
    )

    vi_autostrucured = Variational_InferenceAutoguide(
        pprogram=pprogram_y,
        make_guide_fun=AutoStructured,
        n_steps=n_steps,
        n_samples=n_samples,
        lr=lr,
        print_lr=print_lr
    )

    model_list = [
        vi_diag,
        vi_multivariate_normal,
        vi_laplace,
        vi_autoIAF,
        vi_autostrucured
    ]

    return model_list


def make_MAP_model(
        pprogram_y
        ):
    """
    Create the MAP model.
    Args:
        pprogram_y: a probabilistic program for the response variable
    """
    vi_MAP = Variational_InferenceAutoguide(
        pprogram=pprogram_y,
        make_guide_fun=AutoDelta,
        n_steps=5000,
        n_samples=1,
        lr=1e-1
    )
    return vi_MAP