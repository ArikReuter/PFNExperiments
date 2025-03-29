import torch 

def MSELoss_unsqueezed(input: list[torch.Tensor], target: torch.tensor) -> torch.tensor:
    """
    Compute MSE loss between input and target where the input is unsqueezed
    Args:
        input: torch.tensor: the input tensor
        target: torch.tensor: the target tensor
    Returns:
        torch.tensor: the MSE loss
    """
    input = input[0].squeeze()
    target = target.squeeze()

    assert input.shape == target.shape, f"input shape {input.shape} does not match target shape {target.shape}"

    mse  = torch.nn.MSELoss()(input, target)

    return mse

def MSELoss_first_arg(input: list, target: torch.tensor) -> torch.tensor:
    """
    Compute the MSE loss between the first argument of the input and the target
    Args:
        input: list: the input list
        target: torch.tensor: the target tensor
    """
    pred = input[0]

    assert pred.shape == target.shape, f"pred shape {pred.shape} does not match target shape {target.shape}"

    mse = torch.nn.MSELoss()(pred, target)

    return mse


def nll_loss_full_gaussian(pred, target):
  """
  NLL based on a full covariance matrix
  Args: 
    pred: tuple: the prediction tuple, contains the mean, covariance factor and diagonal for covariance matrix
    target: torch.tensor: the target tensor
  """

  if type(target) == list:
        assert len(target) == 1, "The target is a list but the length is not 1"
        target = target[0]

  mu, cov_factor, cov_diag = pred  

  cov_factor = cov_factor.reshape(mu.shape[0], mu.shape[1], -1)
  cov_diag = cov_diag **2 + 1e-9
  dist = torch.distributions.LowRankMultivariateNormal(
      loc = mu,
      cov_factor = cov_factor,
      cov_diag = cov_diag
      )

  nll = - dist.log_prob(target)
  nll = torch.mean(nll)

  return nll


def nll_loss_full_gaussian_flattened_pred(pred, target):
  """
  NLL based on a full covariance matrix
  Args: 
    pred: long vector: the prediction vector, contains the mean, covariance factor and diagonal for covariance matrix. Has shape (batch_size, 2p + p*p)
    target: torch.tensor: the target tensor. Has shape (batch_size, p) 
  """
  batch_size = pred.shape[0]
  assert target.shape[0] == batch_size, f"pred shape {pred.shape} does not match target shape {target.shape}"
  p = target.shape[1]
  assert pred.shape[1] == 2*p + p*p, f"pred shape {pred.shape} does not match target shape {target.shape}"

  mean = pred[:, :p]
  cov_diag = pred[:, p:2*p]
  cov_factor = pred[:, 2*p:]
  

  cov_factor = cov_factor.reshape(batch_size, p, p)
  cov_diag = cov_diag **2 + 1e-5
  dist = torch.distributions.LowRankMultivariateNormal(
      loc = mean,
      cov_factor = cov_factor,
      cov_diag = cov_diag
      )

  nll = - dist.log_prob(target)
  nll = torch.mean(nll)

  return nll