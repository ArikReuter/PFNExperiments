import torch 

def simulate_X_uniform(n:int, p:int, batch_size:int = 0) -> torch.tensor:
    """
    Simulate covariates from a uniform distribution on the unit interval
    Args:
        n: int: the number of observations
        p: int: the number of covariates
        batch_size: int: the batch size

    Returns:
        torch.tensor: the covariates of shape (n, p) or (batch_size, n, p)
    """

    if batch_size == 0:
        return torch.rand(n, p)
    else:
        return torch.rand(batch_size, n, p)
    

def make_simulate_X_by_loading(
        X_stored: torch.tensor,
        repeat_samples: bool = True
) -> callable:
    """
    Make a function that generates the covariates by loading them from a tensor
    Args:
    X_stored: torch.tensor: the tensor to load the covariates from
    repeat_samples: bool: whether to repeat the samples or not
    """

    def simulate_X_loading(n:int, p:int, batch_size:int = 0) -> torch.tensor:
        """
        Generate the covariates by loading them from a tensor.
        
        Args:
            n: int: the number of observations
            p: int: the number of covariates
            batch_size: int: the batch size (default: 0)

        Returns:
            torch.tensor: the covariates of shape (n, p) or (batch_size, n, p)
        """
        if batch_size == 0:
            batch_size = 1

        if repeat_samples:
            random_indices = torch.randint(0, X_stored.shape[0], (batch_size,))
        else:
            random_indices = torch.randperm(X_stored.shape[0])[:batch_size]
        X = X_stored[random_indices]

        #assert X.shape == (batch_size, n, p), f"X.shape = {X.shape}, expected shape = {(batch_size, n, p)}"

        assert len(X.shape) == 3, f"X.shape = {X.shape}, expected shape = {(batch_size, n, p)}" 

        assert X.shape[1] >= n, f"X.shape = {X.shape}, expected shape = {(batch_size, n, p)}"
        assert X.shape[2] >= p, f"X.shape = {X.shape}, expected shape = {(batch_size, n, p)}"
        assert X.shape[0] >= batch_size, f"X.shape = {X.shape}, expected shape = {(batch_size, n, p)}" 

        X = X[:, :n, :p]



        if batch_size == 1:
            return X.squeeze(0)
        else:
            return X
        
    return simulate_X_loading


def make_simulate_X_by_loading_split(
        X_stored: torch.tensor,
        repeat_samples: bool = True,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        shuffle_X_stored: bool = True
) -> callable:
    """
    Make three functions that generate the covariates by loading them from a tensor split into training, validation, and test sets
    Args:
    X_stored: torch.tensor: the tensor to load the covariates from
    repeat_samples: bool: whether to repeat the samples or not
    train_size: float: the proportion of the training set
    val_size: float: the proportion of the validation set
    test_size: float: the proportion of the test set
    shuffle_X_stored: bool: whether to shuffle the tensor before splitting it or not
    """
    assert train_size + val_size + test_size == 1, "The sum of train_size, val_size, and test_size must be 1"

    if shuffle_X_stored:
        indices = torch.randperm(X_stored.shape[0])
        X_stored = X_stored[indices]

    train_size = int(train_size * X_stored.shape[0])
    val_size = int(val_size * X_stored.shape[0])
    test_size = int(test_size * X_stored.shape[0])

    X_train = X_stored[:train_size]
    X_val = X_stored[train_size:train_size + val_size]
    X_test = X_stored[train_size + val_size:]

    simulate_X_train = make_simulate_X_by_loading(X_train, repeat_samples)
    simulate_X_val = make_simulate_X_by_loading(X_val, repeat_samples)
    simulate_X_test = make_simulate_X_by_loading(X_test, repeat_samples)

    return simulate_X_train, simulate_X_val, simulate_X_test


def simulate_X_uniform_discretized(
        n:int,
        p:int,
        batch_size:int = 1,
        
):
    """
    Simulate covariates from a uniform distribution on the unit interval tha are discretized
    Args:
        n: int: the number of observations
        p: int: the number of covariates
        batch_size: int: the batch size
        n_diff_values_dist: torch.distributions.Categorical: the distribution of the number of different values
    """

    probs = torch.rand(batch_size, p, n)

    if n == 100:
        n1 = int(n*0.2)
        n2 = n - n1 - 1
        prob_vec = [0]*n1 + [1]*n2 + [10]
        prob_vec = torch.tensor(prob_vec)
    

        print(prob_vec)

    else:
        n1 = int(n*0.5)
        n2 = n - n1 - 1
        prob_vec = [0]*n1 + [1]*n2 + [10]
        prob_vec = torch.tensor(prob_vec)

    n_diff_values_dist = torch.distributions.Categorical(prob_vec)

    #randints = torch.distributions.Categorical(probs).sample((n,)).permute(1,0,2)
    

    randints = torch.randint(0, 99, (batch_size, n, p)) + 1
    # randomly permute the integers along the n dimension



    n_diff_values = n_diff_values_dist.sample((batch_size, p)) + 1


    new_randints = randints % n_diff_values.unsqueeze(1) #shape (batch_size, n, p)

    R = torch.rand(batch_size, n, p)

    # Create the indices for the batch dimension and last dimension
    batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).expand(batch_size, n, p)
    last_indices = torch.arange(p).unsqueeze(0).unsqueeze(0).expand(batch_size, n, p)

    # Use advanced indexing to map the values
    P = R[batch_indices, new_randints, last_indices]

    return P, new_randints



def simulate_X_uniform_discretized2(
        n:int,
        p:int,
        batch_size:int = 1,
        
):
    """
    Simulate covariates from a uniform distribution on the unit interval tha are discretized
    Args:
        n: int: the number of observations
        p: int: the number of covariates
        batch_size: int: the batch size
        n_diff_values_dist: torch.distributions.Categorical: the distribution of the number of different values
    """

    if n == 100:
        n1 = int(n*0.2)
        n2 = n - n1 -1
        prob_vec = [0]*n1 + [1]*n2 + [10]
        prob_vec = torch.tensor(prob_vec)
    

        print(len(prob_vec))

    else:
        n1 = int(n*0.5)
        n2 = n - n1 -1
        prob_vec = [0]*n1 + [1]*n2 + [10]
        prob_vec = torch.tensor(prob_vec)


    n_diff_values_dist = torch.distributions.Categorical(prob_vec)

    res = torch.rand(batch_size, n, p)

    for i in range(batch_size):
        for j in range(p):
            n_diff_values = n_diff_values_dist.sample() +1

            randperm = torch.randperm(n) % n_diff_values

            res[i, :, j] = res[i, randperm, j]

    return res


def simulate_X_uniform_discretized3(n: int, p: int, batch_size: int = 1):
    """
    Simulate covariates from a uniform distribution on the unit interval that are discretized.
    
    Args:
        n: int: the number of observations
        p: int: the number of covariates
        batch_size: int: the batch size
    """
    
    if n == 100:
        n1 = int(n * 0.2)
        n2 = n - n1 - 1
        prob_vec = [0] * n1 + [1] * n2 + [10]
        prob_vec = torch.tensor(prob_vec)
    else:
        n1 = int(n * 0.5)
        n2 = n - n1 - 1
        prob_vec = [0] * n1 + [1] * n2 + [10]
        prob_vec = torch.tensor(prob_vec)
    
    n_diff_values_dist = torch.distributions.Categorical(prob_vec)
    
    res = torch.rand(batch_size, n, p)
    
    n_diff_values_samples = n_diff_values_dist.sample((batch_size, p)) + 1
    randperm = torch.randperm(n).unsqueeze(0).unsqueeze(2) % n_diff_values_samples.unsqueeze(1)
    
    res = res.scatter(1, randperm.expand(batch_size, n, p), res.clone())
    
    return res


def simulate_X_uniform_discretized4(n: int, p: int, batch_size: int = 1):
    """
    Simulate covariates from a uniform distribution on the unit interval that are discretized.
    
    Args:
        n: int: the number of observations
        p: int: the number of covariates
        batch_size: int: the batch size
    """
    
    if n == 100:
        n1 = int(n * 0.2)
        n2 = n - n1 - 1
        prob_vec = [0] * n1 + [1] * n2 + [10]
        prob_vec = torch.tensor(prob_vec)
    else:
        n1 = int(n * 0.5)
        n2 = n - n1 - 1
        prob_vec = [0] * n1 + [1] * n2 + [10]
        prob_vec = torch.tensor(prob_vec)
    
    n_diff_values_dist = torch.distributions.Categorical(prob_vec)
    
    datasets = torch.rand(batch_size, n, p)
    

    bs, N, P = datasets.shape
    
    # Sample the number of bins for each dataset and each feature
    n_bins = 10* n_diff_values_dist.sample((bs, P)).int() + 2  # Ensure at least 2 bins for meaningful discretization

    # Create an array to hold the discretized datasets
    discretized_datasets = torch.zeros_like(datasets)

    for i in range(bs):
        for j in range(P):
            # Sample border points and sort them to create bins
            bins = torch.sort(torch.rand(n_bins[i, j] - 1))[0]
            bins = torch.cat((torch.tensor([0.0]), bins, torch.tensor([1.0])))
            
            # Digitize the data points according to the bins
            digitized = torch.bucketize(datasets[i, :, j], bins, right=True) - 1

            # Map the digitized data points back to the bin midpoints
            bin_mids = (bins[:-1] + bins[1:]) / 2.0
            discretized_datasets[i, :, j] = bin_mids[digitized]
    
    return discretized_datasets
    