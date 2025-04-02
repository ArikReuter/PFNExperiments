import sklearn.ensemble
from sklearn.neural_network import MLPClassifier
import torch 
import sklearn 
from pytabkit import RealMLP_TD_Classifier


def compare_samples_classifier_based(P: torch.tensor, 
                                     Q:torch.tensor, 
                                     used_model: sklearn.base.BaseEstimator = sklearn.ensemble.RandomForestClassifier(),
                                     n_folds = 5,
                                     balance_classes = True) -> dict:
    """
    A function that compares two samples from two distributions using a classifier
    Args:
        P: torch.tensor: the samples from the first distribution
        Q: torch.tensor: the samples from the second distribution
        used_model: sklearn.base.BaseEstimator: the classifier to use
        n_folds: int: the number of folds to use in the cross validation
        balance_classes: bool: whether to use as much samples from one class as from the other
    Returns:
        float: the ROC AUC score of the classifier
    """
    try:
        n = P.shape[0]
        m = Q.shape[0]

        if balance_classes:
            n = min(n, m)
            m = n
            P = P[:n]
            Q = Q[:m]

        X = torch.cat([P, Q], dim = 0)
        y = torch.cat([torch.zeros(n), torch.ones(m)]).numpy()

        # shuffle the data
        perm = torch.randperm(X.shape[0])
        X = X[perm]
        y = y[perm]

        cv = sklearn.model_selection.StratifiedKFold(n_splits = n_folds)

        roc_auc_scores = []
        accuracy_scores = []

        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train = X_train.detach().cpu().numpy()
            X_test = X_test.detach().cpu().numpy()

            used_model.fit(X_train, y_train)

            y_pred = used_model.predict(X_test)
            roc_auc_scores.append(sklearn.metrics.roc_auc_score(y_test, y_pred))
            accuracy_scores.append(sklearn.metrics.accuracy_score(y_test, y_pred))
        
        roc = torch.tensor(roc_auc_scores).mean().item()
        res = {"cst_roc_auc": roc}

    except Exception as e:
        print(f"An error occured in compare_samples_classifier_based: {e}")
        res = {"cst_roc_auc": torch.nan}
    return res



def compare_samples_classifier_based_RF(
    P: torch.tensor, 
    Q:torch.tensor, 
    n_folds = 10,
    balance_classes = True) -> dict:
    """
    A function that compares two samples from two distributions using a classifier
    Args:
        P: torch.tensor: the samples from the first distribution
        Q: torch.tensor: the samples from the second distribution
        n_folds: int: the number of folds to use in the cross validation
        balance_classes: bool: whether to use as much samples from one class as from the other
    Returns:
        float: the ROC AUC score of the classifier
    """
    print("Run RF")
    clf = sklearn.ensemble.RandomForestClassifier()
    res = compare_samples_classifier_based(P, Q, clf, n_folds, balance_classes = balance_classes)
    return {"roc_RF": res["cst_roc_auc"]}


def compare_samples_classifier_based_NN_Sklearn_Lueck(
    P: torch.tensor, 
    Q:torch.tensor, 
    n_folds = 10,
    balance_classes = True) -> dict:
    """
    A function that compares two samples from two distributions using a classifier
    Args:
        P: torch.tensor: the samples from the first distribution
        Q: torch.tensor: the samples from the second distribution
        n_folds: int: the number of folds to use in the cross validation
        balance_classes: bool: whether to use as much samples from one class as from the other
    Returns:
        float: the ROC AUC score of the classifier
    """
    n_features = P.shape[1]
    hidden_layer_size = n_features * 10

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * hidden_layer_size, 10 * hidden_layer_size),
        max_iter=10000,
        solver="adam",
    )

    print("Run Sklearn MLP")

    res = compare_samples_classifier_based(P, Q, clf, n_folds, balance_classes = balance_classes)
    return {"roc_NN_Sklearn_Lueck": res["cst_roc_auc"]}


def compare_samples_classifier_based_NN_RealMLP(
    P: torch.tensor, 
    Q:torch.tensor, 
    n_folds = 10,
    balance_classes = True) -> dict:
    """
    A function that compares two samples from two distributions using a classifier
    Args:
        P: torch.tensor: the samples from the first distribution
        Q: torch.tensor: the samples from the second distribution
        n_folds: int: the number of folds to use in the cross validation
        balance_classes: bool: whether to use as much samples from one class as from the other
    Returns:
        float: the ROC AUC score of the classifier
    """

    clf = RealMLP_TD_Classifier() 

    print("Run RealMLP")

    res = compare_samples_classifier_based(P, Q, clf, n_folds, balance_classes = balance_classes)
    return {"roc_NN_RealMLP": res["cst_roc_auc"]}