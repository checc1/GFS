import numpy as np


def entropy(X: np.ndarray, bins: int) -> float:
    """
    Compute the Shannon entropy of a variable X.
    :param X: (np.ndarray) The input variable;
    :return: (float) The Shannon entropy of the input variable.
    """
    binned_dist = np.histogram(X, bins)[0]
    probs = binned_dist / np.sum(binned_dist)
    probs = probs[np.nonzero(probs)]
    entropy_ = - np.sum(probs * np.log(probs))
    return entropy_


def joint_entropy(X: np.ndarray, Y: np.ndarray, bins: int) -> float:
    """
    Compute the joint Shannon entropy between two variables.
    Each variable could be two features expressed as arrays or a feature and a label.
    :param X: (np.ndarray) The first input variable (usually feature);
    :param Y: (np.ndarray) The second input variable (feature or label);
    :return: (float) The joint Shannon entropy of the two variables.
    """
    binned_dist = np.histogram2d(X, Y, bins)[0]
    probs = binned_dist / np.sum(binned_dist)
    probs = probs[np.nonzero(probs)]
    joint_e = - np.sum(probs * np.log(probs))
    return joint_e


def conditional_joint_entropy(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, bins: int) -> float:
    """
    Compute the joint Shannon entropy between three variables.
    :param X: (np.ndarray) The first input variable;
    :param Y: (np.ndarray) The second input variable;
    :param Z: (np.ndarray) The third input variable;
    :param bins: (int) The number of bins for discretization.
    :return: (float) The joint Shannon entropy of the three variables.
    """
    binned_dist = np.histogramdd((X, Y, Z), bins=bins)[0]
    probs = binned_dist / np.sum(binned_dist)
    probs = probs[np.nonzero(probs)]
    joint_e = - np.sum(probs * np.log(probs))
    return joint_e


def conditional_mutual_inf(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, bins: int) -> float:
    """
    Compute the conditional mutual information I(X;Y | Z).
    :param X: (np.ndarray) The first input variable;
    :param Y: (np.ndarray) The second input variable;
    :param Z: (np.ndarray) The conditioning variable;
    :param bins: (int) The number of bins for discretization.
    :return: (float) The conditional mutual information I(X;Y | Z).
    """
    HXZ = joint_entropy(X, Z, bins)
    HYZ = joint_entropy(Y, Z, bins)
    HXYZ = conditional_joint_entropy(X, Y, Z, bins)

    return HXZ + HYZ - HXYZ


def mutual_info(X: np.ndarray, Y: np.ndarray, bins: int) -> float:
    """
    Compute the mutual information between two input variables.
    :param X: (np.ndarray) The first input variable (usually feature);
    :param Y: (np.ndarray) The second input variable (feature or label);
    :return: The mutual information between the two variables.
    """
    HX = entropy(X, bins)
    HY = entropy(Y, bins)
    HXHY = joint_entropy(X, Y, bins)
    H = HX + HY - HXHY
    return H


def threeway_mutual_info(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, bins: int) -> float:
    """
    Compute the three-way mutual information I(X;Y;Z).
    :param X: (np.ndarray) The first input variable;
    :param Y: (np.ndarray) The second input variable;
    :param Z: (np.ndarray) The third input variable;
    :param bins: (int) The number of bins for discretization.
    :return: (float) The three-way mutual information I(X;Y;Z).
    """
    HX = entropy(X, bins)
    HY = entropy(Y, bins)
    HZ = entropy(Z, bins)
    HXYZ = conditional_joint_entropy(X, Y, Z, bins)

    return HX + HY + HZ - HXYZ