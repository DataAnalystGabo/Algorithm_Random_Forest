import numpy as np


def entropy(y: np.ndarray) -> float:
    """
    Calcula la entropía de un conjunto de variables.

    Parámetros:
    - y: vector de variables.

    """

    unique_values, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    probabilities = np.clip(probabilities, 1e-10, None)
    return -np.sum(probabilities * np.log2(probabilities))
