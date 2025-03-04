import numpy as np


def split_data(
    X: np.ndarray, y: np.ndarray, feature: int, threshold: float
) -> np.array:
    """
    Objetivo:
    ----------
    Dividir el dataset en dos subsets de acuerdo al valor del threshold calculado y la feature seleccionada
    por la función `find_best_split`.

    Parámetros:
    -----------
    - X (np.ndarray): Vector multidimensional que contiene las features del conjunto de datos.
    - y (np.ndarray): Vector unidimensional que contiene las etiquetas o valores categóricos del conjunto de datos.

    Retorna:
    --------
    - np.array: subconjunto de datos del vector `X` tal que para la feature seleccionada los valores son menores o iguales al threshold (`left_X`).
    - np.array: subconjunto de datos del vector `y` tal que para la feature seleccionada los valores son menores o iguales al threshold (`left_y`).
    - np.array: subconjunto de datos del vector `X` tal que para la feature seleccionada los valores superan al threshold (`right_X`).
    - np.array: subconjunto de datos del vector `y` tal que para la feature seleccionada los valores superan al threshold (`right_y`).

    Algoritmo:
    ---------------------
    1. Crea una máscara para aquellos valores que son menores o igual threshold y aquellos que lo superan.
    2. Segmenta el dataset aplicando las máscaras.
    3. Retornan lo subsets resultantes.
    """

    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold

    left_X = X[left_mask]
    left_y = y[left_mask]
    right_X = X[right_mask]
    right_y = y[right_mask]

    return left_X, left_y, right_X, right_y
