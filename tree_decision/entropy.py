import numpy as np


def entropy(y: np.ndarray) -> float:
    """
    Objetivo:
    ----------
    Calcular la entropía de un conjunto de valores, una medida del grado de desorden o incertidumbre
    en la distribución de clases.

    Parámetros:
    -----------
    - y (np.ndarray): Vector unidimensional que contiene las etiquetas o valores categóricos del conjunto de datos.

    Retorna:
    --------
    - float: Valor de la entropía, en base 2, del conjunto `y`.

    Detalles del cálculo:
    ---------------------
    1. Se obtienen los valores únicos y sus frecuencias en `y`.
    2. Se calculan las probabilidades relativas de cada valor único.
    3. Se asegura que no haya probabilidades de valor cero usando `np.clip()` para evitar errores numéricos.
    4. Se aplica la fórmula de entropía de Shannon.
    """

    unique_values, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    probabilities = np.clip(probabilities, 1e-10, None)
    return -np.sum(probabilities * np.log2(probabilities))
