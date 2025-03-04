import numpy as np


def entropy(feature_values: np.ndarray) -> float:
    """
    Objetivo:
    ----------
    Calcular la entropía de una característica del dataset; una medida del grado de desorden o incertidumbre
    en la distribución de clases.

    Parámetros:
    -----------
    - `feature_values` (np.ndarray, requerido): Vector unidimensional que contiene los valores de la categoría (variable independiente).

    Retorna:
    --------
    - float: Valor de la entropía, en base 2, del conjunto `feature_values`.

    Algoritmo:
    ---------------------
    1. Obtiene los valores únicos y sus frecuencias en `feature_values`.
    2. Calcula las probabilidades relativas de cada valor único.
    3. Audita que no haya probabilidades de valor cero usando `np.clip()` para evitar errores numéricos.
    4. Aplica la fórmula de entropía de Shannon.
    5. Retorna el valor de la entropía.
    """

    unique_values, counts = np.unique(feature_values, return_counts=True)
    probabilities = counts / len(feature_values)
    probabilities = np.clip(probabilities, 1e-10, None)
    return -np.sum(probabilities * np.log2(probabilities))
