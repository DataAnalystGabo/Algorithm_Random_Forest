"""
    Objetivo:
    ----------
    Determinar la feature (característica) y el threshold (umbral) que generan la división más homogénea 
    posible dentro del dataset, utilizando el criterio de entropía.

    Pasos del algoritmo:
    ---------------------
    1. Inicializa `best_feature`, `best_threshold` y `best_entropy` con valores nulos o infinitos.
    2. Itera sobre cada característica del conjunto de datos `X`.
    3. Ordena los valores únicos de la característica actual para evaluar posibles umbrales.
    4. Para cada par consecutivo de valores ordenados, calcula el umbral como el promedio entre ambos.
    5. Divide `y` en dos subconjuntos:
        - `left_y`: Contiene las etiquetas donde la característica es menor o igual al umbral.
        - `right_y`: Contiene las etiquetas donde la característica es mayor al umbral.
    6. Calcula la entropía de ambos subconjuntos (`left_y` y `right_y`).
    7. Calcula la entropía total ponderada en función del tamaño relativo de cada subconjunto.
    8. Si la entropía total calculada es menor que la mejor encontrada hasta el momento, 
    actualiza `best_feature`, `best_threshold` y `best_entropy`.
    9. Devuelve la mejor característica (`best_feature`), el mejor umbral (`best_threshold`) y la 
    entropía mínima (`best_entropy`).
"""

import numpy as np
from tree_decision.entropy import entropy


def find_best_split(X, y) -> float:
    best_feature = None
    best_threshold = None
    best_entropy = float("inf")

    for feature in range(X.shape[1]):
        sorted_values = np.sort(X[:, feature])
        for i in range(1, len(sorted_values)):
            threshold = (sorted_values[i - 1] + sorted_values[i]) / 2

            left_y = y[X[:, feature] <= threshold]
            right_y = y[X[:, feature] > threshold]
            entropy_left = entropy(left_y)
            entropy_right = entropy(right_y)
            total_entropy = (len(left_y) / len(y)) * entropy_left + (
                len(right_y) / len(y)
            ) * entropy_right

            if total_entropy < best_entropy:
                best_feature = feature
                best_threshold = threshold
                best_entropy = total_entropy

    return best_feature, best_threshold, best_entropy
