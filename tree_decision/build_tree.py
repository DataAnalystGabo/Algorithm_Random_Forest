import numpy as np
from tree_decision.TreeNode import TreeNode
from tree_decision.find_best_split import find_best_split
from tree_decision.split_data import split_data


def build_tree(X: np.ndarray, y: np.ndarray, depth=0, max_depth=5) -> TreeNode:
    """
    Objetivo:
    ----------
    Construir un árbol de decisión de manera recursiva dividiendo los datos en función de la mejor característica y el mejor umbral, hasta que se alcance una condición de parada (pureza de clase o profundidad máxima).

    Parámetros:
    -----------
    - `X` (np.ndarray): Matriz de características de entrada, con forma (n_samples, n_features).
    - `y` (np.ndarray): Vector de etiquetas correspondiente a las muestras de X, con forma (n_samples,).
    - `depth` (int, opcional): Profundidad actual del árbol durante la construcción recursiva (por defecto es 0).
    - `max_depth` (int, opcional): Profundidad máxima permitida para evitar sobreajuste del árbol (por defecto es 5).

    Retorna:
    --------
    TreeNode: Un nodo del árbol de decisión que representa una división de los datos y contiene los nodos hijos si es un nodo interno, o la predicción final si es un nodo hoja.

    Algoritmo:
    ---------------------
    1. Verifica si todos los elementos en `y` son de la misma clase o si se ha alcanzado la profundidad máxima. Si es así, crea un nodo hoja que prediga la clase mayoritaria de y.
    2. Llama a la función `find_best_split(X, y)` para encontrar la mejor característica y umbral para dividir los datos.
    3. Usa el umbral y la característica seleccionados para dividir los datos en dos subconjuntos usando `split_data(X, y, best_feature, best_threshold)`.
    4. Llama recursivamente a `build_tree` para los subconjuntos izquierdo y derecho, incrementando la profundidad de la llamada.
    """

    if len(set(y)) == 1 or depth == max_depth:
        return TreeNode(value=max(set(y), key=list(y).count))

    best_feature, best_threshold, _ = find_best_split(X, y)

    left_X, left_y, right_X, right_y = split_data(X, y, best_feature, best_threshold)

    left_node = build_tree(left_X, left_y, depth + 1, max_depth)
    right_node = build_tree(right_X, right_y, depth + 1, max_depth)

    return TreeNode(
        feature_index=best_feature,
        threshold=best_threshold,
        left=left_node,
        right=right_node,
    )
