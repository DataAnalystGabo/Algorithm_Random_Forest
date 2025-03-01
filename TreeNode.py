class TreeNode:
    def __init__(
        self, feature_index=None, threshold=None, left=None, right=None, value=None
    ):
        """
        Nodo de un Árbol de decisión.

        Parámetros:
        - feature_index: índice de la variable utilizada para dividir el nodo.
        - threshold: valor umbral para la división.
        - left: nodo hijo izquierdo.
        - right: nodo hijo derecho.
        - value: clase predicha si el nodo es hoja.

        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
