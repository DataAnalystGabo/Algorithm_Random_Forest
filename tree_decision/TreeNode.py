class TreeNode:
    def __init__(
        self, feature_index=None, threshold=None, left=None, right=None, value=None
    ):
        """
        Objetivo:
        ----------
        Representar un nodo en un Árbol de Decisión, que puede ser un nodo de decisión o una hoja.

        Parámetros:
        -----------
        - feature_index (int, opcional): Índice de la característica utilizada para dividir el nodo. Si el nodo es una hoja, este valor es `None`.
        - threshold (float, opcional): Umbral de división para la característica seleccionada. Si el nodo es una hoja, este valor es `None`.
        - left (TreeNode, opcional): Referencia al nodo hijo izquierdo. Contiene las observaciones con valores menores o iguales al umbral.
        - right (TreeNode, opcional): Referencia al nodo hijo derecho. Contiene las observaciones con valores mayores al umbral.
        - value (int o float, opcional): Valor predicho si el nodo es una hoja. En un nodo de decisión, este valor es `None`.

        Descripción:
        ------------
        - Un nodo de decisión almacena información sobre la característica y el umbral que mejor divide el conjunto de datos.
        - Un nodo hoja almacena el valor de la predicción final cuando ya no es posible o necesario dividir más.
        """

        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
