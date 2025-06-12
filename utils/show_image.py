import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10

def show_image(image_matrix):
    """
    Grafica una imagen RGB de 32x32 correctamente.
    
    Parámetros:
        image_matrix (numpy.ndarray): Matriz de la imagen con forma (32, 32, 3).
    """
    if image_matrix.shape != (32, 32, 3):
        raise ValueError("La matriz debe tener forma (32, 32, 3).")

    # Asegurar que los valores estén en el rango correcto [0, 255]
    if image_matrix.dtype == np.float32 or image_matrix.dtype == np.float64:
        if np.max(image_matrix) <= 1.0:
            image_matrix = (image_matrix * 255).astype(np.uint8)
    
    # Graficar con interpolación 'nearest' para evitar distorsión
    plt.imshow(image_matrix, interpolation='nearest')
    plt.axis('off')  # Oculta los ejes
    plt.show()
