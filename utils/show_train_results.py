import matplotlib.pyplot as plt

def show_train_results(history):
    """
    Genera dos gráficos en una misma ventana:
    1. Loss vs Val_loss
    2. Accuracy vs Val_accuracy
    
    Parámetros:
        history (keras.callbacks.History): Objeto retornado por model.fit().
    """
    # Obtener las métricas del historial
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    
    # Crear figura con dos subplots
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Loss vs Val_loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Subplot 2: Accuracy vs Val_accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Ajustar layout y mostrar
    plt.tight_layout()
    plt.show()
