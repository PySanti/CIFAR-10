from keras.src.backend.config import max_epochs
from tensorflow import keras
from sklearn.model_selection import train_test_split
import keras_tuner as kt
from utils.model_builder import model_builder

# carga del dataset
(X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

# aplanamiento de targets
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()

# division del conjunto de datos
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, random_state=42, stratify=Y_test, test_size=.5)

# normalizacion

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0

# entrenamiento

tuner = kt.Hyperband(
    model_builder,
    objective='val_accuracy',
    factor=2,
    max_epochs=15,
    project_name="CIFAR-10",
    directory="train_results"
)

tuner.search(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    verbose=2
)


# Mejor modelo encontrado
best_model = tuner.get_best_models(num_models=1)[0]

# Resumen de los mejores hiperparámetros
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]



print(f"""
    Mejor combinacion de hiperparametros

        {best_hps.values}
""")


early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_precision',   # Métrica a monitorear (puede ser 'val_accuracy')
    patience=15,          # Número de épocas sin mejora antes de detener
    restore_best_weights=True , # Restaura los pesos del mejor modelo
    mode="max",
    verbose=1
)


history = best_model.fit(
    X_train, Y_train,
    epochs=30,
    validation_data=(X_val, Y_val),
    callbacks=[early_stopping]
)

test_loss, test_accuracy = best_model.evaluate()
